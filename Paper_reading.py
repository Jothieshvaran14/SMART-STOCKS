# ============================================================
# STABLE MULTIMODAL MODEL FOR FORECASTING
# ============================================================

import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

torch.manual_seed(42)

# ================= CONFIG =================
TICKER = "SPY"
START = "2018-01-01"
END = "2023-12-31"
LOOKBACK = 60
HORIZON = 3
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= PRICE DATA =================
df = yf.download(
    TICKER,
    start=START,
    end=END,
    auto_adjust=True,
    progress=False
)

prices = df["Close"].dropna()

# Use log returns
returns = np.log(prices / prices.shift(1)).dropna()

# ================= TARGETS =================
future_ret = returns.rolling(HORIZON).sum().shift(-HORIZON)
direction = (future_ret > 0).astype(float)

# Align
valid_idx = future_ret.dropna().index
returns = returns.loc[valid_idx]
future_ret = future_ret.loc[valid_idx]
direction = direction.loc[valid_idx]

returns = returns.values.reshape(-1, 1)
future_ret = future_ret.values.reshape(-1)
direction = direction.values.reshape(-1)

# ================= GDELT NEWS =================
def fetch_gdelt_headlines(date):
    date_str = date.strftime("%Y%m%d")
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query=market&mode=ArtList&maxrecords=8"
        f"&format=json&startdatetime={date_str}000000&enddatetime={date_str}235959"
    )
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return []
        return [a["title"] for a in r.json().get("articles", []) if "title" in a]
    except:
        return []

# ================= FINBERT =================
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
).to(DEVICE)
finbert.eval()

def finbert_score(texts):
    if not texts:
        return 0.0
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(DEVICE)
        logits = finbert(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        return (probs[:, 2] - probs[:, 0]).mean().item()

# ================= DAILY SENTIMENT =================
sentiments = []
dates = valid_idx

for d in tqdm(dates, desc="FinBERT sentiment"):
    headlines = fetch_gdelt_headlines(d)
    sentiments.append(finbert_score(headlines))

sentiment = np.array(sentiments).reshape(-1, 1)

# ================= CREATE SEQUENCES =================
def create_sequences(R, S, Y_dir, Y_reg):
    Xp, Xs, Yd, Yr = [], [], [], []

    for i in range(len(R) - LOOKBACK):
        Xp.append(R[i:i+LOOKBACK])
        Xs.append(S[i:i+LOOKBACK])
        Yd.append(Y_dir[i+LOOKBACK])
        Yr.append(Y_reg[i+LOOKBACK])

    return (
        torch.tensor(np.array(Xp), dtype=torch.float32),
        torch.tensor(np.array(Xs), dtype=torch.float32),
        torch.tensor(np.array(Yd), dtype=torch.float32).view(-1,1),
        torch.tensor(np.array(Yr), dtype=torch.float32).view(-1,1),
    )

X_price, X_sent, Y_dir, Y_reg = create_sequences(
    returns,
    sentiment,
    direction,
    future_ret
)

# ================= TRAIN / TEST SPLIT =================
split = int(0.8 * len(X_price))

Xtr_p, Xte_p = X_price[:split], X_price[split:]
Xtr_s, Xte_s = X_sent[:split], X_sent[split:]
Yd_tr, Yd_te = Y_dir[:split], Y_dir[split:]
Yr_tr, Yr_te = Y_reg[:split], Y_reg[split:]

Xtr_p, Xte_p = Xtr_p.to(DEVICE), Xte_p.to(DEVICE)
Xtr_s, Xte_s = Xtr_s.to(DEVICE), Xte_s.to(DEVICE)
Yd_tr, Yd_te = Yd_tr.to(DEVICE), Yd_te.to(DEVICE)
Yr_tr, Yr_te = Yr_tr.to(DEVICE), Yr_te.to(DEVICE)

# ================= MODEL =================
class MultimodalDualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.price_lstm = nn.LSTM(1, 64, batch_first=True)
        self.sent_fc = nn.Linear(1, 16)
        self.shared = nn.Linear(64 + 16, 32)
        self.dir_head = nn.Linear(32, 1)
        self.reg_head = nn.Linear(32, 1)

    def forward(self, xp, xs):
        _, (h, _) = self.price_lstm(xp)
        price_feat = h[-1]

        sent_feat = torch.relu(self.sent_fc(xs.mean(dim=1)))

        fused = torch.cat([price_feat, sent_feat], dim=1)
        shared = torch.relu(self.shared(fused))

        dir_out = torch.sigmoid(self.dir_head(shared))
        reg_out = self.reg_head(shared)

        return dir_out, reg_out

model = MultimodalDualHead().to(DEVICE)

criterion_dir = nn.BCELoss()
criterion_reg = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAIN =================
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    pred_dir, pred_reg = model(Xtr_p, Xtr_s)

    loss_dir = criterion_dir(pred_dir, Yd_tr)
    loss_reg = criterion_reg(pred_reg, Yr_tr)

    loss = loss_dir + loss_reg
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

# ================= EVALUATION =================
model.eval()
with torch.no_grad():
    pred_dir, pred_reg = model(Xte_p, Xte_s)

# Direction accuracy
pred_dir_bin = (pred_dir > 0.5).float()
directional_accuracy = (pred_dir_bin == Yd_te).float().mean().item() * 100

# Regression metrics
rmse = torch.sqrt(torch.mean((pred_reg - Yr_te) ** 2)).item()

mape = torch.mean(
    torch.abs((pred_reg - Yr_te) / (Yr_te + 1e-8))
).item() * 100

print("\n===== FINAL RESULTS =====")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"RMSE (log-return): {rmse:.6f}")
print(f"MAPE (%): {mape:.2f}%")
