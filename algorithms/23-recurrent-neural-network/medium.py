# PyTorch nn.RNN: forecast tomorrow's oil rate for one well from the last 14 days
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

# ---- Synthetic 3 years of daily production: hyperbolic decline + noise + shut-ins ----
days = np.arange(1095)
qi, Di, b = 900.0, 0.004, 0.8
# 🧠 ML: Arps hyperbolic decline q(t) = qi / (1 + b*Di*t)^(1/b) - the classic petroleum engineering curve
base = qi / (1 + b * Di * days) ** (1 / b)
noise = np.random.normal(0, 0.04, len(days)) * base
oil = base + noise
# Random shut-ins (workovers, pump failures): rate drops to ~0 for 2-5 days
for start in np.random.choice(len(days) - 6, 12, replace=False):
    oil[start:start + np.random.randint(2, 6)] = np.random.uniform(0, 20)
# 🐍 PY: np.clip keeps values in a valid range (no negative barrels)
df = pd.DataFrame({"day": days, "oil_bbl_d": np.clip(oil, 0, None)})
print(df.describe().loc[["mean", "min", "max"]].round(1).T)

# 🧠 ML: scale using the TRAIN period only, to avoid leaking future information
SPLIT = int(len(df) * 0.8)
scale = df.oil_bbl_d[:SPLIT].max()
series = (df.oil_bbl_d / scale).to_numpy(dtype=np.float32)

WINDOW = 14


def make_windows(arr, window):
    # 🧠 ML: sliding windows turn a time series into (last 14 days -> next day) training examples
    # 🐍 PY: list comprehension + np.stack builds a 2-D array of overlapping slices
    X = np.stack([arr[i:i + window] for i in range(len(arr) - window)])
    y = arr[window:]
    return X, y


X, y = make_windows(series, WINDOW)
split_idx = SPLIT - WINDOW
# 🧠 ML: time-ordered split - train on the past, test on the future (never shuffle across time)
# 🐍 PY: [..., None] adds a trailing feature dimension: shape becomes (samples, 14, 1)
X_tr, X_te = torch.tensor(X[:split_idx, :, None]), torch.tensor(X[split_idx:, :, None])
y_tr, y_te = torch.tensor(y[:split_idx]), torch.tensor(y[split_idx:])
print(f"\nTrain windows: {len(X_tr)}  Test windows: {len(X_te)}")


class OilRNN(nn.Module):
    # 🐍 PY: a class that inherits from nn.Module; __init__ defines layers, forward defines the math
    def __init__(self, hidden=16):
        super().__init__()
        # 🧠 ML: nn.RNN runs h_t = tanh(W_x x_t + W_h h_{t-1} + b) across all 14 days for us
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, h_last = self.rnn(x)
        # 🧠 ML: predict the CHANGE from the last observed day (residual), from the 14-day summary
        # 🧠 ML: this makes "same as yesterday" the starting point the RNN only has to improve on
        return x[:, -1, 0] + self.head(h_last[-1]).squeeze(-1)


model = OilRNN()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
# 🧠 ML: Huber loss = MSE for small errors, MAE for big ones, so shut-in days don't dominate training
loss_fn = nn.HuberLoss(delta=0.02)
for epoch in range(1, 41):
    model.train()
    # 🐍 PY: torch.randperm shuffles window order (fine WITHIN train - each window keeps its own order)
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        # 🧠 ML: backpropagation through time - gradients flow back across all 14 steps
        loss.backward()
        # 🧠 ML: gradient clipping stops rare "exploding" updates that RNNs are prone to
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d}  train Huber loss={loss.item():.5f}")

model.eval()
# 🐍 PY: torch.no_grad() turns off gradient tracking for faster, memory-light inference
with torch.no_grad():
    pred = model(X_te).numpy() * scale
actual = y_te.numpy() * scale
naive = X_te[:, -1, 0].numpy() * scale
mae = np.mean(np.abs(pred - actual))
naive_mae = np.mean(np.abs(naive - actual))
print(f"\nTest MAE  RNN: {mae:5.1f} bbl/d   naive 'same as yesterday': {naive_mae:5.1f} bbl/d")
# 🧠 ML: shut-ins are unpredictable from rates alone; score "normal" days separately (both sides > 50 bbl/d)
normal = (actual > 50) & (naive > 50)
print(f"Normal days only ({normal.sum()}): RNN {np.mean(np.abs(pred - actual)[normal]):5.1f}"
      f"  naive {np.mean(np.abs(naive - actual)[normal]):5.1f} bbl/d")
print("\nLast 5 test days:")
for p, a in zip(pred[-5:], actual[-5:]):
    print(f"  actual {a:6.1f}  predicted {p:6.1f}")
print(f"\nBusiness read-out: mean test rate {actual.mean():.0f} bbl/d; RNN error {mae:.0f} bbl/d"
      f" ({100 * mae / actual.mean():.1f}%). For a ONE-day horizon it only ties the naive rule -")
print("so always report a naive baseline. The value of sequence models shows up in multi-day forecasts (Hard).")

# ---
# Recurrent Neural Network (RNN) — MEDIUM lab · Oil & Gas: Daily oil production rate forecasting for a well
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
