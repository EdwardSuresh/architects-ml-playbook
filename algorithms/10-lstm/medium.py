import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

# 🧠 ML: 4 years of weekly demand = trend + yearly seasonality + noise (one SKU at one DC)
weeks = np.arange(208)
# 🐍 PY: numpy vector maths - every week computed at once, no loop needed
units_shipped = (800 + 0.8 * weeks + 300 * np.sin(2 * np.pi * weeks / 52)
                 + np.random.normal(0, 100, len(weeks)))
print(f"Weeks of history: {len(units_shipped)}  min={units_shipped.min():.0f}  max={units_shipped.max():.0f}")

# 🧠 ML: scale using TRAIN stats only so no information leaks from the future
split = 156
mu, sd = units_shipped[:split].mean(), units_shipped[:split].std()
scaled = (units_shipped - mu) / sd

# 🧠 ML: sliding window - use the last 12 weeks to predict the next week
WINDOW = 12
def make_windows(series, window):
    # 🐍 PY: list comprehension of slices; series[i:i+window] is 12 weeks, series[i+window] is the target
    X = [series[i:i + window] for i in range(len(series) - window)]
    y = [series[i + window] for i in range(len(series) - window)]
    # 🐍 PY: np.array(...)[..., None] adds a feature axis -> shape (samples, 12, 1)
    return torch.tensor(np.array(X)[..., None], dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = make_windows(scaled, WINDOW)
# 🧠 ML: time-ordered split - train on the past, test on the most recent year (never shuffle across time)
n_train = split - WINDOW
X_tr, y_tr, X_te, y_te = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
print(f"Train windows={len(X_tr)}  Test windows={len(X_te)}")

# 🐍 PY: a subclass of nn.Module; layers are attributes, forward() defines the data flow
class DemandLSTM(nn.Module):
    def __init__(self, hidden=24):
        super().__init__()
        # 🧠 ML: batch_first=True means input shape is (batch, time steps, features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        seq_out, _ = self.lstm(x)
        # 🧠 ML: use only the LAST time step's hidden state as the summary of the 12 weeks
        return self.out(seq_out[:, -1, :]).squeeze(-1)

model = DemandLSTM()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
for epoch in range(1, 201):
    opt.zero_grad()
    loss = loss_fn(model(X_tr), y_tr)
    loss.backward()
    opt.step()
    # 🐍 PY: the % (modulo) operator prints progress every 40 epochs
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  train MSE={loss.item():.4f}")

model.eval()
with torch.no_grad():
    pred = model(X_te).numpy() * sd + mu
actual = y_te.numpy() * sd + mu
# 🧠 ML: naive baseline = "same as last week"; a model must beat this to be worth running
naive = X_te[:, -1, 0].numpy() * sd + mu

def mape(a, p):
    return np.mean(np.abs(a - p) / a) * 100

print(f"\nTest MAPE  LSTM={mape(actual, pred):.1f}%   naive last-week={mape(actual, naive):.1f}%")
print("\nweek  actual  lstm  naive")
# 🐍 PY: zip with range lets us print the first 8 test weeks side by side
for w, a, p, nv in zip(range(split, split + 8), actual, pred, naive):
    print(f"{w:4d} {a:7.0f} {p:5.0f} {nv:6.0f}")
print(f"\nPlanner view: average absolute error = {np.mean(np.abs(actual - pred)):.0f} cases/week")

# ---
# LSTM — MEDIUM lab · Supply Chain: Weekly SKU demand forecasting for a distribution center
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
