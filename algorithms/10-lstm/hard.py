import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
torch.manual_seed(42)
SEED, WINDOW, HORIZON = 42, 26, 4
FEATS = ["units", "promo", "price", "week_sin", "week_cos"]


def make_history(n_weeks=260, seed=SEED):
    """5 years of weekly demand for one SKU at one DC, with promotions and price changes."""
    rng = np.random.default_rng(seed)
    w = np.arange(n_weeks)
    # 🐍 PY: (array < 0.15).astype(int) turns True/False into 1/0 promo flags
    promo = (rng.random(n_weeks) < 0.15).astype(int)
    # 🧠 ML: promos cut price by ~2 dollars, so price and promo are correlated inputs
    price = np.round(12 - 2.0 * promo + rng.normal(0, 0.3, n_weeks), 2)
    season = 250 * np.sin(2 * np.pi * w / 52) + 120 * (w % 52 >= 47)    # holiday peak
    # 🧠 ML: promo lifts demand ~35% and price elasticity lowers it - the LSTM must learn both
    units = (900 + 0.8 * w + season) * (1 + 0.35 * promo) * (price / 12) ** -1.2
    units = units + rng.normal(0, 70, n_weeks)
    df = pd.DataFrame({"week": w, "units": units.round(), "promo": promo, "price": price})
    # 🧠 ML: sine/cosine of week-of-year lets the model know where it is in the seasonal cycle
    df["week_sin"] = np.sin(2 * np.pi * w / 52)
    df["week_cos"] = np.cos(2 * np.pi * w / 52)
    return df


def windows(arr, future_promo, start, end):
    """X = last WINDOW weeks of all features + KNOWN future promo flags; y = next HORIZON weeks."""
    X, y = [], []
    for t in range(max(start, WINDOW), end - HORIZON + 1):
        # 🐍 PY: slice [t - WINDOW:t] grabs the 26 rows before week t, all feature columns
        past = arr[t - WINDOW:t]
        # 🧠 ML: planned promos for the next 4 weeks are known in advance - feed them in
        fut = np.repeat(future_promo[t:t + HORIZON][None, :], WINDOW, axis=0)
        X.append(np.hstack([past, fut]))
        # 🧠 ML: target = the next 4 weeks of scaled units (column 0)
        y.append(arr[t:t + HORIZON, 0])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


class Forecaster(nn.Module):
    def __init__(self, n_in, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        # 🧠 ML: direct multi-step - one head outputs all 4 future weeks at once (no error feedback loop)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


def fit(Xtr, ytr, Xva, yva, hidden, lr, epochs=150):
    # 🧠 ML: reseeding makes every grid candidate start from comparable random weights
    torch.manual_seed(SEED)
    m = Forecaster(Xtr.shape[2], hidden, dropout=0.1)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, state = 1e9, None
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        # 🧠 ML: full-batch training is fine here - only ~150 windows
        nn.functional.mse_loss(m(Xtr), ytr).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(m(Xva), yva).item()
        # 🧠 ML: keep weights from the best validation epoch (early stopping)
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    return m, best


# 🧠 ML: MAPE = average absolute % miss; planners understand it, but it explodes near zero demand
def mape(a, p):
    return float(np.mean(np.abs(a - p) / np.abs(a)) * 100)


df = make_history()
n = len(df)
# 🧠 ML: time-ordered split: train weeks 0-181 | validate 182-207 | test last 52 weeks (never shuffle)
tr_end, va_end = n - 78, n - 52
# 🧠 ML: scale using training weeks only to avoid leaking future level into the model
mu, sd = df[FEATS][:tr_end].mean(), df[FEATS][:tr_end].std()
arr = ((df[FEATS] - mu) / sd).values
Xtr, ytr = windows(arr, df.promo.values, 0, tr_end)
Xva, yva = windows(arr, df.promo.values, tr_end, va_end)
Xte, yte = windows(arr, df.promo.values, va_end, n)
print(f"windows train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}  inputs per step = {Xtr.shape[2]}")

# 🧠 ML: small grid over hidden size and learning rate, chosen on the validation window
results = {}
for hidden in [16, 32]:
    for lr in [0.01, 0.003]:
        model, v = fit(Xtr, ytr, Xva, yva, hidden, lr)
        results[(hidden, lr)] = (v, model)
        print(f"  hidden={hidden:2d} lr={lr:<6} val MSE={v:.3f}")
# 🐍 PY: min with key=lambda picks the dict item whose validation MSE (kv[1][0]) is smallest
(best_h, best_lr), (_, model) = min(results.items(), key=lambda kv: kv[1][0])
print(f"Chosen hidden={best_h} lr={best_lr}")

model.eval()
with torch.no_grad():
    # 🧠 ML: undo scaling so errors are in real cases, not z-scores
    pred = model(Xte).numpy() * sd["units"] + mu["units"]
actual = yte.numpy() * sd["units"] + mu["units"]
# 🧠 ML: seasonal-naive baseline = same week last year (hard to beat for seasonal SKUs)
starts = np.arange(max(va_end, WINDOW), n - HORIZON + 1)
# 🐍 PY: list comprehension slices the same 4 weeks from 52 weeks earlier for each forecast start
snaive = np.array([df.units.values[s - 52:s - 52 + HORIZON] for s in starts])

print("\nMAPE by horizon   LSTM   seasonal-naive")
for h in range(HORIZON):
    print(f"  week +{h + 1}        {mape(actual[:, h], pred[:, h]):5.1f}%   {mape(actual[:, h], snaive[:, h]):5.1f}%")
print(f"  overall        {mape(actual, pred):5.1f}%   {mape(actual, snaive):5.1f}%")

# 🧠 ML: promo weeks are where seasonal-naive fails - check accuracy split by promo flag
promo_mask = np.array([df.promo.values[s] for s in starts]) == 1
print(f"Week+1 MAPE on promo weeks: LSTM {mape(actual[promo_mask, 0], pred[promo_mask, 0]):.1f}%"
      f" vs naive {mape(actual[promo_mask, 0], snaive[promo_mask, 0]):.1f}%")

# 🧠 ML: permutation importance on test - shuffle one input feature, see how much MAPE worsens
base = mape(actual, pred)
rng = np.random.default_rng(SEED)
print("\nFeature importance (MAPE increase when shuffled):")
# 🐍 PY: a dict maps a readable name to the list of input columns that belong to it
groups = {name: [j] for j, name in enumerate(FEATS)}
groups["future_promo"] = list(range(len(FEATS), Xte.shape[2]))
for name, cols in groups.items():
    Xp = Xte.clone()
    Xp[:, :, cols] = Xp[rng.permutation(len(Xp))][:, :, cols]
    with torch.no_grad():
        p2 = model(Xp).numpy() * sd["units"] + mu["units"]
    print(f"  {name:12s} {mape(actual, p2) - base:+5.1f} pts")

# 🧠 ML: safety stock = z * sigma of forecast error * sqrt(lead time); z=1.65 -> 95% service level
lead_weeks = 2
for label, p in [("LSTM", pred), ("seasonal-naive", snaive)]:
    # 🧠 ML: sigma of week+1 forecast error drives how much buffer stock is needed
    sigma = np.std(actual[:, 0] - p[:, 0])
    ss = 1.65 * sigma * np.sqrt(lead_weeks)
    print(f"Safety stock ({label:14s}): sigma={sigma:5.0f}  -> {ss:5.0f} units at 95% service")

print("\nNext 4-week forecast:", [int(x) for x in pred[-1]], " actual:", [int(x) for x in actual[-1]])
# Governance: forecasts drive purchase orders; log model version, monitor MAPE drift weekly,
# and let planners override with a reason code (S&OP process).
print("Governance: version the model, monitor weekly MAPE drift, log planner overrides in S&OP.")
# 🧠 ML: save scaling stats with the weights so next week's data is scaled identically
path = os.path.join(tempfile.mkdtemp(), "sku_lstm.pt")
torch.save({"state": model.state_dict(), "mu": mu.to_dict(), "sd": sd.to_dict()}, path)
print("Saved ->", os.path.basename(path))

# ---
# LSTM — HARD lab · Supply Chain: Weekly SKU demand forecasting for a distribution center
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
