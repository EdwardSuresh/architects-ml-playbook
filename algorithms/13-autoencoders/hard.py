"""Compressor anomaly detection: windowed autoencoder, validation-percentile threshold,
per-sensor attribution, precision/recall on injected faults, alarm persistence."""
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)
SENSORS = ["vibration_mm_s", "discharge_temp_f", "suction_pressure_psi",
           "discharge_pressure_psi", "flow_mmscfd", "motor_current_a"]
WINDOW = 4  # readings per window (10-min readings -> 40 minutes)


def simulate(n_steps):
    """Time-ordered healthy data: load drifts smoothly (not random every row), plus sensor noise."""
    # 🧠 ML: realistic data is autocorrelated - neighbouring readings look alike
    load = np.empty(n_steps)
    load[0] = 0.7
    for t in range(1, n_steps):
        # mean-reverting load: pulled back toward 70% each step, plus a small random push
        load[t] = 0.7 + 0.97 * (load[t - 1] - 0.7) + rng.normal(0, 0.03)
    load = np.clip(load, 0.4, 1.0)
    ambient = 80 + 10 * np.sin(np.arange(n_steps) * 2 * np.pi / 144)  # daily cycle
    return pd.DataFrame({
        "vibration_mm_s": 2 + 3 * load + rng.normal(0, 0.15, n_steps),
        "discharge_temp_f": 150 + 60 * load + 0.5 * ambient + rng.normal(0, 2, n_steps),
        "suction_pressure_psi": 400 - 50 * load + rng.normal(0, 4, n_steps),
        "discharge_pressure_psi": 900 + 300 * load + rng.normal(0, 8, n_steps),
        "flow_mmscfd": 20 + 40 * load + rng.normal(0, 1, n_steps),
        "motor_current_a": 150 + 200 * load + rng.normal(0, 5, n_steps)})


def inject_faults(df):
    """Add three labelled fault episodes into a test period."""
    df, label = df.copy(), np.zeros(len(df), dtype=int)
    # 🐍 PY: a list of tuples (start, end, description, {sensor: offset}) drives the injection
    episodes = [(300, 340, "valve leak", {"flow_mmscfd": -7, "discharge_temp_f": 14}),
                (700, 760, "bearing wear", {"vibration_mm_s": 1.2}),
                (1100, 1130, "pressure sensor drift", {"discharge_pressure_psi": 60})]
    for start, end, _, offsets in episodes:
        for col, delta in offsets.items():
            # 🧠 ML: ramp the fault in gradually, like a real developing failure
            df.loc[start:end - 1, col] += np.linspace(0.3, 1.0, end - start) * delta
        label[start:end] = 1
    return df, label, episodes


def to_windows(arr, w=WINDOW):
    # 🐍 PY: stacking shifted slices builds rows of the last w readings, flattened side by side
    return np.stack([arr[i:len(arr) - w + 1 + i] for i in range(w)], axis=1).reshape(len(arr) - w + 1, -1)


class AE(nn.Module):
    def __init__(self, n_in, code=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 12), nn.Tanh(), nn.Linear(12, code))
        self.dec = nn.Sequential(nn.Linear(code, 12), nn.Tanh(), nn.Linear(12, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(Xtr, Xval, epochs=150, batch=128):
    model = AE(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    best, state = np.inf, None
    for _ in range(epochs):
        for idx in torch.randperm(len(Xtr)).split(batch):
            opt.zero_grad()
            nn.functional.mse_loss(model(Xtr[idx]), Xtr[idx]).backward()
            opt.step()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xval), Xval).item()
        # 🧠 ML: keep the weights with the best validation reconstruction (simple early stopping)
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def sensor_errors(model, X):
    """Squared error per window, averaged per SENSOR across the window's time steps."""
    with torch.no_grad():
        sq = ((model(X) - X) ** 2).numpy()
    # 🧠 ML: reshaping to (windows, time, sensors) lets us attribute the error to each sensor
    return sq.reshape(len(sq), WINDOW, len(SENSORS)).mean(axis=1)


def persist(flags, k=3, n=5):
    # 🧠 ML: alarm only if k of the last n windows are flagged - cuts one-off false alarms
    return pd.Series(flags).rolling(n, min_periods=1).sum().values >= k


def prf(pred, truth):
    tp = (pred & truth).sum()
    precision = tp / max(pred.sum(), 1)
    recall = tp / max(truth.sum(), 1)
    return precision, recall


if __name__ == "__main__":
    # 🧠 ML: train and validate on HEALTHY history only; faults only appear in the test period
    hist = simulate(5000)
    train, val = hist.iloc[:4000], hist.iloc[4000:]
    test, label, episodes = inject_faults(simulate(1400))
    mu, sd = train.mean(), train.std()
    scale = lambda d: ((d - mu) / sd).values
    Xtr, Xval, Xte = (torch.tensor(to_windows(scale(d)), dtype=torch.float32) for d in (train, val, test))
    # A window is faulty if its LAST reading is faulty (we alarm on the newest data)
    y_win = label[WINDOW - 1:].astype(bool)

    model, best_val = train_ae(Xtr, Xval)
    print(f"Windows train/val/test: {len(Xtr)}/{len(Xval)}/{len(Xte)}  best val MSE {best_val:.4f}")

    val_err = sensor_errors(model, Xval).mean(axis=1)
    test_sens = sensor_errors(model, Xte)
    test_err = test_sens.mean(axis=1)

    # 🧠 ML: threshold is a percentile of VALIDATION error - never tuned on the faults we test on
    print("\nPercentile  threshold  precision  recall  (raw / with 3-of-5 persistence)")
    for pct in [95, 99, 99.5]:
        thr = np.percentile(val_err, pct)
        raw = test_err > thr
        p1, r1 = prf(raw, y_win)
        p2, r2 = prf(persist(raw), y_win)
        print(f"{pct:>9}   {thr:8.4f}   {p1:.2f}/{p2:.2f}    {r1:.2f}/{r2:.2f}")

    thr = np.percentile(val_err, 99)
    alarms = persist(test_err > thr)
    print(f"\nChosen threshold: 99th percentile = {thr:.4f}")
    print("\nPer-episode detection and top sensor by error share:")
    for start, end, name, _ in episodes:
        # 🐍 PY: convert reading indices to window indices by subtracting WINDOW - 1
        s, e = start - (WINDOW - 1), end - (WINDOW - 1)
        hit = np.where(alarms[s:e])[0]
        delay = f"{hit[0] * 10} min" if len(hit) else "missed"
        share = test_sens[s:e].mean(axis=0) / test_sens[s:e].mean(axis=0).sum()
        top2 = np.argsort(share)[::-1][:2]
        blame = ", ".join(f"{SENSORS[i]} {share[i]:.0%}" for i in top2)
        print(f"  {name:<22} first alarm after {delay:<8} | {blame}")

    healthy_alarm_hours = (alarms & ~y_win).sum() * 10 / 60
    print(f"\nFalse-alarm time on healthy periods: {healthy_alarm_hours:.1f} h of {(~y_win).sum() * 10 / 60:.0f} h")

    path = os.path.join(tempfile.mkdtemp(), "compressor_ae.pt")
    torch.save({"model": model.state_dict(), "mu": mu.values, "sd": sd.values, "threshold": thr}, path)
    print(f"Saved model, scaler and threshold -> {os.path.basename(path)}")
    print("Safety note: alarms support, not replace, API 618/670 machinery protection trips;")
    print("retrain after overhauls and log every alarm outcome for threshold review.")

# ---
# Autoencoders — HARD lab · Oil & Gas: Gas compressor sensor anomaly detection for predictive maintenance
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
