import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)

# --- 1. Simulate compressor sensor readings (one row per 10 minutes) ---
def make_readings(n, fault=False):
    # 🧠 ML: healthy sensors are driven by two hidden factors - load and ambient temperature
    load = rng.uniform(0.4, 1.0, n)
    ambient = rng.normal(80, 10, n)
    # 🐍 PY: a dict of column -> array; each sensor is a formula of the hidden drivers plus noise
    df = pd.DataFrame({
        "vibration_mm_s": 2.0 + 3.0 * load + rng.normal(0, 0.15, n),
        "discharge_temp_f": 150 + 60 * load + 0.5 * ambient + rng.normal(0, 2, n),
        "suction_pressure_psi": 400 - 50 * load + rng.normal(0, 4, n),
        "discharge_pressure_psi": 900 + 300 * load + rng.normal(0, 8, n),
        "flow_mmscfd": 20 + 40 * load + rng.normal(0, 1, n),
        "motor_current_a": 150 + 200 * load + rng.normal(0, 5, n),
    })
    if fault:
        # Fault: valve leak - flow drops and temperature climbs while load looks normal
        df["flow_mmscfd"] -= rng.uniform(4, 10, n)
        df["discharge_temp_f"] += rng.uniform(8, 20, n)
    return df

train_df = make_readings(3000)
normal_test = make_readings(500)
fault_test = make_readings(50, fault=True)
print(train_df.describe().loc[["mean", "std"]].round(1).to_string())

# --- 2. Scale using training (normal) data only ---
# 🐍 PY: pandas arithmetic is column-wise, so this standardises all 6 sensors at once
mu, sd = train_df.mean(), train_df.std()
# 🐍 PY: a lambda is a tiny one-line function
to_tensor = lambda d: torch.tensor(((d - mu) / sd).values, dtype=torch.float32)
X_train, X_norm, X_fault = to_tensor(train_df), to_tensor(normal_test), to_tensor(fault_test)

# --- 3. Autoencoder: 6 sensors -> 2-number bottleneck -> 6 sensors ---
# 🧠 ML: the bottleneck forces the net to learn only the 'normal' relationships between sensors
model = nn.Sequential(
    nn.Linear(6, 8), nn.Tanh(),
    nn.Linear(8, 2),            # encoder output = the code
    nn.Linear(2, 8), nn.Tanh(),
    nn.Linear(8, 6),            # decoder output = rebuilt sensors
)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 🧠 ML: input and target are the SAME - the model learns to copy through the bottleneck
for epoch in range(301):
    opt.zero_grad()
    loss = loss_fn(model(X_train), X_train)
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  train MSE {loss.item():.4f}")

# --- 4. Reconstruction error and threshold ---
def recon_error(X):
    # 🐍 PY: torch.no_grad() skips gradient bookkeeping during scoring
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1).numpy()

err_train = recon_error(X_train)
# 🧠 ML: alarm when error exceeds the 99th percentile of NORMAL training error
threshold = np.percentile(err_train, 99)
err_norm, err_fault = recon_error(X_norm), recon_error(X_fault)
print(f"\nThreshold (99th pct of normal error): {threshold:.4f}")
print(f"Median error  normal: {np.median(err_norm):.4f}   faulty: {np.median(err_fault):.4f}")

# --- 5. Results ---
false_alarms = (err_norm > threshold).mean()
detected = (err_fault > threshold).mean()
print(f"\nFalse alarm rate on new normal data: {false_alarms:.1%}")
print(f"Valve-leak readings detected:        {detected:.1%}")
print("\nMaintenance read-out: a high error means the sensors no longer agree with each other")
print("the way they do on a healthy compressor - send a technician before the valve fails.")

# ---
# Autoencoders — MEDIUM lab · Oil & Gas: Gas compressor sensor anomaly detection for predictive maintenance
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
