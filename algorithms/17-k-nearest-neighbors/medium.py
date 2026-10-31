# Medium: KNeighborsRegressor predicting transit days from distance, weight and season
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
N = 3000

# 🐍 PY: build columns as numpy arrays, then wrap them in a DataFrame
distance_mi = np.random.uniform(50, 2800, N)
weight_lb = np.random.lognormal(mean=7, sigma=0.6, size=N)
# 🐍 PY: np.random.choice samples month numbers 1-12
ship_month = np.random.choice(np.arange(1, 13), N)
is_peak = np.isin(ship_month, [11, 12]).astype(int)
is_winter = np.isin(ship_month, [1, 2, 12]).astype(int)

# True process: ~1 day per 500 miles, heavy freight slower, peak season and winter add delays
transit_days = (1 + distance_mi / 500 + 0.0002 * weight_lb
                + 1.2 * is_peak + 0.6 * is_winter + np.random.normal(0, 0.6, N))
df = pd.DataFrame({
    "distance_mi": distance_mi.round(0), "weight_lb": weight_lb.round(0),
    "is_peak": is_peak, "is_winter": is_winter,
    # 🐍 PY: np.clip keeps values within [1, inf) and .round(1) tidies decimals
    "transit_days": np.clip(transit_days, 1, None).round(1),
})
print(df.head().to_string(), "\n")
print(df.describe().loc[["mean", "min", "max"]].round(1).to_string(), "\n")

# 🐍 PY: a list of column names selects several DataFrame columns at once
features = ["distance_mi", "weight_lb", "is_peak", "is_winter"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["transit_days"], test_size=0.2, random_state=42)

# 🧠 ML: k-NN is pure distance - without scaling, miles (thousands) would drown out the 0/1 season flags
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# 🧠 ML: baseline = always predict the training average; a model must beat this to be useful
base_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
print(f"Baseline (predict mean) MAE: {base_mae:.2f} days")

print("\n  k | MAE unscaled | MAE scaled")
for k in [1, 5, 15, 40]:
    # 🐍 PY: a tuple of (train, test) pairs lets one loop body handle both versions
    maes = []
    for Xtr, Xte in ((X_train, X_test), (X_train_s, X_test_s)):
        knn = KNeighborsRegressor(n_neighbors=k).fit(Xtr, y_train)
        maes.append(mean_absolute_error(y_test, knn.predict(Xte)))
    print(f"{k:>3} |    {maes[0]:.2f}      |   {maes[1]:.2f}")

# 🧠 ML: small k = noisy (high variance); large k = over-smoothed (high bias); pick a middle value
model = KNeighborsRegressor(n_neighbors=15).fit(X_train_s, y_train)
pred = model.predict(X_test_s)
print(f"\nChosen k=15  MAE {mean_absolute_error(y_test, pred):.2f} days   R2 {r2_score(y_test, pred):.3f}")

# Score a brand-new shipment
new = pd.DataFrame([{"distance_mi": 1400, "weight_lb": 2200, "is_peak": 1, "is_winter": 1}])
est = model.predict(scaler.transform(new))[0]
# 🧠 ML: kneighbors returns the actual past shipments used - built-in explainability
dist, idx = model.kneighbors(scaler.transform(new), n_neighbors=5)
print(f"\nNew shipment 1400 mi, 2200 lb, December -> estimate {est:.1f} days")
print("Five closest past shipments:")
print(X_train.iloc[idx[0]].assign(transit_days=y_train.iloc[idx[0]].values).to_string())

# Business interpretation
within_1 = np.mean(np.abs(pred - y_test) <= 1.0)
print(f"\n{within_1:.0%} of test shipments predicted within +/- 1 day -> good enough to quote customer ETAs.")

# ---
# k-Nearest Neighbors (k-NN) — MEDIUM lab · Supply Chain: Delivery transit-day estimation from similar past shipments
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
