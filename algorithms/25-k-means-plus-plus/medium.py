# k-means vs k-means++ initialisation for choosing distribution-center (DC) sites from customer locations
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

# ---- Synthetic customers: 3,000 store/delivery points around 8 metro areas (lat/lon in degrees) ----
metros = pd.DataFrame({
    "metro": ["Dallas", "Houston", "Oklahoma City", "Tulsa", "Kansas City", "Denver", "Memphis", "Little Rock"],
    "lat": [32.78, 29.76, 35.47, 36.15, 39.10, 39.74, 35.15, 34.75],
    "lon": [-96.80, -95.37, -97.52, -95.99, -94.58, -104.99, -90.05, -92.29],
    "share": [0.22, 0.20, 0.10, 0.08, 0.12, 0.12, 0.09, 0.07],
})
N = 3000
# 🐍 PY: rng.choice with p= picks a metro index for each customer, weighted by market share
idx = rng.choice(len(metros), N, p=metros.share)
customers = pd.DataFrame({
    "customer_id": np.arange(N),
    "lat": metros.lat.to_numpy()[idx] + rng.normal(0, 0.35, N),
    "lon": metros.lon.to_numpy()[idx] + rng.normal(0, 0.45, N),
    "weekly_cases": rng.lognormal(3.0, 0.6, N).round(),
})
print(customers.describe().loc[["mean", "min", "max"], ["lat", "lon", "weekly_cases"]].round(2))

# 🧠 ML: convert degrees to approximate miles so 1 unit east = 1 unit north (k-means uses straight distance)
# 🐍 PY: np.cos needs radians, so np.radians converts the average latitude first
MI_PER_DEG_LAT = 69.0
mi_per_deg_lon = 69.0 * np.cos(np.radians(customers.lat.mean()))
X = np.column_stack([customers.lat * MI_PER_DEG_LAT, customers.lon * mi_per_deg_lon])
# 🧠 ML: hold out 20% of customers to check that DC sites also serve customers we did not fit on
X_fit, X_new = train_test_split(X, test_size=0.2, random_state=42)

K = 8
rows = []
for seed in range(10):
    for init in ("random", "k-means++"):
        # 🧠 ML: n_init=1 shows the effect of ONE initialisation; production uses several and keeps the best
        km = KMeans(n_clusters=K, init=init, n_init=1, random_state=seed, max_iter=300).fit(X_fit)
        # 🧠 ML: inertia_ = sum of squared miles from each customer to its DC; score() on new data = -inertia
        rows.append({"seed": seed, "init": init, "inertia_M": km.inertia_ / 1e6,
                     "iterations": km.n_iter_, "holdout_inertia_M": -km.score(X_new) / 1e6})
res = pd.DataFrame(rows)

# 🐍 PY: pivot turns long rows into a seed x init table for side-by-side comparison
print("\nTraining inertia (millions of sq-miles) per seed:")
print(res.pivot(index="seed", columns="init", values="inertia_M").round(2).to_string())
# 🐍 PY: groupby + agg computes several summaries per init method in one go
summary = res.groupby("init").agg(mean_inertia=("inertia_M", "mean"), worst=("inertia_M", "max"),
                                  sd=("inertia_M", "std"), avg_iters=("iterations", "mean"),
                                  holdout=("holdout_inertia_M", "mean"))
print("\nSummary over 10 seeds:")
print(summary.round(2).to_string())

best_inertia = res.inertia_M.min()
# 🧠 ML: a "bad" run converged to a local optimum >5% worse than the best found - e.g. 2 DCs in one metro
bad = res.assign(bad=res.inertia_M > best_inertia * 1.05).groupby("init").bad.mean()
print(f"\nShare of runs >5% worse than best: random {bad['random']:.0%}, k-means++ {bad['k-means++']:.0%}")

# Final DC sites from the recommended setting (k-means++, 10 restarts)
final = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42).fit(X)
sites = pd.DataFrame(final.cluster_centers_, columns=["lat", "lon"])
sites["lat"] /= MI_PER_DEG_LAT
sites["lon"] /= mi_per_deg_lon
sites["customers"] = np.bincount(final.labels_)
# 🧠 ML: average straight-line distance from each customer to its assigned DC (not squared)
d = np.linalg.norm(X - final.cluster_centers_[final.labels_], axis=1)
sites["avg_miles"] = pd.Series(d).groupby(final.labels_).mean().to_numpy()
print("\nRecommended DC sites (k-means++, n_init=10):")
print(sites.round(2).sort_values("customers", ascending=False).to_string(index=False))
print(f"\nBusiness read-out: average customer is {d.mean():.0f} miles from its DC. Random init often"
      f"\nputs two DCs in one metro and leaves another metro unserved - k-means++ makes that rare.")

# ---
# k-Means++ — BUILD lab · Supply Chain: Choosing distribution center locations
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
