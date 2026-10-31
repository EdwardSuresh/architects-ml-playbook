# MEDIUM: scikit-learn HistGradientBoosting predicts which shipments will deliver late
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

rng = np.random.default_rng(42)
n = 5000
carriers = ["FastFreight", "BlueLine", "CheapHaul", "RegionalX"]
lanes = ["TX-OK", "CA-AZ", "IL-OH", "GA-FL", "NY-PA", "WA-OR"]
# 🐍 PY: rng.choice picks random labels from a list, with optional probabilities p
df = pd.DataFrame({
    "carrier": rng.choice(carriers, n, p=[.3, .3, .2, .2]),
    "lane": rng.choice(lanes, n),
    "distance_mi": rng.integers(50, 2200, n),
    "weather_severity": rng.integers(0, 4, n),       # 0 clear .. 3 severe storm
    "promised_days": rng.integers(1, 6, n),
    "pickup_dow": rng.integers(0, 7, n),             # 0 = Monday
    "dock_wait_hrs": rng.exponential(2, n).round(1),
})
# Hidden truth: tight promise vs distance, storms, a weak carrier, Friday pickups
# 🧠 ML: tightness = miles per promised mile-capacity; >1 means the promise is hard to meet
tightness = df.distance_mi / (df.promised_days * 500)
logit = (-4.3 + 1.6 * tightness + 0.7 * df.weather_severity + 1.1 * (df.carrier == "CheapHaul")
         + 0.5 * (df.pickup_dow == 4) + 0.15 * df.dock_wait_hrs + 0.6 * (df.lane == "IL-OH") * (df.weather_severity >= 2))
df["late"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Shipments: {n}  late rate: {df.late.mean():.1%}")

# 🧠 ML: HistGradientBoosting can use text categories directly when typed as pandas 'category'
# 🐍 PY: .astype('category') stores repeated strings as small integer codes
for c in ["carrier", "lane"]:
    df[c] = df[c].astype("category")
X, y = df.drop(columns="late"), df.late
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 🧠 ML: early_stopping holds out 10% of train and stops adding trees when it stops improving
model = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=400, max_leaf_nodes=15,
                                       categorical_features="from_dtype", early_stopping=True,
                                       validation_fraction=0.1, random_state=42)
model.fit(X_tr, y_tr)
print(f"Trees actually built (early stopping): {model.n_iter_}")

proba = model.predict_proba(X_te)[:, 1]
# 🧠 ML: precision = flagged loads that were late; recall = late loads we flagged
pred = (proba >= 0.5).astype(int)
print(f"ROC-AUC {roc_auc_score(y_te, proba):.3f} | precision {precision_score(y_te, pred):.2f}"
      f" | recall {recall_score(y_te, pred):.2f}")

# 🧠 ML: compare predicted vs actual late rate per carrier - a sanity check planners understand
res = X_te.assign(actual=y_te.values, predicted=proba)
# 🐍 PY: groupby + agg with named tuples produces a tidy summary table
by_carrier = res.groupby("carrier", observed=True).agg(shipments=("actual", "size"),
                                                       actual_late=("actual", "mean"),
                                                       predicted_late=("predicted", "mean"))
print("\nBy carrier:")
print(by_carrier.round(3).to_string())

by_wx = res.groupby("weather_severity").agg(actual_late=("actual", "mean"), predicted=("predicted", "mean"))
print("\nBy weather severity (0 clear .. 3 storm):")
# 🐍 PY: iterrows yields (index, row) pairs - fine for small summary tables
for sev, row in by_wx.iterrows():
    print(f"  {sev}: actual {row.actual_late:.0%} predicted {row.predicted:.0%} {'#' * int(row.predicted * 40)}")

# Business action: proactively notify customers for the riskiest 15% of shipments
# 🧠 ML: rank-based action - planners can only call so many customers, so flag the riskiest 15%
cut = np.quantile(proba, 0.85)
top = proba >= cut
print(f"\nFlag top 15% (risk >= {cut:.2f}): {y_te.values[top].mean():.0%} of them really were late "
      f"vs {y_te.mean():.0%} overall -> catches {y_te.values[top].sum() / y_te.sum():.0%} of all late loads")

# ---
# Gradient Boosting — MEDIUM lab · Supply Chain: Predict late shipments (on-time delivery risk)
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
