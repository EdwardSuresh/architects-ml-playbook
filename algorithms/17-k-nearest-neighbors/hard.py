# Hard: production transit-day estimator - one-hot + scaling pipeline, tuned k-NN, explainable neighbours
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
NUM = ["distance_mi", "weight_lb"]
CAT = ["carrier", "season", "service"]
CARRIER_DELAY = {"CarrierA": 0.0, "CarrierB": 0.5, "CarrierC": 1.2, "CarrierD": -0.3}
SEASON_DELAY = {"Winter": 0.7, "Spring": 0.0, "Summer": 0.2, "Peak": 1.4}


def make_shipments(n=4000):
    """Synthetic shipment history with carrier, season and service level effects."""
    df = pd.DataFrame({
        "ship_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 640, n)), "D"),
        "distance_mi": rng.uniform(50, 2800, n).round(0),
        "weight_lb": rng.lognormal(7, 0.6, n).round(0),
        "carrier": rng.choice(list(CARRIER_DELAY), n, p=[0.4, 0.3, 0.2, 0.1]),
        "service": rng.choice(["Standard", "Expedited"], n, p=[0.8, 0.2]),
    })
    month = df.ship_date.dt.month
    # 🐍 PY: np.select picks a label from the first matching condition (vectorised if/elif)
    df["season"] = np.select([month.isin([11, 12]), month.isin([1, 2]), month.isin([3, 4, 5])],
                             ["Peak", "Winter", "Spring"], default="Summer")
    # 🐍 PY: np.where(cond, a, b) is a vectorised if/else over a whole column
    speed = np.where(df.service == "Expedited", 650, 450)
    # Transit moves in whole "driving days" (step-shaped, not a straight line) - k-NN handles steps well
    df["transit_days"] = (1 + np.ceil(df.distance_mi / speed) + 0.00015 * df.weight_lb
                          + df.carrier.map(CARRIER_DELAY) + df.season.map(SEASON_DELAY)
                          + rng.gamma(2, 0.3, n)).clip(1).round(1)
    return df


def build_pipeline():
    # 🧠 ML: one-hot turns carrier/season into 0/1 columns so they can take part in distance calculations
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("knn", KNeighborsRegressor())])


df = make_shipments()
# 🧠 ML: time-ordered split - train on the past, test on the most recent 20% (how it is used live)
# 🐍 PY: iloc slicing by position works because rows are already sorted by ship_date
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"Train {train.ship_date.min().date()} -> {train.ship_date.max().date()} ({len(train)} rows)")
print(f"Test  {test.ship_date.min().date()} -> {test.ship_date.max().date()} ({len(test)} rows)\n")

# Tune k, weighting and distance metric with 5-fold CV
# 🧠 ML: weights='distance' lets close neighbours count more; p=1 Manhattan, p=2 Euclidean
grid = {"knn__n_neighbors": [3, 7, 15, 30], "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
search = GridSearchCV(build_pipeline(), grid, cv=KFold(5, shuffle=True, random_state=42),
                      scoring="neg_mean_absolute_error", n_jobs=1)
search.fit(train[NUM + CAT], train.transit_days)
print(f"Best params: {search.best_params_}  CV MAE {-search.best_score_:.3f} days")
cv = pd.DataFrame(search.cv_results_)
# 🐍 PY: pivot_table turns long CV rows into a k x weights table
cv["mae"] = -cv.mean_test_score
print(cv[cv.param_knn__p == search.best_params_["knn__p"]]
      .pivot_table(index="param_knn__n_neighbors", columns="param_knn__weights", values="mae")
      .round(3).to_string(), "\n")

# Compare with baselines on the held-out future period
# 🧠 ML: best_estimator_ is the pipeline refit on ALL training data with the winning settings
best = search.best_estimator_
# 🧠 ML: a linear model is a strong baseline - k-NN must earn its place by beating it
lin = Pipeline([("pre", build_pipeline().named_steps["pre"]), ("lr", LinearRegression())])
models = {"Mean baseline": DummyRegressor(), "Linear regression": lin, "Tuned k-NN": best}
print("Model               | Test MAE (days) | within +/-1 day")
for name, m in models.items():
    m.fit(train[NUM + CAT], train.transit_days)
    p = m.predict(test[NUM + CAT])
    # 🧠 ML: share within +/- 1 day is a business-friendly accuracy metric customers understand
    hit = np.mean(np.abs(p - test.transit_days) <= 1)
    print(f"{name:<19} |     {mean_absolute_error(test.transit_days, p):.3f}       |   {hit:.0%}")

# Global explainability
# 🧠 ML: permutation importance = how much worse MAE gets when one column is shuffled
imp = permutation_importance(best, test[NUM + CAT], test.transit_days,
                             scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
print("\nPermutation importance (MAE increase):")
for col, v in sorted(zip(NUM + CAT, imp.importances_mean), key=lambda t: -t[1]):
    print(f"  {col:<12} {v:.3f} " + "#" * int(v * 20))


def explain(model, shipment, history, k=5):
    """Show the actual past shipments the model leaned on - k-NN's natural explanation."""
    # 🐍 PY: named_steps is a dict-like view of the pipeline's stages by name
    pre, knn = model.named_steps["pre"], model.named_steps["knn"]
    # 🧠 ML: transform the query with the SAME fitted preprocessor, then ask for its neighbours
    dist, idx = knn.kneighbors(pre.transform(shipment), n_neighbors=k)
    nbrs = history.iloc[idx[0]][NUM + CAT + ["transit_days"]].copy()
    # 🧠 ML: showing neighbour distance tells the user how similar the evidence really is
    nbrs["distance"] = dist[0].round(3)
    return nbrs


new = pd.DataFrame([{"distance_mi": 1800, "weight_lb": 900, "carrier": "CarrierC",
                     "season": "Peak", "service": "Standard"}])
eta = best.predict(new)[0]
print(f"\nNew shipment: 1800 mi, 900 lb, CarrierC, Peak, Standard -> ETA {eta:.1f} days")
print(explain(best, new, train).to_string(index=False))

# Business decision: quote a buffered promise date
# 🧠 ML: use the error distribution on test data to set a buffer that meets a 90% on-time target
resid = test.transit_days - best.predict(test[NUM + CAT])
# 🐍 PY: np.quantile returns the value below which 90% of the residuals fall
buffer = np.quantile(resid, 0.90)
print(f"\nQuote ETA + {buffer:.1f} days to keep ~90% of promises -> promise {np.ceil(eta + buffer):.0f} days")

path = os.path.join(tempfile.mkdtemp(), "transit_knn.joblib")
joblib.dump(best, path)
print(f"Saved pipeline -> {os.path.basename(path)}")
print("Governance: k-NN stores the training data itself - purge per data-retention rules and "
      "re-fit monthly as carrier performance drifts.")

# ---
# k-Nearest Neighbors (k-NN) — HARD lab · Supply Chain: Delivery transit-day estimation from similar past shipments
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
