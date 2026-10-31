# HARD: production on-time-delivery risk model - time-ordered CV, GBM vs HistGBM, cost-based threshold
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # keep HistGradientBoosting's threads polite on small boxes
import tempfile
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
# 🐍 PY: module-level lists of column names keep the pipeline and the data in sync
CAT = ["carrier", "lane", "equipment"]
NUM = ["distance_mi", "promised_days", "weather_severity", "pickup_dow", "dock_wait_hrs",
       "carrier_late_rate_30d", "lane_volume_7d"]
LATE_PENALTY, EXPEDITE_COST = 800, 180   # $ chargeback per late load vs $ to expedite / reroute


def make_shipments(n=9000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ship_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 365, n)), unit="D"),
        "carrier": rng.choice(["FastFreight", "BlueLine", "CheapHaul", "RegionalX"], n, p=[.3, .3, .2, .2]),
        "lane": rng.choice(["TX-OK", "CA-AZ", "IL-OH", "GA-FL", "NY-PA", "WA-OR"], n),
        "equipment": rng.choice(["dry_van", "reefer", "flatbed"], n, p=[.7, .2, .1]),
        "distance_mi": rng.integers(50, 2200, n),
        "promised_days": rng.integers(1, 6, n),
        "pickup_dow": rng.integers(0, 7, n),
        "dock_wait_hrs": rng.exponential(2, n).round(1),
        "lane_volume_7d": rng.poisson(40, n),
    })
    # 🐍 PY: the .dt accessor exposes date parts (month, weekday) of a datetime column
    month = df.ship_date.dt.month
    # 🧠 ML: seasonality - winter storms in Dec-Feb, hurricanes on GA-FL in Aug-Oct
    storm_p = 0.15 + 0.25 * month.isin([12, 1, 2]) + 0.3 * (month.isin([8, 9, 10]) & (df.lane == "GA-FL"))
    df["weather_severity"] = rng.binomial(3, storm_p)
    df["carrier_late_rate_30d"] = (0.15 + 0.12 * (df.carrier == "CheapHaul") + rng.normal(0, 0.03, n)).clip(0, 1)
    # 🧠 ML: the label is driven by interactions (reefer x storm) - a natural fit for boosted trees
    tight = df.distance_mi / (df.promised_days * 500)
    logit = (-4.2 + 1.7 * tight + 0.8 * df.weather_severity + 4.0 * df.carrier_late_rate_30d
             + 0.6 * (df.equipment == "reefer") * (df.weather_severity >= 2) + 0.5 * (df.pickup_dow == 4)
             + 0.12 * df.dock_wait_hrs + 0.01 * (df.lane_volume_7d - 40))
    df["late"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def make_pipe(kind):
    if kind == "gbm":
        # 🧠 ML: classic GradientBoosting = exact split search, one tree per step; slower but transparent
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", "passthrough", NUM)])
        clf = GradientBoostingClassifier(n_estimators=150, subsample=0.8, random_state=42)
    else:
        # 🧠 ML: HistGradientBoosting bins features into 255 buckets - same idea LightGBM uses, far faster
        pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
                                 ("num", "passthrough", NUM)])
        clf = HistGradientBoostingClassifier(max_iter=200, categorical_features=[0, 1, 2], early_stopping=True,
                                             validation_fraction=0.15, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    df = make_shipments()
    # 🧠 ML: time-ordered split - train Jan-Sep, test Oct-Dec, exactly like deploying on future loads
    train, test = df[df.ship_date < "2025-10-01"], df[df.ship_date >= "2025-10-01"]
    X_tr, y_tr, X_te, y_te = train[CAT + NUM], train.late, test[CAT + NUM], test.late
    print(f"Train {len(train)} loads (late {y_tr.mean():.1%}) | Test {len(test)} loads (late {y_te.mean():.1%})")

    # 🧠 ML: TimeSeriesSplit always validates on later months than it trains on - no peeking ahead
    tscv = TimeSeriesSplit(n_splits=3)
    grids = {
        "gbm": {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]},
        "hist": {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]},
    }
    # 🧠 ML: DummyClassifier predicts the average late rate for everyone - the floor to beat
    results = {"baseline": DummyClassifier(strategy="prior").fit(X_tr, y_tr)}
    # 🐍 PY: dict.items() yields (key, value) pairs we can unpack directly in the for statement
    for kind, grid in grids.items():
        t0 = time.time()
        gs = GridSearchCV(make_pipe(kind), grid, cv=tscv, scoring="neg_log_loss", n_jobs=1).fit(X_tr, y_tr)
        results[kind] = gs.best_estimator_
        print(f"{kind:<5} best {gs.best_params_} | tuned in {time.time() - t0:.1f}s")

    print("\nModel      ROC-AUC  log-loss")
    for name, m in results.items():
        p = m.predict_proba(X_te)[:, 1]
        # 🐍 PY: conditional expression - a constant model has no ranking, so AUC is 0.5 by definition
        auc = roc_auc_score(y_te, p) if name != "baseline" else 0.5
        print(f"{name:<9} {auc:>8.3f} {log_loss(y_te, p):>9.3f}")
    model = results["hist"]
    proba = model.predict_proba(X_te)[:, 1]

    # 🧠 ML: expedite when expected penalty saved exceeds expedite cost; sweep thresholds on the business $
    print("\nThreshold | expedited | late loads caught | net savings $")
    # 🐍 PY: -np.inf is a starting value any real number will beat
    best = (None, -np.inf)
    for t in np.arange(0.2, 0.81, 0.1):
        act = proba >= t
        # Assume expediting prevents 70% of late outcomes
        saved = 0.7 * LATE_PENALTY * (act & (y_te.values == 1)).sum() - EXPEDITE_COST * act.sum()
        print(f"{t:>9.1f} | {act.sum():>9} | {(act & (y_te.values == 1)).sum():>16} | {saved:>12,.0f}")
        if saved > best[1]:
            best = (round(t, 1), saved)
    print(f"Break-even risk = {EXPEDITE_COST / (0.7 * LATE_PENALTY):.2f}; best tested threshold {best[0]}")

    # 🧠 ML: permutation importance answers 'which input, if scrambled, hurts the forecast most?'
    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1)
    print("\nPermutation importance (AUC drop):")
    for i in np.argsort(-pi.importances_mean)[:6]:
        print(f"  {X_te.columns[i]:<22} {pi.importances_mean[i]:+.4f}")

    # Monthly stability check - a light-touch drift monitor
    # 🐍 PY: .assign adds a column without modifying test; groupby month then compares means
    mon = test.assign(p=proba).groupby(test.ship_date.dt.month).agg(actual=("late", "mean"), predicted=("p", "mean"))
    print("\nMonth | actual late | predicted")
    for m, r in mon.iterrows():
        print(f"{m:>5} | {r.actual:>11.1%} | {r.predicted:>9.1%}")
    print("Governance: carrier-scorecard use must be explainable to carriers; retrain monthly, "
          "alert if calibration gap > 5 pts.")

    path = os.path.join(tempfile.mkdtemp(), "ontime_hgb.joblib")
    joblib.dump({"model": model, "threshold": best[0]}, path)
    print(f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    main()

# ---
# Gradient Boosting — HARD lab · Supply Chain: Predict late shipments (on-time delivery risk)
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
