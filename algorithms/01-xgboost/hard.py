# HARD: production-style XGBoost card-fraud scorer - time split, tuning, cost threshold, explainability
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
# 🐍 PY: UPPER_CASE names are a convention for module-level constants that never change
NUM = ["txn_amount", "km_from_home", "hour", "txns_last_1h", "amt_vs_avg", "acct_age_days", "card_present"]
CAT = ["merchant_cat", "channel"]
COST_FN, COST_FP = 1.0, 15.0  # missed fraud costs its amount; a false decline costs ~$15 friction + call


def make_data(n=20000, seed=42):
    """Simulate time-ordered card authorizations with ~1.5% fraud."""
    rng = np.random.default_rng(seed)
    # 🐍 PY: a dict comprehension is not needed here - a plain dict literal maps column -> array
    df = pd.DataFrame({
        "ts": np.sort(rng.uniform(0, 90, n)),  # day number over one quarter
        "txn_amount": rng.lognormal(3.7, 1.1, n).round(2),
        "km_from_home": rng.exponential(30, n).round(1),
        "hour": rng.integers(0, 24, n),
        "txns_last_1h": rng.poisson(0.6, n),
        "amt_vs_avg": rng.lognormal(0, 0.6, n).round(2),
        "acct_age_days": rng.integers(15, 4000, n),
        "card_present": rng.binomial(1, 0.65, n),
        "merchant_cat": rng.choice(["grocery", "fuel", "electronics", "travel", "gift_cards", "dining"],
                                   n, p=[.30, .15, .12, .10, .05, .28]),
        "channel": rng.choice(["pos", "ecom", "atm"], n, p=[.6, .3, .1]),
    })
    # 🐍 PY: .isin() returns True/False per row; .astype(int) turns that into 1/0 flags
    risky_mcc = df["merchant_cat"].isin(["electronics", "gift_cards"]).astype(int)
    burst = (df["txns_last_1h"] >= 3).astype(int)
    far_cnp = ((df["km_from_home"] > 80) & (df["card_present"] == 0)).astype(int)
    # Fraud signals combine non-linearly (rings hit risky merchants at night) - trees capture this
    combo = risky_mcc * (df["hour"] < 6) * (df["channel"] == "ecom")
    logit = (-6.4 + 1.2 * np.log(df["amt_vs_avg"] + 0.1) + 0.8 * risky_mcc + 2.2 * burst
             + 3.0 * far_cnp + 3.5 * combo + 0.6 * (df["hour"] < 5) + 0.3 * (df["channel"] == "ecom")
             - 0.0004 * df["acct_age_days"] + 0.35 * np.log1p(df["txn_amount"]))
    df["is_fraud"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def build_pipeline(model):
    # 🧠 ML: one Pipeline = identical preprocessing in training and in the real-time scoring API
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def best_threshold(y, proba, amounts):
    """Pick the cut-off that minimises total business cost on validation data."""
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        flag = proba >= t
        # 🧠 ML: cost-sensitive thresholding - a missed $900 fraud hurts more than a missed $5 one
        fn_cost = amounts[(~flag) & (y == 1)].sum() * COST_FN
        fp_cost = ((flag) & (y == 0)).sum() * COST_FP
        rows.append((round(t, 2), fn_cost + fp_cost, int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())))
    # 🐍 PY: min with key=lambda picks the tuple whose 2nd item (total cost) is smallest
    return min(rows, key=lambda r: r[1]), rows


def main():
    df = make_data()
    print(f"Authorizations: {len(df):,}  fraud rate: {df.is_fraud.mean():.2%}")
    # 🧠 ML: time-ordered split - train on the past, test on the future (no shuffling = no leakage)
    # 🐍 PY: .iloc slices by row position - first 80% for training, last 20% for testing
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_tr, y_tr, X_te, y_te = train[NUM + CAT], train.is_fraud, test[NUM + CAT], test.is_fraud

    # Baseline: logistic regression with class weighting
    base = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    base.fit(X_tr, y_tr)
    # 🧠 ML: always keep a simple baseline - the complex model must beat it to justify its cost
    p_base = base.predict_proba(X_te)[:, 1]

    # 🧠 ML: scale_pos_weight handles the ~1:65 imbalance instead of throwing away genuine rows
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb = build_pipeline(XGBClassifier(n_estimators=150, tree_method="hist", eval_metric="aucpr",
                                       scale_pos_weight=spw, n_jobs=2, random_state=42))
    # 🐍 PY: keys use the 'step__param' naming so the search reaches inside the Pipeline
    grid = {"clf__max_depth": [3, 4, 5], "clf__learning_rate": [0.03, 0.06, 0.1],
            "clf__subsample": [0.7, 0.9], "clf__min_child_weight": [1, 5]}
    # 🧠 ML: TimeSeriesSplit cross-validation respects time order; PR-AUC is the score for rare fraud
    search = RandomizedSearchCV(xgb, grid, n_iter=4, cv=TimeSeriesSplit(n_splits=3),
                                scoring="average_precision", random_state=42, n_jobs=1)
    search.fit(X_tr, y_tr)
    print("Best params:", {k.replace("clf__", ""): v for k, v in search.best_params_.items()})
    model = search.best_estimator_
    p_xgb = model.predict_proba(X_te)[:, 1]

    # 🐍 PY: looping over a list of (name, value) tuples keeps the comparison table compact
    print("\nModel               ROC-AUC   PR-AUC")
    for name, p in [("LogReg baseline", p_base), ("XGBoost tuned", p_xgb)]:
        print(f"{name:<18} {roc_auc_score(y_te, p):>8.3f} {average_precision_score(y_te, p):>8.3f}")
    print(f"(PR-AUC of random guessing = {y_te.mean():.3f})")

    # Threshold from business cost (in real life tune on a separate validation month)
    amounts = X_te["txn_amount"].values
    (t, cost, tp, fp), rows = best_threshold(y_te.values, p_xgb, amounts)
    print("\nThreshold | total cost $ | fraud caught | false declines")
    for r in rows[::3]:
        print(f"{r[0]:>9} | {r[1]:>12,.0f} | {r[2]:>12} | {r[3]:>14}")
    no_model = amounts[y_te.values == 1].sum()
    print(f"Chosen threshold {t}: cost ${cost:,.0f} vs ${no_model:,.0f} with no model")

    # 🧠 ML: permutation importance = drop in PR-AUC when one raw column is shuffled (model-agnostic)
    pi = permutation_importance(model, X_te, y_te, scoring="average_precision",
                                n_repeats=3, random_state=42, n_jobs=1)
    # 🐍 PY: argsort of the negative gives indices from largest to smallest
    order = np.argsort(-pi.importances_mean)
    print("\nPermutation importance (PR-AUC drop):")
    for i in order[:6]:
        print(f"  {X_te.columns[i]:<14} {pi.importances_mean[i]:+.4f}")

    # 🧠 ML: model-risk note - document, validate independently, monitor drift (SR 11-7 / OCC 2011-12)
    print("\nModel risk: SR 11-7 validation, monthly PSI drift checks, reason codes for declines,")
    print("PCI-DSS for card data; fraud models are not credit decisions but still need governance.")

    out = tempfile.mkdtemp()
    path = os.path.join(out, "card_fraud_xgb.joblib")
    # 🐍 PY: joblib.dump pickles the whole pipeline (scaler + encoder + model) into one file
    joblib.dump({"pipeline": model, "threshold": t}, path)
    print(f"Saved model + threshold to {os.path.basename(path)} ({os.path.getsize(path) // 1024} KB)")


# 🐍 PY: this guard runs main() only when the file is executed, not when it is imported
if __name__ == "__main__":
    main()

# ---
# XGBoost — HARD lab · Financial Services: Real-time credit card fraud detection
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
