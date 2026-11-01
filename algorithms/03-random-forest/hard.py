# PRODUCTION: production SIU referral model - Random Forest pipeline, CV tuning, capacity-based threshold
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
NUM = ["days_policy_to_claim", "claim_amount", "num_witnesses", "prior_claims_3y",
       "vehicle_age_yrs", "days_to_report", "repair_estimate_ratio"]
CAT = ["loss_type", "repair_shop_tier", "state"]
BIN = ["police_report", "injury_claim", "attorney_involved"]
# 🐍 PY: tuple unpacking assigns two constants in one line
SIU_COST, SIU_CAPACITY = 450, 0.08   # $ per investigation; SIU can review 8% of claims


def make_claims(n=8000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "days_policy_to_claim": rng.integers(1, 1500, n),
        "claim_amount": rng.lognormal(8.6, 0.8, n).round(0),
        "num_witnesses": rng.poisson(0.9, n),
        "prior_claims_3y": rng.poisson(0.4, n),
        "vehicle_age_yrs": rng.integers(0, 20, n),
        "days_to_report": rng.exponential(4, n).round(),
        "repair_estimate_ratio": rng.lognormal(0, 0.3, n).round(2),  # estimate / book repair cost
        "loss_type": rng.choice(["rear_end", "theft", "single_vehicle", "parking", "hail"], n),
        "repair_shop_tier": rng.choice(["network", "independent", "flagged"], n, p=[.6, .35, .05]),
        "state": rng.choice(["TX", "FL", "OK", "CA", "NY"], n),
        "police_report": rng.binomial(1, 0.65, n),
        "injury_claim": rng.binomial(1, 0.25, n),
        "attorney_involved": rng.binomial(1, 0.15, n),
    })
    # Real data has gaps: 5% of repair estimates missing
    # 🐍 PY: .loc[mask, col] = value overwrites only the rows where the mask is True
    df.loc[rng.random(n) < 0.05, "repair_estimate_ratio"] = np.nan
    p1 = (df.days_policy_to_claim < 120) & (df.injury_claim == 1)
    p2 = (df.police_report == 0) & (df.num_witnesses == 0) & (df.claim_amount > 7000)
    p3 = (df.repair_shop_tier == "flagged") & (df.repair_estimate_ratio.fillna(1) > 1.3)
    logit = (-4.6 + 3.6 * p1 + 3.6 * p2 + 3.8 * p3 + 0.6 * df.attorney_involved
             + 0.3 * df.prior_claims_3y + 0.04 * df.days_to_report + 0.5 * (df.loss_type == "theft"))
    df["fraud"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def preprocessor(scale=False):
    # 🧠 ML: trees need no scaling, but missing values must be imputed and categories one-hot encoded
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("bin", "passthrough", BIN),
    ])


def lift_table(y, score, cuts=(0.02, 0.05, 0.08, 0.15)):
    order = np.argsort(-score)
    for c in cuts:
        k = int(len(y) * c)
        hits = y[order[:k]].sum()
        # 🐍 PY: yield turns this function into a generator - rows are produced one at a time
        yield c, k, hits, hits / max(y.sum(), 1)


def main():
    df = make_claims()
    X, y = df.drop(columns="fraud"), df.fraud
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"Claims {len(df):,} | fraud rate {y.mean():.1%} | train {len(X_tr)} / test {len(X_te)}")

    # 🧠 ML: baseline logistic regression - if the forest cannot beat this, prefer the simpler model
    base = Pipeline([("pre", preprocessor(scale=True)),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(X_tr, y_tr)

    rf = Pipeline([("pre", preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                                  n_jobs=2, random_state=42))])
    # 🧠 ML: tune tree depth, leaf size and features-per-split; these control the bias/variance trade-off
    space = {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [3, 10, 25],
             "clf__max_features": ["sqrt", 0.5]}
    search = RandomizedSearchCV(rf, space, n_iter=5, scoring="average_precision", random_state=42,
                                cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_
    print("Best RF params:", {k.split("__")[1]: v for k, v in search.best_params_.items()})

    print("\nModel                 ROC-AUC  PR-AUC")
    scores = {}
    for name, m in [("LogReg baseline", base), ("Random Forest", model)]:
        scores[name] = m.predict_proba(X_te)[:, 1]
        print(f"{name:<20} {roc_auc_score(y_te, scores[name]):>8.3f} {average_precision_score(y_te, scores[name]):>7.3f}")

    # 🧠 ML: the threshold is set by SIU capacity, not 0.5 - refer the top-scoring 8% of claims
    p = scores["Random Forest"]
    thr = np.quantile(p, 1 - SIU_CAPACITY)
    print(f"\nCapacity threshold (top {SIU_CAPACITY:.0%}): score >= {thr:.3f}")
    print("Referred | claims | fraud caught | share of all fraud")
    for c, k, hits, share in lift_table(y_te.values, p):
        print(f"top {c:>4.0%} | {k:>6} | {hits:>12} | {share:>8.0%}")

    # 🧠 ML: translate model output into money - fraud payouts avoided minus investigation cost
    flagged = p >= thr
    fraud_paid_avoided = X_te.claim_amount[flagged & (y_te == 1)].sum()
    cost = flagged.sum() * SIU_COST
    print(f"Fraud payouts avoided ${fraud_paid_avoided:,.0f} - SIU cost ${cost:,.0f} "
          f"= net ${fraud_paid_avoided - cost:,.0f}")

    # 🧠 ML: permutation importance on raw columns is less biased than impurity importance
    pi = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=3,
                                random_state=42, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    print("\nPermutation importance (PR-AUC drop):")
    for col, v in imp.head(6).items():
        print(f"  {col:<22} {v:+.4f}")

    # 🧠 ML: fairness check - referral rates by state should be explainable by risk, not geography alone
    # 🐍 PY: groupby on another Series aligned by index, then mean() of booleans = referral rate
    rates = pd.Series(flagged, index=X_te.index).groupby(X_te.state).mean()
    print("\nReferral rate by state:", ", ".join(f"{s} {r:.0%}" for s, r in rates.items()))
    print("Governance: NAIC AI Model Bulletin + state unfair-claims laws -> model only PRIORITISES;")
    print("a human SIU investigator decides. Log reasons, monitor drift, re-validate yearly.")

    path = os.path.join(tempfile.mkdtemp(), "siu_rf.joblib")
    # 🐍 PY: a dict bundles the model with metadata so scoring code knows the threshold used
    joblib.dump({"model": model, "threshold": float(thr), "capacity": SIU_CAPACITY}, path)
    print(f"Saved {os.path.basename(path)} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()

# ---
# Random Forest — PRODUCTION lab · Insurance: Auto insurance claims fraud detection (SIU referral)
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
