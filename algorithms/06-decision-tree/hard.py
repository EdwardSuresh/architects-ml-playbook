import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42
LABELS = ["Accept", "Refer", "Decline"]
NUM = ["roof_age_yrs", "prior_claims_5yr", "dist_fire_station_mi", "credit_insurance_score",
       "home_value_k", "year_built"]
CAT = ["construction", "wildfire_zone"]


# 🐍 PY: default arguments (n=4000) let callers override the size without changing the function
def make_book(n=4000, seed=SEED):
    """Synthetic homeowners applications with a hidden underwriting guideline."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "roof_age_yrs": rng.integers(0, 35, n),
        "prior_claims_5yr": rng.poisson(0.6, n),
        "dist_fire_station_mi": np.round(rng.gamma(2, 2.5, n), 1),
        "credit_insurance_score": rng.integers(500, 900, n),
        "home_value_k": rng.integers(120, 900, n),
        "year_built": rng.integers(1940, 2024, n),
        "construction": rng.choice(["frame", "masonry", "superior"], n, p=[0.6, 0.3, 0.1]),
        "wildfire_zone": rng.choice(["low", "moderate", "high"], n, p=[0.65, 0.25, 0.10]),
    })
    # 🧠 ML: hidden guideline = the 'truth' underwriters followed; the tree must rediscover it
    risk = ((df.roof_age_yrs > 20) * 2 + df.prior_claims_5yr * 1.5
            + df.wildfire_zone.map({"low": 0, "moderate": 1, "high": 2.5})
            + (df.construction == "frame") * 0.7 + (df.dist_fire_station_mi > 10)
            + (df.credit_insurance_score < 620) * 1.5 + rng.normal(0, 0.7, n))
    # 🐍 PY: np.select returns the label of the first True condition, else the default
    df["decision"] = np.select([risk < 2.5, risk < 5], ["Accept", "Refer"], default="Decline")
    return df


def build_pipeline():
    # 🧠 ML: trees need no scaling, but text categories must become 0/1 columns first
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM),
    ], verbose_feature_names_out=False)
    # 🧠 ML: class_weight='balanced' makes rare Declines count more so the tree does not ignore them
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    return Pipeline([("pre", pre), ("tree", tree)])


def business_cost(y_true, y_pred):
    # 🧠 ML: cost matrix rows=true, cols=decision; auto-accepting a Decline risk is the worst error
    cost = np.array([[0, 40, 300],      # true Accept: refer wastes underwriter time, decline loses premium
                     [150, 0, 120],     # true Refer
                     [2500, 200, 0]])   # true Decline: accepting it = expected claim leakage
    # 🐍 PY: cm * cost multiplies element by element; .sum() totals dollars across all cells
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return int((cm * cost).sum())


df = make_book()
X, y = df.drop(columns="decision"), df["decision"]
# 🧠 ML: stratify keeps the Accept/Refer/Decline mix identical in train and test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print("Class mix (train):", y_tr.value_counts().to_dict())

# 🧠 ML: tune depth and leaf size with 5-fold CV; macro-F1 treats all three decisions equally
grid = {"tree__max_depth": [3, 4, 5, 6, 8],
        "tree__min_samples_leaf": [10, 25, 50],
        "tree__ccp_alpha": [0.0, 0.001]}
# 🧠 ML: 5 folds = 5 different train/validate splits; the average score is more trustworthy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = GridSearchCV(build_pipeline(), grid, scoring="f1_macro", cv=cv, n_jobs=1)
search.fit(X_tr, y_tr)
best = search.best_estimator_
print("Best params:", search.best_params_, f"CV macro-F1={search.best_score_:.3f}")

# 🧠 ML: the test set was never seen during tuning - this is the honest performance estimate
pred = best.predict(X_te)
print(f"Test macro-F1: {f1_score(y_te, pred, average='macro'):.3f}")
print("Confusion (rows=true A/R/D):", confusion_matrix(y_te, pred, labels=LABELS).tolist())

# 🧠 ML: baseline = today's manual rule 'refer everything with any prior claim, else accept'
baseline = np.where(X_te.prior_claims_5yr > 0, "Refer", "Accept")
print(f"Business cost  model=${business_cost(y_te, pred):,}  vs  rule-of-thumb=${business_cost(y_te, baseline):,}")

# 🧠 ML: permutation importance = drop in score when one column is shuffled (model-agnostic check)
pi = permutation_importance(best, X_te, y_te, scoring="f1_macro", n_repeats=5, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
print("\nPermutation importance (macro-F1 drop):")
for col, v in imp.head(6).items():
    print(f"  {col:24s} {v:+.3f} {'#' * max(0, int(v * 60))}")

# 🧠 ML: a shallow 'reason code' view: the printed tree IS the model, auditable line by line
# 🐍 PY: named_steps is a dict-like view of the pipeline stages by their names
names = list(best.named_steps["pre"].get_feature_names_out())
rules = export_text(best.named_steps["tree"], feature_names=names, max_depth=2)
print("\nTop rules (depth<=2):")
print("\n".join(rules.splitlines()[:12]))

# 🧠 ML: decision_path gives the exact rules one applicant passed through - an adverse-action reason
app = X_te.iloc[[0]]
t = best.named_steps["tree"]
Xa = best.named_steps["pre"].transform(app)
# 🐍 PY: .indices lists node ids on the path; the last one is the leaf, so we skip it below
node_ids = t.decision_path(Xa).indices
print(f"\nApplicant 0 -> {best.predict(app)[0]}; path:")
for nid in node_ids[:-1]:
    f, thr = t.tree_.feature[nid], t.tree_.threshold[nid]
    side = "<=" if Xa[0, f] <= thr else ">"
    print(f"  {names[f]} = {Xa[0, f]:.1f} {side} {thr:.1f}")

# Regulatory note: NAIC model-governance bulletins + state rules on credit-based insurance scores
# require explainable decisions, documented variables, and periodic unfair-discrimination testing.
print("\nModel-risk note: credit_insurance_score use must follow state law; log rules + version.")
# 🧠 ML: save the WHOLE pipeline (encoder + tree) so scoring uses identical preprocessing
out = os.path.join(tempfile.mkdtemp(), "ho_underwriting_tree.joblib")
joblib.dump(best, out)
print("Saved model ->", os.path.basename(out), f"({os.path.getsize(out)} bytes)")

# ---
# Decision Tree — HARD lab · Insurance: Homeowners underwriting decision: Accept / Refer / Decline
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
