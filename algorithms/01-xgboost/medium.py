# MEDIUM: XGBoost on 5,000 synthetic card authorizations with ~1-2% fraud
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

# 🐍 PY: default_rng(42) is numpy's modern random generator; the seed makes results repeatable
rng = np.random.default_rng(42)
n = 5000

# 🐍 PY: a DataFrame built from a dict - each key becomes a column name
df = pd.DataFrame({
    "txn_amount": rng.lognormal(3.6, 1.0, n).round(2),
    "merchant_risk": rng.uniform(0, 1, n).round(3),
    "km_from_home": rng.exponential(25, n).round(1),
    "hour": rng.integers(0, 24, n),
    "card_present": rng.binomial(1, 0.7, n),
    "txns_last_24h": rng.poisson(3, n),
    "acct_age_days": rng.integers(30, 3650, n),
})

# Hidden "true" fraud pattern, including a non-linear interaction trees can discover
night = (df["hour"] < 5).astype(int)
far_cnp = ((df["km_from_home"] > 60) & (df["card_present"] == 0)).astype(int)
logit = (-10.0 + 0.8 * np.log1p(df["txn_amount"]) + 3.0 * df["merchant_risk"] + 1.8 * night
         + 3.5 * far_cnp + 0.3 * df["txns_last_24h"] - 0.0005 * df["acct_age_days"])
# 🧠 ML: the sigmoid turns a score into a probability; binomial draws 1 (fraud) with that probability
df["is_fraud"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Rows: {len(df)}  fraud cases: {df.is_fraud.sum()}  fraud rate: {df.is_fraud.mean():.2%}")

X = df.drop(columns="is_fraud")
y = df["is_fraud"]
# 🧠 ML: stratify=y keeps the same tiny fraud % in train and test so the test set is not fraud-free
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 🧠 ML: scale_pos_weight up-weights rare fraud rows (~ genuine count / fraud count) to fight imbalance
spw = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                      n_jobs=2, random_state=42)
model.fit(X_train, y_train)
print(f"scale_pos_weight used: {spw:.1f}")

# 🐍 PY: [:, 1] slices column 1 = probability of the positive class (fraud)
proba = model.predict_proba(X_test)[:, 1]
# 🧠 ML: ROC-AUC ranks all cases; PR-AUC focuses on the rare class and is the honest metric for fraud
print(f"\nROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(f"PR-AUC : {average_precision_score(y_test, proba):.3f}  (random model = {y_test.mean():.3f})")

print("\nThreshold | caught fraud | missed | false alarms | precision")
for t in [0.5, 0.7, 0.9]:
    # 🐍 PY: .ravel() flattens the 2x2 matrix so we can unpack 4 numbers in one line
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= t).astype(int)).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0
    print(f"{t:>9} | {tp:>12} | {fn:>6} | {fp:>12} | {prec:>9.2f}")

# 🧠 ML: importance shows which signals drive the score - useful for analysts and model validators
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop drivers:")
for col, v in imp.head(5).items():
    print(f"  {col:<15} {'#' * int(v * 50)} {v:.2f}")

# Business view: dollars of fraud stopped at the 0.5 threshold
flag = proba >= 0.5
# 🐍 PY: boolean masks combine with & ; .sum() on dollars gives totals
caught = X_test.loc[flag & (y_test == 1).values, "txn_amount"].sum()
missed = X_test.loc[~flag & (y_test == 1).values, "txn_amount"].sum()
print(f"\nAt threshold 0.5: fraud $ stopped = ${caught:,.0f}, fraud $ missed = ${missed:,.0f}")
print(f"Customers inconvenienced (declines/step-up) = {int(flag.sum())} of {len(flag)} swipes")

# ---
# XGBoost — MEDIUM lab · Financial Services: Real-time credit card fraud detection
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
