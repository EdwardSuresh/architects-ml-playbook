# MEDIUM: Random Forest on 4,000 synthetic auto claims to prioritise SIU referrals
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

rng = np.random.default_rng(42)
n = 4000
df = pd.DataFrame({
    "days_policy_to_claim": rng.integers(1, 1500, n),
    "claim_amount": rng.lognormal(8.6, 0.8, n).round(0),
    "police_report": rng.binomial(1, 0.65, n),
    "num_witnesses": rng.poisson(0.9, n),
    "prior_claims_3y": rng.poisson(0.4, n),
    "injury_claim": rng.binomial(1, 0.25, n),
    "vehicle_age_yrs": rng.integers(0, 20, n),
    "attorney_involved": rng.binomial(1, 0.15, n),
})
# Hidden fraud rule set: brand-new policy + injury claim, or no report + no witness + big amount
pattern1 = (df.days_policy_to_claim < 120) & (df.injury_claim == 1)
pattern2 = (df.police_report == 0) & (df.num_witnesses == 0) & (df.claim_amount > 7000)
logit = (-4.2 + 3.0 * pattern1 + 3.0 * pattern2 + 1.2 * df.attorney_involved
         + 0.5 * df.prior_claims_3y + 0.02 * df.vehicle_age_yrs)
df["siu_confirmed_fraud"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Claims: {n}  confirmed fraud rate: {df.siu_confirmed_fraud.mean():.1%}")

# 🐍 PY: drop(columns=...) returns a new DataFrame without the label - the original df is unchanged
X = df.drop(columns="siu_confirmed_fraud")
y = df["siu_confirmed_fraud"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 🧠 ML: class_weight='balanced_subsample' re-weights rare fraud inside every bootstrap sample
# 🧠 ML: min_samples_leaf=10 stops trees memorising single claims (less overfitting)
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features="sqrt",
                            class_weight="balanced_subsample", oob_score=True,
                            n_jobs=2, random_state=42)
rf.fit(X_tr, y_tr)
# 🧠 ML: out-of-bag score = accuracy on rows each tree did NOT see - a free built-in validation
print(f"Out-of-bag accuracy: {rf.oob_score_:.3f}")

proba = rf.predict_proba(X_te)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
# 🧠 ML: precision = share of referrals that are fraud; recall = share of all fraud we referred
print(classification_report(y_te, (proba >= 0.5).astype(int), target_names=["legit", "fraud"], digits=2))

# 🧠 ML: SIU has limited investigators - rank claims by score and check the top slice (lift)
# 🐍 PY: argsort()[::-1] gives indices from highest to lowest score; slicing takes the top k
order = np.argsort(proba)[::-1]
base_rate = y_te.mean()
print("SIU capacity | fraud found | hit rate | lift vs random")
for pct in [0.05, 0.10, 0.20]:
    k = int(len(order) * pct)
    hits = y_te.values[order[:k]].sum()
    print(f"top {pct:>4.0%}     | {hits:>11} | {hits / k:>8.1%} | {hits / k / base_rate:>6.1f}x")

# 🧠 ML: impurity importance favours many-valued columns (vehicle_age) - verify with permutation importance
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop red-flag drivers:")
for col, v in imp.head(5).items():
    print(f"  {col:<22} {'#' * int(v * 60)} {v:.3f}")

k = int(len(order) * 0.10)
# 🐍 PY: .loc selects rows by index label; .query filters with a readable string expression
saved = df.loc[X_te.index[order[:k]]].query("siu_confirmed_fraud == 1").claim_amount.sum()
print(f"\nInvestigating top 10% ({k} claims) targets ${saved:,.0f} of fraudulent payouts")

# ---
# Random Forest — MEDIUM lab · Insurance: Auto insurance claims fraud detection (SIU referral)
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
