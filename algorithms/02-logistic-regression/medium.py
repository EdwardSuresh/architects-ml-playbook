# BUILD: scikit-learn logistic regression for consumer loan probability of default (PD)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

rng = np.random.default_rng(42)
n = 4000
# 🐍 PY: np.clip keeps simulated values inside realistic bounds
df = pd.DataFrame({
    "credit_score": np.clip(rng.normal(690, 60, n), 450, 850).round(),
    "debt_to_income": np.clip(rng.normal(32, 10, n), 2, 70).round(1),
    "annual_income_k": np.clip(rng.lognormal(4.1, 0.45, n), 15, 400).round(1),
    "loan_amount_k": np.clip(rng.lognormal(2.6, 0.5, n), 1, 60).round(1),
    "delinq_2y": rng.poisson(0.3, n),
    "months_employed": rng.integers(0, 240, n),
})
logit = (-2.4 - 0.018 * (df.credit_score - 690) + 0.05 * (df.debt_to_income - 32)
         + 0.55 * df.delinq_2y - 0.004 * df.months_employed
         + 0.4 * np.log(df.loan_amount_k / df.annual_income_k * 10))
df["default_12m"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Loans: {n}  12-month default rate: {df.default_12m.mean():.1%}")

# 🐍 PY: keeping column names in one list avoids typos when selecting the same columns again
features = ["credit_score", "debt_to_income", "annual_income_k", "loan_amount_k",
            "delinq_2y", "months_employed"]
X_tr, X_te, y_tr, y_te = train_test_split(df[features], df.default_12m, test_size=0.25,
                                          stratify=df.default_12m, random_state=42)
# 🧠 ML: fit the scaler on TRAIN only, then apply to test - otherwise test info leaks into training
scaler = StandardScaler().fit(X_tr)
model = LogisticRegression(max_iter=1000)
model.fit(scaler.transform(X_tr), y_tr)

# 🐍 PY: method chaining - transform the test rows, then predict, in one expression
pd_te = model.predict_proba(scaler.transform(X_te))[:, 1]
auc = roc_auc_score(y_te, pd_te)
# 🧠 ML: Gini = 2*AUC - 1 is the number credit-risk teams quote; Brier score checks calibration
print(f"AUC {auc:.3f} | Gini {2 * auc - 1:.3f} | Brier {brier_score_loss(y_te, pd_te):.3f}")

# 🧠 ML: coefficients on scaled data are comparable - sign = direction, size = strength
coefs = pd.Series(model.coef_[0], index=features).sort_values()
print("\nCoefficient (per 1 std) and odds ratio:")
for f, c in coefs.items():
    print(f"  {f:<16} {c:+.3f}   odds x{np.exp(c):.2f}")

# 🧠 ML: calibration check - average predicted PD vs actual default rate, by risk grade (quintile)
# 🐍 PY: pd.qcut splits values into equal-sized buckets; groupby then aggregates each bucket
res = pd.DataFrame({"pd": pd_te, "actual": y_te.values})
res["grade"] = pd.qcut(res.pd, 5, labels=["A-best", "B", "C", "D", "E-worst"])
table = res.groupby("grade", observed=True).agg(loans=("pd", "size"), predicted=("pd", "mean"),
                                                 actual=("actual", "mean"))
print("\nRisk grade calibration:")
print(table.round(3).to_string())

# Business interpretation: approve if PD < 10%
# 🐍 PY: comparing an array to a number gives a boolean mask; mask.mean() = share of True values
approve = pd_te < 0.10
print(f"\nPolicy PD < 10%: approve {approve.mean():.0%} of applicants, "
      f"bad rate among approved {y_te.values[approve].mean():.1%} "
      f"vs {y_te.values[~approve].mean():.1%} among declined")

# ---
# Logistic Regression — BUILD lab · Financial Services: Consumer loan default prediction and credit scorecard
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
