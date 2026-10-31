# MEDIUM: scikit-learn linear regression for expected auto claim cost (pure premium) per policy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(42)
n = 4000
df = pd.DataFrame({
    "driver_age": rng.integers(18, 80, n),
    "annual_miles_k": rng.normal(12, 4, n).clip(2, 30).round(1),
    "vehicle_value_k": rng.lognormal(3.2, 0.4, n).round(1),
    "prior_claims_3y": rng.poisson(0.3, n),
    "credit_tier": rng.choice(["A", "B", "C"], n, p=[.5, .35, .15]),
    "territory": rng.choice(["urban", "suburban", "rural"], n, p=[.35, .45, .2]),
})
# Hidden truth: young drivers costlier, more miles, pricier car, prior claims, urban territory
# 🐍 PY: np.maximum compares element-wise - drivers over 30 get 0 extra risk
young = np.maximum(0, 30 - df.driver_age)
df["claim_cost"] = (420 + 28 * young + 22 * df.annual_miles_k + 6 * df.vehicle_value_k
                    + 210 * df.prior_claims_3y + df.territory.map({"urban": 260, "suburban": 90, "rural": 0})
                    + df.credit_tier.map({"A": 0, "B": 70, "C": 160}) + rng.normal(0, 180, n)).round(0)
print(f"Policies: {n}  mean annual claim cost: ${df.claim_cost.mean():,.0f}")

num = ["driver_age", "annual_miles_k", "vehicle_value_k", "prior_claims_3y"]
cat = ["credit_tier", "territory"]
X_tr, X_te, y_tr, y_te = train_test_split(df[num + cat], df.claim_cost, test_size=0.25, random_state=42)

# 🧠 ML: one-hot turns each category into 0/1 columns; drop='first' avoids perfectly redundant columns
# 🐍 PY: a ColumnTransformer sends different columns to different preprocessing steps
pre = ColumnTransformer([("num", "passthrough", num),
                         ("cat", OneHotEncoder(drop="first"), cat)])
model = Pipeline([("pre", pre), ("lr", LinearRegression())])
model.fit(X_tr, y_tr)

# 🧠 ML: MAE = average dollar miss per policy; R-squared = share of cost variation explained
pred = model.predict(X_te)
print(f"Test MAE ${mean_absolute_error(y_te, pred):,.0f} | R-squared {r2_score(y_te, pred):.3f}")

# 🧠 ML: each coefficient is the $ change in expected cost for +1 unit, holding everything else fixed
names = model.named_steps["pre"].get_feature_names_out()
coefs = model.named_steps["lr"].coef_
print(f"\nIntercept: ${model.named_steps['lr'].intercept_:,.0f}")
# 🐍 PY: str.split('__')[1] strips the 'num__' / 'cat__' prefix added by ColumnTransformer
for nm, c in zip(names, coefs):
    print(f"  {nm.split('__')[1]:<18} {c:>+8.1f} $")

# 🧠 ML: driver_age is linear here, but true risk only rises below 30 - residuals by age band reveal it
res = X_te.assign(actual=y_te.values, pred=pred)
res["age_band"] = pd.cut(res.driver_age, [17, 25, 35, 50, 80])
check = res.groupby("age_band", observed=True)[["actual", "pred"]].mean()
print("\nAge band | actual $ | predicted $ | gap")
# 🐍 PY: itertuples is a fast way to loop rows as named tuples
for row in check.itertuples():
    print(f"{str(row.Index):<9}| {row.actual:>8.0f} | {row.pred:>11.0f} | {row.actual - row.pred:>+5.0f}")

# Business: pure premium + 30% expense & profit load for two quotes
# 🐍 PY: a small DataFrame of new quotes must use the same column names as the training data
quotes = pd.DataFrame({"driver_age": [20, 45], "annual_miles_k": [15, 10], "vehicle_value_k": [18, 35],
                       "prior_claims_3y": [1, 0], "credit_tier": ["B", "A"], "territory": ["urban", "rural"]})
for (_, q), pp in zip(quotes.iterrows(), model.predict(quotes)):
    print(f"Quote age {q.driver_age}, {q.territory}: pure premium ${pp:,.0f} -> charged ${pp * 1.3:,.0f}")

# ---
# Linear Regression — MEDIUM lab · Insurance: Auto insurance expected claim cost and premium pricing
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
