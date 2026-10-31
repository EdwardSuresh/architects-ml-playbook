# EASY: linear regression by hand - expected yearly claim cost for an auto policy
import numpy as np

# 🐍 PY: a list of dicts - each dict is one policyholder with named fields
policies = [
    {"driver_age": 19, "annual_miles_k": 14, "claim_cost": 1650},
    {"driver_age": 22, "annual_miles_k": 12, "claim_cost": 1380},
    {"driver_age": 25, "annual_miles_k": 15, "claim_cost": 1290},
    {"driver_age": 30, "annual_miles_k": 10, "claim_cost": 900},
    {"driver_age": 35, "annual_miles_k": 12, "claim_cost": 870},
    {"driver_age": 40, "annual_miles_k": 8, "claim_cost": 640},
    {"driver_age": 45, "annual_miles_k": 11, "claim_cost": 700},
    {"driver_age": 50, "annual_miles_k": 9, "claim_cost": 560},
    {"driver_age": 55, "annual_miles_k": 13, "claim_cost": 690},
    {"driver_age": 60, "annual_miles_k": 7, "claim_cost": 480},
]
# 🐍 PY: list comprehensions pull one field out of every dict
age = np.array([p["driver_age"] for p in policies], dtype=float)
miles = np.array([p["annual_miles_k"] for p in policies], dtype=float)
cost = np.array([p["claim_cost"] for p in policies], dtype=float)

# 🧠 ML: linear regression = find the straight-line formula cost = b0 + b1*age + b2*miles with least error
# 🐍 PY: np.column_stack builds the design matrix; the column of ones lets the model learn an intercept b0
X = np.column_stack([np.ones(len(age)), age, miles])
# 🧠 ML: least squares picks the coefficients that minimise the sum of squared errors
coef, *_ = np.linalg.lstsq(X, cost, rcond=None)
b0, b_age, b_miles = coef
print(f"cost = {b0:.0f} {b_age:+.1f} x age {b_miles:+.1f} x miles_k")
print(f"-> each extra year of age changes expected cost by ${b_age:.1f}")
print(f"-> each extra 1,000 miles driven adds ${b_miles:.1f}\n")

# 🐍 PY: @ multiplies the matrix by the coefficient vector -> one prediction per row
pred = X @ coef
print(" age | miles_k | actual | predicted | error")
# 🐍 PY: zip walks four arrays together in one loop
for a, m, c, p in zip(age, miles, cost, pred):
    print(f"{a:>4.0f} | {m:>7.0f} | {c:>6.0f} | {p:>9.0f} | {c - p:>+5.0f}")

# 🧠 ML: R-squared = share of the variation in cost explained by the model (1.0 = perfect)
# 🧠 ML: notice errors are positive at BOTH ends (age 19 and 60) - a curve, not a line, fits risk vs age
r2 = 1 - np.sum((cost - pred) ** 2) / np.sum((cost - cost.mean()) ** 2)
print(f"\nR-squared: {r2:.3f}")

# Price a new policy: pure premium = expected claim cost, then add a 25% expense/profit load
new_driver = np.array([1, 28, 16])
pure = new_driver @ coef
print(f"New 28-year-old driving 16k miles: expected claims ${pure:.0f}, premium ${pure * 1.25:.0f}")

# ---
# Linear Regression — EASY lab · Insurance: Auto insurance expected claim cost and premium pricing
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
