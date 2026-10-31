# EASY: gradient boosting from scratch - each tiny tree fixes the last one's errors on late-shipment risk
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# 🐍 PY: two parallel lists - distance in miles and a storm flag (1 = storm on the lane) per shipment
distance_mi = [120, 300, 450, 800, 950, 1200, 1500, 1800, 200, 650, 1100, 1700]
storm = [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
# 1 = delivered late, 0 = on time
late = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
# 🐍 PY: zip pairs items position by position; list(...) materialises the pairs as rows
X = np.array(list(zip(distance_mi, storm)), dtype=float)

# 🧠 ML: start with one constant guess for everyone - the overall late rate
pred = np.full(len(late), late.mean())
print(f"Starting guess for every shipment: {late.mean():.2f}\n")
learning_rate = 0.5
trees = []

for round_no in range(1, 6):
    # 🧠 ML: residual = what the current model still gets wrong (actual minus predicted)
    residual = late - pred
    # 🧠 ML: fit a tiny 'stump' (depth 1 tree) to the residuals, not to the original label
    stump = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, residual)
    # 🧠 ML: add only a fraction (learning rate) of the correction - small steps avoid overshooting
    pred = pred + learning_rate * stump.predict(X)
    trees.append(stump)
    # 🐍 PY: ['distance', 'storm'][index] picks a name by position from a literal list
    feat = ["distance", "storm"][stump.tree_.feature[0]]
    err = np.mean((late - pred) ** 2)
    print(f"round {round_no}: split on {feat:<8} at {stump.tree_.threshold[0]:>7.1f} | squared error {err:.3f}")

print("\nShipment | miles | storm | actual | predicted risk")
# 🐍 PY: enumerate gives index i; slicing [:6] shows just the first 6 rows
for i, (d, s) in enumerate(X[:6]):
    print(f"{i:>8} | {d:>5.0f} | {s:>5.0f} | {late[i]:>6} | {pred[i]:>6.2f}")

# Score a brand-new 1,400-mile shipment through a storm by summing all trees
new = np.array([[1400, 1]])
# 🐍 PY: sum() over a generator expression adds up each tree's contribution
raw = late.mean() + learning_rate * sum(t.predict(new)[0] for t in trees)
# 🧠 ML: squared-error boosting can overshoot 1.0; real classifiers boost log-odds (log-loss) instead
risk = min(max(raw, 0.0), 1.0)
print(f"\nNew 1,400 mi storm lane -> raw score {raw:.2f}, clipped risk {risk:.2f} -> "
      f"{'expedite / warn customer' if risk > 0.5 else 'standard'}")

# ---
# Gradient Boosting — EASY lab · Supply Chain: Predict late shipments (on-time delivery risk)
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
