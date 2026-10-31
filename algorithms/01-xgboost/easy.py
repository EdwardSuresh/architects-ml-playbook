# EASY: XGBoost learns to spot card fraud from 18 hand-typed authorizations
import numpy as np
from xgboost import XGBClassifier

# 🐍 PY: a list of lists - each inner list is one swipe: [amount_usd, km_from_home, hour, is_online]
swipes = [
    [12.5, 2, 9, 0], [45.0, 5, 13, 0], [8.9, 1, 8, 0], [60.0, 3, 18, 0],
    [25.0, 4, 12, 1], [33.3, 6, 20, 1], [15.0, 2, 10, 0], [75.0, 8, 19, 0],
    [22.0, 3, 14, 1], [40.0, 5, 17, 0], [18.0, 1, 11, 0], [55.0, 7, 16, 1],
    [820.0, 3, 15, 0],
    [950.0, 4200, 3, 1], [780.0, 3100, 2, 1], [120.0, 5000, 4, 1], [640.0, 9, 1, 0],
    [35.0, 4800, 3, 1],
]
# 🐍 PY: [0] * 13 repeats a list; + joins lists -> 13 genuine swipes then 5 fraud swipes
labels = [0] * 13 + [1] * 5
# 🐍 PY: np.array turns Python lists into fast numeric arrays that ML libraries expect
X = np.array(swipes)
y = np.array(labels)
print("Data shape:", X.shape, "| fraud rows:", y.sum(), "| fraud rate:", f"{y.mean():.0%}")

# 🧠 ML: XGBoost builds many small trees; each new tree corrects the mistakes of all trees before it
# 🧠 ML: max_depth=2 keeps each tree tiny; learning_rate=0.3 means each tree only nudges the score
model = XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.3,
                      min_child_weight=0.5, random_state=42)
model.fit(X, y)

# Two brand-new swipes the model has never seen: a coffee and a 3am online purchase far from home
new_swipes = np.array([[30.0, 3, 12, 0], [880.0, 3900, 3, 1]])

# 🧠 ML: watch the fraud probability sharpen as more trees are added (iteration_range = use first k trees)
print("\nTrees used | P(fraud) coffee swipe | P(fraud) 3am online swipe")
for k in [1, 3, 5, 10, 20]:
    p = model.predict_proba(new_swipes, iteration_range=(0, k))[:, 1]
    # 🐍 PY: f-string format spec :>10 right-aligns in 10 characters, .3f = 3 decimals
    print(f"{k:>10} | {p[0]:>20.3f} | {p[1]:>25.3f}")

# 🧠 ML: feature importance = how much each column helped the trees separate fraud from genuine
# 🧠 ML: with only 18 rows one clean split can separate everything, so one feature may take all credit
names = ["amount_usd", "km_from_home", "hour_of_day", "is_online"]
print("\nFeature importance (gain share):")
# 🐍 PY: zip pairs names with scores; sorted(key=lambda ...) orders them, the minus sign = descending
for name, imp in sorted(zip(names, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:<13} {'#' * int(imp * 40):<40} {imp:.2f}")

# 🧠 ML: the bank chooses a cut-off; above it the authorization is declined or sent to step-up checks
probs = model.predict_proba(new_swipes)[:, 1]
# 🐍 PY: a list comprehension with an inline if/else builds one decision per swipe
decisions = ["DECLINE/REVIEW" if p > 0.5 else "APPROVE" for p in probs]
print("\nDecisions:", decisions)

# ---
# XGBoost — EASY lab · Financial Services: Real-time credit card fraud detection
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
