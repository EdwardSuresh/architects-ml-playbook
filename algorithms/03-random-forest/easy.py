# CONCEPT: a hand-built "forest" of 3 tiny trees votes on whether an auto claim goes to SIU
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 🐍 PY: a dict of lists - each key is a column, each position is one claim
claims = {
    "days_policy_to_claim": [400, 20, 900, 15, 650, 30, 1200, 10, 300, 45, 800, 25, 500, 12],
    "claim_amount_k":       [3.2, 18.0, 2.1, 22.5, 4.0, 15.0, 5.5, 25.0, 2.8, 4.1, 6.0, 19.5, 3.5, 3.0],
    "police_report":        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    "witnesses":            [2, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 1],
}
is_fraud = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0])
# 🐍 PY: list(claims) gives the keys; np.column_stack glues the lists side by side as columns
names = list(claims)
X = np.column_stack([claims[k] for k in names])
print("Claims:", X.shape[0], "| known fraud:", is_fraud.sum())

# 🧠 ML: Random Forest = many decision trees, each trained on a random resample of rows (bagging)
# 🧠 ML: max_features=2 means each split only looks at 2 random columns -> trees disagree usefully
rf = RandomForestClassifier(n_estimators=3, max_depth=2, max_features=2, random_state=42)
rf.fit(X, is_fraud)

new_claim = np.array([[18, 21.0, 0, 0]])  # new policy, big claim, no police report, no witnesses
print("\nEach tree's vote on the new claim:")
# 🐍 PY: enumerate gives a counter (i) alongside each item (tree) in the loop
for i, tree in enumerate(rf.estimators_):
    # 🧠 ML: each tree saw a different bootstrap sample, so it may learn a different rule
    vote = tree.predict(new_claim)[0]
    root = names[tree.tree_.feature[0]]
    print(f"  tree {i + 1}: first question asks about '{root}' -> vote {'FRAUD' if vote else 'ok'}")

# 🧠 ML: the forest averages the trees' probabilities - the 'wisdom of the crowd'
p = rf.predict_proba(new_claim)[0, 1]
print(f"\nForest fraud probability: {p:.2f} -> {'Refer to SIU' if p >= 0.5 else 'Fast-track payment'}")

# Now a bigger forest of 100 trees for a steadier answer
rf100 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X, is_fraud)
honest_claim = np.array([[700, 3.0, 1, 2]])
print(f"100-tree forest: suspicious claim {rf100.predict_proba(new_claim)[0, 1]:.2f}, "
      f"routine claim {rf100.predict_proba(honest_claim)[0, 1]:.2f}")

# 🧠 ML: feature importance averages how much each column cleaned up the splits across all trees
print("\nWhat the forest relies on:")
for n, imp in sorted(zip(names, rf100.feature_importances_), key=lambda t: t[1], reverse=True):
    print(f"  {n:<22} {'*' * int(imp * 30)} {imp:.2f}")

# ---
# Random Forest — CONCEPT lab · Insurance: Auto insurance claims fraud detection (SIU referral)
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
