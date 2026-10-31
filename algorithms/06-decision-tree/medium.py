import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🧠 ML: fixed seed = same synthetic book of business every run, so results are reproducible
np.random.seed(42)
n = 3000

# 🐍 PY: a dict of column-name -> numpy array becomes a pandas DataFrame in one line
df = pd.DataFrame({
    "roof_age_yrs": np.random.randint(0, 35, n),
    "prior_claims_5yr": np.random.poisson(0.6, n),
    "wildfire_zone": np.random.binomial(1, 0.2, n),
    "dist_fire_station_mi": np.round(np.random.gamma(2, 2.5, n), 1),
    "credit_insurance_score": np.random.randint(500, 900, n),
    "home_value_k": np.random.randint(120, 900, n),
})

# 🧠 ML: hidden "true" underwriting risk points - the tree must rediscover this logic from data
risk = (
    (df.roof_age_yrs > 20) * 2 + df.prior_claims_5yr * 1.5 + df.wildfire_zone * 2
    + (df.dist_fire_station_mi > 10) * 1 + (df.credit_insurance_score < 620) * 1.5
    + np.random.normal(0, 0.8, n)
)
# 🐍 PY: np.select picks a label per row from the first condition that is True
df["decision"] = np.select([risk < 2.5, risk < 5], ["Accept", "Refer"], default="Decline")
print("Decision mix:", df.decision.value_counts().to_dict())

# 🐍 PY: drop() returns a copy without the label column; X = inputs, y = answer
X = df.drop(columns="decision")
y = df["decision"]
# 🧠 ML: stratify keeps the Accept/Refer/Decline mix the same in train and test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 🧠 ML: min_samples_leaf=30 stops the tree making rules from a handful of policies (overfitting)
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)
tree.fit(X_tr, y_tr)
pred = tree.predict(X_te)

print(f"Train accuracy: {tree.score(X_tr, y_tr):.3f}")
print(f"Test accuracy : {accuracy_score(y_te, pred):.3f}\n")
labels = ["Accept", "Refer", "Decline"]
# 🧠 ML: confusion matrix rows = true decision, columns = model decision
cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
print("Confusion matrix (rows=true):\n", cm, "\n")
print(classification_report(y_te, pred, digits=2))

# 🧠 ML: feature_importances_ = how much each column reduced impurity across all splits
imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance:")
# 🐍 PY: .items() yields (index, value) pairs; '#' * k builds a simple ASCII bar
for name, v in imp.items():
    print(f"  {name:24s} {v:.3f} {'#' * int(v * 40)}")

# 🧠 ML: only the top 2 levels (depth 0-1) printed so an underwriter can read the core rules
rules = export_text(tree, feature_names=list(X.columns), max_depth=1)
print("\nTop of the rule book:\n" + rules)

# Business read-out
dec_as_acc = ((y_te == "Decline") & (pred == "Accept")).sum()
print(f"Business check: {dec_as_acc} true-Decline risks auto-accepted out of {len(y_te)} test apps")

# ---
# Decision Tree — MEDIUM lab · Insurance: Homeowners underwriting decision: Accept / Refer / Decline
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
