from sklearn.tree import DecisionTreeClassifier, export_text

# 🐍 PY: a list of lists = a tiny table typed by hand; each inner list is one homeowners application
# columns: roof_age_yrs, prior_claims_5yr, wildfire_zone (0/1), home_value_k
applications = [
    [5, 0, 0, 250], [3, 0, 0, 410], [8, 1, 0, 300], [12, 0, 0, 180],
    [18, 1, 0, 220], [22, 2, 0, 275], [15, 0, 1, 350], [6, 0, 1, 520],
    [25, 3, 1, 190], [28, 2, 0, 160], [20, 3, 1, 240], [10, 1, 1, 330],
]
# 🐍 PY: a matching list of answers (the label), one per application, same order
decisions = ["Accept", "Accept", "Accept", "Accept",
             "Refer", "Refer", "Refer", "Refer",
             "Decline", "Decline", "Decline", "Refer"]
features = ["roof_age_yrs", "prior_claims_5yr", "wildfire_zone", "home_value_k"]

# 🧠 ML: Gini impurity = chance two random rows in a group disagree; 0 means the group is "pure"
def gini(labels):
    # 🐍 PY: set() removes duplicates so we loop once per distinct decision
    shares = [labels.count(c) / len(labels) for c in set(labels)]
    return 1 - sum(p * p for p in shares)

print(f"Gini of all 12 applications: {gini(decisions):.3f}")

# 🧠 ML: try one candidate question by hand: "roof_age_yrs <= 13?" and measure how pure each side gets
# 🐍 PY: zip pairs each application with its decision so we can filter both together
left = [d for row, d in zip(applications, decisions) if row[0] <= 13]
right = [d for row, d in zip(applications, decisions) if row[0] > 13]
print(f"roof <= 13 -> {left}  gini={gini(left):.3f}")
print(f"roof  > 13 -> {right} gini={gini(right):.3f}")

# 🧠 ML: the tree repeats this search over every column and threshold, keeping the purest split
# 🧠 ML: max_depth=3 keeps the rule book short enough for an underwriter to read
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(applications, decisions)

# 🧠 ML: export_text prints the learned if-else rules - this is the "explainable" part
print("\nLearned underwriting rules:")
print(export_text(tree, feature_names=features))

# 🐍 PY: predict expects a list of rows, so wrap a single new application in an outer list
new_app = [[14, 2, 1, 280]]
print("New application", new_app[0], "->", tree.predict(new_app)[0])

# ---
# Decision Tree — EASY lab · Insurance: Homeowners underwriting decision: Accept / Refer / Decline
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
