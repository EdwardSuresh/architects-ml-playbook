import numpy as np
from sklearn.ensemble import IsolationForest

# 🐍 PY: a list of tuples = 10 wire transfers typed by hand: (amount_usd, hour_of_day, new_country 0/1)
txns = [
    (1200, 10, 0), (850, 11, 0), (1500, 14, 0), (990, 9, 0), (1100, 15, 0),
    (1300, 13, 0), (700, 16, 0), (1250, 10, 0), (9800, 3, 1), (1050, 12, 0),
]
# 🐍 PY: np.array turns the list into a 10x3 numeric grid that sklearn can read
X = np.array(txns)
print("Shape of data:", X.shape)

# 🧠 ML: core idea - an odd point gets isolated with FEW random cuts; a normal one needs many
rng = np.random.default_rng(42)
amounts = X[:, 0]
# 🐍 PY: a function that keeps cutting at random until only the target value is left
def cuts_to_isolate(values, target, rng):
    cuts = 0
    while len(values) > 1:
        # 🧠 ML: pick a random split point between the current min and max
        split = rng.uniform(values.min(), values.max())
        # 🐍 PY: boolean mask keeps only the side of the split that holds our target
        values = values[values < split] if target < split else values[values >= split]
        cuts += 1
    return cuts

# 🧠 ML: average over many random trees, exactly what a forest does
for idx in [0, 8]:
    # 🐍 PY: list comprehension repeats the experiment 200 times
    runs = [cuts_to_isolate(amounts, amounts[idx], rng) for _ in range(200)]
    print(f"txn {idx} amount={amounts[idx]:>6.0f}: avg cuts to isolate = {np.mean(runs):.2f}")

# 🧠 ML: contamination=0.1 tells the model we expect about 1 in 10 to be suspicious
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)
# 🧠 ML: predict returns -1 for anomaly, +1 for normal; score_samples: lower = more unusual
labels = model.predict(X)
scores = model.score_samples(X)

print("\n  #  amount hour newCtry  score  flag")
# 🐍 PY: enumerate gives the row number; zip walks three sequences side by side
for i, (row, s, lab) in enumerate(zip(txns, scores, labels)):
    flag = "ALERT" if lab == -1 else ""
    print(f"{i:3d} {row[0]:7d} {row[1]:4d} {row[2]:7d} {s:6.3f}  {flag}")

# ---
# Isolation Forest — EASY lab · Financial Services: AML suspicious transaction detection for the analyst alert queue
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
