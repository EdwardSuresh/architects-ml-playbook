import numpy as np

# 🐍 PY: each row is one bank customer: [avg_monthly_balance_k, card_txns_per_month]
customers = np.array([
    [1.2, 45], [0.8, 50], [1.5, 40], [1.0, 55],      # young, low balance, card-heavy
    [25.0, 8], [30.0, 5], [22.0, 10], [28.0, 6],     # older savers, high balance, few txns
    [8.0, 25], [10.0, 22], [9.0, 28], [7.5, 20],     # mid-life families
])
# 🐍 PY: a list of strings, one per customer, so we can label rows when printing
names = [f"C{i:02d}" for i in range(len(customers))]

# 🧠 ML: scale columns so 'balance in $k' and 'txn count' contribute fairly to distance
# 🐍 PY: numpy broadcasting subtracts each column's mean from every row at once
X = (customers - customers.mean(axis=0)) / customers.std(axis=0)

k = 3
# 🧠 ML: start with k centroids picked from the data (rows 0, 1, 4 - a deliberately poor start)
centroids = X[[0, 1, 4]].copy()

# 🐍 PY: range(5) loops five times; k-Means usually settles in a handful of rounds
for it in range(5):
    # 🧠 ML: distance from every customer to every centroid -> shape (12, 3)
    # 🐍 PY: X[:, None, :] adds an axis so broadcasting compares all pairs in one line
    dist = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    # 🧠 ML: assignment step - each customer joins its nearest centroid
    labels = dist.argmin(axis=1)
    # 🧠 ML: update step - each centroid moves to the average of its members
    new_c = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    # 🧠 ML: inertia = total squared distance to own centroid; smaller = tighter segments
    inertia = ((X - new_c[labels]) ** 2).sum()
    print(f"iter {it}: labels={labels.tolist()} inertia={inertia:.3f}")
    # 🐍 PY: np.allclose checks if two arrays are (almost) equal -> stop when nothing moves
    if np.allclose(new_c, centroids):
        print("Converged - centroids stopped moving.")
        break
    centroids = new_c

print("\nSegment profiles (original units):")
for j in range(k):
    members = customers[labels == j]
    # 🐍 PY: list comprehension with a filter picks the names in this segment
    who = [n for n, l in zip(names, labels) if l == j]
    print(f"Segment {j}: avg balance ${members[:, 0].mean():5.1f}k, "
          f"{members[:, 1].mean():4.1f} card txns/mo -> {who}")

# ---
# k-Means Clustering — CONCEPT lab · Financial Services: Retail banking customer segmentation for personalized products
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
