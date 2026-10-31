# Easy: group 5 stocks by how alike their returns move - agglomerative merging by hand
import numpy as np

# 🐍 PY: dict of ticker -> list of 8 monthly returns in % (synthetic)
returns = {
    "BANK1": [2.1, -1.5, 3.0, 0.5, -2.2, 1.8, 0.9, -0.4],
    "BANK2": [1.9, -1.2, 2.7, 0.8, -2.5, 1.5, 1.1, -0.6],
    "TECH1": [4.5, 2.0, -3.1, 5.2, 1.0, -2.4, 3.8, 2.2],
    "TECH2": [4.0, 2.4, -2.8, 4.6, 1.5, -2.0, 3.1, 2.6],
    "UTIL1": [0.3, 0.6, 0.2, -0.1, 0.5, 0.4, -0.2, 0.1],
}
# 🐍 PY: list(dict) gives the keys in insertion order
names = list(returns)
R = np.array([returns[n] for n in names])

# 🧠 ML: correlation distance d = sqrt(0.5 * (1 - corr)): 0 = move together, 1 = move opposite
corr = np.corrcoef(R)
D = np.sqrt(0.5 * (1 - corr))
print("Correlation matrix:")
for n, row in zip(names, corr):
    # 🐍 PY: a generator inside join formats every number in the row
    print(f"  {n:<6}" + " ".join(f"{v:6.2f}" for v in row))

# 🧠 ML: start with every stock as its own cluster; repeatedly merge the two closest clusters
# 🐍 PY: each cluster is a tuple of member names so it can be used as a dict key / set element
clusters = [(n,) for n in names]

def cluster_dist(a, b):
    # 🧠 ML: AVERAGE linkage = mean distance between every member of a and every member of b
    # 🐍 PY: nested generator expression loops over both clusters' members
    return np.mean([D[names.index(x), names.index(y)] for x in a for y in b])

step = 1
while len(clusters) > 1:
    # 🐍 PY: build all (distance, i, j) pairs and min() picks the smallest distance first
    d, i, j = min((cluster_dist(clusters[i], clusters[j]), i, j)
                  for i in range(len(clusters)) for j in range(i + 1, len(clusters)))
    merged = clusters[i] + clusters[j]
    print(f"Step {step}: merge {clusters[i]} + {clusters[j]} at distance {d:.3f}")
    # 🐍 PY: keep every cluster except i and j, then add the new merged tuple
    clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    step += 1

# 🧠 ML: cutting the tree at a height (e.g. 0.5) gives the groups - one pick per group diversifies
print("\nReading: the two tech names merge first, then the banks; the utility joins last -> best diversifier.")

# ---
# Hierarchical Clustering — EASY lab · Financial Services: Portfolio diversification with correlation clustering and HRP
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
