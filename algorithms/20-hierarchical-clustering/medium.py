# Medium: cluster 20 stocks by return correlation with scipy linkage and cut into sector groups
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

np.random.seed(42)
N_DAYS = 750  # about 3 years of trading days

# 🐍 PY: dict of sector -> list of tickers (synthetic names)
sectors = {
    "Banks": ["BNK1", "BNK2", "BNK3", "BNK4", "BNK5"],
    "Tech": ["TEC1", "TEC2", "TEC3", "TEC4", "TEC5"],
    "Energy": ["ENG1", "ENG2", "ENG3", "ENG4", "ENG5"],
    "Utilities": ["UTL1", "UTL2", "UTL3", "UTL4", "UTL5"],
}
sector_vol = {"Banks": 0.012, "Tech": 0.016, "Energy": 0.018, "Utilities": 0.007}

# 🧠 ML: one-factor-per-sector model: return = market + sector factor + stock-specific noise
market = np.random.normal(0.0003, 0.008, N_DAYS)
cols = {}
for sector, tickers in sectors.items():
    factor = np.random.normal(0, sector_vol[sector], N_DAYS)
    for t in tickers:
        cols[t] = 0.6 * market + factor + np.random.normal(0, 0.008, N_DAYS)
# 🐍 PY: DataFrame from a dict of equal-length arrays -> one column per ticker
returns = pd.DataFrame(cols)
# 🐍 PY: shuffle the column order so the algorithm cannot "cheat" by position
returns = returns[np.random.permutation(returns.columns)]
print(f"Daily returns: {returns.shape[0]} days x {returns.shape[1]} stocks")
print(returns.iloc[:3, :6].round(4).to_string(), "\n")

# 🧠 ML: correlation -> distance. d = sqrt(0.5*(1-rho)) is a proper metric (0 = identical moves)
corr = returns.corr()
dist = np.sqrt(0.5 * (1 - corr))
# 🐍 PY: squareform converts the square matrix to the condensed 1-D form scipy expects
condensed = squareform(dist.values, checks=False)

# 🧠 ML: Ward merges the pair adding least within-cluster variance; valid here since this distance is Euclidean
# (sqrt(0.5*(1-rho)) is proportional to the straight-line gap between standardised return series)
Z = linkage(condensed, method="ward")
print("Last 5 merges (cluster_a, cluster_b, height, size):")
for row in Z[-5:]:
    print(f"  {int(row[0]):>3} + {int(row[1]):>3}  at height {row[2]:.3f}  -> {int(row[3])} stocks")

# 🧠 ML: cutting the tree into k groups - fcluster with criterion='maxclust'
labels = fcluster(Z, t=4, criterion="maxclust")
groups = pd.Series(labels, index=corr.index, name="cluster")
print("\nClusters found (cut at 4):")
# 🐍 PY: groupby on a Series then iterate (key, sub-series) pairs
for c, members in groups.groupby(groups):
    print(f"  Cluster {c}: {sorted(members.index)}")

# Text dendrogram: leaf order puts similar stocks next to each other
# 🐍 PY: no_plot=True returns the layout as a dict without drawing anything
leaf_order = dendrogram(Z, labels=list(corr.index), no_plot=True)["ivl"]
print("\nDendrogram leaf order:", " ".join(leaf_order))

# Check against the true sectors
# 🐍 PY: dict comprehension inverts sector->tickers into ticker->sector
true_sector = {t: s for s, ts in sectors.items() for t in ts}
check = pd.crosstab(groups, pd.Series(groups.index.map(true_sector), index=groups.index, name="true_sector"))
print("\nCluster vs true sector:\n" + check.to_string())

# Business interpretation: average correlation inside vs between clusters
same = groups.values[:, None] == groups.values[None, :]
off_diag = ~np.eye(len(groups), dtype=bool)
print(f"\nAvg correlation inside clusters : {corr.values[same & off_diag].mean():.2f}")
print(f"Avg correlation across clusters : {corr.values[~same].mean():.2f}")
print("Holding 5 stocks from ONE cluster is close to holding one stock - spread picks across clusters.")

# ---
# Hierarchical Clustering — MEDIUM lab · Financial Services: Portfolio diversification with correlation clustering and HRP
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
