import numpy as np

# Daily changes in Treasury yields, in basis points (1 bp = 0.01%), for 10 trading days
# 🐍 PY: a list of strings - we use it later only for printing nice column headers
maturities = ["3M", "2Y", "10Y", "30Y"]
# 🐍 PY: np.array turns a list of lists into a 2-D table: 10 rows (days) x 4 columns (maturities)
changes_bp = np.array([
    [ 2.0,  4.0,  5.0,  5.0],
    [-1.0, -3.0, -4.0, -4.0],
    [ 0.5,  2.0,  1.0,  0.0],
    [ 3.0,  6.0,  7.0,  6.0],
    [-2.0, -4.0, -6.0, -6.0],
    [ 1.0,  0.0, -2.0, -3.0],
    [ 0.0,  1.0,  3.0,  4.0],
    [-1.5, -5.0, -5.0, -4.0],
    [ 2.5,  3.0,  4.0,  3.0],
    [-0.5,  1.0,  0.0, -1.0],
])
print("Data shape (days, maturities):", changes_bp.shape)

# 🧠 ML: PCA works on deviations from the average, so subtract each column's mean first
# 🐍 PY: axis=0 means 'down the rows', giving one mean per column; broadcasting subtracts it from every row
centered = changes_bp - changes_bp.mean(axis=0)

# 🧠 ML: the covariance matrix says how each pair of maturities moves together
cov = np.cov(centered, rowvar=False)
print("\nCovariance matrix (bp^2):")
print(np.round(cov, 1))

# 🧠 ML: eigenvectors = directions of movement; eigenvalues = how much variance each direction carries
eigvals, eigvecs = np.linalg.eigh(cov)
# 🐍 PY: eigh returns smallest first; [::-1] reverses the order so the biggest factor comes first
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
# 🧠 ML: an eigenvector's sign is arbitrary; flip so the 30Y loading is positive (easier to read)
eigvecs = eigvecs * np.sign(eigvecs[-1, :])

share = eigvals / eigvals.sum()
names = ["PC1 level?", "PC2 slope?", "PC3 curve?", "PC4 noise"]
print("\nFactor       variance share   loadings on", maturities)
# 🐍 PY: enumerate gives the position i and the item together; f-strings format numbers neatly
for i, name in enumerate(names):
    print(f"{name:<12} {share[i]:>8.1%}        {np.round(eigvecs[:, i], 2)}")

# 🧠 ML: projecting each day onto PC1 gives one 'level move' number per day
pc1_scores = centered @ eigvecs[:, 0]
print("\nDay-by-day PC1 score (how much the whole curve shifted):")
print(np.round(pc1_scores, 1))

# ---
# Principal Component Analysis (PCA) — EASY lab · Financial Services: Treasury yield curve risk factors for rate hedging
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
