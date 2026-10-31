# t-SNE on 1,200 synthetic card customers: can investigators SEE the fraud ring? (Medium)
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)

# 🐍 PY: a helper function that builds one customer segment as a DataFrame
def make_segment(n, amt, night, foreign, velocity, merchants, label):
    return pd.DataFrame({
        "avg_txn_amount": rng.normal(amt, amt * 0.4, n),
        "night_txn_share": np.clip(rng.normal(night, 0.12, n), 0, 1),
        "foreign_txn_share": np.clip(rng.normal(foreign, 0.12, n), 0, 1),
        "txn_per_day": rng.normal(velocity, velocity * 0.35, n),
        "distinct_merchants_7d": rng.poisson(merchants, n),
        "segment": label,
    })

# 🧠 ML: four behaviour groups - t-SNE never sees 'segment', we only use it to colour the map afterwards
# 🐍 PY: pd.concat stacks DataFrames vertically; ignore_index renumbers rows 0..n-1
df = pd.concat([
    make_segment(600, 60, 0.05, 0.02, 2.0, 8, "Retail"),
    make_segment(350, 250, 0.10, 0.30, 1.0, 5, "Traveler"),
    make_segment(200, 25, 0.40, 0.01, 6.0, 20, "Student"),
    make_segment(50, 320, 0.30, 0.35, 4.0, 14, "FraudRing"),
], ignore_index=True)
features = [c for c in df.columns if c != "segment"]
print(df.groupby("segment")[features].mean().round(2).to_string())

# 🧠 ML: scale first - otherwise dollars dominate shares and the map just sorts by amount
X = StandardScaler().fit_transform(df[features])

# 🧠 ML: perplexity ~ how many neighbours each point 'cares about' (5-50 typical)
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
emb = tsne.fit_transform(X)
print(f"\nFinal KL divergence: {tsne.kl_divergence_:.3f}")

# 🐍 PY: ASCII scatter plot - a list of lists of characters acts as our canvas
def ascii_scatter(emb, labels, width=60, height=18):
    grid = [[" "] * width for _ in range(height)]
    # 🐍 PY: min/max per column, then rescale coordinates to integer grid cells
    lo, hi = emb.min(axis=0), emb.max(axis=0)
    cells = ((emb - lo) / (hi - lo) * [width - 1, height - 1]).astype(int)
    # 🐍 PY: draw the rare class last so it is not hidden under the big ones
    order = np.argsort(labels == "FraudRing")
    for i in order:
        x, y = cells[i]
        grid[height - 1 - y][x] = labels[i][0]
    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")

print("\nt-SNE map (R=Retail T=Traveler S=Student F=FraudRing)")
ascii_scatter(emb, df["segment"].to_numpy(dtype=str))

# 🧠 ML: quantitative check - share of each point's 10 nearest map-neighbours with the same segment
nn = NearestNeighbors(n_neighbors=11).fit(emb)
_, idx = nn.kneighbors(emb)
# 🐍 PY: to_numpy(dtype=str) gives a plain numpy string array (fast fancy indexing)
lab = df["segment"].to_numpy(dtype=str)
# 🐍 PY: idx[:, 1:] skips column 0, which is the point itself
purity = (lab[idx[:, 1:]] == lab[:, None]).mean(axis=1)
df["knn_purity"] = purity
print("\n10-NN purity in the 2-D map, by segment:")
print(df.groupby("segment")["knn_purity"].mean().round(3).to_string())

fr = df.loc[df.segment == "FraudRing", "knn_purity"].mean()
print(f"\nBusiness read: {fr:.0%} of a fraud-ring card's map neighbours are also fraud-ring cards,")
print("so investigators can lasso that island and review those accounts together.")

# ---
# t-SNE — MEDIUM lab · Financial Services: Visualising card-transaction behaviour to reveal fraud clusters
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
