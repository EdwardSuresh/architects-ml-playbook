# t-SNE for fraud investigators: PCA pre-reduction, perplexity sweep, trust checks, PCA comparison (Hard)
import time
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
SEGMENTS = {  # name: (count, map letter)
    "Retail": (400, "R"), "Traveler": (200, "T"), "Student": (140, "S"),
    "AccountTakeover": (40, "A"), "CardTesting": (25, "C"),
}
N_FEATURES = 20


def make_data():
    """20 behavioural features; only 6 of them actually separate the segments (rest is noise)."""
    # 🧠 ML: realistic data is high-dimensional and noisy - that is exactly why we pre-reduce with PCA
    # 🐍 PY: one row of 6 'signal' means per segment (amount, night, foreign, velocity, mcc, decline)
    signal = np.array([[0, 0, 0, 0, 0, 0], [2.5, 0, 3, -1, 0, 0], [-1.5, 2.5, 0, 2, 1, 0],
                       [3, 2, 2.5, 1.5, 2.5, 1], [-2, 1.5, 0, 4, 3, 3.5]], dtype=float)
    frames = []
    for k, (name, (n, _)) in enumerate(SEGMENTS.items()):
        X = rng.normal(0, 1, size=(n, N_FEATURES))
        # 🐍 PY: slice assignment adds the segment's signal to the first 6 columns only
        X[:, :6] += signal[k]
        f = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(N_FEATURES)])
        f["segment"] = name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    names = ["txn_amount_z", "night_share_z", "foreign_share_z", "velocity_z", "mcc_entropy_z", "decline_rate_z"]
    # 🐍 PY: dict(zip(...)) builds a rename map old->new in one line
    return df.rename(columns=dict(zip(df.columns[:6], names)))


def knn_purity(emb, labels, k=10):
    """Share of each point's k nearest map-neighbours that share its label."""
    # 🧠 ML: purity measures how well known classes form islands - it is a sanity check, not a model metric
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb)
    return (labels[idx[:, 1:]] == labels[:, None]).mean(axis=1)


def ascii_scatter(emb, labels, letters, width=64, height=18):
    grid = [[" "] * width for _ in range(height)]
    lo, hi = emb.min(axis=0), emb.max(axis=0)
    cells = ((emb - lo) / (hi - lo + 1e-9) * [width - 1, height - 1]).astype(int)
    # 🐍 PY: sort so the big 'Retail' class is drawn first and rare fraud classes stay visible on top
    order = np.argsort([labels[i] != "Retail" for i in range(len(labels))], kind="stable")
    for i in order:
        x, y = cells[i]
        grid[height - 1 - y][x] = letters[labels[i]]
    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")


def main():
    df = make_data()
    feats = [c for c in df.columns if c != "segment"]
    labels = df["segment"].to_numpy(dtype=str)
    letters = {k: v[1] for k, v in SEGMENTS.items()}
    # 🐍 PY: value_counts().to_dict() turns the class counts into a compact dict for printing
    print(f"Customers: {len(df)}, features: {len(feats)}, mix: {df['segment'].value_counts().to_dict()}")

    # 🧠 ML: Pipeline = scale -> PCA to 10 dims. PCA strips noise and makes t-SNE faster and more stable
    prep = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=10, random_state=SEED))])
    Z = prep.fit_transform(df[feats])
    ev = prep.named_steps["pca"].explained_variance_ratio_
    print(f"\nPCA(10) keeps {ev.sum():.1%} of variance; first 3 PCs: {np.round(ev[:3], 3)}")

    # 🧠 ML: baseline = plain PCA to 2-D (linear, fast, distances ARE meaningful)
    pca2 = Z[:, :2]
    rows = [{"method": "PCA-2D", "perplexity": "-", "trust@10": trustworthiness(Z, pca2, n_neighbors=10),
             "purity@10": knn_purity(pca2, labels).mean(), "secs": 0.0}]

    # 🧠 ML: perplexity sweep - small = very local view, large = more global structure; no single right value
    best, best_score = None, -1
    for perp in [5, 30, 50]:
        t0 = time.time()
        emb = TSNE(n_components=2, perplexity=perp, init="pca", max_iter=600, random_state=SEED).fit_transform(Z)
        # 🧠 ML: trustworthiness (0-1) penalises map-neighbours that were NOT neighbours in the original space
        tr = trustworthiness(Z, emb, n_neighbors=10)
        pu = knn_purity(emb, labels).mean()
        rows.append({"method": "t-SNE", "perplexity": perp, "trust@10": tr, "purity@10": pu,
                     "secs": time.time() - t0})
        if tr + pu > best_score:
            best, best_score, best_perp = emb, tr + pu, perp
    res = pd.DataFrame(rows)
    print("\nMethod comparison (higher trust/purity = better local fidelity):")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nBest t-SNE map (perplexity={best_perp}) R=Retail T=Traveler S=Student A=AcctTakeover C=CardTest")
    ascii_scatter(best, labels, letters, height=11)
    print("PCA 2-D map for comparison:")
    ascii_scatter(pca2, labels, letters, height=6)

    # 🧠 ML: per-class purity tells investigators which fraud types form clean islands worth bulk review
    df["purity"] = knn_purity(best, labels)
    # 🐍 PY: groupby -> mean -> round -> to_dict gives {segment: purity} on one printable line
    print("\nPer-segment 10-NN purity on best t-SNE map:", df.groupby("segment")["purity"].mean().round(2).to_dict())

    # 🧠 ML: business triage - fraud cards that do NOT sit in their own island are what a map alone misses
    # 🐍 PY: np.isin tests membership against a list, giving a boolean mask
    is_fraud = np.isin(labels, ["AccountTakeover", "CardTesting"])
    hidden = (is_fraud & (df["purity"].to_numpy() < 0.5)).sum()
    print(f"\nFraud cards whose map neighbours are mostly other segments: {hidden} of {is_fraud.sum()}")

    # 🐍 PY: tempfile.mkdtemp gives a safe scratch folder; joblib stores numpy-heavy objects efficiently
    out = tempfile.mkdtemp()
    joblib.dump({"prep": prep, "embedding": best, "perplexity": best_perp}, f"{out}/tsne_fraud_map.joblib")
    print(f"Saved preprocessing pipeline + embedding to {out}")

    print("\nMODEL-RISK NOTE (SR 11-7 / investigator guidance):")
    print(" - t-SNE is an EXPLORATION tool. Cluster sizes and gaps between islands are NOT meaningful distances.")
    print(" - It has no transform() for new cards: never score live transactions with a t-SNE coordinate.")
    print(" - Use islands to form hypotheses, then confirm with a validated, documented fraud model + case review.")
    print(" - Record perplexity, seed and PCA settings so the picture shown to investigators is reproducible.")


if __name__ == "__main__":
    main()

# ---
# t-SNE — HARD lab · Financial Services: Visualising card-transaction behaviour to reveal fraud clusters
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
