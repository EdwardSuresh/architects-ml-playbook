import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
SEED = 42
SKEWED = ["avg_balance_k", "monthly_deposits_k", "card_spend_k"]
OTHER = ["digital_login_share", "loan_balance_ratio", "num_products", "overdrafts_12m"]
FEATS = SKEWED + OTHER


def make_customers(seed=SEED):
    rng = np.random.default_rng(seed)
    # (n, balance, deposits, card spend, digital, loan ratio, products, overdraft rate, mean age)
    # 🧠 ML: four hidden personas - a good segmentation should rediscover them without being told
    personas = [(1800, 2.5, 3.0, 1.8, 0.9, 0.1, 1.5, 0.5, 27), (1400, 12, 7.5, 2.5, 0.6, 0.7, 3.5, 0.3, 42),
                (1000, 55, 12, 3.5, 0.3, 0.1, 2.5, 0.05, 63), (800, 0.8, 2.2, 0.6, 0.25, 1.0, 1.2, 4.0, 48)]
    frames = []
    for n, bal, dep, spend, dig, loan, prod, od, age in personas:
        frames.append(pd.DataFrame({
            # 🧠 ML: lognormal gives long right tails, like real bank balances
            "avg_balance_k": rng.lognormal(np.log(bal), 0.5, n),
            "monthly_deposits_k": rng.lognormal(np.log(dep), 0.35, n),
            "card_spend_k": rng.lognormal(np.log(spend), 0.4, n),
            "digital_login_share": np.clip(rng.normal(dig, 0.12, n), 0, 1),
            "loan_balance_ratio": np.clip(rng.normal(loan, 0.15, n), 0, 1.5),
            "num_products": np.clip(rng.poisson(prod, n), 1, 8),
            "overdrafts_12m": rng.poisson(od, n),
            "age": np.clip(rng.normal(age, 7, n), 18, 90).round(),  # NOT a model input (fairness)
        }))
    return pd.concat(frames, ignore_index=True)


def build_pipeline(k, seed=SEED):
    # 🧠 ML: log1p squashes long-tailed money columns so a few millionaires do not own a cluster
    pre = ColumnTransformer([("log", FunctionTransformer(np.log1p), SKEWED),
                             ("keep", "passthrough", OTHER)])
    # 🧠 ML: k-means++ init + n_init restarts avoid the bad-start trap seen in the easy level
    return Pipeline([("pre", pre), ("scale", StandardScaler()),
                     ("km", KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed))])


# 🐍 PY: range(2, 9) is a default argument; it yields k = 2..8
def choose_k(df, ks=range(2, 9)):
    rows = []
    for k in ks:
        pipe = build_pipeline(k).fit(df[FEATS])
        # 🐍 PY: pipe[:-1] slices the pipeline to everything except KMeans - the scaled features
        Z = pipe[:-1].transform(df[FEATS])
        # 🧠 ML: silhouette on a 2,000-row sample keeps it fast; it is O(n squared) on full data
        sil = silhouette_score(Z, pipe["km"].labels_, sample_size=2000, random_state=SEED)
        rows.append((k, pipe["km"].inertia_, sil))
    out = pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])
    # 🧠 ML: elbow = where the % drop in inertia from adding one more cluster becomes small
    out["inertia_drop_%"] = (-out.inertia.pct_change() * 100).round(1)
    return out


def stability(df, k, seeds=(1, 2, 3, 4)):
    # 🧠 ML: refit on 80% bootstraps; ARI near 1 = segments are real, not a random artefact
    ref = build_pipeline(k).fit(df[FEATS])
    scores = []
    for s in seeds:
        sub = df.sample(frac=0.8, random_state=s)
        alt = build_pipeline(k, seed=s).fit(sub[FEATS])
        scores.append(adjusted_rand_score(ref.predict(sub[FEATS]), alt.predict(sub[FEATS])))
    return np.mean(scores)


df = make_customers()
print(f"Customers={len(df)}  model features={len(FEATS)} (age held out for fairness review)")
table = choose_k(df)
print(table.round(3).to_string(index=False))
# 🐍 PY: idxmax returns the row label of the highest silhouette; .loc then reads its k
best_k = int(table.loc[table.silhouette.idxmax(), "k"])
print(f"Chosen k={best_k}  bootstrap stability ARI={stability(df, best_k):.3f}")

# 🧠 ML: refit on all customers with the chosen k; predict assigns each to its nearest centroid
model = build_pipeline(best_k).fit(df[FEATS])
df["segment"] = model.predict(df[FEATS])

# 🧠 ML: baseline = the classic 'balance tier' segmentation many banks start with
# 🐍 PY: pd.qcut cuts balances into equal-sized quantile buckets (the old-school tiers)
tiers = pd.qcut(df.avg_balance_k, best_k, labels=False)
Zall = model[:-1].transform(df[FEATS])
print(f"Silhouette  k-means={silhouette_score(Zall, df.segment, sample_size=2000, random_state=SEED):.3f}"
      f"  vs balance-tier baseline={silhouette_score(Zall, tiers, sample_size=2000, random_state=SEED):.3f}")

# 🧠 ML: explain segments by z-score of each feature vs the whole book (+ = above average)
Z = pd.DataFrame(model[:-1].transform(df[FEATS]), columns=FEATS)
Z["segment"] = df.segment
print("\nSegment fingerprint (z-scores):")
print(Z.groupby("segment").mean().round(2).to_string())

# 🧠 ML: surrogate tree turns clusters into readable rules for marketing and model-risk review
# 🧠 ML: fidelity = how often the readable tree agrees with k-Means; above 0.95 is a trustworthy summary
sur = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(df[FEATS], df.segment)
print(f"\nSurrogate tree fidelity={sur.score(df[FEATS], df.segment):.3f}; first rules:")
print("\n".join(export_text(sur, feature_names=FEATS, decimals=1).splitlines()[:8]))

# 🧠 ML: business value - expected campaign margin per segment (response rate x margin x size)
offers = {"wealth": (0.06, 900), "refi": (0.04, 1200), "card": (0.08, 150), "hardship": (0.10, 60)}


def pick_offer(r):
    if r.avg_balance_k > 30:
        return "wealth"
    if r.overdrafts_12m > 2:
        return "hardship"
    return "refi" if r.loan_balance_ratio > 0.4 else "card"


# 🧠 ML: medians resist outliers better than means when profiling skewed money columns
prof = df.groupby("segment")[FEATS + ["age"]].median()
prof["size"] = df.segment.value_counts().sort_index()
print("\nseg  size  med_age  offer     expected_margin")
for seg, r in prof.iterrows():
    o = pick_offer(r)
    rate, margin = offers[o]
    print(f"{seg:3d} {int(r['size']):5d} {r.age:7.0f}  {o:9s} ${r['size'] * rate * margin:>10,.0f}")

# Fair lending: segments must not proxy for protected classes (ECOA/Reg B, UDAAP). Age is reviewed
# AFTER clustering; a segment that is mostly one age band needs a compliance sign-off before credit offers.
print("\nFairness check: age is held out; review segments that skew by age before credit offers.")
# 🧠 ML: save the full pipeline so next month's customers get the same transforms and centroids
path = os.path.join(tempfile.mkdtemp(), "retail_segments_kmeans.joblib")
joblib.dump(model, path)
print("Saved ->", os.path.basename(path))

# ---
# k-Means Clustering — HARD lab · Financial Services: Retail banking customer segmentation for personalized products
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
