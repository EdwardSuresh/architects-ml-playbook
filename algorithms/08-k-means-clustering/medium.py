import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

# 🐍 PY: a list of dicts describes each hidden persona; we generate customers from them
personas = [
    dict(n=900, age=27, bal=2.5, card=45, digital=0.9, loan=0.1, prod=1.5),   # digital young
    dict(n=700, age=42, bal=12.0, card=30, digital=0.6, loan=0.7, prod=3.5),  # mortgage families
    dict(n=500, age=63, bal=55.0, card=10, digital=0.3, loan=0.1, prod=2.5),  # affluent savers
    dict(n=400, age=48, bal=0.8, card=12, digital=0.25, loan=1.0, prod=1.2),   # stretched borrowers
]
frames = []
for p in personas:
    n = p["n"]
    # 🐍 PY: np.clip keeps generated numbers inside sensible business ranges
    frames.append(pd.DataFrame({
        "age": np.clip(np.random.normal(p["age"], 6, n), 18, 90).round(),
        "avg_balance_k": np.abs(np.random.normal(p["bal"], p["bal"] * 0.35, n)).round(2),
        "card_txns_month": np.random.poisson(p["card"], n),
        "digital_login_share": np.clip(np.random.normal(p["digital"], 0.12, n), 0, 1).round(2),
        "loan_balance_ratio": np.clip(np.random.normal(p["loan"], 0.15, n), 0, 1.5).round(2),
        "num_products": np.clip(np.random.poisson(p["prod"], n), 1, 8),
    }))
df = pd.concat(frames, ignore_index=True)
print(f"Customers: {len(df)}  features: {list(df.columns)}")

# 🧠 ML: k-Means uses distance, so every feature must be on the same scale (mean 0, std 1)
scaler = StandardScaler()
X = scaler.fit_transform(df)

# 🧠 ML: elbow = inertia drops fast then flattens; silhouette = how well-separated (-1..1, higher better)
print("\n k   inertia  silhouette")
results = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    # 🧠 ML: silhouette on a 1500-row sample is accurate enough and much faster
    sil = silhouette_score(X, km.labels_, sample_size=1500, random_state=42)
    results[k] = sil
    print(f"{k:2d} {km.inertia_:9.0f}  {sil:8.3f}  {'*' * int(sil * 40)}")

# 🐍 PY: max with key= returns the dict key whose value is largest
best_k = max(results, key=results.get)
print(f"\nBest k by silhouette: {best_k}")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["segment"] = km.labels_

# 🐍 PY: groupby().mean() builds one profile row per segment; size() counts members
profile = df.groupby("segment").mean().round(2)
profile["customers"] = df.groupby("segment").size()
print("\nSegment profiles:")
print(profile.to_string())

# 🧠 ML: clusters have no names - a human reads the profile and assigns a business meaning
print("\nSuggested offers:")
for seg, row in profile.iterrows():
    if row.avg_balance_k > 30:
        offer = "Wealth advisory / CDs"
    elif row.loan_balance_ratio > 0.4 and row.avg_balance_k < 5:
        offer = "Debt consolidation + budgeting tools"
    elif row.loan_balance_ratio > 0.4:
        offer = "Mortgage refinance + 529 college saver"
    else:
        offer = "Cash-back card + mobile-first savings"
    print(f"  Segment {seg} ({int(row.customers)} cust, age {row.age:.0f}): {offer}")

# ---
# k-Means Clustering — BUILD lab · Financial Services: Retail banking customer segmentation for personalized products
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
