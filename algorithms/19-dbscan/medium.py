# Build: sklearn DBSCAN on synthetic hailstorm claim locations around a metro area
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

np.random.seed(42)
# Hail swaths are long and thin (storm tracks), not round - a shape DBSCAN handles well
# 🐍 PY: a list of dicts describes each synthetic hail swath
swaths = [
    {"name": "North swath", "start": (36.20, -97.15), "end": (36.24, -97.00), "n": 450},
    {"name": "East swath",  "start": (36.08, -96.98), "end": (36.15, -96.92), "n": 300},
    {"name": "Town core",   "start": (36.11, -97.07), "end": (36.12, -97.06), "n": 200},
]
frames = []
for sw in swaths:
    t = np.random.rand(sw["n"])
    # 🐍 PY: tuple unpacking pulls lat/lon out of the start and end tuples
    (lat0, lon0), (lat1, lon1) = sw["start"], sw["end"]
    lat = lat0 + t * (lat1 - lat0) + np.random.normal(0, 0.004, sw["n"])
    lon = lon0 + t * (lon1 - lon0) + np.random.normal(0, 0.004, sw["n"])
    frames.append(pd.DataFrame({"lat": lat, "lon": lon, "true_area": sw["name"]}))
# Scattered claims: wind-only damage, late reports, or possibly suspicious
n_scatter = 60
frames.append(pd.DataFrame({"lat": np.random.uniform(36.00, 36.30, n_scatter),
                            "lon": np.random.uniform(-97.20, -96.85, n_scatter),
                            "true_area": "Scattered"}))
claims = pd.concat(frames, ignore_index=True)
claims["claim_amount"] = np.random.lognormal(mean=9.3, sigma=0.5, size=len(claims)).round(0)
print(f"{len(claims)} claims loaded. Sample:\n{claims.head(3).to_string()}\n")

# 🧠 ML: convert degrees to km so eps means something physical (1 deg lat ~ 111 km)
lat_ref = np.radians(claims.lat.mean())
X = np.column_stack([claims.lat * 111.0, claims.lon * 111.0 * np.cos(lat_ref)])

# 🧠 ML: eps = neighbourhood radius (km), min_samples = claims needed to call a spot dense
print(" eps_km | clusters | noise claims")
for eps in [0.3, 0.6, 1.0, 2.0]:
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
    # 🐍 PY: set() removes duplicates; subtract the noise label -1 to count real clusters
    n_clusters = len(set(lab) - {-1})
    print(f"  {eps:>4} |    {n_clusters:>2}    |   {np.sum(lab == -1):>4}")

model = DBSCAN(eps=0.6, min_samples=10)
claims["cluster"] = model.fit_predict(X)
# 🐍 PY: a boolean mask keeps only rows where the condition is True
mask = claims.cluster != -1
# 🧠 ML: silhouette (-1 to 1) measures how well separated clusters are; computed without noise points
print(f"\nChosen eps=0.6 km. Silhouette (non-noise) = {silhouette_score(X[mask], claims.cluster[mask]):.3f}")

# 🐍 PY: groupby + agg with named outputs builds a per-hotspot summary table
summary = (claims[mask].groupby("cluster")
           .agg(claims=("claim_amount", "size"), total_usd=("claim_amount", "sum"),
                centre_lat=("lat", "mean"), centre_lon=("lon", "mean"))
           .sort_values("total_usd", ascending=False))
summary["total_usd"] = summary.total_usd.map("${:,.0f}".format)
print("\nHotspots ranked by exposure:")
print(summary.round(4).to_string())

# 🐍 PY: pd.crosstab counts how true areas map onto found clusters
print("\nTrue area vs DBSCAN label:")
print(pd.crosstab(claims.true_area, claims.cluster).to_string())

# Business interpretation
noise = claims[claims.cluster == -1]
print(f"\n{mask.sum()} claims sit in {claims.cluster[mask].nunique()} hotspots -> deploy adjuster teams to each centre.")
print(f"{len(noise)} isolated claims (${noise.claim_amount.sum():,.0f}) -> desk review; "
      "a hail claim far from any swath deserves a closer look.")

# ---
# DBSCAN — BUILD lab · Insurance: Catastrophe claims hotspots after a hailstorm or tornado
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
