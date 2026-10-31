# Hard: catastrophe-claim hotspot engine - haversine DBSCAN, k-distance eps, k-Means comparison, noise triage
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
EARTH_KM = 6371.0
ADJUSTER_CLAIMS_PER_DAY = 6


def simulate_cat_event():
    """Tornado track + two hail cores + scattered claims (some suspicious) after one storm."""
    parts = []
    # 🐍 PY: tuples of (name, lat0, lon0, lat1, lon1, n, spread_deg, mean_log_amount)
    specs = [("Tornado track", 36.05, -97.25, 36.18, -96.95, 350, 0.0025, 10.6),
             ("Hail core W", 36.25, -97.10, 36.27, -97.06, 400, 0.006, 9.2),
             ("Hail core E", 36.10, -96.85, 36.12, -96.82, 250, 0.006, 9.3)]
    for name, la0, lo0, la1, lo1, n, sd, mu in specs:
        t = rng.random(n)
        parts.append(pd.DataFrame({
            "lat": la0 + t * (la1 - la0) + rng.normal(0, sd, n),
            "lon": lo0 + t * (lo1 - lo0) + rng.normal(0, sd, n),
            "claim_usd": rng.lognormal(mu, 0.5, n), "event_zone": name}))
    n = 50
    parts.append(pd.DataFrame({"lat": rng.uniform(35.9, 36.4, n), "lon": rng.uniform(-97.4, -96.7, n),
                               "claim_usd": rng.lognormal(9.6, 0.8, n), "event_zone": "Scattered"}))
    df = pd.concat(parts, ignore_index=True)
    df["days_after_event"] = rng.integers(0, 30, len(df))
    df["policy_age_days"] = rng.integers(30, 3000, len(df))
    # A few claims on brand-new policies far from the storm - classic red flag
    df.loc[df.event_zone == "Scattered", "policy_age_days"] = rng.choice([10, 25, 45, 400, 1500], n)
    df["claim_id"] = [f"CLM{i:05d}" for i in range(len(df))]
    return df


def k_distance_eps(coords_rad, k):
    """Return the 'knee' of the sorted k-th-neighbour distance curve, in km."""
    # 🧠 ML: haversine works on (lat, lon) in RADIANS and returns great-circle angle; x Earth radius = km
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(coords_rad)
    d, _ = nn.kneighbors(coords_rad)
    kd = np.sort(d[:, -1]) * EARTH_KM
    # 🧠 ML: knee = point furthest from the straight line joining first and last points of the curve
    # 🐍 PY: linspace gives evenly spaced values; here the diagonal line we compare the curve to
    x = np.linspace(0, 1, len(kd))
    y = (kd - kd.min()) / (kd.max() - kd.min())
    knee = int(np.argmax(np.abs(y - x)))
    return kd, kd[knee]


def hotspot_report(df, label_col):
    g = df[df[label_col] != -1].groupby(label_col)
    # 🐍 PY: named aggregation - each keyword becomes an output column
    rep = g.agg(claims=("claim_id", "size"), total_usd=("claim_usd", "sum"),
                avg_usd=("claim_usd", "mean"), lat=("lat", "mean"), lon=("lon", "mean"))
    # 🧠 ML: turn cluster size into a resource number managers can act on
    rep["adjuster_days"] = np.ceil(rep.claims / ADJUSTER_CLAIMS_PER_DAY).astype(int)
    return rep.sort_values("total_usd", ascending=False)


df = simulate_cat_event()
# 🐍 PY: np.radians converts degrees to radians, which the haversine metric requires
coords = np.radians(df[["lat", "lon"]].to_numpy())
print(f"{len(df)} claims after the event, total incurred ${df.claim_usd.sum():,.0f}\n")

# 1. Pick eps with the k-distance plot (text version)
MIN_SAMPLES = 12
kd, eps_km = k_distance_eps(coords, MIN_SAMPLES)
print(f"k-distance curve (k={MIN_SAMPLES}) percentiles, km:")
for q in [10, 50, 80, 90, 95, 99]:
    v = np.percentile(kd, q)
    # 🐍 PY: min() caps the ASCII bar so a huge tail value does not flood the screen
    print(f"  p{q:<2} {v:5.2f} " + "#" * min(int(v * 10), 40))
print(f"Knee suggests eps = {eps_km:.2f} km\n")

# 2. DBSCAN with haversine + ball_tree
# 🧠 ML: eps must be in radians for haversine -> divide km by Earth radius
db = DBSCAN(eps=eps_km / EARTH_KM, min_samples=MIN_SAMPLES, metric="haversine", algorithm="ball_tree")
df["dbscan"] = db.fit_predict(coords)
# 🐍 PY: nunique counts distinct labels; subtract one if noise (-1) is present
n_clusters = df.dbscan.nunique() - (1 if -1 in df.dbscan.values else 0)
print(f"DBSCAN found {n_clusters} hotspots, {np.sum(df.dbscan == -1)} noise claims")
print(hotspot_report(df, "dbscan").round({"total_usd": 0, "avg_usd": 0, "lat": 3, "lon": 3}).to_string())

# 3. Sensitivity to eps
print("\n eps_km | hotspots | noise")
for f in [0.5, 1.0, 2.0]:
    lab = DBSCAN(eps=eps_km * f / EARTH_KM, min_samples=MIN_SAMPLES, metric="haversine",
                 algorithm="ball_tree").fit_predict(coords)
    print(f"  {eps_km * f:5.2f} |    {len(set(lab) - {-1}):>2}    | {np.sum(lab == -1):>4}")

# 4. Compare with k-Means (k = DBSCAN cluster count)
# 🧠 ML: k-Means must assign EVERY claim and assumes round blobs - it cannot flag outliers or follow a track
km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
xy = np.column_stack([df.lat * 111, df.lon * 111 * np.cos(np.radians(df.lat.mean()))])
df["kmeans"] = km.fit_predict(xy)
# 🧠 ML: ARI compares two labelings and ignores label names; 1.0 = identical grouping
truth = df.event_zone.astype("category").cat.codes
print(f"\nAgreement with true storm zones (Adjusted Rand Index, 1 = perfect):")
print(f"  DBSCAN  {adjusted_rand_score(truth, df.dbscan):.3f}")
print(f"  k-Means {adjusted_rand_score(truth, df.kmeans):.3f}")
scattered_km = df[df.event_zone == "Scattered"].kmeans.value_counts().sum()
print(f"  k-Means forced all {scattered_km} scattered claims into a hotspot; "
      f"DBSCAN flagged {np.sum((df.event_zone == 'Scattered') & (df.dbscan == -1))} of them as noise")

# 5. Noise triage - a simple rule-based priority score for desk review
# 🧠 ML: noise != fraud. It is a review queue; combine with business rules for prioritisation
# 🐍 PY: .copy() avoids pandas SettingWithCopy surprises when we add columns
noise = df[df.dbscan == -1].copy()
# 🐍 PY: boolean Series add up as 0/1, giving a quick points-based score
noise["review_score"] = ((noise.policy_age_days < 60).astype(int) * 2
                         + (noise.claim_usd > df.claim_usd.quantile(0.9)).astype(int)
                         + (noise.days_after_event > 20).astype(int))
cols = ["claim_id", "lat", "lon", "claim_usd", "policy_age_days", "days_after_event", "review_score"]
print(f"\nTop noise claims for SIU/desk review (of {len(noise)}):")
print(noise.sort_values(["review_score", "claim_usd"], ascending=False)[cols].head(6)
      .round({"lat": 3, "lon": 3, "claim_usd": 0}).to_string(index=False))

# 6. Business decision: adjuster deployment
rep = hotspot_report(df, "dbscan")
print(f"\nDeploy plan: {rep.adjuster_days.sum()} adjuster-days across {len(rep)} hotspots; "
      f"largest hotspot holds {rep.total_usd.iloc[0] / df.claim_usd.sum():.0%} of incurred $")

path = os.path.join(tempfile.mkdtemp(), "cat_hotspots.joblib")
joblib.dump({"eps_km": eps_km, "min_samples": MIN_SAMPLES, "hotspots": rep}, path)
print(f"Saved hotspot config -> {os.path.basename(path)}")
print("Compliance: noise flags trigger review, not denial - state unfair claims settlement rules"
      " (NAIC model act) require prompt, fair handling of every claim.")

# ---
# DBSCAN — HARD lab · Insurance: Catastrophe claims hotspots after a hailstorm or tornado
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
