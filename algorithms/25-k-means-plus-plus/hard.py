# Production-style DC network design: demand-weighted k-means++ (sample_weight), choose k by total
# annual cost (fixed DC cost + transport cost on haversine miles), stability check, site report, save.
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
EARTH_RADIUS_MI = 3958.8
FIXED_COST_PER_DC = 1_400_000      # $/year: lease, staff, WMS licence for one DC
COST_PER_CASE_MILE = 0.012         # $ per case per mile (truckload + last mile blended)
CIRCUITY = 1.20                    # road miles are ~20% longer than straight-line miles
SERVICE_RADIUS_MI = 150            # next-day delivery promise


def make_customers(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    metros = pd.DataFrame({
        "lat": [32.78, 29.76, 35.47, 36.15, 39.10, 39.74, 35.15, 34.75, 30.27, 38.63, 41.26, 37.69],
        "lon": [-96.80, -95.37, -97.52, -95.99, -94.58, -104.99, -90.05, -92.29, -97.74, -90.20,
                -95.93, -97.34],
        "pop_share": [.17, .16, .07, .06, .09, .10, .06, .04, .09, .08, .04, .04],
    })
    idx = rng.choice(len(metros), n, p=metros.pop_share)
    # 🧠 ML: big metros also order more per customer - demand is NOT spread evenly across points
    size_boost = 1 + 3 * metros.pop_share.to_numpy()[idx]
    return pd.DataFrame({
        "lat": metros.lat.to_numpy()[idx] + rng.normal(0, 0.4, n),
        "lon": metros.lon.to_numpy()[idx] + rng.normal(0, 0.5, n),
        "annual_cases": (rng.lognormal(6.5, 0.8, n) * size_boost).round(),
    })


def haversine_mi(lat1, lon1, lat2, lon2):
    # 🧠 ML: great-circle distance on a sphere - the honest "as the crow flies" miles between two lat/lons
    # 🐍 PY: np.radians works element-wise on whole arrays, so this computes thousands of distances at once
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


class DCPlanner:
    """Clusters customers in an equal-area projection, then costs the network with haversine miles."""

    def __init__(self, customers):
        self.c = customers
        self.lat0 = customers.lat.mean()
        # 🧠 ML: k-means needs Euclidean space; this local projection makes 1 unit ~= 1 mile both ways
        self.X = np.column_stack([customers.lat * 69.0,
                                  customers.lon * 69.0 * np.cos(np.radians(self.lat0))])

    def to_latlon(self, centers):
        return centers[:, 0] / 69.0, centers[:, 1] / (69.0 * np.cos(np.radians(self.lat0)))

    # 🧠 ML: caveat - k-means minimises SQUARED distance; transport cost is linear in miles, so the
    # 🧠 ML: centroids are near-optimal, not optimal (a Weiszfeld / MIP step can refine them)
    def fit(self, k, weighted=True, seed=42):
        # 🧠 ML: sample_weight = annual cases -> centroids are pulled toward heavy-demand customers
        w = self.c.annual_cases.to_numpy() if weighted else None
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=seed)
        return km.fit(self.X, sample_weight=w)

    def cost(self, km):
        dc_lat, dc_lon = self.to_latlon(km.cluster_centers_)
        # 🧠 ML: re-assign each customer to the nearest DC by REAL miles (haversine), not projected units
        d = haversine_mi(self.c.lat.to_numpy()[:, None], self.c.lon.to_numpy()[:, None],
                         dc_lat[None, :], dc_lon[None, :])
        assign = d.argmin(axis=1)
        miles = d[np.arange(len(d)), assign] * CIRCUITY
        transport = float((miles * self.c.annual_cases * COST_PER_CASE_MILE).sum())
        fixed = FIXED_COST_PER_DC * km.n_clusters
        service = float(np.average(miles <= SERVICE_RADIUS_MI, weights=self.c.annual_cases))
        return {"k": km.n_clusters, "fixed_M": fixed / 1e6, "transport_M": transport / 1e6,
                "total_M": (fixed + transport) / 1e6, "service_pct": 100 * service,
                "assign": assign, "miles": miles, "dc_lat": dc_lat, "dc_lon": dc_lon}


if __name__ == "__main__":
    cust = make_customers()
    planner = DCPlanner(cust)
    print(f"{len(cust)} customers, {cust.annual_cases.sum() / 1e6:.2f}M cases/year")

    print("\n== Choosing k by total annual cost (demand-weighted k-means++) ==")
    print("  k  fixed$M  transport$M  total$M  served<150mi")
    results = []
    for k in range(2, 13):
        r = planner.cost(planner.fit(k))
        results.append(r)
        bar = "#" * int((r["total_M"] - 10) * 2) if r["total_M"] > 10 else ""
        print(f" {k:2d}  {r['fixed_M']:7.1f}  {r['transport_M']:11.2f}  {r['total_M']:7.2f}   "
              f"{r['service_pct']:5.1f}%  {bar}")
    # 🧠 ML: the elbow is a business trade-off - each extra DC adds fixed cost but cuts transport cost
    best = min(results, key=lambda r: r["total_M"])
    k_best = best["k"]
    print(f"  -> cheapest network: k={k_best} at ${best['total_M']:.2f}M/year")

    # 🧠 ML: if business requires a service level, pick the cheapest k that meets it (a constraint)
    meets = [r for r in results if r["service_pct"] >= 95]
    if meets:
        k_sla = min(meets, key=lambda r: r["total_M"])
        print(f"  -> cheapest k meeting 95% within {SERVICE_RADIUS_MI} mi: k={k_sla['k']}"
              f" at ${k_sla['total_M']:.2f}M/year")

    print(f"\n== Weighted vs unweighted clustering at k={k_best} ==")
    unw = planner.cost(planner.fit(k_best, weighted=False))
    print(f"  unweighted transport ${unw['transport_M']:.2f}M  vs  weighted ${best['transport_M']:.2f}M"
          f"  (saving ${1e3 * (unw['transport_M'] - best['transport_M']):.0f}K/year)")

    # 🧠 ML: stability - do different random seeds give the same customer-to-DC assignments?
    labels = [planner.fit(k_best, seed=s).labels_ for s in range(4)]
    ari = [adjusted_rand_score(labels[0], l) for l in labels[1:]]
    print(f"\nStability across seeds (adjusted Rand index, 1.0 = identical): {np.round(ari, 3).tolist()}")

    print(f"\n== Recommended DC sites (k={k_best}) ==")
    # 🐍 PY: building a DataFrame from a dict of arrays, then groupby on the assignment column
    rep = pd.DataFrame({"dc": best["assign"], "cases": cust.annual_cases, "miles": best["miles"]})
    site = rep.groupby("dc").agg(customers=("cases", "size"), cases_K=("cases", lambda s: s.sum() / 1e3),
                                 avg_road_mi=("miles", "mean"), max_road_mi=("miles", "max"))
    site["lat"], site["lon"] = best["dc_lat"].round(2), best["dc_lon"].round(2)
    site["demand_share"] = (100 * site.cases_K / site.cases_K.sum()).round(1)
    print(site.sort_values("cases_K", ascending=False).round(1).to_string())

    print("\nModel-risk note: centroids are a STARTING point - snap each to real candidate sites (labour,"
          "\nzoning, interstate access, incentives) and re-run cost with a road-network router before"
          "\ncommitting capital. Re-validate when demand shifts >10% in any region.")

    path = os.path.join(tempfile.mkdtemp(), "dc_plan.joblib")
    # 🐍 PY: joblib.dump stores the fitted KMeans plus the chosen sites as one bundle
    joblib.dump({"kmeans": planner.fit(k_best), "sites": site.reset_index().to_dict("records")}, path)
    print(f"Saved model + site table -> {os.path.basename(path)} in a temp dir")

# ---
# k-Means++ — HARD lab · Supply Chain: Choosing distribution center locations
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
