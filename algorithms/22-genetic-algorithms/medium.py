# Genetic Algorithm for a 15-stop delivery route (one truck, return to depot) with numpy + pandas
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_STOPS = 15

# ---- Synthetic customer stops in a 40 x 40 mile service area; depot in the middle ----
# 🐍 PY: pandas DataFrame = a spreadsheet-like table with named columns
stops = pd.DataFrame({
    "stop_id": [f"C{i:02d}" for i in range(1, N_STOPS + 1)],
    "x_mi": rng.uniform(0, 40, N_STOPS).round(1),
    "y_mi": rng.uniform(0, 40, N_STOPS).round(1),
    "cases_ordered": rng.integers(5, 60, N_STOPS),
})
depot = np.array([20.0, 20.0])
# 🐍 PY: np.vstack stacks the depot row on top of the customer coordinates -> index 0 is the depot
pts = np.vstack([depot, stops[["x_mi", "y_mi"]].to_numpy()])
# 🧠 ML: precompute the full distance matrix once; fitness evaluations then become cheap lookups
# 🐍 PY: broadcasting pts[:, None, :] - pts[None, :, :] gives every pair difference in one shot
dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))


def route_length(perm):
    """perm = order of customers 1..15; truck starts and ends at depot 0."""
    # 🐍 PY: np.concatenate builds [0, perm..., 0]; fancy indexing then sums each leg's distance
    path = np.concatenate(([0], perm, [0]))
    return dist[path[:-1], path[1:]].sum()


def crossover(p1, p2):
    # 🧠 ML: keep a random slice of parent 1, fill remaining stops in parent 2's order (valid route)
    a, b = sorted(rng.choice(N_STOPS, 2, replace=False))
    middle = p1[a:b]
    rest = [s for s in p2 if s not in middle]
    return np.array(rest[:a] + list(middle) + rest[a:])


def mutate(route, rate=0.2):
    # 🧠 ML: inversion mutation - reverse a segment; it un-crosses tangled route legs
    r = route.copy()
    if rng.random() < rate:
        a, b = sorted(rng.choice(N_STOPS, 2, replace=False))
        r[a:b + 1] = r[a:b + 1][::-1]
    return r


POP, GENS, KEEP = 120, 300, 30
# 🧠 ML: initial population = 120 random routes (each a permutation of stops 1..15)
population = [rng.permutation(np.arange(1, N_STOPS + 1)) for _ in range(POP)]
history = []
for gen in range(GENS):
    # 🐍 PY: np.array of lengths + argsort gives indices from shortest to longest route
    lengths = np.array([route_length(r) for r in population])
    order = np.argsort(lengths)
    population = [population[i] for i in order]
    history.append((gen, lengths[order[0]], lengths.mean()))
    # 🧠 ML: selection - the 30 shortest routes survive and become parents
    parents = population[:KEEP]
    children = []
    while len(children) < POP - KEEP:
        i, j = rng.choice(KEEP, 2, replace=False)
        children.append(mutate(crossover(parents[i], parents[j])))
    population = parents + children

hist = pd.DataFrame(history, columns=["generation", "best_miles", "avg_miles"])
print("GA progress (every 50 generations):")
print(hist.iloc[::50].round(1).to_string(index=False))

best = population[0]
best_len = route_length(best)
# 🧠 ML: baseline = average random route; shows how much the GA actually helped
random_len = np.mean([route_length(rng.permutation(np.arange(1, N_STOPS + 1))) for _ in range(2000)])
print(f"\nBest GA route  : {best_len:6.1f} miles")
print(f"Avg random route: {random_len:6.1f} miles")
# 🐍 PY: .iloc with (best - 1) maps stop numbers 1..15 back to DataFrame rows 0..14
seq = " -> ".join(stops.stop_id.iloc[best - 1])
print(f"\nStop sequence: Depot -> {seq} -> Depot")

COST_PER_MILE, DAYS_PER_YEAR = 2.10, 250
saving = (random_len - best_len) * COST_PER_MILE * DAYS_PER_YEAR
print(f"\nBusiness read-out: at ${COST_PER_MILE}/mile and {DAYS_PER_YEAR} delivery days, the GA route saves"
      f" about ${saving:,.0f}/year vs an unplanned order of stops.")

# ---
# Genetic Algorithms — MEDIUM lab · Supply Chain: Delivery route optimization for one truck
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
