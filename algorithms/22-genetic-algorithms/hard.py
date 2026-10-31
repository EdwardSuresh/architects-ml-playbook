# Production-style GA for one-truck delivery routing with customer time windows.
# Order crossover (OX), tournament selection, elitism, 2-opt-style mutation, lateness penalty,
# nearest-neighbour baseline, parameter sweep, and route export for the dispatch system.
import os
import json
import tempfile
from dataclasses import dataclass
import numpy as np

SPEED_MPH = 30.0
SERVICE_MIN = 10.0            # minutes unloading at each stop
LATE_PENALTY_PER_MIN = 2.0    # $ per minute late (driver overtime + customer credit)
COST_PER_MILE = 2.10


@dataclass
class Problem:
    dist: np.ndarray          # (n+1)x(n+1) miles, index 0 = depot
    win_open: np.ndarray      # minutes after 8:00 am the customer can receive
    win_close: np.ndarray     # minutes after 8:00 am the window shuts

    @property
    def n(self):
        return len(self.win_open) - 1


def make_problem(n=15, seed=7):
    rng = np.random.default_rng(seed)
    pts = np.vstack([[15, 15], rng.uniform(0, 30, (n, 2))])
    dist = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    # 🧠 ML: half the stops get a morning window, half afternoon - realistic retail receiving docks
    opens = np.where(rng.random(n + 1) < 0.5, 0.0, 180.0)
    opens[0] = 0.0
    closes = opens + 300.0
    closes[0] = 1e9
    return Problem(dist, opens, closes)


def evaluate(route, P):
    """Return (total_cost, miles, minutes_late). route = permutation of 1..n."""
    t, miles, late, prev = 0.0, 0.0, 0.0, 0
    for stop in route:
        d = P.dist[prev, stop]
        miles += d
        t += d / SPEED_MPH * 60
        # 🐍 PY: max() means the truck waits if it arrives before the window opens
        t = max(t, P.win_open[stop])
        late += max(0.0, t - P.win_close[stop])
        t += SERVICE_MIN
        prev = stop
    miles += P.dist[prev, 0]
    # 🧠 ML: penalty method - turn the hard time-window constraint into a cost the GA can trade off
    return miles * COST_PER_MILE + late * LATE_PENALTY_PER_MIN, miles, late


def order_crossover(p1, p2, rng):
    # 🧠 ML: OX keeps a slice of parent 1 in place, then fills the gaps in parent 2's cyclic order
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    # 🐍 PY: np.roll rotates parent 2 so filling starts right after the copied slice
    kept = set(p1[a:b].tolist())
    fill = [g for g in np.roll(p2, -b) if g not in kept]
    idx = [(b + k) % n for k in range(n - (b - a))]
    child[idx] = fill
    return child


def inversion_mutation(r, rate, rng):
    # 🧠 ML: reversing a segment = a 2-opt move; the most useful single mutation for routing
    r = r.copy()
    if rng.random() < rate:
        a, b = sorted(rng.choice(len(r), 2, replace=False))
        r[a:b + 1] = r[a:b + 1][::-1]
    return r


def tournament(pop, costs, k, rng):
    # 🧠 ML: tournament selection - pick k at random, the cheapest wins; k controls selection pressure
    contenders = rng.choice(len(pop), k, replace=False)
    return pop[contenders[np.argmin(costs[contenders])]]


def run_ga(P, pop_size=100, gens=250, k=3, mut_rate=0.3, elite=4, seed=0, seed_route=None):
    rng = np.random.default_rng(seed)
    pop = [rng.permutation(np.arange(1, P.n + 1)) for _ in range(pop_size)]
    if seed_route is not None:
        # 🧠 ML: seeding with a heuristic route speeds convergence; the GA can only improve on it
        pop[0] = np.array(seed_route)
    for _ in range(gens):
        costs = np.array([evaluate(r, P)[0] for r in pop])
        order = np.argsort(costs)
        # 🧠 ML: elitism - copy the best few unchanged so the best-so-far is never lost
        new_pop = [pop[i] for i in order[:elite]]
        while len(new_pop) < pop_size:
            c = order_crossover(tournament(pop, costs, k, rng), tournament(pop, costs, k, rng), rng)
            new_pop.append(inversion_mutation(c, mut_rate, rng))
        pop = new_pop
    costs = np.array([evaluate(r, P)[0] for r in pop])
    return pop[int(np.argmin(costs))]


def nearest_neighbour(P):
    # Baseline dispatchers often use: always drive to the closest unvisited stop
    left, route, cur = set(range(1, P.n + 1)), [], 0
    while left:
        # 🐍 PY: min over a set with key= finds the closest remaining stop
        nxt = min(left, key=lambda s: P.dist[cur, s])
        route.append(nxt)
        left.remove(nxt)
        cur = nxt
    return np.array(route)


def report(name, route, P):
    cost, miles, late = evaluate(route, P)
    print(f"  {name:28s} cost=${cost:7.1f}  miles={miles:6.1f}  late_min={late:6.1f}")
    return cost


if __name__ == "__main__":
    P = make_problem()
    print(f"Problem: {P.n} stops, {int((P.win_open[1:] > 0).sum())} with afternoon windows\n")

    print("== Parameter sweep (population x tournament size x mutation rate), 2 seeds, 80 gens ==")
    sweep = []
    for pop_size in (40, 80):
        for k in (2, 5):
            for mut in (0.1, 0.4):
                # 🧠 ML: GAs are stochastic - always average over several seeds before judging a setting
                c = [evaluate(run_ga(P, pop_size, 80, k, mut, seed=s), P)[0] for s in range(2)]
                sweep.append(((pop_size, k, mut), np.mean(c), np.std(c)))
                print(f"  pop={pop_size:3d} k={k} mut={mut:.1f}  mean_cost=${np.mean(c):7.1f}  sd=${np.std(c):5.1f}")
    # 🐍 PY: min over a list of tuples with key=lambda picks the row with the lowest mean cost
    (best_pop, best_k, best_mut), _, _ = min(sweep, key=lambda r: r[1])
    print(f"  -> chosen pop={best_pop}, k={best_k}, mut={best_mut}")

    print("\n== Final comparison ==")
    nn = nearest_neighbour(P)
    nn_cost = report("Nearest neighbour", nn, P)
    ga = run_ga(P, best_pop, 200, best_k, best_mut, seed=42)
    ga_cost = report("GA (tuned, random start)", ga, P)
    ga_seeded = run_ga(P, best_pop, 200, best_k, best_mut, seed=42, seed_route=nn)
    seeded_cost = report("GA (seeded with NN)", ga_seeded, P)

    best_route = ga_seeded if seeded_cost <= ga_cost else ga
    best_cost = min(ga_cost, seeded_cost)
    print(f"\nGA saving vs nearest neighbour: ${nn_cost - best_cost:.1f}/day "
          f"(~${(nn_cost - best_cost) * 250:,.0f}/year at 250 delivery days)")
    # 🧠 ML: read the trade-off - NN minimises miles greedily but ignores windows; the GA optimises total cost
    print("  Most of that saving is avoided lateness penalty, not fewer miles - check the penalty rate"
          " with the business.")

    # Explainability: show arrival times vs windows so a dispatcher can sanity-check the plan
    print("\nFirst 8 stops of the chosen route (arrival vs window, minutes after 8:00):")
    t, prev = 0.0, 0
    for stop in best_route[:8]:
        t = max(t + P.dist[prev, stop] / SPEED_MPH * 60, P.win_open[stop])
        flag = "LATE" if t > P.win_close[stop] else "ok"
        print(f"  stop {stop:2d}: arrive {t:5.0f}  window [{P.win_open[stop]:3.0f},{P.win_close[stop]:3.0f}]  {flag}")
        t += SERVICE_MIN
        prev = stop

    # Safety/compliance: FMCSA hours-of-service caps a driver's day - check before dispatch
    _, miles, _ = evaluate(best_route, P)
    print(f"\nCompliance note: route drive time {miles / SPEED_MPH:.1f}h; verify against FMCSA hours-of-service"
          " and keep a human dispatcher able to override the plan.")

    out = os.path.join(tempfile.mkdtemp(), "route_plan.json")
    # 🐍 PY: json.dump needs plain Python ints, so convert numpy ints with int()
    with open(out, "w") as f:
        json.dump({"route": [int(s) for s in best_route], "cost": round(float(best_cost), 2),
                   "params": {"pop": best_pop, "k": best_k, "mut": best_mut}}, f)
    print(f"Saved route plan -> {os.path.basename(out)} in a temp dir")

# ---
# Genetic Algorithms — HARD lab · Supply Chain: Delivery route optimization for one truck
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
