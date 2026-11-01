# Concept: model a pump's health as a Markov chain and simulate "just keep running it"
import numpy as np

np.random.seed(42)
# 🐍 PY: a list of state names - the position in the list is the state number
states = ["Good", "Worn", "Critical", "Failed"]

# 🧠 ML: transition matrix P[i][j] = chance of moving from state i to state j in one week of running
# Each row must add up to 1.0 (the pump has to end up somewhere)
P_run = np.array([
    [0.85, 0.15, 0.00, 0.00],   # from Good
    [0.00, 0.80, 0.18, 0.02],   # from Worn
    [0.00, 0.00, 0.65, 0.35],   # from Critical
    [0.00, 0.00, 0.00, 1.00],   # from Failed (stays failed until someone acts)
])
# 🐍 PY: .sum(axis=1) adds across each row; np.allclose checks all are ~1.0
print("Rows sum to 1:", np.allclose(P_run.sum(axis=1), 1.0))

# 🐍 PY: dict mapping state name -> weekly cost in $ (lost production, extra energy, etc.)
weekly_cost = {"Good": 0, "Worn": 1_000, "Critical": 4_000, "Failed": 25_000}

# 🧠 ML: Markov property - the next state depends only on the current state, not the history
def simulate(weeks=20):
    s = 0  # start in Good
    path, total = [], 0
    for _ in range(weeks):
        # 🐍 PY: np.random.choice picks the next state using row s as probabilities
        s = np.random.choice(4, p=P_run[s])
        path.append(states[s])
        total += weekly_cost[states[s]]
    return path, total

path, cost = simulate()
# 🐍 PY: str.join builds one string; [:1] slices the first letter of each state name
print("One 20-week run:", " ".join(p[:1] for p in path))
print(f"Cost of that run: ${cost:,}")

# 🧠 ML: multiply the start vector by P repeatedly to get the probability of each state after n weeks
dist = np.array([1.0, 0, 0, 0])
for week in range(1, 21):
    dist = dist @ P_run
    if week in (1, 5, 10, 20):
        # 🐍 PY: zip pairs names with probabilities; :.2f formats each to two decimals
        print(f"Week {week:>2}: " + "  ".join(f"{n}={p:.2f}" for n, p in zip(states, dist)))

# Monte Carlo average over many runs
# 🐍 PY: list comprehension runs the simulation 1000 times and keeps only the cost (index [1])
costs = [simulate()[1] for _ in range(1000)]
print(f"\nAverage 20-week cost if we never maintain: ${np.mean(costs):,.0f}")
print("Next step: add actions (Repair, Replace) and let an MDP choose the cheapest policy.")

# ---
# Markov Decision Process (MDP) — CONCEPT lab · Oil & Gas: Pump maintenance policy optimisation
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
