# Build: learn transition probabilities from maintenance logs, then solve the MDP with value iteration
import numpy as np
import pandas as pd

np.random.seed(42)
STATES = ["Good", "Worn", "Critical", "Failed"]
ACTIONS = ["Run", "Repair", "Replace"]
S, A = len(STATES), len(ACTIONS)

# Hidden "true" physics used only to generate the synthetic log
TRUE_P = {
    "Run":     [[0.85, 0.15, 0, 0], [0, 0.80, 0.18, 0.02], [0, 0, 0.65, 0.35], [0, 0, 0, 1.0]],
    "Repair":  [[0.90, 0.10, 0, 0], [0.70, 0.30, 0, 0], [0.10, 0.60, 0.30, 0], [0, 0.30, 0.50, 0.20]],
    "Replace": [[1.0, 0, 0, 0]] * 4,
}

# 1. Simulate 4000 weekly records from a fleet of 40 pumps (what a CMMS system would export)
rows = []
for pump in range(40):
    s = 0
    for week in range(100):
        # Field crews historically chose actions somewhat randomly - good for learning all transitions
        a = np.random.choice(A, p=[0.70, 0.20, 0.10]) if s < 3 else np.random.choice([1, 2])
        s_next = np.random.choice(S, p=TRUE_P[ACTIONS[a]][s])
        # 🐍 PY: append a dict per row - pandas turns a list of dicts into a table in one call
        rows.append({"pump_id": f"P{pump:02d}", "week": week, "state": STATES[s],
                     "action": ACTIONS[a], "next_state": STATES[s_next]})
        s = s_next
log = pd.DataFrame(rows)
print(f"Maintenance log: {len(log)} rows, {log.pump_id.nunique()} pumps")
print(log.head(4).to_string(index=False), "\n")

# 2. Estimate P(s' | s, a) by counting
# 🧠 ML: maximum-likelihood transition estimate = count(s,a,s') / count(s,a)
P = np.zeros((A, S, S))
# 🐍 PY: groupby on three columns + size() counts how often each (state, action, next) happened
counts = log.groupby(["action", "state", "next_state"]).size()
for (a, s, s2), n in counts.items():
    P[ACTIONS.index(a), STATES.index(s), STATES.index(s2)] = n
row_tot = P.sum(axis=2, keepdims=True)
# 🐍 PY: np.divide with where= avoids divide-by-zero for (state, action) pairs never seen
P = np.divide(P, row_tot, out=np.zeros_like(P), where=row_tot > 0)
# 🧠 ML: never-seen pairs (e.g. Run while Failed) get a conservative assumption - the pump stays put
for a in range(A):
    for s in range(S):
        if row_tot[a, s, 0] == 0:
            P[a, s, s] = 1.0
            print(f"No data for ({STATES[s]}, {ACTIONS[a]}) -> assume it stays {STATES[s]}")
print("Estimated P(next | state, Run):")
print(pd.DataFrame(P[0], index=STATES, columns=STATES).round(2).to_string(), "\n")

# 3. Costs per week in $ (we minimise cost, so reward = -cost)
operating = np.array([0, 1_000, 4_000, 25_000])        # lost output by state
action_cost = np.array([0, 6_000, 20_000])              # Run, Repair, Replace
# 🐍 PY: broadcasting - (S,1) + (A,) -> an S x A cost table in one line
C = operating[:, None] + action_cost[None, :]

# 4. Value iteration
# 🧠 ML: Bellman update V(s) = min_a [ cost(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]
gamma, V = 0.95, np.zeros(S)
for it in range(1, 1000):
    # 🐍 PY: einsum "ask,k->as" multiplies each P[a,s,:] by V and sums over next states
    Q = C.T + gamma * np.einsum("ask,k->as", P, V)
    V_new = Q.min(axis=0)
    # 🧠 ML: stop when values barely change - the policy has converged
    if np.max(np.abs(V_new - V)) < 1e-3:
        break
    V = V_new
print(f"Value iteration converged in {it} sweeps (gamma={gamma})")
policy = Q.argmin(axis=0)
print("\nState     | Best action | Expected discounted cost")
for s in range(S):
    print(f"{STATES[s]:<9} | {ACTIONS[policy[s]]:<11} | ${V[s]:>10,.0f}")

# 5. Business interpretation
print("\nQ-table ($, lower is better):")
print(pd.DataFrame(Q.T, index=STATES, columns=ACTIONS).round(0).astype(int).to_string())
print("\nReading: keep running healthy pumps, intervene before failure - the policy shows exactly"
      " where paying for maintenance beats risking a $25k/week outage.")

# ---
# Markov Decision Process (MDP) — BUILD lab · Oil & Gas: Pump maintenance policy optimisation
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
