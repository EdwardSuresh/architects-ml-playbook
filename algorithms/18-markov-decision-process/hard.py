# Hard: pump maintenance MDP with price regimes - policy vs value iteration, discount sensitivity,
# and Monte Carlo comparison against run-to-failure and fixed-schedule baselines
import os
import time
import tempfile
import joblib
import numpy as np

HEALTH = ["Good", "Worn", "Critical", "Failed"]
PRICE = ["LowPrice", "HighPrice"]
ACTIONS = ["Run", "Repair", "Replace"]
# 🐍 PY: nested comprehension builds the 8 combined state names like "Worn/HighPrice"
STATE_NAMES = [f"{h}/{p}" for p in PRICE for h in HEALTH]
H, NP, A = len(HEALTH), len(PRICE), len(ACTIONS)
S = H * NP


class PumpMDP:
    """Health x oil-price-regime MDP. Cost = lost production + maintenance spend."""

    def __init__(self, repair_cost=6_000, replace_cost=20_000):
        health_P = np.array([
            [[0.85, 0.15, 0, 0], [0, 0.80, 0.18, 0.02], [0, 0, 0.65, 0.35], [0, 0, 0, 1.0]],     # Run
            [[0.90, 0.10, 0, 0], [0.70, 0.30, 0, 0], [0.10, 0.60, 0.30, 0], [0, 0.30, 0.50, 0.20]],  # Repair
            [[1.0, 0, 0, 0]] * 4,                                                               # Replace
        ])
        # 🧠 ML: the price regime evolves on its own (exogenous), independent of what we do to the pump
        price_P = np.array([[0.95, 0.05], [0.08, 0.92]])
        # 🧠 ML: joint transition = health transition x price transition (Kronecker product)
        self.P = np.stack([np.kron(price_P, health_P[a]) for a in range(A)])
        lost_bbl = np.array([0, 40, 160, 1000])          # barrels lost per week by health
        oil_price = np.array([45, 85])                    # $/bbl by regime
        # 🧠 ML: production loss also happens while the crew repairs/replaces (downtime)
        downtime_bbl = np.array([0, 150, 400])
        # 🐍 PY: index p * H + h flattens (price, health) into one state number
        self.C = np.zeros((S, A))
        for p in range(NP):
            for h in range(H):
                for a in range(A):
                    self.C[p * H + h, a] = ((lost_bbl[h] + downtime_bbl[a]) * oil_price[p]
                                            + [0, repair_cost, replace_cost][a])

    def q_values(self, V, gamma):
        # 🐍 PY: einsum sums P[a,s,k] * V[k] over k, giving an (A,S) array
        return self.C.T + gamma * np.einsum("ask,k->as", self.P, V)


def value_iteration(mdp, gamma, tol=1e-4):
    # 🧠 ML: start from zero values; each sweep looks one more week into the future
    V, sweeps = np.zeros(S), 0
    while True:
        sweeps += 1
        V_new = mdp.q_values(V, gamma).min(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            return mdp.q_values(V_new, gamma).argmin(axis=0), V_new, sweeps
        V = V_new


def policy_iteration(mdp, gamma):
    policy, rounds = np.zeros(S, dtype=int), 0
    while True:
        rounds += 1
        # 🧠 ML: policy evaluation solves the linear system V = C_pi + gamma * P_pi V exactly
        # 🐍 PY: fancy indexing picks, for every state, the transition row of the action the policy chooses
        P_pi = mdp.P[policy, np.arange(S)]
        C_pi = mdp.C[np.arange(S), policy]
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, C_pi)
        # 🧠 ML: policy improvement - switch each state to its greedy action
        new_policy = mdp.q_values(V, gamma).argmin(axis=0)
        if np.array_equal(new_policy, policy):
            return policy, V, rounds
        policy = new_policy


def simulate(mdp, policy_fn, weeks=104, runs=1000, seed=42):
    """Monte Carlo: average undiscounted 2-year cost of a decision rule."""
    rng = np.random.default_rng(seed)
    totals, failures = np.zeros(runs), np.zeros(runs)
    for r in range(runs):
        s = 0
        for t in range(weeks):
            # 🧠 ML: the simulator is policy-agnostic - any rule with signature (state, week) can be evaluated
            a = policy_fn(s, t)
            totals[r] += mdp.C[s, a]
            failures[r] += (s % H == 3)
            # 🧠 ML: sample the next state from the model - this is Monte Carlo evaluation
            s = rng.choice(S, p=mdp.P[a, s])
    return totals.mean(), totals.std(), failures.mean()


mdp = PumpMDP()
gamma = 0.97

# 1. Compare the two solvers
# 🐍 PY: perf_counter is a high-resolution clock for timing code
t0 = time.perf_counter()
pol_vi, V_vi, sweeps = value_iteration(mdp, gamma)
t_vi = time.perf_counter() - t0
t0 = time.perf_counter()
pol_pi, V_pi, rounds = policy_iteration(mdp, gamma)
t_pi = time.perf_counter() - t0
print(f"Value iteration : {sweeps:>4} sweeps  {t_vi * 1000:6.1f} ms")
print(f"Policy iteration: {rounds:>4} rounds  {t_pi * 1000:6.1f} ms")
print(f"Same policy? {np.array_equal(pol_vi, pol_pi)}   max value gap ${np.max(np.abs(V_vi - V_pi)):.2f}\n")
print("State               | Optimal action | Discounted cost")
for s in range(S):
    print(f"{STATE_NAMES[s]:<19} | {ACTIONS[pol_pi[s]]:<14} | ${V_pi[s]:>10,.0f}")

# 2. Discount sensitivity
# 🧠 ML: gamma near 1 = long-term planner; small gamma = short-sighted manager who avoids spend today
SHORT = ["Run", "Fix", "New"]  # Run / Repair / Replace
print("\nGamma | Policy (LowPrice: G W C F | HighPrice: G W C F)")
for g in [0.50, 0.80, 0.90, 0.97, 0.99]:
    pol, _, _ = policy_iteration(mdp, g)
    # 🐍 PY: slicing pol[:H] and pol[H:] splits the policy into the two price regimes
    lo = " ".join(SHORT[a] for a in pol[:H])
    hi = " ".join(SHORT[a] for a in pol[H:])
    print(f" {g:.2f} | {lo}  |  {hi}")

# 3. Monte Carlo evaluation vs baselines
# 🐍 PY: lambdas make small decision rules with the same (state, week) signature
baselines = {
    "Run to failure": lambda s, t: 2 if s % H == 3 else 0,
    "Fixed schedule (8wk)": lambda s, t: 2 if s % H == 3 else (1 if t % 8 == 7 else 0),
    "MDP optimal policy": lambda s, t: int(pol_pi[s]),
}
print("\nMonte Carlo, 1000 pumps x 104 weeks:")
print("Strategy               | Avg 2-yr cost | Std dev   | Failed weeks")
results = {}
for name, fn in baselines.items():
    mean, sd, fails = simulate(mdp, fn)
    results[name] = mean
    print(f"{name:<22} | ${mean:>11,.0f} | ${sd:>8,.0f} | {fails:5.2f}")
saving = results["Run to failure"] - results["MDP optimal policy"]
print(f"\nMDP policy saves ${saving:,.0f} per pump over 2 years vs run-to-failure (synthetic costs).")

# 4. Robustness: what if repair cost doubles?
# 🧠 ML: sensitivity test - which decisions flip if a cost assumption is wrong?
alt_pol, _, _ = policy_iteration(PumpMDP(repair_cost=12_000), gamma)
changed = [STATE_NAMES[s] for s in range(S) if alt_pol[s] != pol_pi[s]]
print(f"If repair cost doubles, policy changes in: {changed or 'no states'}")

path = os.path.join(tempfile.mkdtemp(), "pump_policy.joblib")
joblib.dump({"states": STATE_NAMES, "actions": ACTIONS, "policy": pol_pi, "gamma": gamma}, path)
print(f"Saved policy -> {os.path.basename(path)}")
print("Safety note: a cost-optimal policy must never override API/OSHA/PHMSA safety limits -"
      " Critical units on sour or high-pressure service need mandatory shutdown rules on top.")

# ---
# Markov Decision Process (MDP) — HARD lab · Oil & Gas: Pump maintenance policy optimisation
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
