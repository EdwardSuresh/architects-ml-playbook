# Production-style Q-learning for SKU reorder policy with a 2-day supplier lead time,
# hyperparameter tuning, and evaluation against a tuned (s, S) baseline on common random demand.
import os
import tempfile
import itertools
import numpy as np


class InventoryEnv:
    """Single SKU, daily review, orders arrive LEAD_TIME days after being placed."""
    ACTIONS = (0, 5, 10)

    def __init__(self, max_stock=25, lead_time=2, hold=0.5, stockout=8.0, order_fixed=6.0,
                 unit_cost=1.0, seed=0):
        self.max_stock, self.lead_time = max_stock, lead_time
        self.hold, self.stockout = hold, stockout
        self.order_fixed, self.unit_cost = order_fixed, unit_cost
        self.rng = np.random.default_rng(seed)

    def reset(self, stock=None):
        # 🐍 PY: "x if cond else y" is Python's one-line conditional expression
        self.on_hand = int(self.rng.integers(0, self.max_stock + 1)) if stock is None else stock
        # 🧠 ML: the pipeline (orders in transit) is part of the state - without it the problem is not Markov
        self.pipeline = [0] * self.lead_time
        return self.state()

    def state(self):
        # 🐍 PY: tuple(...) is hashable, so it can be used as a dictionary key for the Q-table
        return (self.on_hand,) + tuple(self.pipeline)

    def demand(self, day):
        # 🧠 ML: weekly seasonality (busier Fri/Sat) makes the problem harder than a fixed-mean demand
        mean = 4.0 + 2.0 * (day % 7 in (4, 5))
        return self.rng.poisson(mean)

    def step(self, action_idx, day, demand=None):
        qty = self.ACTIONS[action_idx]
        # 🐍 PY: list.pop(0) removes the oldest in-transit order: it arrives today
        arriving = self.pipeline.pop(0)
        self.pipeline.append(qty)
        self.on_hand = min(self.on_hand + arriving, self.max_stock)
        d = self.demand(day) if demand is None else demand
        sold = min(self.on_hand, d)
        lost = d - sold
        self.on_hand -= sold
        cost = (self.hold * self.on_hand + self.stockout * lost
                + (self.order_fixed + self.unit_cost * qty if qty else 0.0))
        return self.state(), -cost, cost


def full_state(env, day):
    # 🧠 ML: state = inventory position (on hand + in transit), capped at 40 -> only 41 states
    # 🧠 ML: we tested adding day-of-week too: 7x more states, slower to learn, worse with this budget
    position = env.on_hand + sum(env.pipeline)
    return (min(position, 40),)


def train_q(alpha0, eps_decay, episodes=3000, days=56, gamma=0.9, seed=1):
    env = InventoryEnv(seed=seed)
    # 🐍 PY: a dict keyed by state tuples; setdefault creates a zero row the first time a state is seen
    Q, visits = {}, {}
    eps = 1.0
    for ep in range(episodes):
        env.reset()
        s = full_state(env, 0)
        for day in range(days):
            q_s = Q.setdefault(s, np.zeros(3))
            n_s = visits.setdefault(s, np.zeros(3))
            a = int(env.rng.integers(3)) if env.rng.random() < eps else int(np.argmax(q_s))
            _, r, _ = env.step(a, day)
            s2 = full_state(env, day + 1)
            # 🧠 ML: per-(state,action) decaying learning rate alpha0 / n^0.6 - big steps early, small later
            n_s[a] += 1
            alpha = max(0.01, alpha0 / n_s[a] ** 0.6)
            # 🧠 ML: off-policy TD target uses the greedy max even when we explored
            q_s[a] += alpha * (r + gamma * Q.get(s2, np.zeros(3)).max() - q_s[a])
            s = s2
        # 🧠 ML: exponential epsilon decay with a floor, so exploration never fully stops in training
        eps = max(0.02, eps * eps_decay)
    return Q


def evaluate(policy_fn, demand_paths, start_stock=12):
    """Replay the SAME demand paths for every policy (common random numbers = fair comparison)."""
    env = InventoryEnv(seed=0)
    totals = []
    for path in demand_paths:
        env.reset(stock=start_stock)
        cost = 0.0
        for day, d in enumerate(path):
            a = policy_fn(env, day)
            _, _, c = env.step(a, day, demand=d)
            cost += c
        totals.append(cost)
    return np.mean(totals), np.std(totals)


def q_policy(Q):
    # 🐍 PY: a closure - the inner function "remembers" Q from the outer function
    def act(env, day):
        q = Q.get(full_state(env, day))
        return 0 if q is None else int(np.argmax(q))
    return act


def ss_policy(s_low, S_high):
    """Classic (s, S): if inventory position <= s, order up toward S (rounded to allowed lot sizes)."""
    def act(env, day):
        position = env.on_hand + sum(env.pipeline)
        if position > s_low:
            return 0
        gap = S_high - position
        # 🧠 ML: pick the allowed lot size closest to the gap - the baseline must respect the same actions
        return int(np.argmin([abs(gap - q) for q in InventoryEnv.ACTIONS]))
    return act


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    days = 56
    # 🧠 ML: separate demand paths for tuning vs final test - avoid tuning on the test set
    tune_paths = [[rng.poisson(4.0 + 2.0 * (d % 7 in (4, 5))) for d in range(days)] for _ in range(150)]
    test_paths = [[rng.poisson(4.0 + 2.0 * (d % 7 in (4, 5))) for d in range(days)] for _ in range(400)]

    print("== Tuning Q-learning (alpha0 x epsilon decay) on validation demand ==")
    results = {}
    # 🐍 PY: itertools.product builds every combination of the two lists (a small grid search)
    for alpha0, decay in itertools.product([0.3, 0.6], [0.997, 0.999]):
        Q = train_q(alpha0, decay)
        mean_cost, _ = evaluate(q_policy(Q), tune_paths)
        results[(alpha0, decay)] = (mean_cost, Q)
        print(f"  alpha0={alpha0:<4} eps_decay={decay:<6} states_seen={len(Q):4d}  val_cost=${mean_cost:7.1f}")
    # 🐍 PY: min with key= picks the dict key whose stored cost is smallest
    best_cfg = min(results, key=lambda k: results[k][0])
    best_Q = results[best_cfg][1]
    print(f"  -> best config alpha0={best_cfg[0]}, eps_decay={best_cfg[1]}")

    print("\n== Tuning (s, S) baseline on the same validation demand ==")
    ss_grid = [(s, S) for s in range(6, 22, 2) for S in range(s + 5, 31, 3)]
    ss_scores = {cfg: evaluate(ss_policy(*cfg), tune_paths)[0] for cfg in ss_grid}
    best_ss = min(ss_scores, key=ss_scores.get)
    print(f"  searched {len(ss_grid)} (s,S) pairs -> best s={best_ss[0]}, S={best_ss[1]}"
          f" val_cost=${ss_scores[best_ss]:.1f}")

    print("\n== Final test on 400 unseen 8-week demand paths ==")
    rows = [("Q-learning (tuned)", q_policy(best_Q)),
            (f"(s,S)=({best_ss[0]},{best_ss[1]})", ss_policy(*best_ss)),
            ("Always order 5", lambda env, day: 1)]
    test_costs = {}
    for name, fn in rows:
        m, sd = evaluate(fn, test_paths)
        test_costs[name] = m
        bar = "#" * int(m / 40)
        print(f"  {name:20s} mean=${m:7.1f}  sd=${sd:5.1f}  {bar}")
    q_cost = test_costs["Q-learning (tuned)"]
    ss_cost = test_costs[f"(s,S)=({best_ss[0]},{best_ss[1]})"]
    print(f"\nQ-learning vs (s,S): {100 * (ss_cost - q_cost) / ss_cost:+.1f}% cost saving per 8 weeks")

    # Explainability: a tabular policy is fully readable - compare it to the (s,S) rule directly
    print("\nLearned order qty by inventory position (on hand + in transit):")
    # 🐍 PY: a list comprehension builds the row; a missing state means never visited -> show '-'
    acts = ["-" if best_Q.get((p,)) is None else str(InventoryEnv.ACTIONS[int(np.argmax(best_Q[(p,)]))])
            for p in range(0, 31)]
    print("  position: " + " ".join(f"{p:>2}" for p in range(0, 31, 1)))
    print("  order   : " + " ".join(f"{a:>2}" for a in acts))

    # 🧠 ML: model-risk note - an RL policy is a model; it needs validation, monitoring and a fallback
    print("\nModel-risk note: keep (s,S) as a fallback, cap order size, monitor service level weekly,"
          "\nand retrain when the demand pattern drifts (new promo calendar, new supplier lead time).")

    out_dir = tempfile.mkdtemp()
    path = os.path.join(out_dir, "q_table.npy")
    # 🐍 PY: np.save with allow_pickle stores the dict; production would use a versioned model registry
    np.save(path, np.array([best_Q, best_cfg, best_ss], dtype=object), allow_pickle=True)
    print(f"Saved Q-table + configs -> {os.path.basename(path)} in a temp dir")

# ---
# Q-Learning — PRODUCTION lab · Supply Chain: Inventory reorder policy for one SKU
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
