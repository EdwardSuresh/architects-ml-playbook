# Tabular Q-learning for a single-SKU inventory reorder policy (stock 0..20, order 0/5/10)
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---- Environment settings (one SKU in one warehouse) ----
MAX_STOCK = 20
ACTIONS = [0, 5, 10]                 # units we can order each day
HOLDING_COST_PER_UNIT = 0.5          # $ per unit left on the shelf overnight
STOCKOUT_COST_PER_UNIT = 8.0         # $ per unit of demand we could not serve (lost margin + goodwill)
ORDER_FIXED_COST = 6.0               # $ per order placed (truck, paperwork)
MEAN_DAILY_DEMAND = 4.0


def step(stock, order_qty):
    """One day: receive order, see random demand, pay costs. Returns next_stock, reward, demand."""
    # 🧠 ML: demand is random (Poisson), so the agent must learn a policy that works on average
    demand = rng.poisson(MEAN_DAILY_DEMAND)
    # 🐍 PY: min() caps stock at shelf capacity; extra units are simply not accepted
    available = min(stock + order_qty, MAX_STOCK)
    sold = min(available, demand)
    next_stock = available - sold
    lost = demand - sold
    cost = HOLDING_COST_PER_UNIT * next_stock + STOCKOUT_COST_PER_UNIT * lost
    cost += ORDER_FIXED_COST if order_qty > 0 else 0.0
    # 🧠 ML: reward = -cost, so the agent maximising reward is minimising cost
    return next_stock, -cost, demand


# 🧠 ML: Q-table has one row per stock level (0..20) and one column per action
Q = np.zeros((MAX_STOCK + 1, len(ACTIONS)))
alpha, gamma = 0.1, 0.95
epsilon = 1.0
EPISODES, DAYS = 1500, 30

episode_costs = []
for ep in range(EPISODES):
    # 🐍 PY: rng.integers(a, b) returns a random int in [a, b) - random starting stock each episode
    stock = int(rng.integers(0, MAX_STOCK + 1))
    total_cost = 0.0
    for _ in range(DAYS):
        # 🧠 ML: epsilon-greedy - explore with probability epsilon, otherwise exploit best Q
        if rng.random() < epsilon:
            a = int(rng.integers(len(ACTIONS)))
        else:
            a = int(np.argmax(Q[stock]))
        nxt, reward, _ = step(stock, ACTIONS[a])
        # 🧠 ML: Bellman update - move Q toward (reward + discounted best value of next state)
        Q[stock, a] += alpha * (reward + gamma * Q[nxt].max() - Q[stock, a])
        total_cost -= reward
        stock = nxt
    episode_costs.append(total_cost)
    # 🧠 ML: decay exploration so the agent gradually trusts what it has learned
    epsilon = max(0.05, epsilon * 0.997)

# 🐍 PY: slicing [:100] and [-100:] takes first and last 100 episodes
print(f"Avg 30-day cost, first 100 episodes: ${np.mean(episode_costs[:100]):.1f}")
print(f"Avg 30-day cost, last 100 episodes : ${np.mean(episode_costs[-100:]):.1f}")

# 🐍 PY: build a pandas DataFrame so the learned policy prints as a neat table
policy = pd.DataFrame({
    "stock": np.arange(MAX_STOCK + 1),
    "best_order": [ACTIONS[int(np.argmax(Q[s]))] for s in range(MAX_STOCK + 1)],
    "Q_order0": Q[:, 0].round(1),
    "Q_order5": Q[:, 1].round(1),
    "Q_order10": Q[:, 2].round(1),
})
print("\nLearned policy (every 2nd stock level):")
print(policy.iloc[::2].to_string(index=False))


def evaluate(policy_fn, runs=300):
    """Run the greedy policy with no exploration and return mean 30-day cost."""
    costs = []
    for _ in range(runs):
        stock, c = 10, 0.0
        for _ in range(DAYS):
            stock, r, _ = step(stock, policy_fn(stock))
            c -= r
        costs.append(c)
    return np.mean(costs)


# 🐍 PY: lambda creates a tiny one-line function
q_cost = evaluate(lambda s: ACTIONS[int(np.argmax(Q[s]))])
never_cost = evaluate(lambda s: 0)
always_cost = evaluate(lambda s: 5)
print("\nPolicy evaluation (mean cost over 30 days, 300 runs):")
print(f"  Q-learning policy : ${q_cost:7.1f}")
print(f"  Always order 5    : ${always_cost:7.1f}")
print(f"  Never order       : ${never_cost:7.1f}")
# 🐍 PY: " ".join over a generator turns the 21 best actions into one compact line
print("\nFull policy, stock 0..20:", " ".join(str(x) for x in policy.best_order))
first_no_order = int(policy.loc[policy.best_order == 0, "stock"].min())
print(f"Business read-out: the agent mostly reorders when stock < {first_no_order} units,"
      f" saving ${always_cost - q_cost:.0f} per month vs a fixed 'order 5 daily' rule.")

# ---
# Q-Learning — MEDIUM lab · Supply Chain: Inventory reorder policy for one SKU
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
