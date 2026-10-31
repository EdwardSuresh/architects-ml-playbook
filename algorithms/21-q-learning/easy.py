# Q-Learning by hand: a toy 4-state inventory reorder problem for one SKU
import numpy as np

np.random.seed(42)
# 🐍 PY: a list holds the 4 possible stock levels (states) and 3 possible order sizes (actions)
states = [0, 5, 10, 15]
actions = [0, 5, 10]
# 🧠 ML: the Q-table stores "how good" each (state, action) pair is; it starts at all zeros
Q = np.zeros((len(states), len(actions)))
# 🧠 ML: alpha = learning rate, gamma = how much we value future rewards
alpha, gamma, epsilon = 0.5, 0.9, 0.2
HOLD_COST, STOCKOUT_COST, ORDER_COST = 1.0, 10.0, 2.0


def step(stock, order):
    # 🐍 PY: np.random.choice picks one random demand value from the list
    demand = np.random.choice([0, 5, 10])
    # 🐍 PY: min() caps stock at warehouse capacity of 15 units
    available = min(stock + order, 15)
    sold = min(available, demand)
    left = available - sold
    lost = demand - sold
    # 🧠 ML: reward is negative cost, so maximising reward = minimising cost
    reward = -(HOLD_COST * left + STOCKOUT_COST * lost + (ORDER_COST if order > 0 else 0))
    return left, reward, demand


s = 10
for day in range(1, 9):
    # 🐍 PY: states.index() converts a stock value (e.g. 10) into a row number (e.g. 2)
    si = states.index(s)
    # 🧠 ML: epsilon-greedy - mostly pick the best known action, sometimes explore a random one
    if np.random.rand() < epsilon:
        ai = np.random.randint(len(actions))
    else:
        ai = int(np.argmax(Q[si]))
    s2, r, d = step(s, actions[ai])
    s2i = states.index(s2)
    # 🧠 ML: the Q-learning update: nudge Q toward reward + discounted best future value
    target = r + gamma * Q[s2i].max()
    Q[si, ai] += alpha * (target - Q[si, ai])
    # 🐍 PY: f-strings with :>3 right-align numbers so the printout lines up
    print(f"Day {day}: stock={s:>2} order={actions[ai]:>2} demand={d:>2} -> stock={s2:>2} "
          f"reward={r:>6.1f} Q[{s},{actions[ai]}]={Q[si, ai]:>6.2f}")
    s = s2

print("\nQ-table (rows = stock level, cols = order size)")
print("stock | order0  order5 order10 | best")
# 🐍 PY: enumerate gives both the row number and the stock value
for i, st in enumerate(states):
    row = "  ".join(f"{v:6.2f}" for v in Q[i])
    print(f"{st:>5} | {row} | order {actions[int(np.argmax(Q[i]))]}")

# ---
# Q-Learning — EASY lab · Supply Chain: Inventory reorder policy for one SKU
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
