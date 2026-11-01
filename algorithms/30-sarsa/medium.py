# Tabular SARSA: a warehouse robot learns a safe route past a forklift lane (Build)
import numpy as np

np.random.seed(42)
ROWS, COLS = 4, 12
START, GOAL = (3, 0), (3, 11)          # bottom-left dock -> bottom-right packing station
# 🐍 PY: a set comprehension - every cell on the bottom row between start and goal is the forklift lane
HAZARD = {(3, c) for c in range(1, 11)}
MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
ARROWS = "^>v<"


def step(state, action):
    dr, dc = MOVES[action]
    # 🐍 PY: min/max clamp keeps the robot inside the warehouse walls
    r = min(max(state[0] + dr, 0), ROWS - 1)
    c = min(max(state[1] + dc, 0), COLS - 1)
    if (r, c) in HAZARD:
        # 🧠 ML: entering the forklift lane = near-miss/collision: big penalty, robot reset to the dock
        return START, -100, True
    return (r, c), -1, (r, c) == GOAL   # -1 per move encourages short routes


def choose(Q, s, eps):
    # 🧠 ML: epsilon-greedy - explore with probability eps, otherwise take the best known action
    if np.random.rand() < eps:
        return np.random.randint(4)
    # 🐍 PY: random tie-breaking among equal best actions avoids always picking 'up'
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(np.random.choice(best))


# 🧠 ML: Q-table = one row of 4 action-values per grid cell, shape (4, 12, 4)
Q = np.zeros((ROWS, COLS, 4))
alpha, gamma, eps = 0.5, 1.0, 0.1
history = []
for episode in range(500):
    s, total, collided = START, 0, False
    a = choose(Q, s, eps)
    for _ in range(200):
        s2, r, done = step(s, a)
        total += r
        collided |= (r == -100)
        a2 = choose(Q, s2, eps)
        # 🧠 ML: SARSA update uses the NEXT action a2 actually chosen by the same epsilon-greedy policy
        # 🐍 PY: conditional expression - a finished episode has no future value to add
        target = r if done else r + gamma * Q[s2][a2]
        Q[s][a] += alpha * (target - Q[s][a])
        if done:
            break
        # 🐍 PY: tuple assignment swaps in the new state and action in one line
        s, a = s2, a2
    history.append((total, collided))

# 🐍 PY: slicing the last 100 entries and unpacking with zip(*...)
rewards, cols = zip(*history[-100:])
print(f"Last 100 training episodes: avg reward {np.mean(rewards):.1f}, collisions {sum(cols)}")
for start in (0, 100, 200, 300, 400):
    chunk = history[start:start + 100]
    print(f"  episodes {start:3d}-{start + 99:3d}: avg reward {np.mean([h[0] for h in chunk]):7.1f}, "
          f"collisions {sum(h[1] for h in chunk):3d}")

# Follow the greedy policy and draw the route
grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
for (r, c) in HAZARD:
    grid[r][c] = "F"
s, steps = START, 0
while s != GOAL and steps < 50:
    a = int(np.argmax(Q[s]))
    grid[s[0]][s[1]] = ARROWS[a]
    s, _, _ = step(s, a)
    steps += 1
grid[START[0]][START[1]], grid[GOAL[0]][GOAL[1]] = "S", "G"
print(f"\nGreedy route learned by SARSA ({steps} moves). F = forklift lane:")
print("\n".join(" ".join(row) for row in grid))
print("\nBusiness read: SARSA keeps a buffer row away from the forklift lane, accepting a few extra")
print("moves because the robot's own random exploration could otherwise drift into traffic.")

# ---
# SARSA — BUILD lab · Supply Chain: Warehouse robot learning a safe route past a forklift lane
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
