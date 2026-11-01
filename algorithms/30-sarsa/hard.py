# SARSA vs Q-learning (and Expected SARSA) for a warehouse robot near a forklift lane (Production)
import os
import tempfile
import numpy as np

ROWS, COLS = 5, 12
START, GOAL = (4, 0), (4, 11)
HAZARD = {(4, c) for c in range(1, 11)}              # forklift lane along the bottom
RACKS = {(2, c) for c in range(3, 9) if c != 6}      # storage racks the robot cannot enter
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = "^>v<"
COLLISION, MOVE_COST, SLIP = -100.0, -1.0, 0.0  # SLIP = chance of wheel slip (try 0.05)


class Warehouse:
    """Grid with racks, a forklift lane and optional wheel slip (robot moves randomly SLIP of the time)."""
    def __init__(self, rng):
        self.rng = rng

    def step(self, s, a):
        # 🧠 ML: with SLIP > 0 transitions become stochastic - the same action may give a different result
        if self.rng.random() < SLIP:
            a = int(self.rng.integers(4))
        r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
        # 🐍 PY: chained comparison 0 <= r < ROWS reads like maths
        if not (0 <= r < ROWS and 0 <= c < COLS) or (r, c) in RACKS:
            r, c = s                                   # bump into wall/rack: stay put
        if (r, c) in HAZARD:
            return START, COLLISION, True, True
        return (r, c), MOVE_COST, (r, c) == GOAL, False


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def td_target(method, Q, r, s2, a2, done, gamma, eps):
    # 🧠 ML: the ONLY difference between the three algorithms is this one line of maths
    if done:
        return r
    if method == "SARSA":            # on-policy: value of the action we will really take
        return r + gamma * Q[s2][a2]
    if method == "Q-learning":       # off-policy: value of the best action, ignoring exploration
        return r + gamma * Q[s2].max()
    # Expected SARSA: average over the epsilon-greedy policy's action probabilities
    probs = np.full(4, eps / 4)
    probs[Q[s2].argmax()] += 1 - eps
    return r + gamma * probs @ Q[s2]


def train(method, episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    env, Q = Warehouse(rng), np.zeros((ROWS, COLS, 4))
    log = []
    for _ in range(episodes):
        s, total, hits = START, 0.0, 0
        a = eps_greedy(Q, s, eps, rng)
        for _ in range(300):
            s2, r, done, hit = env.step(s, a)
            a2 = eps_greedy(Q, s2, eps, rng)
            Q[s][a] += alpha * (td_target(method, Q, r, s2, a2, done, gamma, eps) - Q[s][a])
            total, hits = total + r, hits + hit
            if done:
                break
            s, a = s2, a2
        log.append((total, hits))
    return Q, np.array(log)


def evaluate(Q, eps, n=300, seed=123):
    # 🧠 ML: online evaluation - the robot still explores a little in production (eps>0) and still slips
    rng = np.random.default_rng(seed)
    env, rewards, hits, steps = Warehouse(rng), [], 0, []
    for _ in range(n):
        s, total = START, 0.0
        for k in range(300):
            s, r, done, hit = env.step(s, eps_greedy(Q, s, eps, rng))
            total, hits = total + r, hits + hit
            if done:
                break
        rewards.append(total)
        steps.append(k + 1)
    return np.mean(rewards), hits / n, np.mean(steps)


def draw_route(Q):
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for r, c in HAZARD:
        grid[r][c] = "F"
    for r, c in RACKS:
        grid[r][c] = "#"
    s, seen = START, set()
    # 🐍 PY: a 'seen' set guards against an infinite loop if the greedy policy cycles
    while s != GOAL and s not in seen:
        seen.add(s)
        a = int(Q[s].argmax())
        grid[s[0]][s[1]] = ARROWS[a]
        r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
        if not (0 <= r < ROWS and 0 <= c < COLS) or (r, c) in RACKS:
            break
        s = (r, c)
    grid[START[0]][START[1]], grid[GOAL[0]][GOAL[1]] = "S", "G"
    return "\n".join("  " + " ".join(row) for row in grid), len(seen)


def main():
    methods = ["SARSA", "Q-learning", "Expected SARSA"]
    # 🧠 ML: average over 5 seeds - RL results from a single run are notoriously noisy
    print("Training (eps=0.1, alpha=0.5), mean over 5 seeds, last 100 episodes:")
    print(f"{'method':15s} {'train reward':>12s} {'collisions/ep':>14s}")
    tables = {}
    for m in methods:
        logs = [train(m, seed=sd) for sd in range(5)]
        tables[m] = logs[0][0]
        last = np.concatenate([lg[-100:] for _, lg in logs])
        print(f"{m:15s} {last[:, 0].mean():12.1f} {last[:, 1].mean():14.3f}")

    # 🧠 ML: tuning sweep - how does the exploration rate change the gap between SARSA and Q-learning?
    print("\nExploration sweep - evaluated WITH the same eps the robot keeps in operation:")
    print(f"{'eps':>5s} {'method':15s} {'avg reward':>11s} {'collisions/trip':>16s} {'avg moves':>10s}")
    for eps in (0.0, 0.05, 0.1, 0.2):
        for m in ("SARSA", "Q-learning"):
            Q, _ = train(m, eps=max(eps, 0.02), seed=1)
            rew, col, mv = evaluate(Q, eps)
            print(f"{eps:5.2f} {m:15s} {rew:11.1f} {col:16.3f} {mv:10.1f}")

    for m in ("SARSA", "Q-learning"):
        route, n = draw_route(tables[m])
        print(f"\n{m} greedy route ({n} moves). F=forklift lane, #=rack:")
        print(route)

    # 🧠 ML: business cost - convert collisions to dollars (equipment damage, stoppage, OSHA reporting)
    cost_per_hit, trips_per_day = 2500, 400
    print("\nDaily risk cost at eps=0.1 (400 trips/day, $2,500 per forklift incident):")
    for m in ("SARSA", "Q-learning"):
        _, col, mv = evaluate(tables[m], 0.1)
        print(f"  {m:11s}: {col * trips_per_day:5.1f} incidents/day -> ${col * trips_per_day * cost_per_hit:9,.0f}"
              f"  (avg {mv:.1f} moves/trip)")

    out = tempfile.mkdtemp()
    # 🐍 PY: np.savez stores several named arrays in one compressed file
    np.savez(os.path.join(out, "warehouse_q_tables.npz"), **{m.replace(" ", "_"): tables[m] for m in methods})
    print(f"\nSaved Q-tables to {out}")
    print("SAFETY NOTE: RL routing is a planning aid, not a safety system. Physical separation, speed limits,")
    print("lidar e-stops and ANSI/RIA R15.08 / OSHA 1910.178 controls must stay in place regardless of policy.")


if __name__ == "__main__":
    main()

# ---
# SARSA — PRODUCTION lab · Supply Chain: Warehouse robot learning a safe route past a forklift lane
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
