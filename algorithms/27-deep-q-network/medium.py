# DQN with a replay buffer: daily Inject / Hold / Withdraw decisions for a gas storage field (Medium)
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
ACTIONS = ["Inject", "Hold", "Withdraw"]


class GasStorageEnv:
    """One storage year from 1 April in 120 decision steps of 3 days. Cheap summer gas, dear winter gas."""
    CAPACITY, RATE, FEE, STEPS = 100.0, 8.0, 0.01, 120  # thousand MMBtu, per step, $/unit/step, steps/yr

    def reset(self):
        self.day, self.storage, self.cash = 0, 0.0, 0.0
        # 🧠 ML: seasonal price curve: ~$2 around July, ~$4 around January, random noise each year
        t = np.arange(self.STEPS + 1)
        self.prices = 3 - np.cos(2 * np.pi * (t - 30) / self.STEPS) + np.random.normal(0, 0.15, self.STEPS + 1)
        return self._state()

    def _state(self):
        # 🐍 PY: np.array of 4 floats - scaled price, fill level, and the calendar as sin/cos
        ang = 2 * np.pi * self.day / self.STEPS
        return np.array([self.prices[self.day] - 3, self.storage / self.CAPACITY, np.sin(ang), np.cos(ang)],
                        dtype=np.float32)

    def step(self, action):
        p = self.prices[self.day]
        # 🐍 PY: min/max clamp the volume so we never overfill or go below empty
        vol = 0.0
        if action == 0:
            vol = min(self.RATE, self.CAPACITY - self.storage)
        elif action == 2:
            vol = -min(self.RATE, self.storage)
        self.storage += vol
        self.day += 1
        # 🧠 ML: mark-to-market reward = gain in value of gas held (cash + inventory at today's price) - fee
        # 🧠 ML: a dense signal like this is far easier to learn from than 'cash only when we finally sell'
        reward = self.storage * (self.prices[self.day] - p) - self.FEE * self.storage
        self.cash += -vol * p
        return self._state(), reward / 10.0, self.day >= self.STEPS  # reward scaled for stable learning


# 🧠 ML: the Q-network maps a 4-number state to 3 Q-values (expected future profit of each action)
qnet = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
opt = torch.optim.Adam(qnet.parameters(), lr=1e-3)
# 🐍 PY: deque(maxlen=...) is a fixed-size queue - oldest memories drop off automatically
replay = deque(maxlen=20000)
env, gamma, eps, batch = GasStorageEnv(), 0.97, 0.10, 64


def act(state, eps):
    # 🧠 ML: epsilon-greedy - mostly exploit the best Q, sometimes try a random action to explore
    if random.random() < eps:
        return random.randrange(3)
    with torch.no_grad():
        return int(qnet(torch.from_numpy(state)).argmax())


for ep in range(1, 61):
    s, done, total = env.reset(), False, 0.0
    while not done:
        a = act(s, eps)
        s2, r, done = env.step(a)
        # 🐍 PY: append a tuple to the buffer; later we sample random mini-batches from it
        replay.append((s, a, r, s2, done))
        s, total = s2, total + r
        if len(replay) >= 500 and env.day % 2 == 0:
            # 🧠 ML: random sampling breaks the day-to-day correlation that destabilises training
            S, A, R, S2, D = zip(*random.sample(replay, batch))
            S, S2 = torch.tensor(np.array(S)), torch.tensor(np.array(S2))
            A, R, D = torch.tensor(A), torch.tensor(R, dtype=torch.float32), torch.tensor(D, dtype=torch.float32)
            q = qnet(S).gather(1, A.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target = R + gamma * (1 - D) * qnet(S2).max(1).values
            loss = nn.functional.smooth_l1_loss(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if ep % 10 == 0:
        print(f"episode {ep:2d}  mark-to-market P&L=${total * 10:7.1f}k  buffer={len(replay)}")

# 🧠 ML: evaluate greedily (eps=0) on a fresh price year and show the monthly pattern
s, done, log = env.reset(), False, []
while not done:
    a = act(s, 0.0)
    log.append((env.day, env.prices[env.day], a, env.storage))
    s, r, done = env.step(a)
MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
print("\nMonth  avg_price  most_common_action  end_storage")
for m in range(12):
    # 🐍 PY: slicing a list of tuples - 10 steps of 3 days = one month
    chunk = log[m * 10:(m + 1) * 10]
    acts = [c[2] for c in chunk]
    print(f"{MONTHS[m]:>5s}  ${np.mean([c[1] for c in chunk]):8.2f}  {ACTIONS[max(set(acts), key=acts.count)]:>18s}"
          f"  {env.storage if m == 11 else log[(m + 1) * 10][3]:10.0f}")
value = env.cash + env.storage * env.prices[-1]
print(f"\nBusiness read: year-end value (cash + gas left) = ${value:.0f}k on a 100k MMBtu field.")
print("A good policy fills in the cheap summer months and withdraws into the winter price peak.")

# ---
# Deep Q-Network (DQN) — MEDIUM lab · Oil & Gas: Natural gas storage: daily Inject / Hold / Withdraw decisions
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
