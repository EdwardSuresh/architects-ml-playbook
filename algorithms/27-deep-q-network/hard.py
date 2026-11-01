# DQN with target network + epsilon decay vs a tuned buy-low-sell-high rule for gas storage (Production)
import os
import random
import tempfile
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
ACTIONS = ["Inject", "Hold", "Withdraw"]


class GasStorageEnv:
    """A storage year from 1 April in 120 steps of 3 days, with fuel costs and price regime shocks."""
    CAPACITY, RATE, FEE, STEPS = 100.0, 8.0, 0.01, 120
    INJ_COST, WDR_COST = 0.04, 0.02  # $/MMBtu compressor fuel + variable O&M

    def __init__(self, seed):
        # 🐍 PY: each env owns its own RandomState so train and test price paths never overlap
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.t, self.storage, self.cash = 0, 0.0, 0.0
        steps = np.arange(self.STEPS + 1)
        # 🧠 ML: richer prices = random winter premium + AR(1) noise (shocks persist for a while)
        amp = self.rng.uniform(0.6, 1.4)
        noise = np.zeros(self.STEPS + 1)
        for i in range(1, self.STEPS + 1):
            noise[i] = 0.85 * noise[i - 1] + self.rng.normal(0, 0.12)
        self.prices = np.maximum(0.5, 3 - amp * np.cos(2 * np.pi * (steps - 30) / self.STEPS) + noise)
        return self.state()

    def state(self):
        ang = 2 * np.pi * self.t / self.STEPS
        return np.array([self.prices[self.t] - 3, self.storage / self.CAPACITY, np.sin(ang), np.cos(ang),
                         1 - self.t / self.STEPS], dtype=np.float32)

    def step(self, a):
        p = self.prices[self.t]
        vol = min(self.RATE, self.CAPACITY - self.storage) if a == 0 else (-min(self.RATE, self.storage) if a == 2 else 0)
        cost = self.INJ_COST * max(vol, 0) + self.WDR_COST * max(-vol, 0) + self.FEE * self.storage
        self.storage += vol
        self.cash += -vol * p - cost
        self.t += 1
        # 🧠 ML: mark-to-market reward: sums (telescopes) to final cash + inventory value
        reward = self.storage * (self.prices[self.t] - p) - cost
        return self.state(), reward / 10.0, self.t >= self.STEPS

    def value(self):
        return self.cash + self.storage * self.prices[self.t]


def make_qnet():
    return nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))


def train_dqn(episodes=120, gamma=0.97, lr=1e-3, batch=64, sync_every=200):
    env, qnet, target = GasStorageEnv(SEED), make_qnet(), make_qnet()
    # 🧠 ML: target network = a frozen copy used to compute targets; stops the 'chasing your own tail' problem
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    replay, updates = deque(maxlen=30000), 0
    for ep in range(episodes):
        # 🧠 ML: epsilon decays from 1.0 (pure exploration) to 0.05 over the first 70% of training
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))
        s, done, ep_r = env.reset(), False, 0.0
        while not done:
            if random.random() < eps:
                a = random.randrange(3)
            else:
                with torch.no_grad():
                    a = int(qnet(torch.from_numpy(s)).argmax())
            s2, r, done = env.step(a)
            replay.append((s, a, r, s2, float(done)))
            s, ep_r = s2, ep_r + r
            if len(replay) < 1000 or env.t % 2:
                continue
            S, A, R, S2, D = map(np.array, zip(*random.sample(replay, batch)))
            S, S2 = torch.from_numpy(S), torch.from_numpy(S2)
            A, R, D = torch.from_numpy(A), torch.tensor(R, dtype=torch.float32), torch.tensor(D, dtype=torch.float32)
            q = qnet(S).gather(1, A.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # 🧠 ML: Double-DQN target: online net CHOOSES the next action, target net SCORES it
                a_next = qnet(S2).argmax(1, keepdim=True)
                y = R + gamma * (1 - D) * target(S2).gather(1, a_next).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            # 🧠 ML: gradient clipping guards against rare exploding updates from price spikes
            nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
            opt.step()
            updates += 1
            if updates % sync_every == 0:
                target.load_state_dict(qnet.state_dict())
        if (ep + 1) % 30 == 0:
            print(f"  episode {ep + 1:3d}  eps={eps:.2f}  train value=${ep_r * 10:6.1f}k  updates={updates}")
    return qnet


def evaluate(policy, n_years=200, seed=999):
    # 🐍 PY: 'policy' is any function state->action, so the DQN and the rule share one evaluator
    env, vals = GasStorageEnv(seed), []
    for _ in range(n_years):
        s, done = env.reset(), False
        while not done:
            s, _, done = env.step(policy(s, env))
        vals.append(env.value())
    return np.array(vals)


def rule_policy(buy_below, sell_above):
    # 🐍 PY: a closure - the inner function 'remembers' the two thresholds
    def policy(s, env):
        p = env.prices[env.t]
        return 0 if p < buy_below else (2 if p > sell_above else 1)
    return policy


def main():
    t0 = time.time()
    # 🧠 ML: give the baseline a fair chance - tune its thresholds on TRAINING years, not test years
    grid = [(b, s) for b in (2.2, 2.5, 2.8) for s in (3.2, 3.5, 3.8)]
    scores = {g: evaluate(rule_policy(*g), n_years=60, seed=SEED).mean() for g in grid}
    best_rule = max(scores, key=scores.get)
    print(f"Best rule on training years: buy < ${best_rule[0]}, sell > ${best_rule[1]} "
          f"(avg ${scores[best_rule]:.1f}k)")

    print("Training Double DQN with target network...")
    qnet = train_dqn()
    print(f"Training + tuning time: {time.time() - t0:.1f}s")

    def dqn_policy(s, env):
        with torch.no_grad():
            return int(qnet(torch.from_numpy(s)).argmax())

    # 🧠 ML: evaluate on 200 unseen price years - the only honest way to compare policies
    res = {"Do nothing": evaluate(lambda s, e: 1), "Rule (tuned)": evaluate(rule_policy(*best_rule)),
           "DQN": evaluate(dqn_policy)}
    print("\nOut-of-sample results over 200 price years ($k per year):")
    print(f"{'policy':14s} {'mean':>7s} {'p5':>7s} {'p95':>7s} {'loss yrs':>9s}")
    for name, v in res.items():
        print(f"{name:14s} {v.mean():7.1f} {np.percentile(v, 5):7.1f} {np.percentile(v, 95):7.1f} "
              f"{(v < 0).mean():9.1%}")
    diff = res["DQN"] - res["Rule (tuned)"]
    print(f"DQN beats rule in {(diff > 0).mean():.0%} of years; mean uplift ${diff.mean():.1f}k")

    # 🧠 ML: explainability - what does the network do at key (season, fill level) points?
    print("\nPolicy map (rows = fill level, cols = month index, price at seasonal average):")
    env = GasStorageEnv(1)
    env.reset()
    print("fill  " + " ".join(f"{m:>2d}" for m in range(1, 13)))
    for fill in (0.0, 0.5, 1.0):
        row = []
        for m in range(12):
            t = m * 10 + 5
            ang = 2 * np.pi * t / 120
            p = 3 - np.cos(2 * np.pi * (t - 30) / 120)
            s = torch.tensor([p - 3, fill, np.sin(ang), np.cos(ang), 1 - t / 120], dtype=torch.float32)
            row.append(" " + ACTIONS[int(qnet(s).argmax())][0])
        print(f"{fill:4.1f} " + " ".join(row))
    print("(I=Inject H=Hold W=Withdraw; month 1 = April)")

    out = os.path.join(tempfile.mkdtemp(), "gas_storage_dqn.pt")
    # 🐍 PY: state_dict() is a plain dict of tensors - the portable way to save torch weights
    torch.save(qnet.state_dict(), out)
    print(f"\nSaved Q-network weights to {out}")
    print("MODEL-RISK NOTE: treat the agent as decision SUPPORT. Pipeline nominations, FERC/tariff limits,")
    print("ratchets and credit limits must be hard constraints outside the model; re-validate when the")
    print("forward curve regime changes (SR 11-7 style ongoing monitoring for trading models).")


if __name__ == "__main__":
    main()

# ---
# Deep Q-Network (DQN) — PRODUCTION lab · Oil & Gas: Natural gas storage: daily Inject / Hold / Withdraw decisions
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
