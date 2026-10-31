# A2C portfolio allocator with risk-adjusted reward, transaction costs and Sharpe vs 60/40 (Hard)
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
SEED = 42
torch.manual_seed(SEED)
PORTFOLIOS = np.array([(0.9, 0.1, 0.0), (0.7, 0.3, 0.0), (0.6, 0.4, 0.0), (0.4, 0.5, 0.1), (0.1, 0.4, 0.5)])
TC = 0.0010  # 10 bps per unit of turnover (commissions + spread + market impact)


def simulate_market(n, rng):
    """Regime-switching returns for stocks, bonds, cash. Bear regimes: stocks fall, bonds rally."""
    regime, out = 0, np.zeros((n, 3))
    for t in range(n):
        if rng.rand() < (1 / 30 if regime == 0 else 1 / 9):
            regime = 1 - regime
        s = rng.normal(0.010, 0.035) if regime == 0 else rng.normal(-0.012, 0.060)
        b = rng.normal(0.003, 0.012) if regime == 0 else rng.normal(0.006, 0.015)
        out[t] = (s, b, 0.002)
    return out


class PortfolioEnv:
    """State = market features + current holdings, so the agent can 'see' its transaction cost."""
    def __init__(self, rets, risk_aversion):
        self.rets, self.lam = rets, risk_aversion

    def features(self, t, w):
        r = self.rets
        # 🐍 PY: np.concatenate glues the market signals and the current weights into one vector
        mkt = [r[t - 3:t, 0].sum(), r[t - 6:t, 0].std(), r[t - 12:t, 0].sum(), r[t - 3:t, 1].sum(), r[t - 1, 0]]
        return np.concatenate([np.array(mkt) * 10, w]).astype(np.float32)

    def batch_features(self, ts, W):
        # 🐍 PY: np.stack builds a (batch, 8) matrix so the network scores many episodes in one call
        return torch.from_numpy(np.stack([self.features(t, w) for t, w in zip(ts, W)]))

    def step(self, t, w_old, a):
        w_new = PORTFOLIOS[a]
        # 🧠 ML: transaction cost = turnover x cost rate; discourages flip-flopping every month
        cost = TC * np.abs(w_new - w_old).sum()
        port = w_new @ self.rets[t] - cost
        # 🧠 ML: risk-adjusted reward = return - lambda * squared deviation (a monthly mean-variance utility)
        reward = port - self.lam * (port - 0.005) ** 2
        return reward * 100, port, w_new


class ActorCritic(nn.Module):
    def __init__(self, n_in, n_act, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor, self.critic = nn.Linear(hidden, n_act), nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def train(rets, lam, epochs=150, T=24, B=16, gamma=0.6, lr=3e-3, seed=SEED):
    torch.manual_seed(seed)
    rng, env = np.random.RandomState(seed), PortfolioEnv(rets, lam)
    model = ActorCritic(8, len(PORTFOLIOS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        # 🧠 ML: B parallel episodes (different start months) = a less noisy gradient per update
        t0 = rng.randint(12, len(rets) - T - 1, size=B)
        W = np.tile(PORTFOLIOS[2], (B, 1))
        logps, vals, rews, ents = [], [], [], []
        # 🧠 ML: roll out T months; holdings carry over, so the actor's choice changes the next state
        for k in range(T):
            logits, v = model(env.batch_features(t0 + k, W))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().numpy()
            # 🐍 PY: zip over three arrays at once, then unpack the list of tuples with zip(*...)
            r, _, W = map(np.array, zip(*[env.step(t, w, ai) for t, w, ai in zip(t0 + k, W, a)]))
            logps.append(dist.log_prob(torch.from_numpy(a)))
            vals.append(v)
            rews.append(torch.tensor(r, dtype=torch.float32))
            ents.append(dist.entropy())
        with torch.no_grad():
            R = model(env.batch_features(t0 + T, W))[1]
        # 🧠 ML: n-step returns computed backwards from the critic's bootstrap value of the final state
        rets_n = []
        for r in reversed(rews):
            R = r + gamma * R
            rets_n.insert(0, R)
        rets_n, vals, logps = torch.stack(rets_n), torch.stack(vals), torch.stack(logps)
        # 🧠 ML: advantage = n-step return - critic's estimate; detach so the actor doesn't train the critic
        adv = rets_n - vals
        loss = -(logps * adv.detach()).mean() + 0.5 * adv.pow(2).mean() - 0.01 * torch.stack(ents).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def backtest(rets, policy, lam=0.0):
    env, w, port, turns = PortfolioEnv(rets, lam), PORTFOLIOS[2], [], []
    for t in range(12, len(rets)):
        a = policy(env.features(t, w), w)
        turns.append(np.abs(PORTFOLIOS[a] - w).sum())
        _, p, w = env.step(t, w, a)
        port.append(p)
    p = np.array(port)
    # 🧠 ML: Sharpe = excess return over cash / volatility, annualised (x sqrt(12) for monthly data)
    sharpe = (p.mean() - 0.002) / p.std() * np.sqrt(12)
    dd = (np.cumprod(1 + p) / np.maximum.accumulate(np.cumprod(1 + p)) - 1).min()
    return {"ann_ret%": p.mean() * 1200, "ann_vol%": p.std() * np.sqrt(12) * 100, "sharpe": sharpe,
            "max_dd%": dd * 100, "turnover/yr": np.mean(turns) * 12}


def greedy(model):
    def pol(x, w):
        with torch.no_grad():
            return int(model(torch.from_numpy(x))[0].argmax())
    return pol


def main():
    rng = np.random.RandomState(SEED)
    train_r, valid_r, test_r = simulate_market(1200, rng), simulate_market(360, rng), simulate_market(480, rng)
    # 🧠 ML: tune the risk-aversion lambda on a VALIDATION period; test period is touched only once
    print("Tuning risk aversion (validation Sharpe):")
    best = None
    for lam in (0.0, 3.0, 8.0):
        m = train(train_r, lam)
        s = backtest(valid_r, greedy(m))["sharpe"]
        print(f"  lambda={lam:3.1f}  valid Sharpe={s:5.2f}")
        if best is None or s > best[1]:
            best = (lam, s, m)
    lam, _, model = best
    print(f"Chosen lambda = {lam}")

    # 🐍 PY: dict of name -> policy function; lambdas ignore the state for the static baselines
    policies = {"A2C agent": greedy(model), "60/40 static": lambda x, w: 2, "Always 90/10": lambda x, w: 0}
    print("\nTest period (40 unseen years), after 10 bps transaction costs:")
    print(f"{'strategy':14s} {'ann_ret%':>9s} {'ann_vol%':>9s} {'sharpe':>7s} {'max_dd%':>8s} {'turnover/yr':>12s}")
    for name, pol in policies.items():
        m = backtest(test_r, pol)
        print(f"{name:14s} " + " ".join(f"{m[k]:{9 if 'ann' in k else 7 if k == 'sharpe' else 8 if k == 'max_dd%' else 12}.2f}"
                                       for k in m))

    # 🧠 ML: explainability - how does the actor's probability mass shift from calm to stressed markets?
    w0 = PORTFOLIOS[2]
    calm = np.concatenate([np.array([0.03, 0.02, 0.12, 0.01, 0.01]) * 10, w0]).astype(np.float32)
    stress = np.concatenate([np.array([-0.10, 0.07, -0.15, 0.03, -0.05]) * 10, w0]).astype(np.float32)
    for label, x in [("calm uptrend", calm), ("stressed selloff", stress)]:
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(x))[0], -1).numpy()
        # 🐍 PY: np.round on the whole probability vector; one probability per model portfolio
        print(f"{label:17s} P(90/10, 70/30, 60/40, 40/50/10, 10/40/50) = {np.round(probs, 2)}")

    path = os.path.join(tempfile.mkdtemp(), "a2c_allocator.pt")
    torch.save({"state_dict": model.state_dict(), "lambda": lam, "portfolios": PORTFOLIOS.tolist()}, path)
    print(f"\nSaved model to {path}")
    print("MODEL-RISK NOTE (SR 11-7): synthetic regimes are not history. Validate on real out-of-sample data,")
    print("stress-test 2008/2020/2022-style scenarios, cap allocation changes, and keep a human PM sign-off.")
    print("Suitability (Reg BI / fiduciary duty) still applies: the agent cannot know a client's risk tolerance.")


if __name__ == "__main__":
    main()

# ---
# Actor-Critic — HARD lab · Financial Services: Monthly Stocks / Bonds / Cash allocation
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
