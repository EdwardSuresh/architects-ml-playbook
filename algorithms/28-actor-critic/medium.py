# Advantage Actor-Critic (A2C): monthly Stocks/Bonds/Cash allocation from 5 model portfolios (Build)
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

# 🐍 PY: list of tuples (stocks, bonds, cash) - the 5 allowed model portfolios = discrete actions
PORTFOLIOS = [(0.9, 0.1, 0.0), (0.7, 0.3, 0.0), (0.5, 0.4, 0.1), (0.3, 0.5, 0.2), (0.0, 0.3, 0.7)]


def simulate_market(n_months):
    """Regime-switching monthly returns: bull markets last ~2 years, bear markets ~8 months."""
    # 🧠 ML: the agent never sees the regime directly - it must infer it from recent returns
    regime, rows = 0, []
    for _ in range(n_months):
        if np.random.rand() < (1 / 24 if regime == 0 else 1 / 8):
            regime = 1 - regime
        mu_s, sd_s = (0.012, 0.035) if regime == 0 else (-0.015, 0.06)
        stock = np.random.normal(mu_s, sd_s)
        bond = np.random.normal(0.003 if regime == 0 else 0.006, 0.012)
        rows.append((stock, bond, 0.002))
    return np.array(rows)


def make_states(rets):
    # 🐍 PY: list comprehension building features from a rolling window of the last 3 and 6 months
    feats = [[rets[t - 3:t, 0].sum(), rets[t - 6:t, 0].std(), rets[t - 3:t, 1].sum(), rets[t - 1, 0]]
             for t in range(6, len(rets))]
    return torch.tensor(np.array(feats) * 10, dtype=torch.float32), rets[6:]


class ActorCritic(nn.Module):
    # 🐍 PY: a class that inherits nn.Module; __init__ builds layers, forward defines the computation
    def __init__(self, n_in, n_act):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, 32), nn.Tanh())
        # 🧠 ML: two heads share one body: the ACTOR outputs action logits, the CRITIC outputs V(s)
        self.actor = nn.Linear(32, n_act)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


train_s, train_r = make_states(simulate_market(1206))  # ~100 years of synthetic months
test_s, test_r = make_states(simulate_market(366))     # 30 unseen years
weights = torch.tensor(PORTFOLIOS, dtype=torch.float32)
model = ActorCritic(4, len(PORTFOLIOS))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
gamma, T = 0.5, 24  # learn from 24-month rollouts

for epoch in range(1, 801):
    start = np.random.randint(0, len(train_s) - T - 1)
    s = train_s[start:start + T + 1]
    logits, values = model(s)
    # 🧠 ML: Categorical turns logits into a probability distribution we can sample from
    dist = torch.distributions.Categorical(logits=logits[:-1])
    a = dist.sample()
    # 🧠 ML: reward = portfolio return in % (weights . asset returns for that month)
    r = (weights[a] * torch.tensor(train_r[start:start + T], dtype=torch.float32)).sum(1) * 100
    # 🧠 ML: one-step TD target and advantage = r + gamma*V(s') - V(s)
    target = r + gamma * values[1:].detach()
    adv = target - values[:-1]
    actor_loss = -(dist.log_prob(a) * adv.detach()).mean()
    critic_loss = adv.pow(2).mean()
    loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 160 == 0:
        print(f"epoch {epoch:3d}  avg reward {r.mean():+.3f}%/mo  critic loss {critic_loss.item():.3f}")

# 🧠 ML: evaluate greedily on unseen months versus a static 60/40 stocks/bonds portfolio
with torch.no_grad():
    acts = model(test_s)[0].argmax(1)
agent = (weights[acts].numpy() * test_r).sum(1)
bench = test_r @ np.array([0.6, 0.4, 0.0])
# 🐍 PY: a small helper defined with lambda for annualised return and volatility
ann = lambda x: (x.mean() * 12 * 100, x.std() * np.sqrt(12) * 100)
print("\nTest period (30 years)   ann_return  ann_vol")
for name, x in [("A2C agent", agent), ("60/40 static", bench)]:
    print(f"{name:22s} {ann(x)[0]:9.2f}% {ann(x)[1]:7.2f}%")
print("\nHow often each model portfolio was chosen:")
for i, w in enumerate(PORTFOLIOS):
    print(f"  stocks {w[0]:.0%} / bonds {w[1]:.0%} / cash {w[2]:.0%}: {(acts == i).float().mean():.0%}")
print("Business read: the agent mostly holds the growth mix and switches to the defensive mix in some months.")
print("It earned more than 60/40 here but with more volatility - the Production level adds risk and cost penalties.")

# ---
# Actor-Critic — BUILD lab · Financial Services: Monthly Stocks / Bonds / Cash allocation
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
