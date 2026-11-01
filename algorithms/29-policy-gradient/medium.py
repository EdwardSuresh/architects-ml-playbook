# REINFORCE with a feature-conditioned softmax policy: personalised renewal discounts (Build)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)
DISCOUNTS = np.array([0.00, 0.05, 0.10])


def make_customers(n):
    # 🐍 PY: a DataFrame from a dict of equal-length numpy arrays
    return pd.DataFrame({
        "tenure_years": rng.integers(1, 15, n),
        "annual_premium": rng.normal(1400, 400, n).clip(400, 3000).round(0),
        "claims_last_3y": rng.poisson(0.4, n),
        "competitor_quote_gap": rng.normal(0.0, 0.08, n).round(3),  # + means rival is cheaper
    })


def renew_probability(df, discount):
    # 🧠 ML: the hidden 'world': loyal customers barely react to discounts, shoppers react a lot
    z = (0.6 + 0.12 * df.tenure_years - 0.3 * df.claims_last_3y - 12 * df.competitor_quote_gap
         + discount * (30 - 3 * df.tenure_years).clip(lower=1))
    return 1 / (1 + np.exp(-z))


def features(df):
    # 🧠 ML: scale features to similar ranges so gradient steps treat them fairly
    X = np.c_[df.tenure_years / 10, df.annual_premium / 1000, df.claims_last_3y, df.competitor_quote_gap * 10]
    return torch.tensor(X, dtype=torch.float32)


train, test = make_customers(4000), make_customers(1000)
X_train, X_test = features(train), features(test)
# 🧠 ML: policy network = linear layer -> 3 logits -> softmax probabilities over the discount levels
policy = nn.Linear(4, 3)
opt = torch.optim.Adam(policy.parameters(), lr=0.03)

for epoch in range(1, 401):
    idx = rng.choice(len(train), 256, replace=False)
    dist = torch.distributions.Categorical(logits=policy(X_train[idx]))
    a = dist.sample()
    batch = train.iloc[idx]
    d = DISCOUNTS[a.numpy()]
    # 🧠 ML: simulate the customer's choice; reward = retained premium net of the discount
    renewed = rng.random(len(idx)) < renew_probability(batch, d).to_numpy()
    reward = torch.tensor(renewed * batch.annual_premium.to_numpy() * (1 - d) / 1000, dtype=torch.float32)
    # 🧠 ML: REINFORCE loss = -(log pi(a|x) * reward); minimising it raises probability of rewarded actions
    loss = -(dist.log_prob(a) * reward).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  avg reward ${reward.mean().item() * 1000:7.1f}  "
              f"action mix (0/5/10%) {np.round(np.bincount(a.numpy(), minlength=3) / len(idx), 2)}")


def expected_value(df, d):
    # 🐍 PY: vectorised pandas maths - no loop over customers needed
    return (renew_probability(df, d) * df.annual_premium * (1 - d)).mean()


with torch.no_grad():
    choice = policy(X_test).argmax(1).numpy()
print("\nTest set: expected retained premium per customer")
for i, dsc in enumerate(DISCOUNTS):
    print(f"  everyone gets {dsc:4.0%}:          ${expected_value(test, np.full(len(test), dsc)):8.2f}")
print(f"  learned personalised policy:  ${expected_value(test, DISCOUNTS[choice]):8.2f}")
# 🧠 ML: the 'oracle' knows the hidden world - an upper bound we can't reach in real life
oracle = np.max([renew_probability(test, dd) * test.annual_premium * (1 - dd) for dd in DISCOUNTS], axis=0)
print(f"  oracle (perfect knowledge):   ${oracle.mean():8.2f}")

test["offer"] = DISCOUNTS[choice]
# 🐍 PY: pd.cut buckets tenure into bands; groupby + mean summarises the offer per band
test["tenure_band"] = pd.cut(test.tenure_years, [0, 3, 7, 15], labels=["1-3y", "4-7y", "8-14y"])
print("\nAverage discount offered by tenure band:")
print(test.groupby("tenure_band", observed=True)["offer"].mean().map("{:.1%}".format).to_string())

# ---
# Policy Gradient (REINFORCE) — BUILD lab · Insurance: Renewal retention offers: which discount for which policyholder
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
