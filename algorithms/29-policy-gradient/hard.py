# REINFORCE with learned baseline + entropy bonus for renewal discounts, with fairness checks (Hard)
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
SEED = 42
DISCOUNTS = np.array([0.00, 0.05, 0.10])
FEATS = ["tenure_years", "annual_premium", "claims_last_3y", "competitor_quote_gap", "multi_policy"]


def make_book(n, rng):
    df = pd.DataFrame({
        "tenure_years": rng.integers(1, 15, n),
        "annual_premium": rng.normal(1400, 400, n).clip(400, 3000).round(0),
        "claims_last_3y": rng.poisson(0.4, n),
        "competitor_quote_gap": rng.normal(0.0, 0.08, n),
        "multi_policy": rng.random(n) < 0.35,
        # 🧠 ML: age_band is NOT a model input - we keep it only to AUDIT outcomes for fairness
        "age_band": rng.choice(["18-34", "35-54", "55+"], n, p=[0.3, 0.4, 0.3]),
    })
    return df


def renew_prob(df, d):
    # 🧠 ML: hidden response curve - shoppers (short tenure, rival cheaper) are discount-sensitive
    sens = (30 - 3 * df.tenure_years).clip(lower=1) + 40 * df.competitor_quote_gap.clip(lower=0)
    z = (0.6 + 0.12 * df.tenure_years - 0.3 * df.claims_last_3y - 12 * df.competitor_quote_gap
         + 0.8 * df.multi_policy + d * sens)
    return 1 / (1 + np.exp(-z))


def to_tensor(df, mu=None, sd=None):
    X = df[FEATS].astype(float).to_numpy()
    # 🐍 PY: default None args let us reuse TRAIN mean/std on the test set (no leakage)
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) if sd is None else sd
    return torch.tensor((X - mu) / sd, dtype=torch.float32), mu, sd


class Policy(nn.Module):
    def __init__(self, n_in, n_act=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, n_act))
        # 🧠 ML: separate value head = learned baseline b(x): 'what reward do we expect for this customer?'
        self.base = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x), self.base(x).squeeze(-1)


def train(df, X, use_baseline, ent_coef, epochs=300, batch=256, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = Policy(X.shape[1])
    opt = torch.optim.Adam(pol.parameters(), lr=0.01)
    grad_norms = []
    for _ in range(epochs):
        idx = rng.choice(len(df), batch, replace=False)
        logits, b = pol(X[idx])
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        sub, d = df.iloc[idx], DISCOUNTS[a.numpy()]
        renewed = rng.random(batch) < renew_prob(sub, d).to_numpy()
        r = torch.tensor(renewed * sub.annual_premium.to_numpy() * (1 - d) / 1000, dtype=torch.float32)
        # 🧠 ML: advantage = reward - baseline. Same expected gradient, MUCH lower variance
        adv = r - b.detach() if use_baseline else r
        pg_loss = -(dist.log_prob(a) * adv).mean()
        # 🧠 ML: entropy bonus keeps the policy from collapsing onto one discount too early
        loss = pg_loss - ent_coef * dist.entropy().mean() + (b - r).pow(2).mean()
        opt.zero_grad()
        pg_loss.backward(retain_graph=True)
        # 🐍 PY: generator expression inside sum() - total gradient size of the policy layer
        grad_norms.append(sum(p.grad.norm().item() ** 2 for p in pol.net.parameters()) ** 0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol, np.std(grad_norms[-100:])


def expected_premium(df, d):
    return float((renew_prob(df, d) * df.annual_premium * (1 - d)).mean())


def main():
    rng = np.random.default_rng(SEED)
    train_df, valid_df, test_df = make_book(6000, rng), make_book(2000, rng), make_book(2000, rng)
    X_tr, mu, sd = to_tensor(train_df)
    X_va, _, _ = to_tensor(valid_df, mu, sd)
    X_te, _, _ = to_tensor(test_df, mu, sd)

    # 🧠 ML: ablation + tuning - does the baseline help, and how much entropy bonus is right?
    print("config                         grad-norm std  valid $/customer")
    results = {}
    for use_b, ent in [(False, 0.0), (True, 0.0), (True, 0.01), (True, 0.05)]:
        pol, gstd = train(train_df, X_tr, use_b, ent)
        with torch.no_grad():
            val = expected_premium(valid_df, DISCOUNTS[pol(X_va)[0].argmax(1).numpy()])
        key = f"baseline={use_b!s:5s} entropy={ent:.2f}"
        results[key] = (pol, val)
        print(f"{key:30s} {gstd:13.3f} {val:16.2f}")
    # 🧠 ML: pick the config on the VALIDATION book; the test book is scored once, below
    best_key = max(results, key=lambda k: results[k][1])
    pol = results[best_key][0]
    with torch.no_grad():
        choice = pol(X_te)[0].argmax(1).numpy()
    best_val = expected_premium(test_df, DISCOUNTS[choice])
    print(f"Chosen: {best_key}")

    print("\nStrategy comparison on held-out book (expected retained premium net of discount):")
    for d in DISCOUNTS:
        print(f"  fixed {d:4.0%} for everyone   ${expected_premium(test_df, np.full(len(test_df), d)):8.2f}")
    print(f"  REINFORCE policy         ${best_val:8.2f}")
    fixed_best = max(expected_premium(test_df, np.full(len(test_df), d)) for d in DISCOUNTS)
    print(f"  uplift vs best fixed: ${best_val - fixed_best:.2f}/customer "
          f"= ${(best_val - fixed_best) * 500_000 / 1e6:.2f}M on a 500k-policy book")

    # 🧠 ML: explainability - average offer by driver (a simple partial-dependence-style table)
    test_df["offer"] = DISCOUNTS[choice]
    test_df["tenure_band"] = pd.cut(test_df.tenure_years, [0, 3, 7, 15], labels=["1-3y", "4-7y", "8-14y"])
    test_df["rival_cheaper"] = test_df.competitor_quote_gap > 0.05
    print("\nAverage offer by tenure band and whether a rival quote is >5% cheaper:")
    print(test_df.pivot_table(index="tenure_band", columns="rival_cheaper", values="offer",
                              observed=True).map(lambda v: f"{v:.1%}").to_string())

    # 🧠 ML: fairness audit - are outcomes skewed by age, even though age was never an input?
    test_df["p_renew"] = renew_prob(test_df, test_df.offer)
    audit = test_df.groupby("age_band").agg(avg_offer=("offer", "mean"), avg_premium=("annual_premium", "mean"),
                                            renew_rate=("p_renew", "mean"))
    print("\nFairness audit by age band (not a model input):")
    print(audit.round(3).to_string())

    path = os.path.join(tempfile.mkdtemp(), "renewal_policy.pt")
    torch.save({"state_dict": pol.state_dict(), "mu": mu, "sd": sd}, path)
    print(f"\nSaved policy to {path}")
    print("REGULATORY NOTE: many US states restrict or ban 'price optimization' - setting renewal prices")
    print("by predicted willingness to stay rather than risk (see NAIC 2015 white paper, CA/NY/FL bulletins).")
    print("Where banned, use this only for non-price retention actions; elsewhere, file rating rules, keep")
    print("discounts within approved bands, and test for disparate impact before deployment.")


if __name__ == "__main__":
    main()

# ---
# Policy Gradient (REINFORCE) — HARD lab · Insurance: Renewal retention offers: which discount for which policyholder
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
