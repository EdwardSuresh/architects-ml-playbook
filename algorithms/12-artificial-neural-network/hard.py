"""Life-insurance lapse model: PyTorch MLP vs logistic regression, class weights, early stopping,
permutation importance, calibration and an outreach-budget threshold."""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)
FEATURES = ["policy_age_years", "insured_age", "annual_premium_usd", "premium_increase_pct",
            "missed_payments_12m", "on_autopay", "agent_contacts_12m", "cash_value_usd"]


def make_policies(n=5000):
    df = pd.DataFrame({
        "policy_age_years": rng.integers(0, 25, n), "insured_age": rng.integers(22, 75, n),
        "annual_premium_usd": rng.lognormal(7.3, 0.5, n), "premium_increase_pct": rng.choice([0, 0, 0, 5, 10, 20], n),
        "missed_payments_12m": rng.poisson(0.5, n), "on_autopay": rng.integers(0, 2, n),
        "agent_contacts_12m": rng.poisson(1.5, n), "cash_value_usd": rng.lognormal(8, 1.0, n)})
    # 🧠 ML: interactions and a U-shape - patterns a straight-line model cannot fully capture
    z = (-2.8 + 0.9 * df.missed_payments_12m + 0.05 * df.premium_increase_pct - 1.1 * df.on_autopay
         - 0.25 * df.agent_contacts_12m + 1.0 * ((df.policy_age_years < 3) | (df.policy_age_years > 20))
         + 0.06 * df.premium_increase_pct * (1 - df.on_autopay) - 0.3 * np.log(df.cash_value_usd / 3000))
    df["lapsed"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


class LapseMLP(nn.Module):
    """Two hidden layers with ReLU and dropout; outputs one logit per policy."""
    def __init__(self, n_in, h1=32, h2=16, p_drop=0.1):
        super().__init__()
        # 🧠 ML: Dropout randomly switches off neurons in training, a cheap guard against overfitting
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(p_drop),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xval, yval, epochs=200, patience=15, lr=3e-3):
    model = LapseMLP(Xtr.shape[1])
    # 🧠 ML: pos_weight up-weights the rarer lapsed class so the net does not just predict 'stays'
    pos_weight = torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.tensor(Xval, dtype=torch.float32), torch.tensor(yval, dtype=torch.float32)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        # 🧠 ML: mini-batches of 128 - each step sees a random slice, which speeds and regularises training
        for idx in torch.randperm(len(Xt)).split(128):
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        # 🧠 ML: early stopping - keep the weights from the epoch with the best validation loss
        if val_loss < best - 1e-4:
            best, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    print(f"MLP early-stopped at epoch {epoch + 1}, best val loss {best:.4f}")
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def permutation_importance(model, X, y, n_repeats=5):
    """Drop in ROC-AUC when one column is shuffled = how much the model relies on it."""
    base = roc_auc_score(y, predict(model, X))
    out = {}
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, predict(model, Xp)))
        out[name] = np.mean(drops)
    return pd.Series(out).sort_values(ascending=False)


def calibration_table(y, p, bins=5):
    # 🧠 ML: calibration = do policies scored 30% actually lapse about 30% of the time?
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": p, "actual": y, "bin": q}).groupby("bin").mean().round(3)


if __name__ == "__main__":
    df = make_policies()
    df["annual_premium_usd"] = np.log(df.annual_premium_usd)
    df["cash_value_usd"] = np.log(df.cash_value_usd)
    X, y = df[FEATURES].values, df.lapsed.values
    print(f"Policies {len(df)}, lapse rate {y.mean():.1%}")
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(Xtmp, ytmp, test_size=0.2, stratify=ytmp, random_state=42)
    # 🧠 ML: fit the scaler on TRAIN only, then apply to val/test - avoids leaking test statistics
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xval, Xte = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

    mlp = train_mlp(Xtr, ytr, Xval, yval)
    p_mlp = predict(mlp, Xte)
    # 🧠 ML: baseline - a well-understood logistic regression with the same class weighting
    logit = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, ytr)
    p_lr = logit.predict_proba(Xte)[:, 1]

    print("\nModel        ROC-AUC  PR-AUC  Brier")
    for name, p in [("Logistic", p_lr), ("MLP", p_mlp)]:
        print(f"{name:<12} {roc_auc_score(yte, p):.3f}    {average_precision_score(yte, p):.3f}   "
              f"{brier_score_loss(yte, p):.3f}")

    print("\nPermutation importance (ROC-AUC drop) for the MLP:")
    for feat, val in permutation_importance(mlp, Xte, yte).items():
        print(f"  {feat:<22} {val:+.4f} {'#' * int(max(val, 0) * 200)}")

    print("\nCalibration of MLP (class weights inflate probabilities):")
    print(calibration_table(yte, p_mlp).to_string())
    # 🧠 ML: simple fix - rescale odds by the class weight so scores become real lapse probabilities
    w = (ytr == 0).sum() / (ytr == 1).sum()
    odds = p_mlp / (1 - p_mlp) / w
    p_cal = odds / (1 + odds)
    print(f"Mean predicted after odds correction {p_cal.mean():.3f} vs actual {yte.mean():.3f}; "
          f"Brier {brier_score_loss(yte, p_cal):.3f}")

    # 🧠 ML: business threshold - retention team can call 15% of the book; value saved per lapse
    budget = int(0.15 * len(yte))
    call_cost, saved_value, save_rate = 40, 1800, 0.25
    top = np.argsort(-p_cal)[:budget]
    caught = yte[top].sum()
    net = caught * save_rate * saved_value - budget * call_cost
    print(f"\nCall top {budget} policies: reach {caught}/{yte.sum()} lapses "
          f"({caught / yte.sum():.0%}); expected net value ${net:,.0f}")

    path = os.path.join(tempfile.mkdtemp(), "lapse_mlp.pt")
    torch.save({"state_dict": mlp.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, path)
    print(f"Saved model -> {os.path.basename(path)}")
    print("Governance: NAIC AI Model Bulletin - document data, test for unfair discrimination,")
    print("keep the logistic baseline as a challenger, and monitor calibration quarterly.")

# ---
# Artificial Neural Network (ANN / MLP) — HARD lab · Insurance: Life insurance policy lapse prediction for retention outreach
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
