import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
n = 4000

# --- 1. Synthetic life-insurance policy book ---
# 🐍 PY: a dict of column -> numpy array becomes a DataFrame in one step
df = pd.DataFrame({
    "policy_age_years": rng.integers(0, 25, n),
    "insured_age": rng.integers(22, 75, n),
    "annual_premium_usd": rng.lognormal(7.3, 0.5, n).round(0),
    "premium_increase_pct": rng.choice([0, 0, 0, 5, 10, 20], n),
    "missed_payments_12m": rng.poisson(0.5, n),
    "on_autopay": rng.integers(0, 2, n),
    "agent_contacts_12m": rng.poisson(1.5, n),
    "cash_value_usd": rng.lognormal(8, 1.0, n).round(0),
})

# 🧠 ML: the true lapse rule is NON-linear (interactions, a U-shape in policy age) - good for a neural net
z = (-2.2 + 0.9 * df.missed_payments_12m + 0.06 * df.premium_increase_pct
     - 1.1 * df.on_autopay - 0.25 * df.agent_contacts_12m
     + 0.8 * ((df.policy_age_years < 3) | (df.policy_age_years > 20))
     + 0.04 * df.premium_increase_pct * (1 - df.on_autopay)
     - 0.3 * np.log(df.cash_value_usd / 3000))
# 🐍 PY: comparing a random uniform number to a probability draws a 0/1 outcome per row
df["lapsed"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
print(df.head(3).to_string(index=False))
print(f"\nPolicies: {n}, lapse rate: {df.lapsed.mean():.1%}")

# --- 2. Train/test split ---
X = df.drop(columns="lapsed")
y = df["lapsed"]
# 🧠 ML: stratify=y keeps the same lapse rate in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# --- 3. Neural network with scaling ---
# 🧠 ML: neural nets need scaled inputs - premiums in thousands would swamp 0/1 autopay flags
# 🧠 ML: hidden_layer_sizes=(16, 8) = two hidden layers with 16 and 8 neurons
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", alpha=1e-3,
                  max_iter=500, early_stopping=True, n_iter_no_change=25,
                  random_state=42),
)
model.fit(X_train, y_train)
# 🐍 PY: [-1] grabs the last step of the pipeline; n_iter_ = epochs actually run
print(f"Training stopped after {model[-1].n_iter_} epochs")

# --- 4. Evaluate ---
# 🧠 ML: predict_proba gives P(lapse); column 1 is the positive class
proba = model.predict_proba(X_test)[:, 1]
print(f"\nTest ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(classification_report(y_test, proba >= 0.5, target_names=["in_force", "lapsed"], digits=3))

# --- 5. Business interpretation: who should the retention team call? ---
# 🐍 PY: assign adds a new column without modifying X_test in place
scored = X_test.assign(p_lapse=proba, lapsed=y_test.values)
top = scored.sort_values("p_lapse", ascending=False).head(len(scored) // 10)
print(f"Top 10% riskiest policies: actual lapse rate {top.lapsed.mean():.1%} "
      f"vs book average {scored.lapsed.mean():.1%}")
print(f"Calling them reaches {top.lapsed.sum() / scored.lapsed.sum():.0%} of all lapses "
      f"with {len(top)} calls.")
# 🐍 PY: groupby + mean shows the average predicted risk per autopay status
print("\nAverage predicted lapse risk by autopay:")
print(scored.groupby("on_autopay")["p_lapse"].mean().round(3).to_string())

# ---
# Artificial Neural Network (ANN / MLP) — MEDIUM lab · Insurance: Life insurance policy lapse prediction for retention outreach
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
