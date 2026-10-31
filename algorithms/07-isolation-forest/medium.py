import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

np.random.seed(42)
n_normal, n_bad = 4900, 100

# 🐍 PY: helper that builds one block of transactions so we can reuse it for normal and suspicious rows
def make_txns(n, amt_mu, amt_sd, night_p, cash_p, xborder_p, velocity_lam, label):
    return pd.DataFrame({
        "txn_amount": np.abs(np.random.lognormal(amt_mu, amt_sd, n)).round(2),
        "is_night": np.random.binomial(1, night_p, n),
        "cash_deposit": np.random.binomial(1, cash_p, n),
        "cross_border": np.random.binomial(1, xborder_p, n),
        "txns_last_24h": np.random.poisson(velocity_lam, n),
        "acct_age_days": np.random.randint(30, 4000, n) if label == 0 else np.random.randint(5, 900, n),
        "is_suspicious": label,
    })

# 🧠 ML: suspicious = larger amounts (thousands), more cash, more cross-border, high velocity, new accounts
normal = make_txns(n_normal, 6.0, 1.0, 0.05, 0.10, 0.03, 2, 0)
bad = make_txns(n_bad, 8.6, 0.35, 0.25, 0.45, 0.25, 5, 1)
# 🐍 PY: pd.concat stacks DataFrames; sample(frac=1) shuffles all rows
df = pd.concat([normal, bad]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Rows: {len(df)}  suspicious rate: {df.is_suspicious.mean():.1%}")

# 🧠 ML: log of amount tames the long tail so one $500k wire does not dominate the cuts
df["log_amount"] = np.log1p(df.txn_amount)
feats = ["log_amount", "is_night", "cash_deposit", "cross_border", "txns_last_24h", "acct_age_days"]
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df.is_suspicious)

# 🧠 ML: unsupervised - we fit on features only; labels are kept aside just to grade the result
model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
model.fit(train[feats])

# 🧠 ML: negate score_samples so that a HIGHER number = MORE suspicious (easier for analysts)
test = test.copy()
test["risk_score"] = -model.score_samples(test[feats])
# 🐍 PY: (predict == -1) gives True/False; astype(int) turns it into 1/0
test["alert"] = (model.predict(test[feats]) == -1).astype(int)

auc = roc_auc_score(test.is_suspicious, test.risk_score)
p = precision_score(test.is_suspicious, test.alert)
r = recall_score(test.is_suspicious, test.alert)
print(f"ROC-AUC: {auc:.3f}   alert precision: {p:.2f}   alert recall: {r:.2f}")
print(f"Alerts raised: {test.alert.sum()} of {len(test)} test transactions")

# 🧠 ML: analysts work a ranked queue, so precision in the top-K matters more than overall accuracy
for k in [10, 30, 50]:
    # 🐍 PY: nlargest(k, col) returns the k rows with the highest risk score
    topk = test.nlargest(k, "risk_score")
    print(f"  top-{k:<3d} queue: {topk.is_suspicious.sum():2d} truly suspicious ({topk.is_suspicious.mean():.0%})")

print("\nTop 5 alerts for the AML analyst:")
cols = ["txn_amount", "cash_deposit", "cross_border", "txns_last_24h", "acct_age_days", "risk_score"]
print(test.nlargest(5, "risk_score")[cols].round(3).to_string(index=False))

# 🐍 PY: groupby + mean compares the average profile of alerted vs non-alerted transactions
print("\nAverage profile by alert flag:")
print(test.groupby("alert")[["txn_amount", "cash_deposit", "txns_last_24h"]].mean().round(2))

# ---
# Isolation Forest — MEDIUM lab · Financial Services: AML suspicious transaction detection for the analyst alert queue
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
