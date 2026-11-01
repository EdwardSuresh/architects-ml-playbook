import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
SEED = 42
FEATS = ["log_amount", "amt_vs_cust_avg", "is_night", "cash_deposit", "cross_border",
         "txns_last_3d", "near_10k", "acct_age_days"]


def simulate(n_cust=600, days=60, bad_cust=12, seed=SEED):
    """Daily transactions per customer; a few customers start laundering in the last 20 days."""
    rng = np.random.default_rng(seed)
    rows = []
    # 🐍 PY: a set gives fast 'c in bad_ids' membership checks inside the loop
    bad_ids = set(rng.choice(n_cust, bad_cust, replace=False))
    for c in range(n_cust):
        base = rng.lognormal(6, 0.8)
        age = rng.integers(30, 4000)
        for d in range(days):
            # 🧠 ML: laundering customers transact more often in their final 20 days (velocity)
            active_p = 0.5 if (c in bad_ids and d >= days - 20) else 0.35
            if rng.random() > active_p:
                continue
            # 🧠 ML: labels exist only to GRADE the model; Isolation Forest never sees them
            launder = c in bad_ids and d >= days - 20 and rng.random() < 0.6
            amt = rng.uniform(2500, 9990) if launder else base * rng.lognormal(0, 0.5)
            rows.append(dict(cust_id=c, day=d, txn_amount=round(amt, 2), acct_age_days=age + d,
                             is_night=int(rng.random() < (0.25 if launder else 0.08)),
                             cash_deposit=int(rng.random() < (0.5 if launder else 0.15)),
                             cross_border=int(rng.random() < (0.2 if launder else 0.05)),
                             is_suspicious=int(launder)))
    return pd.DataFrame(rows)


def add_features(df):
    df = df.sort_values(["cust_id", "day"]).copy()
    # 🧠 ML: log1p compresses huge amounts so random cuts are not wasted on the long tail
    df["log_amount"] = np.log1p(df.txn_amount)
    # 🧠 ML: behaviour vs the customer's OWN history catches anomalies a global rule would miss
    # 🐍 PY: shift() moves values down one row so today's amount is excluded from its own history
    hist = df.groupby("cust_id").txn_amount.transform(lambda s: s.shift().expanding().median())
    df["amt_vs_cust_avg"] = (df.txn_amount / hist.fillna(df.txn_amount)).clip(0, 50)
    # 🧠 ML: velocity feature = how many txns this customer made in the last few days
    # 🐍 PY: searchsorted on the sorted day array counts rows with day in [d-2, d] for each row
    df["txns_last_3d"] = df.groupby("cust_id").day.transform(
        lambda s: np.arange(len(s)) - np.searchsorted(s.values, s.values - 2) + 1)
    # 🧠 ML: structuring red flag - amounts just below the $10k CTR reporting threshold
    df["near_10k"] = df.txn_amount.between(8000, 9999).astype(int)
    return df


# 🐍 PY: np.argsort(-score) sorts highest-risk first; [:k] keeps the analyst's daily capacity
def queue_metrics(y, score, k):
    order = np.argsort(-score)[:k]
    return y[order].sum(), y.sum()


df = add_features(simulate())
# 🧠 ML: time-ordered split - fit on days 0-39 (mostly clean history), score the newest 20 days
train, test = df[df.day < 40], df[df.day >= 40]
print(f"Train txns={len(train)}  Test txns={len(test)}  test suspicious={test.is_suspicious.sum()}")

# 🧠 ML: baseline = classic AML rule 'cash deposit over $8k' used by many legacy rule engines
rule = ((test.cash_deposit == 1) & (test.txn_amount > 8000)).astype(int).values
y = test.is_suspicious.values

# 🧠 ML: tune n_estimators / max_samples / max_features; score with PR-AUC because positives are rare
best, best_ap = None, -1
for n_est in [100, 300]:
    for max_s in [256, 1024]:
        for max_f in [0.6, 1.0]:
            # 🧠 ML: max_features < 1 lets each tree see a random subset of features (more diversity)
            m = Pipeline([("sel", FunctionTransformer(lambda d: d[FEATS].values)),
                          ("iso", IsolationForest(n_estimators=n_est, max_samples=max_s,
                                                  max_features=max_f, random_state=SEED))])
            m.fit(train)
            ap = average_precision_score(y, -m.score_samples(test))
            if ap > best_ap:
                best, best_ap, best_p = m, ap, (n_est, max_s, max_f)
print(f"Best (n_estimators, max_samples, max_features) = {best_p}  PR-AUC={best_ap:.3f}")
# Note: in production you tune on a separate validation window, not the final test window.

# 🧠 ML: negate so a higher number means more suspicious; ROC-AUC checks overall ranking quality
score = -best.score_samples(test)
print(f"ROC-AUC={roc_auc_score(y, score):.3f}")

# 🧠 ML: business threshold = analyst capacity; 25 analyst-hours/day at ~1 alert/hour
for k in [25, 50, 100]:
    hit, tot = queue_metrics(y, score, k)
    print(f"  top-{k:<3d} alerts -> {hit:3d}/{tot} suspicious caught  precision={hit / k:.0%}")
# 🐍 PY: & on two 0/1 arrays is element-wise AND - counts alerts that were truly suspicious
rule_hits = (rule & y).sum()
print(f"  rule baseline   -> {rule_hits:3d}/{y.sum()} caught using {rule.sum()} alerts"
      f"  precision={rule_hits / max(rule.sum(), 1):.0%}")

# 🧠 ML: explain one alert - how much the score drops when each feature is reset to the median
# 🧠 ML: reset one feature at a time to the 'typical' median; the bigger the score drop, the bigger the reason
top_idx = int(np.argmax(score))
row = test.iloc[[top_idx]].copy()
med = train[FEATS].median()
contrib = {}
for f in FEATS:
    r2 = row.copy()
    r2[f] = med[f]
    contrib[f] = score[top_idx] - (-best.score_samples(r2)[0])
print(f"\nWhy is cust {row.cust_id.iat[0]} day {row.day.iat[0]} top of queue? (score={score[top_idx]:.3f})")
for f, v in sorted(contrib.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {f:16s} value={row[f].iat[0]:>9.2f}  score lift={v:+.3f}")

# 🧠 ML: cost view - missed SAR is far costlier than an extra analyst review
cost_review, cost_miss = 40, 5000
hit, tot = queue_metrics(y, score, 100)
print(f"\nCost @top-100: model=${100 * cost_review + (tot - hit) * cost_miss:,}"
      f"  rule=${rule.sum() * cost_review + (tot - rule_hits) * cost_miss:,}")

# Model-risk: BSA/AML models fall under SR 11-7 validation; FinCEN expects documented tuning,
# below-the-line testing, and analyst feedback. Scores route to humans; a person decides on SAR filing.
print("Governance: SR 11-7 validation + BSA/AML tuning docs; analyst, not model, files the SAR.")
# 🧠 ML: persist the fitted forest so nightly scoring uses the exact validated model
path = os.path.join(tempfile.mkdtemp(), "aml_iforest.joblib")
joblib.dump(best.named_steps["iso"], path)
print("Saved ->", os.path.basename(path))

# ---
# Isolation Forest — PRODUCTION lab · Financial Services: AML suspicious transaction detection for the analyst alert queue
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
