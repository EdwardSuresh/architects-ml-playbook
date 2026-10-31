# Hard: Hierarchical Risk Parity (HRP) - correlation clustering, linkage comparison, out-of-sample risk
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, cophenet
from scipy.spatial.distance import squareform

rng = np.random.default_rng(42)
SECTORS = {"Banks": 0.012, "Tech": 0.017, "Energy": 0.020, "Health": 0.010, "Utilities": 0.006}
N_PER, N_DAYS = 5, 1000


def simulate_returns():
    """Sector-factor model with a mid-sample volatility regime change (stress period)."""
    market = rng.normal(0.0003, 0.009, N_DAYS)
    # 🐍 PY: np.r_ concatenates arrays - first half ones, second half 1.6
    stress = np.r_[np.ones(N_DAYS // 2), np.full(N_DAYS - N_DAYS // 2, 1.6)]
    cols = {}
    for sector, vol in SECTORS.items():
        factor = rng.normal(0, vol, N_DAYS)
        for i in range(1, N_PER + 1):
            # 🧠 ML: each stock has its own idiosyncratic risk level - unequal vols matter for allocation
            idio = rng.uniform(0.004, 0.015)
            cols[f"{sector[:3].upper()}{i}"] = stress * (0.7 * market + factor + rng.normal(0, idio, N_DAYS))
    df = pd.DataFrame(cols)
    # 🐍 PY: shuffle columns so clustering cannot rely on the original sector order
    return df[rng.permutation(df.columns)]


def corr_distance(returns):
    # 🧠 ML: d = sqrt(0.5 * (1 - rho)) turns correlation into a true distance metric
    return np.sqrt(0.5 * (1 - returns.corr()))


def inverse_variance_weights(cov):
    # 🧠 ML: inverse-variance = give each asset weight proportional to 1 / its variance
    iv = 1 / np.diag(cov)
    return iv / iv.sum()


def cluster_variance(cov, idx):
    # 🐍 PY: np.ix_ selects the sub-matrix for the given rows AND columns
    sub = cov[np.ix_(idx, idx)]
    w = inverse_variance_weights(sub)
    return float(w @ sub @ w)


def hrp_weights(returns, method="single"):
    """Lopez de Prado-style HRP: cluster, quasi-diagonalise, recursive bisection."""
    cov = returns.cov().values
    Z = linkage(squareform(corr_distance(returns).values, checks=False), method=method)
    # 🧠 ML: quasi-diagonalisation - order assets so correlated ones sit next to each other
    order = list(leaves_list(Z))
    # 🐍 PY: start every asset at weight 1.0, then multiply down as clusters are split
    w = pd.Series(1.0, index=order)
    # 🐍 PY: a list of lists acts as a work-queue of clusters still to split
    queue = [order]
    while queue:
        queue = [c[s:e] for c in queue for s, e in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        # 🐍 PY: step through the queue two at a time - each pair is the left/right half of one split
        for i in range(0, len(queue), 2):
            left, right = queue[i], queue[i + 1]
            v_l, v_r = cluster_variance(cov, left), cluster_variance(cov, right)
            # 🧠 ML: give MORE money to the LESS risky half (risk-parity split)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return pd.Series(w.sort_index().values, index=returns.columns[w.sort_index().index])


def portfolio_stats(weights, returns):
    port = returns.values @ weights
    return port.std() * np.sqrt(252), (port.mean() * 252) / (port.std() * np.sqrt(252))


rets = simulate_returns()
# 🧠 ML: time split - estimate on the first half (calm), test on the second half (stress)
train, test = rets.iloc[: N_DAYS // 2], rets.iloc[N_DAYS // 2:]
print(f"{rets.shape[1]} stocks, train {len(train)} days, test {len(test)} days (higher-vol regime)\n")

# 1. Linkage comparison via cophenetic correlation and sector purity
# 🧠 ML: cophenetic corr = how faithfully the tree's merge heights preserve the original distances
cond = squareform(corr_distance(train).values, checks=False)
true_sector = [c[:3] for c in train.columns]
print("Linkage  | cophenetic corr | sector purity at 5 clusters")
for method in ["single", "average", "complete", "ward"]:
    Z = linkage(cond, method=method)
    coph, _ = cophenet(Z, cond)
    labels = fcluster(Z, 5, criterion="maxclust")
    # 🐍 PY: for each cluster, the share taken by its most common true sector, averaged by size
    purity = pd.crosstab(labels, true_sector).max(axis=1).sum() / len(labels)
    print(f"{method:<8} |      {coph:.3f}      |   {purity:.2f}")

# 2. Show the ordered clusters (quasi-diagonal order)
Z = linkage(cond, method="ward")
order = train.columns[leaves_list(Z)]
print("\nWard leaf order:", " ".join(order))

# 3. Build portfolios on TRAIN
# 🧠 ML: every portfolio is built ONLY from training data, then judged on the unseen test period
cov = train.cov().values
portfolios = {
    "Equal weight": np.full(train.shape[1], 1 / train.shape[1]),
    "Inverse variance": inverse_variance_weights(cov),
    "HRP (single)": hrp_weights(train, "single").values,
    "HRP (ward)": hrp_weights(train, "ward").values,
}
# Minimum-variance via inverse covariance - classic optimiser that HRP aims to stabilise
# 🧠 ML: inverting a noisy covariance matrix amplifies errors - the weakness HRP is designed to avoid
inv = np.linalg.inv(cov)
mv = inv @ np.ones(len(cov)) / (np.ones(len(cov)) @ inv @ np.ones(len(cov)))
portfolios["Min-variance (unconstrained)"] = mv

# 4. Evaluate in-sample and out-of-sample
print("\nPortfolio                    | train vol | test vol | test Sharpe | max weight | min weight")
for name, w in portfolios.items():
    # 🐍 PY: underscore _ ignores a return value we do not need
    tv, _ = portfolio_stats(w, train)
    ov, sr = portfolio_stats(w, test)
    print(f"{name:<28} |  {tv:6.2%}  | {ov:6.2%}  |   {sr:5.2f}     |  {w.max():6.2%}   | {w.min():7.2%}")

# 5. Explainability: weight by sector cluster
hrp = pd.Series(portfolios["HRP (ward)"], index=train.columns)
# 🐍 PY: groupby a function of the index (first 3 letters) aggregates weights per sector
print("\nHRP (ward) weight by sector:")
for sec, wt in hrp.groupby(lambda t: t[:3]).sum().sort_values(ascending=False).items():
    print(f"  {sec:<4} {wt:6.2%} " + "#" * int(wt * 100))

# 6. Stability: refit each method on the TEST window and measure how much the weights would move
# 🧠 ML: turnover = sum of |weight change|; unstable weights mean high trading cost and model risk
cov_te = test.cov().values
inv_te = np.linalg.inv(cov_te)
refit = {"HRP (ward)": hrp_weights(test, "ward").values,
         "Min-variance (unconstrained)": inv_te @ np.ones(len(cov_te)) / np.sum(inv_te)}
for name, w_new in refit.items():
    print(f"Rebalance turnover {name:<28}: {np.abs(w_new - portfolios[name]).sum():6.1%}")
same = np.allclose(portfolios["HRP (single)"], portfolios["HRP (ward)"])
print(f"HRP single vs ward identical? {same} (clean sector blocks give the same leaf order)")

# 7. Concentration check - effective number of bets
for name in ["Equal weight", "HRP (ward)", "Min-variance (unconstrained)"]:
    w = portfolios[name]
    # 🧠 ML: effective N = 1 / sum(w^2); 25 means perfectly spread, low values mean concentration
    print(f"Effective N ({name}): {1 / np.sum(w ** 2):5.1f}")

path = os.path.join(tempfile.mkdtemp(), "hrp_weights.joblib")
joblib.dump({"weights": hrp, "linkage": "ward", "as_of_day": N_DAYS // 2}, path)
print(f"\nSaved HRP weights -> {os.path.basename(path)}")
print("Model-risk note (SR 11-7): document distance, linkage and lookback choices; backtest across"
      " regimes; check weight stability before every rebalance.")

# ---
# Hierarchical Clustering — HARD lab · Financial Services: Portfolio diversification with correlation clustering and HRP
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
