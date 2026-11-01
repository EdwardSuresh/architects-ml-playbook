"""PCA yield-curve risk factors: choose k, interpret, PCA-VaR, factor hedge, stability check."""
import os
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)
TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
YEARS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


def simulate_curve_moves(n_days=2000):
    """Daily yield changes (bp) driven by level/slope/curve plus a calmer and a stressed regime."""
    shapes = np.vstack([np.ones(len(YEARS)),
                        np.log(YEARS) / np.log(30),
                        np.exp(-((np.log(YEARS) - np.log(3)) ** 2))])
    # 🧠 ML: volatility regimes make factor stability worth checking - real markets do this too
    vol_mult = np.where(np.arange(n_days) > 1400, 1.8, 1.0)[:, None]
    shocks = np.random.normal(0, [6.0, 2.5, 1.2], (n_days, 3)) * vol_mult
    noise = np.random.normal(0, 0.6, (n_days, len(YEARS)))
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    return pd.DataFrame(shocks @ shapes + noise, index=dates, columns=TENORS)


def choose_k(train, test, target=0.99, max_k=6):
    """Pick the smallest k reaching target explained variance; show holdout reconstruction error."""
    rows = []
    for k in range(1, max_k + 1):
        # 🧠 ML: centre only (no std scaling) - all tenors share the bp unit, so big movers should dominate
        pipe = Pipeline([("center", StandardScaler(with_std=False)), ("pca", PCA(n_components=k))])
        pipe.fit(train)
        rebuilt = pipe.inverse_transform(pipe.transform(test))
        rmse = np.sqrt(((test.values - rebuilt) ** 2).mean())
        cum = pipe.named_steps["pca"].explained_variance_ratio_.sum()
        rows.append({"k": k, "cum_var": cum, "test_rmse_bp": rmse})
    table = pd.DataFrame(rows)
    k_star = int(table.loc[table.cum_var >= target, "k"].min())
    return table, k_star


def oriented_loadings(pca):
    """Return loadings with a fixed sign convention: positive at 30Y."""
    L = pd.DataFrame(pca.components_.T, index=TENORS,
                     columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
    # 🧠 ML: eigenvector signs are arbitrary, so fix them before comparing windows or reporting
    return L * np.sign(L.loc["30Y"])


def pca_var(dv01, L, factor_var, z=2.326):
    """99% one-day parametric VaR using only the PCA factors. P&L = -DV01 . dy."""
    # 🧠 ML: exposure to each factor = sum over tenors of DV01 x loading
    expo = L.T.values @ dv01
    pnl_sd = np.sqrt(np.sum(expo ** 2 * factor_var))
    return z * pnl_sd, expo


def stability(df, window=250, step=250):
    """Cosine similarity of rolling-window loadings vs full-sample loadings."""
    full = oriented_loadings(PCA(3).fit(df - df.mean()))
    out = []
    for start in range(0, len(df) - window + 1, step):
        win = df.iloc[start:start + window]
        L = oriented_loadings(PCA(3).fit(win - win.mean()))
        # 🧠 ML: |cosine| near 1 means the factor shape is the same as the long-run shape
        cos = [abs(np.dot(L[c], full[c])) for c in full.columns]
        out.append({"window_start": win.index[0].date(), "PC1": cos[0], "PC2": cos[1],
                    "PC3": cos[2], "vol_bp": win.values.std()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = simulate_curve_moves()
    # 🧠 ML: time-ordered split: fit on the past, evaluate on the most recent 400 days
    train, test = df.iloc[:-400], df.iloc[-400:]

    table, k = choose_k(train, test)
    print("Choosing number of factors (target 99% variance):")
    print(table.round(4).to_string(index=False))
    print(f"-> chosen k = {k}\n")

    pipe = Pipeline([("center", StandardScaler(with_std=False)), ("pca", PCA(n_components=k))])
    pipe.fit(train)
    pca = pipe.named_steps["pca"]
    L = oriented_loadings(pca)
    print("Loadings:")
    print(L.round(2).T.to_string())
    print("Interpretation: PC1=level (all +), PC2=slope (short - / long +), PC3=curvature (belly vs wings)\n")

    # Bond portfolio: key-rate DV01 in USD per 1bp RISE (long bonds lose money when yields rise)
    # 🧠 ML: DV01 is the 'feature weight' of each tenor in the P&L - risk = DV01 times rate move
    dv01 = pd.Series(0.0, index=TENORS)
    dv01[["2Y", "5Y", "10Y", "30Y"]] = [12_000, 25_000, 40_000, 18_000]
    factor_var = pca.explained_variance_
    var_before, expo = pca_var(dv01.values, L, factor_var)
    print("Portfolio factor exposures ($ per 1-unit factor move):", np.round(expo, 0))
    print(f"PCA 99% 1-day VaR before hedge: ${var_before:,.0f}")

    # 🧠 ML: factor hedge - pick hedge sizes so the portfolio has ZERO exposure to every factor
    hedge_tenors = ["2Y", "10Y", "30Y"][:k]
    H = L.loc[hedge_tenors].values.T  # (factors x hedge instruments), each hedge = 1 USD DV01
    hedge_dv01 = np.linalg.solve(H, -expo)
    hedged = dv01.copy()
    for t, h in zip(hedge_tenors, hedge_dv01):
        hedged[t] += h
    var_after, _ = pca_var(hedged.values, L, factor_var)
    print("Hedge DV01 by tenor (negative = sell futures/pay swaps):",
          {t: round(h) for t, h in zip(hedge_tenors, hedge_dv01)})
    print(f"PCA 99% 1-day VaR after hedge:  ${var_after:,.0f}")

    # 🧠 ML: backtest on unseen days using FULL tenor moves, not the PCA approximation
    pnl_before = -(test.values @ dv01.values)
    pnl_after = -(test.values @ hedged.values)
    print(f"\nOut-of-sample daily P&L std: before ${pnl_before.std():,.0f}  after ${pnl_after.std():,.0f}")
    breaches = (pnl_before < -var_before).mean()
    print(f"VaR breach rate (unhedged) on test: {breaches:.2%} (expect ~1%; higher = regime shift)")

    stab = stability(df)
    print("\nRolling-window stability (|cosine| vs full-sample loadings):")
    print(stab.round(3).to_string(index=False))
    weak = stab[stab[["PC1", "PC2", "PC3"]].min(axis=1) < 0.95]
    print(f"Windows with a factor shape below 0.95 similarity: {len(weak)} of {len(stab)}")

    # 🧠 ML: save the fitted pipeline so the risk engine applies the SAME factors every morning
    path = os.path.join(tempfile.mkdtemp(), "yield_pca.joblib")
    joblib.dump(pipe, path)
    print(f"\nSaved pipeline -> {os.path.basename(path)}")
    print("Model-risk note (SR 11-7): document factor choice, re-estimate window, VaR backtest")
    print("breaches and stability results; recalibrate when volatility regime changes.")

# ---
# Principal Component Analysis (PCA) — PRODUCTION lab · Financial Services: Treasury yield curve risk factors for rate hedging
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
