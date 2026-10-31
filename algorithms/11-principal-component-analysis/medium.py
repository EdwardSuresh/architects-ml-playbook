import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

np.random.seed(42)

# --- 1. Simulate 1,500 trading days of daily yield changes (bp) for 8 maturities ---
# 🐍 PY: a dict of maturity label -> years, used to build smooth factor shapes
tenors = {"3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "20Y": 20, "30Y": 30}
years = np.array(list(tenors.values()))
n_days = 1500

# 🧠 ML: real curves move as a few hidden drivers - we bake in level, slope and curvature shapes
level_shape = np.ones(len(years))
slope_shape = np.log(years) / np.log(30)
curve_shape = np.exp(-((np.log(years) - np.log(3)) ** 2))
# 🐍 PY: np.random.normal(mean, std, size) draws daily shocks for each hidden driver
level = np.random.normal(0, 6.0, n_days)
slope = np.random.normal(0, 2.5, n_days)
curve = np.random.normal(0, 1.2, n_days)
noise = np.random.normal(0, 0.6, (n_days, len(years)))
# 🐍 PY: np.outer builds a (days x maturities) table: each day's shock times the factor shape
moves = (np.outer(level, level_shape) + np.outer(slope, slope_shape)
         + np.outer(curve, curve_shape) + noise)

# 🐍 PY: a DataFrame gives named columns like chg_10Y_bp instead of raw column numbers
cols = [f"chg_{t}_bp" for t in tenors]
df = pd.DataFrame(moves, columns=cols)
df.insert(0, "trade_date", pd.bdate_range("2019-01-02", periods=n_days))
print("Sample rows:")
print(df.head(3).to_string(index=False, float_format="%.2f"))

# --- 2. Split: fit factors on earlier days, check on later days (no shuffling for time data) ---
# 🧠 ML: shuffle=False keeps time order so we test on 'future' days the model never saw
train, test = train_test_split(df[cols], test_size=0.25, shuffle=False)
print(f"\nTrain days: {len(train)}, Test days: {len(test)}")

# --- 3. Fit PCA ---
# 🧠 ML: no scaling here on purpose - all columns are in the same unit (bp), so size matters
pca = PCA(n_components=3)
pca.fit(train)
print("\nExplained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
print(f"Top 3 factors together explain {pca.explained_variance_ratio_.sum():.2%} of daily moves")

# 🧠 ML: loadings (components_) show how each maturity reacts to one unit of each factor
loadings = pd.DataFrame(pca.components_.T, index=list(tenors), columns=["PC1", "PC2", "PC3"])
# 🐍 PY: multiply each column by the sign of its 30Y value so the shapes print the same way every run
loadings = loadings * np.sign(loadings.loc["30Y"])
print("\nLoadings (rows = maturities):")
print(loadings.round(3).to_string())

# --- 4. How well do 3 numbers rebuild the full 8-point curve on unseen days? ---
# 🧠 ML: transform -> 3 factor scores; inverse_transform -> back to 8 maturities
rebuilt = pca.inverse_transform(pca.transform(test))
err = test.values - rebuilt
rmse_bp = np.sqrt((err ** 2).mean())
print(f"\nTest reconstruction RMSE: {rmse_bp:.2f} bp (typical daily move ~{test.values.std():.1f} bp)")

# --- 5. Name the factors with simple rules ---
pc1, pc2, pc3 = loadings["PC1"], loadings["PC2"], loadings["PC3"]
# 🐍 PY: boolean expressions like (pc1 > 0).all() return True/False for every-element checks
print("\nFactor interpretation:")
print(f"  PC1 all same sign?       {(pc1 > 0).all()}  -> LEVEL (parallel shift)")
print(f"  PC2 short vs long sign:  {pc2['3M']:+.2f} vs {pc2['30Y']:+.2f}  -> SLOPE (steepen/flatten)")
print(f"  PC3 belly vs wings:      {pc3['2Y']:+.2f} vs {pc3['3M']:+.2f}/{pc3['30Y']:+.2f}  -> CURVATURE (butterfly)")

# --- 6. Business view ---
print("\nBusiness read-out for the rates desk:")
print(f"  Hedging the LEVEL factor alone covers {pca.explained_variance_ratio_[0]:.0%} of daily risk.")
print(f"  Adding SLOPE and CURVE hedges brings that to {pca.explained_variance_ratio_.sum():.0%}.")
print("  So 3 hedge trades can replace hedging 8 maturities separately.")

# ---
# Principal Component Analysis (PCA) — MEDIUM lab · Financial Services: Treasury yield curve risk factors for rate hedging
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
