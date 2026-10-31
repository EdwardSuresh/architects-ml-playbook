# HARD: auto pure-premium pricing - OLS vs engineered Ridge vs Tweedie GLM (actuarial standard)
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, TweedieRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_tweedie_deviance

warnings.filterwarnings("ignore")
# 🐍 PY: constants at the top make it obvious which columns the pipeline expects
NUM = ["driver_age", "annual_miles_k", "vehicle_value_k", "prior_claims_3y", "years_licensed"]
CAT = ["territory", "vehicle_use", "credit_tier"]


def make_book(n=20000, seed=42):
    """Simulate a policy book: most policies have $0 claims, a few have large losses."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 85, n)
    df = pd.DataFrame({
        "driver_age": age,
        "years_licensed": np.clip(age - 17 - rng.integers(0, 5, n), 0, None),
        "annual_miles_k": rng.normal(12, 4, n).clip(2, 30).round(1),
        "vehicle_value_k": rng.lognormal(3.2, 0.4, n).round(1),
        "prior_claims_3y": rng.poisson(0.25, n),
        "territory": rng.choice(["urban", "suburban", "rural"], n, p=[.35, .45, .2]),
        "vehicle_use": rng.choice(["commute", "pleasure", "business"], n, p=[.6, .3, .1]),
        "credit_tier": rng.choice(["A", "B", "C"], n, p=[.5, .35, .15]),
        "exposure": rng.uniform(0.25, 1.0, n).round(2),   # fraction of a policy-year in force
    })
    # 🧠 ML: frequency is multiplicative (log-linear) with a U-shaped age curve - typical for auto
    log_freq = (np.log(0.08) + 1.2 * np.exp(-(age - 18) / 8) + 0.6 * np.clip(age - 70, 0, None) / 15
                + 0.05 * (df.annual_miles_k - 12) + 0.4 * df.prior_claims_3y
                + df.territory.map({"urban": 0.5, "suburban": 0.1, "rural": 0.0})
                + df.vehicle_use.map({"commute": 0.1, "pleasure": 0.0, "business": 0.3})
                + df.credit_tier.map({"A": 0.0, "B": 0.2, "C": 0.45}))
    # 🧠 ML: claim counts are Poisson, scaled by exposure - half a year in force = half the expected claims
    counts = rng.poisson(np.exp(log_freq) * df.exposure)
    # Severity: gamma-distributed $ per claim, larger for expensive vehicles
    sev_mean = 4200 * (df.vehicle_value_k / 25) ** 0.3
    # 🐍 PY: a list comprehension with a conditional sums k random severities only where k > 0
    loss = [rng.gamma(2.0, m / 2.0, k).sum() if k else 0.0 for k, m in zip(counts, sev_mean)]
    df["pure_premium"] = np.array(loss) / df.exposure   # $ per full policy-year
    return df


def preprocessor(splines=False):
    num_step = StandardScaler()
    if splines:
        # 🧠 ML: splines let a LINEAR model bend - key to capturing the U-shaped age curve
        num_step = ColumnTransformer([("age", SplineTransformer(n_knots=5, degree=3), ["driver_age"])],
                                     remainder=StandardScaler())
    return ColumnTransformer([("num", num_step, NUM), ("cat", OneHotEncoder(drop="first"), CAT)])


def lorenz_gini(y, pred, w):
    """Ordered Lorenz / Gini - how well the model ranks risk (actuaries' favourite lift metric)."""
    # 🧠 ML: sort policies from safest to riskiest predicted, then see how fast actual losses accumulate
    order = np.argsort(pred)
    cum_loss = np.cumsum((y * w)[order]) / np.sum(y * w)
    cum_expo = np.cumsum(w[order]) / np.sum(w)
    return 1 - 2 * np.trapezoid(cum_loss, cum_expo)


def main():
    df = make_book()
    print(f"Policies {len(df):,} | with a claim {np.mean(df.pure_premium > 0):.1%} | "
          f"avg pure premium ${np.average(df.pure_premium, weights=df.exposure):,.0f}")
    X, y, w = df[NUM + CAT], df.pure_premium.values, df.exposure.values
    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, w, test_size=0.25, random_state=42)

    # 🧠 ML: OLS predicts the mean directly; it can go negative and treats a $0 year like any other value
    models = {
        "OLS (plain linear)": Pipeline([("pre", preprocessor()), ("m", LinearRegression())]),
        "Ridge + age splines": Pipeline([("pre", preprocessor(True)), ("m", RidgeCV(alphas=[0.1, 1, 10, 100]))]),
    }
    # 🧠 ML: Tweedie GLM with log link: power 1-2 = compound Poisson-gamma, handles the pile of $0 claims
    glm = Pipeline([("pre", preprocessor(True)), ("m", TweedieRegressor(link="log", max_iter=1000))])
    grid = {"m__power": [1.3, 1.5, 1.7], "m__alpha": [0.0001, 0.001, 0.01]}
    # 🐍 PY: fit params for a pipeline step are passed as <step>__<param>, here the exposure weights
    gs = GridSearchCV(glm, grid, cv=KFold(3, shuffle=True, random_state=42),
                      scoring="neg_mean_poisson_deviance", n_jobs=1).fit(X_tr, y_tr, m__sample_weight=w_tr)
    print("Tweedie best:", {k[3:]: v for k, v in gs.best_params_.items()})
    models["Tweedie GLM (log link)"] = gs.best_estimator_

    print("\nModel                   MAE $   Tweedie dev   Gini  min pred $")
    preds = {}
    for name, m in models.items():
        if name != "Tweedie GLM (log link)":
            m.fit(X_tr, y_tr, m__sample_weight=w_tr)
        p = m.predict(X_te)
        preds[name] = p
        # 🧠 ML: Tweedie deviance needs positive predictions - OLS can predict negative premiums!
        dev = mean_tweedie_deviance(y_te, np.clip(p, 1, None), power=1.5, sample_weight=w_te)
        print(f"{name:<23} {mean_absolute_error(y_te, p, sample_weight=w_te):>6,.0f} {dev:>12.1f} "
              f"{lorenz_gini(y_te, p, w_te):>6.3f} {p.min():>10,.0f}")

    best = models["Tweedie GLM (log link)"]
    p = preds["Tweedie GLM (log link)"]
    # 🧠 ML: balance check - total predicted premium should match total actual losses on test
    print(f"\nBalance: predicted ${np.sum(p * w_te):,.0f} vs actual ${np.sum(y_te * w_te):,.0f}")

    # 🧠 ML: log link -> exp(coef) is a multiplicative rating relativity, exactly how rate filings read
    names = best.named_steps["pre"].get_feature_names_out()
    coefs = best.named_steps["m"].coef_
    print("\nRating relativities (categorical factors):")
    for nm, c in zip(names, coefs):
        if nm.startswith("cat__"):
            print(f"  {nm[5:]:<20} x{np.exp(c):.2f}")

    # Age curve the GLM learned, via predictions for a reference driver
    ref = X_te.iloc[[0] * 6].copy()
    ref["driver_age"] = [18, 22, 30, 45, 65, 80]
    ref["years_licensed"] = (ref.driver_age - 17).clip(0, None)
    print("\nAge | indicated pure premium (reference driver)")
    for a, v in zip(ref.driver_age, best.predict(ref)):
        print(f"{a:>3} | ${v:>7,.0f} {'#' * int(v / 100)}")

    # Decile lift: does the riskiest tenth really cost more?
    # 🐍 PY: qcut with labels=False returns decile numbers 0-9 instead of interval labels
    dec = pd.qcut(p, 10, labels=False)
    lift = pd.DataFrame({"d": dec, "loss": y_te * w_te, "w": w_te}).groupby("d").sum()
    lr = (lift.loss / lift.w).values
    print(f"\nActual loss cost, lowest vs highest predicted decile: ${lr[0]:,.0f} vs ${lr[-1]:,.0f} ({lr[-1] / lr[0]:.1f}x)")

    print("\nRegulatory: rates must not be excessive, inadequate or unfairly discriminatory (state rating laws);")
    print("credit-based factors are restricted in some states; NAIC AI Bulletin expects governance + testing.")
    path = os.path.join(tempfile.mkdtemp(), "auto_pure_premium_glm.joblib")
    joblib.dump(best, path)
    print(f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    main()

# ---
# Linear Regression — HARD lab · Insurance: Auto insurance expected claim cost and premium pricing
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
