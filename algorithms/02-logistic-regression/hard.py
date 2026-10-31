# HARD: production credit scorecard - WoE binning, tuned logistic regression, points, adverse action
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
FEATS = ["credit_score", "debt_to_income", "utilization_pct", "delinq_2y", "months_employed", "inquiries_6m"]
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 8   # 600 points = 8:1 good:bad odds; +20 points doubles odds


def make_data(n=8000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "credit_score": np.clip(rng.normal(690, 65, n), 450, 850).round(),
        "debt_to_income": np.clip(rng.normal(33, 11, n), 1, 75).round(1),
        "utilization_pct": np.clip(rng.beta(1.1, 1.4, n) * 100, 0, 100).round(1),
        "delinq_2y": rng.poisson(0.35, n),
        "months_employed": rng.integers(0, 300, n),
        "inquiries_6m": rng.poisson(1.2, n),
    })
    # 🧠 ML: utilization has a U-shape risk (0% and >80% both riskier) - WoE bins capture this, raw LR cannot
    util_risk = 1.3 * (df.utilization_pct > 80) + 0.8 * (df.utilization_pct < 8)
    logit = (-2.6 - 0.017 * (df.credit_score - 690) + 0.045 * (df.debt_to_income - 33) + util_risk
             + 0.5 * df.delinq_2y - 0.003 * df.months_employed + 0.2 * df.inquiries_6m)
    df["bad"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


# 🐍 PY: inheriting BaseEstimator + TransformerMixin makes our class work inside sklearn Pipelines
class WoEBinner(BaseEstimator, TransformerMixin):
    """Quantile-bin each feature and replace the bin with its Weight of Evidence."""

    # 🐍 PY: __init__ only stores settings; sklearn requires learning to happen in fit()
    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X, y):
        # 🐍 PY: attributes ending in _ are learned during fit (a scikit-learn convention)
        self.edges_, self.woe_, self.iv_ = {}, {}, {}
        y = np.asarray(y)
        for col in X.columns:
            edges = np.unique(np.quantile(X[col], np.linspace(0, 1, self.n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            b = np.digitize(X[col], edges[1:-1])
            goods, bads = np.bincount(b[y == 0], minlength=len(edges) - 1), np.bincount(b[y == 1], minlength=len(edges) - 1)
            # 🧠 ML: WoE = ln(%good / %bad) per bin; +0.5 smoothing avoids divide-by-zero on empty bins
            pg, pb = (goods + 0.5) / goods.sum(), (bads + 0.5) / bads.sum()
            woe = np.log(pg / pb)
            self.edges_[col], self.woe_[col] = edges, woe
            # 🧠 ML: Information Value sums how well the bins separate goods from bads (>0.1 = useful)
            self.iv_[col] = float(np.sum((pg - pb) * woe))
        return self

    def transform(self, X):
        # 🐍 PY: dict comprehension builds {column: WoE values} for every column in one line
        out = {c: self.woe_[c][np.digitize(X[c], self.edges_[c][1:-1])] for c in X.columns}
        return pd.DataFrame(out, index=X.index)


def to_points(pipe, X):
    """Convert each feature's WoE x coefficient into scorecard points (higher score = safer)."""
    # 🧠 ML: factor = points per unit of log-odds; offset anchors BASE_SCORE at BASE_ODDS
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    woe = pipe.named_steps["woe"].transform(X)
    lr = pipe.named_steps["lr"]
    # 🧠 ML: score = offset - factor * log-odds(bad); split the log-odds across features to get points
    pts = -factor * woe * lr.coef_[0]
    base_pts = offset - factor * lr.intercept_[0]
    return pts, base_pts


def main():
    df = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATS], df.bad, test_size=0.3, stratify=df.bad, random_state=42)
    print(f"Applications: {len(df):,}  bad rate: {df.bad.mean():.1%}")

    pipe = Pipeline([("woe", WoEBinner()), ("lr", LogisticRegression(max_iter=1000))])
    # 🧠 ML: tune bin count and regularization C with 5-fold stratified cross-validation on AUC
    grid = {"woe__n_bins": [4, 6, 8], "lr__C": [0.01, 0.1, 1.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(X_tr, y_tr)
    model = search.best_estimator_
    print(f"Best params: {search.best_params_}  CV AUC {search.best_score_:.3f}")

    # Baseline: plain logistic regression on raw (unbinned) features
    # 🧠 ML: baseline on raw features shows what WoE binning adds (it captures non-linear U-shapes)
    raw = LogisticRegression(max_iter=2000).fit((X_tr - X_tr.mean()) / X_tr.std(), y_tr)
    auc_raw = roc_auc_score(y_te, raw.predict_proba((X_te - X_tr.mean()) / X_tr.std())[:, 1])
    auc_woe = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"Test AUC  raw LR {auc_raw:.3f} (Gini {2*auc_raw-1:.3f}) | WoE scorecard {auc_woe:.3f} (Gini {2*auc_woe-1:.3f})")

    binner = model.named_steps["woe"]
    print("\nFeature           IV     coef")
    for f, c in zip(FEATS, model.named_steps["lr"].coef_[0]):
        print(f"  {f:<15} {binner.iv_[f]:.3f}  {c:+.3f}")

    pts, base = to_points(model, X_te)
    score = (pts.sum(axis=1) + base).round()
    print(f"\nScore range on test set: {score.min():.0f} - {score.max():.0f}")
    # 🧠 ML: a scorecard should rank-order - bad rate must fall as the score band rises
    bands = pd.cut(score, [0, 560, 600, 640, 680, 2000], labels=["<560", "560-599", "600-639", "640-679", "680+"])
    tbl = pd.DataFrame({"band": bands, "bad": y_te.values}).groupby("band", observed=True)["bad"].agg(["size", "mean"])
    print(tbl.rename(columns={"size": "apps", "mean": "bad_rate"}).round(3).to_string())

    # Cut-off decision: expected profit = 0.9k margin per good loan, 6k loss per bad loan
    # 🧠 ML: the cut-off is a business decision - trade approval volume against expected losses
    print("\nCut-off | approve % | bad rate | expected profit $k")
    for cut in [560, 580, 600, 620, 640]:
        ok = score >= cut
        prof = 0.9 * ((y_te.values == 0) & ok).sum() - 6.0 * ((y_te.values == 1) & ok).sum()
        print(f"{cut:>7} | {ok.mean():>9.0%} | {y_te.values[ok].mean():>8.1%} | {prof:>12,.0f}")

    # 🧠 ML: adverse action reasons (ECOA / Reg B) = features where the applicant lost most points vs max
    max_pts = pts.max()
    i = int(np.argmin(score.values))
    # 🐍 PY: Series arithmetic aligns on column names; nlargest picks the biggest shortfalls
    shortfall = (max_pts - pts.iloc[i]).nlargest(3)
    print(f"\nDeclined applicant score {score.iloc[i]:.0f}. Top adverse action reasons:")
    for f, gap in shortfall.items():
        print(f"  - {f} (value {X_te.iloc[i][f]}), {gap:.0f} points below best bin")

    print("\nFair lending: no protected attributes used; run disparate-impact tests on proxies (ECOA/FHA).")
    print("Model risk: SR 11-7 documentation, independent validation, annual PSI/KS monitoring.")
    path = os.path.join(tempfile.mkdtemp(), "pd_scorecard.joblib")
    joblib.dump({"pipeline": model, "pdo": PDO, "base_score": BASE_SCORE}, path)
    print(f"Saved scorecard to {os.path.basename(path)}")


if __name__ == "__main__":
    main()

# ---
# Logistic Regression — HARD lab · Financial Services: Consumer loan default prediction and credit scorecard
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
