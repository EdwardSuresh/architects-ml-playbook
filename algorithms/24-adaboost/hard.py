# Production-style stuck-pipe early warning: AdaBoost tuned with grouped CV (by well), imbalance weights,
# staged learning curve, GradientBoosting comparison, cost-based alarm threshold, permutation importance.
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
COST_MISSED_EVENT = 250_000   # $ - fishing job + sidetrack rig time when stuck pipe is not prevented
COST_FALSE_ALARM = 4_000      # $ - ~1 hour of rig time spent circulating/reaming unnecessarily
FEATURES = ["hookload_klb", "torque_kftlb", "rop_ft_hr", "mud_weight_ppg", "standpipe_psi",
            "torque_trend_10min", "hookload_trend_10min"]


def make_wells(n_wells=24, rows_per_well=250, seed=42):
    rng = np.random.default_rng(seed)
    frames = []
    for w in range(n_wells):
        # 🧠 ML: each well has its own baseline (depth, formation) - rows from one well are correlated
        base_hl, base_mw = rng.normal(230, 20), rng.normal(10.5, 0.5)
        n = rows_per_well
        d = pd.DataFrame({
            "well_id": w,
            "hookload_klb": rng.normal(base_hl, 12, n),
            "torque_kftlb": rng.normal(14, 2.5, n),
            "rop_ft_hr": rng.normal(70, 18, n).clip(5, None),
            "mud_weight_ppg": rng.normal(base_mw, 0.3, n),
            "standpipe_psi": rng.normal(3200, 250, n),
            "torque_trend_10min": rng.normal(0, 0.8, n),
            "hookload_trend_10min": rng.normal(0, 4, n),
        })
        # 🧠 ML: risk depends on CHANGES vs the well's own normal, plus an interaction (overpull AND slow ROP)
        logit = (-5.0 + 1.8 * d.torque_trend_10min + 0.25 * d.hookload_trend_10min
                 + 1.2 * (d.mud_weight_ppg - 10.5) + 1.5 * ((d.rop_ft_hr < 45) & (d.hookload_trend_10min > 3)))
        d["stuck_pipe_next_hr"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        frames.append(d)
    # 🐍 PY: pd.concat stacks a list of DataFrames vertically into one table
    return pd.concat(frames, ignore_index=True)


def expected_cost(y_true, proba, thr):
    pred = proba >= thr
    missed = int(((~pred) & (y_true == 1)).sum())
    false_alarms = int((pred & (y_true == 0)).sum())
    return missed * COST_MISSED_EVENT + false_alarms * COST_FALSE_ALARM, missed, false_alarms


if __name__ == "__main__":
    df = make_wells()
    # 🧠 ML: hold out whole WELLS for testing - a random row split would leak each well's baseline
    test_wells = df.well_id >= 18
    train, test = df[~test_wells], df[test_wells]
    X_tr, y_tr, g_tr = train[FEATURES], train.stuck_pipe_next_hr, train.well_id
    X_te, y_te = test[FEATURES], test.stuck_pipe_next_hr
    print(f"Train: {len(train)} rows / 18 wells, event rate {y_tr.mean():.1%}  |  "
          f"Test: {len(test)} rows / 6 wells, event rate {y_te.mean():.1%}")

    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
    grid = {"n_estimators": [50, 150], "learning_rate": [0.1, 0.5],
            "estimator__max_depth": [1, 2]}
    # 🧠 ML: GroupKFold keeps every row of a well in the same fold -> honest CV for unseen wells
    # 🧠 ML: average precision (PR-AUC) is the right metric when positives are rare
    search = GridSearchCV(ada, grid, cv=GroupKFold(n_splits=3), scoring="average_precision", n_jobs=1)
    search.fit(X_tr, y_tr, groups=g_tr)
    print("\n== AdaBoost grid search (3-fold GroupKFold, PR-AUC) ==")
    res = pd.DataFrame(search.cv_results_)
    cols = ["param_n_estimators", "param_learning_rate", "param_estimator__max_depth", "mean_test_score"]
    print(res[cols].sort_values("mean_test_score", ascending=False).head(5).round(3).to_string(index=False))
    best_ada = search.best_estimator_

    # 🧠 ML: imbalance option - 'balanced' sample weights up-weight rare stuck-pipe rows during training
    sw = compute_sample_weight("balanced", y_tr)
    ada_bal = AdaBoostClassifier(**{k: v for k, v in best_ada.get_params().items()
                                    if k in ("estimator", "n_estimators", "learning_rate", "random_state")})
    ada_bal.fit(X_tr, y_tr, sample_weight=sw)

    # 🧠 ML: staged_predict_proba replays the ensemble one stump at a time = free learning curve
    print("\n== Staged learning curve on test wells (ROC-AUC as stumps are added) ==")
    n_total = best_ada.n_estimators
    checkpoints = sorted({1, 5, 20, 50, n_total // 2, n_total})
    for i, p in enumerate(best_ada.staged_predict_proba(X_te), start=1):
        if i in checkpoints:
            auc = roc_auc_score(y_te, p[:, 1])
            print(f"  {i:4d} estimators  AUC={auc:.3f} " + "#" * int((auc - 0.5) * 80))

    # 🧠 ML: gradient boosting fits residuals with a smooth loss - often more robust to label noise
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
    gb.fit(X_tr, y_tr, sample_weight=sw)

    print("\n== Model comparison on 6 unseen wells ==")
    models = {"AdaBoost (tuned)": best_ada, "AdaBoost + balanced": ada_bal, "GradientBoosting": gb}
    probas = {}
    for name, m in models.items():
        probas[name] = m.predict_proba(X_te)[:, 1]
        print(f"  {name:20s} ROC-AUC={roc_auc_score(y_te, probas[name]):.3f}  "
              f"PR-AUC={average_precision_score(y_te, probas[name]):.3f}")

    # 🧠 ML: ranking metrics barely move with weights - weighting mainly shifts the score scale (threshold)
    # 🧠 ML: pick the alarm threshold by $ cost, not accuracy - a missed event costs ~60x a false alarm
    print("\n== Alarm threshold by expected cost (AdaBoost) ==")
    p = probas["AdaBoost (tuned)"]
    # 🐍 PY: np.quantile picks thresholds spread across the score range, avoiding empty bins
    thresholds = np.unique(np.round(np.quantile(p, np.linspace(0.3, 0.98, 8)), 3))
    rows = [(t, *expected_cost(y_te.to_numpy(), p, t)) for t in thresholds]
    for t, c, m, fa in rows:
        print(f"  thr={t:.3f}  missed={m:3d}  false_alarms={fa:4d}  cost=${c / 1e6:5.2f}M")
    best_thr, best_cost, _, _ = min(rows, key=lambda r: r[1])
    no_model = int(y_te.sum()) * COST_MISSED_EVENT
    print(f"  -> threshold {best_thr:.3f}: ${best_cost / 1e6:.2f}M vs ${no_model / 1e6:.2f}M with no warning system")

    # 🧠 ML: permutation importance = drop in PR-AUC when a feature is shuffled (model-agnostic)
    pi = permutation_importance(best_ada, X_te, y_te, scoring="average_precision", n_repeats=3,
                                random_state=42, n_jobs=1)
    print("\n== Permutation importance (PR-AUC drop) ==")
    for f, v in sorted(zip(FEATURES, pi.importances_mean), key=lambda x: -x[1]):
        print(f"  {f:22s} {v:+.3f}")

    print("\nSafety note: this is an ADVISORY tool - the driller keeps authority (API RP 59 well-control"
          "\npractice, BSEE SEMS). Log every alarm and response; re-validate per field and formation.")
    path = os.path.join(tempfile.mkdtemp(), "stuck_pipe_ada.joblib")
    # 🐍 PY: joblib.dump serialises the whole fitted sklearn object, plus metadata in a dict
    joblib.dump({"model": best_ada, "threshold": float(best_thr), "features": FEATURES}, path)
    print(f"Saved model + threshold -> {os.path.basename(path)} in a temp dir")

# ---
# AdaBoost — PRODUCTION lab · Oil & Gas: Drilling stuck-pipe early warning
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
