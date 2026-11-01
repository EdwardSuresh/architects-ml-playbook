# Production: production lithology classifier - tuned RBF SVM, calibrated, tested on a blind well
import os
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
FEATURES = ["gamma_ray_api", "density_g_cc", "neutron_porosity", "resistivity_ohm_m"]
PROFILE = {  # (mean, std) per log for each rock type
    "Sandstone": [(55, 20), (2.38, 0.09), (0.20, 0.06), (20, 12)],
    "Shale":     [(95, 25), (2.48, 0.08), (0.30, 0.07), (5, 4)],
    "Limestone": [(40, 18), (2.58, 0.08), (0.10, 0.06), (40, 30)],
}


def simulate_well(name, n_beds=25, top_ft=5000.0):
    """Return a depth-ordered log for one well made of stacked rock beds."""
    # 🐍 PY: multiple assignment sets two variables in one line
    rows, depth = [], top_ft
    for _ in range(n_beds):
        rock = RNG.choice(list(PROFILE), p=[0.35, 0.45, 0.20])
        # 🧠 ML: neighbouring samples in a bed are correlated - this is why random splits are too optimistic
        thickness = int(RNG.integers(5, 30))
        # 🧠 ML: a bed-level shift mimics real geology - the same rock reads slightly differently bed to bed
        bias = [RNG.normal(0, s * 0.3) for _, s in PROFILE[rock]]
        for _ in range(thickness):
            vals = [RNG.normal(m + b, s) for (m, s), b in zip(PROFILE[rock], bias)]
            rows.append([name, depth, *vals, rock])
            depth += 0.5
    # 🐍 PY: *vals unpacks the list into the row so all columns sit at one level
    return pd.DataFrame(rows, columns=["well", "depth_ft", *FEATURES, "lithology"])


def log_resistivity(X):
    """Resistivity spans decades, so take log10 of the last column."""
    X = np.array(X, dtype=float, copy=True)
    # 🐍 PY: X[:, 3] selects every row of column 3 (resistivity)
    X[:, 3] = np.log10(np.clip(X[:, 3], 0.5, None))
    return X


def build_pipeline():
    # 🧠 ML: putting transform + scaler + SVM in one Pipeline means CV refits scaling on each fold (no leakage)
    return Pipeline([
        ("logres", FunctionTransformer(log_resistivity)),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    ])


# 1. Data: four training wells, one blind well kept completely aside
wells = pd.concat([simulate_well(f"W{i}") for i in range(1, 6)], ignore_index=True)
# 🧠 ML: hold out a WHOLE well - the honest way to test a model meant for new wells
train = wells[wells.well != "W5"].sort_values(["well", "depth_ft"])
blind = wells[wells.well == "W5"].sort_values("depth_ft")
print(f"Train rows {len(train)} from wells {sorted(train.well.unique())}, blind well W5 rows {len(blind)}")
# 🐍 PY: dict comprehension converts pandas keys/values to plain str/float for clean printing
mix = train.lithology.value_counts(normalize=True).round(2)
print("Train mix:", {str(k): float(v) for k, v in mix.items()})

# 2. Grid search C and gamma with stratified 5-fold CV
# 🧠 ML: C = penalty for margin violations (high C = tighter fit); gamma = reach of each support vector
grid = {"svm__C": [0.3, 1, 10, 50], "svm__gamma": [0.03, 0.1, 0.3, 1.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 🧠 ML: macro-F1 averages F1 over rock types, so the rarer Limestone counts as much as Shale
search = GridSearchCV(build_pipeline(), grid, cv=cv, scoring="f1_macro", n_jobs=1)
search.fit(train[FEATURES], train.lithology)
print(f"\nBest params {search.best_params_}  CV macro-F1 {search.best_score_:.3f}")
res = pd.DataFrame(search.cv_results_)
# 🐍 PY: pivot_table reshapes the long results into a C x gamma grid for easy reading
print(res.pivot_table(index="param_svm__C", columns="param_svm__gamma",
                      values="mean_test_score").round(3).to_string())

# 3. Probability calibration
# 🧠 ML: SVM scores are not probabilities; sigmoid (Platt) calibration maps them to trustworthy 0-1 values
best_pipe = search.best_estimator_
calibrated = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=5)
calibrated.fit(train[FEATURES], train.lithology)

# 4. Blind-well test - the honest estimate of performance in a new well
# 🧠 ML: a 'always predict the majority rock' baseline shows how much the SVM really adds
baseline = DummyClassifier(strategy="most_frequent").fit(train[FEATURES], train.lithology)
proba = calibrated.predict_proba(blind[FEATURES])
# 🐍 PY: argmax(axis=1) finds the most likely class column for every row
pred = calibrated.classes_[proba.argmax(axis=1)]
print("\nBlind well W5 results:")
for name, p in [("Baseline (always majority)", baseline.predict(blind[FEATURES])), ("Calibrated SVM", pred)]:
    print(f"  {name:<27} acc {accuracy_score(blind.lithology, p):.3f}  "
          f"macro-F1 {f1_score(blind.lithology, p, average='macro'):.3f}")
print(f"  Calibrated log-loss {log_loss(blind.lithology, proba, labels=calibrated.classes_):.3f}")
cm = confusion_matrix(blind.lithology, pred, labels=calibrated.classes_)
print(pd.DataFrame(cm, index=calibrated.classes_, columns=calibrated.classes_).to_string())

# 5. Explainability via permutation importance on the blind well
# 🧠 ML: shuffle one log at a time; the bigger the F1 drop, the more the model relies on that log
imp = permutation_importance(best_pipe, blind[FEATURES], blind.lithology, scoring="f1_macro",
                             n_repeats=5, random_state=42)
print("\nPermutation importance (drop in macro-F1):")
for f, m in sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1]):
    print(f"  {f:<18} {m:.3f} " + "#" * int(m * 100))

# 6. Business decision: send low-confidence depths to a petrophysicist
# 🧠 ML: calibrated max-probability is a confidence; a threshold trades review effort for error risk
# 🐍 PY: assign returns a copy of the DataFrame with new columns added
blind = blind.assign(pred=pred, conf=proba.max(axis=1))
print("\nThreshold | auto-labelled | errors in auto | depths to review")
for t in [0.5, 0.7, 0.8, 0.9]:
    auto = blind[blind.conf >= t]
    errs = int((auto.pred != auto.lithology).sum())
    print(f"   {t:.1f}    |   {len(auto):>4}        |   {errs:>3}          |   {len(blind) - len(auto):>4}")

# 7. Depth-ordered strip - 12 samples around the first misclassified depth
first_miss = int(np.argmax((blind.pred != blind.lithology).values))
print("\nDepth log (blind well, around first miss):")
for _, r in blind.iloc[max(first_miss - 2, 0): first_miss + 2].iterrows():
    flag = "" if r.pred == r.lithology else "  <-- miss"
    print(f"  {r.depth_ft:7.1f} ft  true {r.lithology:<9} pred {r.pred:<9} conf {r.conf:.2f}{flag}")

# 8. Save the model for deployment
out = os.path.join(tempfile.mkdtemp(), "lithology_svm.joblib")
# 🧠 ML: save the WHOLE calibrated pipeline so scoring repeats the exact same preprocessing
joblib.dump(calibrated, out)
print(f"\nSaved calibrated model -> {os.path.basename(out)}")
print("Model-risk note: validate on blind wells per basin; document logs, tool vendors and"
      " depth ranges used; re-calibrate when a new field or logging suite is added.")

# ---
# Support Vector Machine (SVM) — PRODUCTION lab · Oil & Gas: Lithology classification from well logs
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
