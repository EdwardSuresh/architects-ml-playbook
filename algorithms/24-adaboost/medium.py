# AdaBoostClassifier: early warning of stuck pipe from 1-minute drilling parameters
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
N = 4000

# ---- Synthetic rig-sensor snapshots (one row = a 1-minute summary while drilling) ----
# 🐍 PY: a dict of numpy arrays turns straight into a DataFrame, one key per column
df = pd.DataFrame({
    "hookload_klb": rng.normal(230, 15, N),
    "torque_kftlb": rng.normal(14, 2.5, N),
    "rop_ft_hr": rng.normal(70, 18, N).clip(5, None),
    "mud_weight_ppg": rng.normal(10.5, 0.6, N),
    "standpipe_psi": rng.normal(3200, 250, N),
    "torque_trend_10min": rng.normal(0, 0.8, N),      # change in torque over last 10 minutes
})
# 🧠 ML: hidden "true" risk - rising torque, high hookload (overpull), falling ROP and heavy mud raise risk
logit = (-4.5 + 2.0 * df.torque_trend_10min + 0.10 * (df.hookload_klb - 230)
         - 0.05 * (df.rop_ft_hr - 70) + 1.5 * (df.mud_weight_ppg - 10.5)
         + 0.003 * (df.standpipe_psi - 3200))
# 🐍 PY: 1 / (1 + exp(-x)) is the logistic function; rng.random() < p draws a 0/1 outcome
df["stuck_pipe_next_hr"] = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
print(f"Rows: {N}   stuck-pipe rate: {df.stuck_pipe_next_hr.mean():.1%}")

X = df.drop(columns="stuck_pipe_next_hr")
y = df.stuck_pipe_next_hr
# 🧠 ML: stratify=y keeps the same rare-event rate in train and test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 🧠 ML: AdaBoost = many depth-1 trees ("stumps"), each focusing on rows the previous ones got wrong
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42,
)
model.fit(X_tr, y_tr)

# 🧠 ML: predict_proba gives a risk score; we use it for AUC and a custom alarm threshold
proba = model.predict_proba(X_te)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

# 🧠 ML: AdaBoost probabilities bunch up near 0.5, so small threshold moves matter a lot
# 🧠 ML: with rare events the default 0.5 cut-off misses many; lower it to raise recall
for thr in (0.5, 0.45):
    pred = (proba >= thr).astype(int)
    # 🐍 PY: .ravel() flattens the 2x2 confusion matrix so we can unpack 4 numbers at once
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"\nThreshold {thr}: caught {tp}/{tp + fn} events, {fp} false alarms")
    if thr == 0.45:
        print(classification_report(y_te, pred, target_names=["normal", "stuck"], digits=3))

# 🧠 ML: AdaBoost's feature_importances_ = how much weighted error each feature's stumps removed
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance:")
# 🐍 PY: .items() yields (index, value) pairs from a pandas Series
for name, v in imp.items():
    print(f"  {name:20s} {v:.3f} " + "#" * int(v * 60))

print("\nBusiness read-out: lowering the threshold from 0.5 to 0.45 roughly doubles events caught;"
      "\neach false alarm costs a short pause to circulate/ream, far cheaper than a fishing job.")

# ---
# AdaBoost — MEDIUM lab · Oil & Gas: Drilling stuck-pipe early warning
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
