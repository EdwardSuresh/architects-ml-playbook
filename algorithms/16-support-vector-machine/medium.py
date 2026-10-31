# Medium: RBF SVM classifying Sandstone / Shale / Limestone from four well logs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

np.random.seed(42)
N_PER_CLASS = 600

# 🐍 PY: a dict of dicts holds typical (mean, std) log responses per rock type
profiles = {
    "Sandstone": {"gamma_ray_api": (55, 20), "density_g_cc": (2.38, 0.09),
                  "neutron_porosity": (0.20, 0.06), "resistivity_ohm_m": (20, 12)},
    "Shale":     {"gamma_ray_api": (95, 25), "density_g_cc": (2.48, 0.08),
                  "neutron_porosity": (0.30, 0.07), "resistivity_ohm_m": (5, 4)},
    "Limestone": {"gamma_ray_api": (40, 18), "density_g_cc": (2.58, 0.08),
                  "neutron_porosity": (0.10, 0.06), "resistivity_ohm_m": (40, 30)},
}

frames = []
# 🐍 PY: .items() yields (key, value) pairs so we loop over rock name and its profile together
for rock, cols in profiles.items():
    # 🐍 PY: dict comprehension draws N random values per column from a normal distribution
    data = {c: np.random.normal(m, s, N_PER_CLASS) for c, (m, s) in cols.items()}
    df_rock = pd.DataFrame(data)
    df_rock["lithology"] = rock
    frames.append(df_rock)
# 🐍 PY: pd.concat stacks the three tables; ignore_index renumbers rows 0..N-1
df = pd.concat(frames, ignore_index=True)
# 🧠 ML: resistivity is skewed (log-normal in reality), so model log10 of it
df["resistivity_ohm_m"] = df["resistivity_ohm_m"].clip(lower=0.5)
df["log_resistivity"] = np.log10(df["resistivity_ohm_m"])

features = ["gamma_ray_api", "density_g_cc", "neutron_porosity", "log_resistivity"]
# 🐍 PY: .values turns DataFrame columns into a plain numpy array
X, y = df[features].values, df["lithology"].values
print("Rows per class:\n", df["lithology"].value_counts().to_string(), "\n")

# 🧠 ML: stratify=y keeps the same rock mix in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# 🧠 ML: fit the scaler on TRAIN only, then apply to test - avoids leaking test info
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# 🧠 ML: RBF kernel bends the boundary; gamma = how local each point's influence is
for scaled in (False, True):
    Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, y_train)
    acc = accuracy_score(y_test, clf.predict(Xte))
    print(f"RBF SVM  scaled={str(scaled):<5}  test accuracy = {acc:.3f}")

model = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_train_s, y_train)
# 🧠 ML: n_support_ shows how many training samples each class needs to define the boundary
y_pred = model.predict(X_test_s)
print(f"\nSupport vectors per class: {dict(zip(model.classes_, model.n_support_.tolist()))}")
print("\nClassification report (test):")
print(classification_report(y_test, y_pred, digits=3))

# 🐍 PY: wrap the numpy confusion matrix in a DataFrame for readable row/column labels
cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=model.classes_),
                  index=[f"true_{c}" for c in model.classes_],
                  columns=[f"pred_{c}" for c in model.classes_])
print("Confusion matrix:\n", cm.to_string())

# Business interpretation
shale_as_sand = cm.loc["true_Shale", "pred_Sandstone"]
print(f"\nShale called Sandstone: {shale_as_sand} samples."
      " These are the costly errors - a petrophysicist might flag a non-reservoir zone as pay.")

# ---
# Support Vector Machine (SVM) — MEDIUM lab · Oil & Gas: Lithology classification from well logs
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
