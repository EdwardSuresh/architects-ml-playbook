# Easy: separate Sandstone from Shale with a straight line (linear SVM)
import numpy as np
from sklearn.svm import SVC

# 🐍 PY: a list of tuples - each tuple is one depth sample: (gamma_ray_API, density_g_cc, label)
samples = [
    (35, 2.30, "Sandstone"), (40, 2.32, "Sandstone"), (45, 2.28, "Sandstone"),
    (50, 2.35, "Sandstone"), (38, 2.25, "Sandstone"), (55, 2.33, "Sandstone"),
    (95, 2.52, "Shale"), (110, 2.55, "Shale"), (120, 2.50, "Shale"),
    (88, 2.48, "Shale"), (130, 2.58, "Shale"), (100, 2.45, "Shale"),
]
# 🐍 PY: list comprehension pulls the first two numbers of each tuple into a 2-D numpy array
X = np.array([[gr, rho] for gr, rho, _ in samples])
# 🐍 PY: conditional expression inside a comprehension -> 1 for Shale, 0 for Sandstone
y = np.array([1 if lab == "Shale" else 0 for _, _, lab in samples])

# Put both features on a similar scale: GR is ~100s, density is ~2.x
# 🧠 ML: SVM measures distances, so a feature with big numbers would dominate; we rescale by hand
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# 🧠 ML: kernel="linear" means the boundary is a straight line; C controls how hard we punish mistakes
model = SVC(kernel="linear", C=1.0)
model.fit(X_scaled, y)

# 🧠 ML: support vectors are the few samples sitting closest to the line - they alone define it
print("Support vector indices:", model.support_)
for i in model.support_:
    gr, rho, lab = samples[i]
    # 🐍 PY: f-string with format specs (:>4 right-align, :.2f two decimals)
    print(f"  sample {i:>2}: GR={gr:>4} API  RHOB={rho:.2f}  -> {lab}")

# 🧠 ML: the line is w . x + b = 0 ; the margin (street width) is 2 / ||w||
w, b = model.coef_[0], model.intercept_[0]
print(f"\nWeights w = {np.round(w, 3)}, bias b = {b:.3f}")
print(f"Margin width (scaled units) = {2 / np.linalg.norm(w):.3f}")

# Score a new depth: decision value sign tells the side of the line
new_depths = [(60, 2.36), (85, 2.44), (115, 2.53)]
for gr, rho in new_depths:
    # 🐍 PY: build a 1-row array and scale it with the SAME mean/std as training
    x = (np.array([[gr, rho]]) - X.mean(axis=0)) / X.std(axis=0)
    score = model.decision_function(x)[0]
    # 🧠 ML: |score| < 1 means the point falls inside the margin - the model is less sure
    label = "Shale" if score > 0 else "Sandstone"
    zone = "inside margin" if abs(score) < 1 else "clear side"
    print(f"GR={gr:>3} RHOB={rho:.2f} -> score {score:+.2f} -> {label:<9} ({zone})")

# ---
# Support Vector Machine (SVM) — EASY lab · Oil & Gas: Lithology classification from well logs
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
