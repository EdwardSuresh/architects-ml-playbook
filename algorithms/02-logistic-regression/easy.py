# EASY: logistic regression from scratch - probability a loan applicant defaults
import numpy as np

# 🐍 PY: each tuple is one past borrower: (debt_to_income %, missed_payments_last_2y, defaulted 0/1)
borrowers = [
    (15, 0, 0), (20, 0, 0), (25, 1, 0), (18, 0, 0), (30, 0, 0), (22, 1, 0),
    (35, 2, 0), (28, 0, 0), (45, 3, 1), (50, 2, 1), (40, 4, 1), (55, 1, 1),
    (38, 3, 1), (60, 5, 1), (33, 2, 1), (42, 0, 0),
    (24, 2, 1), (47, 1, 0), (29, 3, 0), (36, 1, 1),
]
# 🐍 PY: np.array on a list of tuples gives a 2-D table; [:, :2] = all rows, first two columns
data = np.array(borrowers, dtype=float)
X, y = data[:, :2], data[:, 2]
# 🧠 ML: scaling features (z-score) makes gradient descent converge smoothly
mu, sd = X.mean(axis=0), X.std(axis=0)
Xs = (X - mu) / sd

# 🧠 ML: the sigmoid squashes any number into a 0-1 probability
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 🐍 PY: np.zeros(2) creates [0., 0.] - one weight per feature; b is the intercept (bias)
w, b, lr = np.zeros(2), 0.0, 0.5
for step in range(201):
    # 🐍 PY: @ is matrix multiplication: each row's features times the weights, summed
    p = sigmoid(Xs @ w + b)
    # 🧠 ML: log-loss punishes confident wrong answers; we nudge weights downhill on it
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    w -= lr * (Xs.T @ (p - y)) / len(y)
    b -= lr * np.mean(p - y)
    # 🐍 PY: % is the remainder operator - print every 50th step only
    if step % 50 == 0:
        print(f"step {step:>3}  log-loss {loss:.3f}  weights DTI={w[0]:+.2f} missed={w[1]:+.2f}")

# 🧠 ML: positive weight = raises default risk; exp(weight) = odds multiplier per 1 std-dev increase
print(f"\nOdds multiplier per +1 std of DTI: {np.exp(w[0]):.2f}x, of missed payments: {np.exp(w[1]):.2f}x")

# Score two new applicants
new = np.array([[20, 0], [48, 3]], dtype=float)
probs = sigmoid(((new - mu) / sd) @ w + b)
# 🐍 PY: zip walks two sequences side by side
for (dti, missed), pd_ in zip(new, probs):
    verdict = "DECLINE" if pd_ > 0.5 else "APPROVE"
    print(f"DTI {dti:.0f}%, missed {missed:.0f} -> probability of default {pd_:.1%} -> {verdict}")

# ---
# Logistic Regression — EASY lab · Financial Services: Consumer loan default prediction and credit scorecard
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
