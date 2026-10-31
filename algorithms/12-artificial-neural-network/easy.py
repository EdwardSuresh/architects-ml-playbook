import numpy as np

# 8 life-insurance policyholders: [missed_payments_last_year, premium_increase_pct/10]
# 🐍 PY: each inner list is one customer; np.array makes it a 2-D table of floats
X = np.array([[0, 0.0], [0, 0.5], [1, 0.2], [0, 1.0],
              [2, 1.5], [3, 0.8], [2, 2.0], [4, 1.2]], dtype=float)
# 🐍 PY: 1 = the policy lapsed (customer stopped paying), 0 = still in force
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# 🧠 ML: one neuron = weighted sum of inputs + bias, squashed by sigmoid into a probability
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

np.random.seed(42)
# 🐍 PY: np.random.randn(2) gives two small random starting weights
w = np.random.randn(2) * 0.1
b = 0.0
lr = 0.5  # learning rate: how big each correction step is

# 🐍 PY: range(301) loops epochs 0..300; one epoch = one pass over all 8 customers
for epoch in range(301):
    # Forward pass: predict a lapse probability for every customer at once
    # 🐍 PY: X @ w is matrix-vector multiply; adding b broadcasts the bias to all 8 rows
    p = sigmoid(X @ w + b)
    # 🧠 ML: log-loss punishes confident wrong answers heavily
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    # 🧠 ML: backprop for one neuron - the gradient of log-loss w.r.t. z is simply (p - y)
    error = p - y
    grad_w = X.T @ error / len(y)
    grad_b = error.mean()
    # 🧠 ML: gradient descent nudges weights against the gradient to reduce loss
    w -= lr * grad_w
    b -= lr * grad_b
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  loss={loss:.3f}  w={np.round(w, 2)}  b={b:.2f}")

print("\nCustomer  missed  prem_inc  lapsed  P(lapse)")
# 🐍 PY: zip walks X rows, labels and predictions in step; enumerate adds a row number
for i, (row, label, prob) in enumerate(zip(X, y, sigmoid(X @ w + b))):
    print(f"{i:>8}  {row[0]:>6.0f}  {row[1]*10:>7.0f}%  {label:>6.0f}  {prob:>8.2f}")

new_customer = np.array([1, 1.0])  # 1 missed payment, 10% premium increase
print(f"\nNew customer lapse probability: {sigmoid(new_customer @ w + b):.2f}")

# ---
# Artificial Neural Network (ANN / MLP) — EASY lab · Insurance: Life insurance policy lapse prediction for retention outreach
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
