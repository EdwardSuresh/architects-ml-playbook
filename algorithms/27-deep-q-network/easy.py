# A tiny neural Q-function for gas storage: Inject / Hold / Withdraw (Easy, plain numpy)
import numpy as np

np.random.seed(42)
ACTIONS = ["Inject", "Hold", "Withdraw"]
# 🐍 PY: each transition is a tuple (state, action, reward, next_state)
# state = [scaled gas price, storage fill fraction]
transitions = [
    # reward for Inject = winter forward value of the gas minus today's purchase cost (mark-to-market)
    (np.array([0.20, 0.10]), 0, +3.0, np.array([0.25, 0.30])),  # cheap summer gas -> inject
    (np.array([0.20, 0.10]), 1, -0.2, np.array([0.25, 0.10])),  # hold in summer -> only a storage fee
    (np.array([0.90, 0.80]), 2, +9.0, np.array([0.85, 0.60])),  # pricey winter gas -> withdraw (we earn)
    (np.array([0.50, 0.50]), 1, -0.2, np.array([0.55, 0.50])),  # hold -> small storage fee
    (np.array([0.20, 0.10]), 2, -5.0, np.array([0.25, 0.10])),  # withdraw when empty-ish and cheap -> penalty
]
gamma, lr = 0.5, 0.02  # modest discount keeps this tiny bootstrapped problem stable

# 🧠 ML: Q-network = one hidden layer: 2 inputs -> 8 hidden -> 3 outputs (one Q-value per action)
W1 = np.random.randn(2, 8) * 0.5
b1 = np.zeros(8)
W2 = np.random.randn(8, 3) * 0.5
b2 = np.zeros(3)

# 🐍 PY: a function returning two values as a tuple
def forward(s):
    h = np.maximum(0, s @ W1 + b1)  # 🧠 ML: ReLU hidden layer
    return h, h @ W2 + b2           # 🐍 PY: '@' is matrix multiplication

for epoch in range(301):
    loss = 0.0
    # 🐍 PY: tuple unpacking straight inside the for-loop header
    for s, a, r, s2 in transitions:
        h, q = forward(s)
        _, q_next = forward(s2)
        # 🧠 ML: Bellman target = reward now + discounted best future value
        target = r + gamma * q_next.max()
        err = q[a] - target
        loss += err ** 2
        # 🧠 ML: gradient only flows through the chosen action's output (backprop by hand)
        dq = np.zeros(3)
        dq[a] = err
        dh = (W2 @ dq) * (h > 0)
        # 🐍 PY: np.outer builds the weight-gradient matrix; '-=' updates arrays in place
        W2 -= lr * np.outer(h, dq)
        b2 -= lr * dq
        W1 -= lr * np.outer(s, dh)
        b1 -= lr * dh
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  squared TD error = {loss:.3f}")

# 🧠 ML: the greedy policy simply picks the action with the highest predicted Q
for label, s in [("Summer, cheap, nearly empty", [0.2, 0.1]), ("Winter, pricey, fairly full", [0.9, 0.8])]:
    _, q = forward(np.array(s))
    print(f"{label:30s} Q={np.round(q, 2)} -> {ACTIONS[int(q.argmax())]}")

# ---
# Deep Q-Network (DQN) — EASY lab · Oil & Gas: Natural gas storage: daily Inject / Hold / Withdraw decisions
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
