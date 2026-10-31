import numpy as np

np.random.seed(42)
# 🐍 PY: weekly units shipped for one SKU, scaled to 0-1 (1.0 = 1,000 cases) - a plain list
weekly_demand = [0.42, 0.45, 0.50, 0.61, 0.58]

# 🧠 ML: an LSTM cell keeps two memories - h (short-term output) and c (long-term cell state)
hidden = 2
# 🐍 PY: np.zeros makes a vector of 0s; both memories start empty
h = np.zeros(hidden)
c = np.zeros(hidden)

# 🧠 ML: four gates, each has weights for the input x and the previous h, plus a bias
# 🐍 PY: a dict comprehension builds W (input weights), U (hidden weights) and b per gate
gates = ["forget", "input", "candidate", "output"]
W = {g: np.random.randn(hidden, 1) * 0.5 for g in gates}
U = {g: np.random.randn(hidden, hidden) * 0.5 for g in gates}
b = {g: np.zeros(hidden) for g in gates}
# 🧠 ML: forget-gate bias of +1 is a common trick - "remember by default"
b["forget"] += 1.0

# 🐍 PY: small helper functions defined with def; sigmoid squashes any number into 0..1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gate(name, x, h, act):
    return act(W[name] @ x + U[name] @ h + b[name])

# 🐍 PY: enumerate(..., start=1) numbers the weeks from 1 instead of 0
for week, demand in enumerate(weekly_demand, start=1):
    x = np.array([demand])
    # 🧠 ML: forget gate f - how much old memory to keep (0 = wipe, 1 = keep all)
    f = gate("forget", x, h, sigmoid)
    # 🧠 ML: input gate i - how much of the new information to write
    i = gate("input", x, h, sigmoid)
    # 🧠 ML: candidate g - the new information itself, in -1..1 via tanh
    g = gate("candidate", x, h, np.tanh)
    # 🧠 ML: output gate o - how much of the memory to reveal as this week's output
    o = gate("output", x, h, sigmoid)
    # 🧠 ML: the two key LSTM equations - update long memory, then derive the short memory
    c = f * c + i * g
    h = o * np.tanh(c)
    print(f"week {week}: demand={demand:.2f} f={np.round(f, 2)} i={np.round(i, 2)} "
          f"c={np.round(c, 3)} h={np.round(h, 3)}")

# 🧠 ML: in a real model a final linear layer turns h into next week's forecast
w_out, b_out = np.array([0.6, 0.4]), 0.55
# 🐍 PY: the @ operator between two 1-D arrays is a dot product (a single number)
forecast = (w_out @ h + b_out) * 1000
print(f"\nUntrained forecast for week 6: {forecast:.0f} cases (training would tune W, U, b)")

# ---
# LSTM — EASY lab · Supply Chain: Weekly SKU demand forecasting for a distribution center
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
