# t-SNE by hand: how similar are 6 card customers? (Concept)
import numpy as np

np.set_printoptions(precision=3, suppress=True)
# 🐍 PY: a list of tuples -> each tuple is one customer: (avg txn $ in hundreds, night-time txn share)
customers = [(0.5, 0.10), (0.6, 0.12), (0.55, 0.08), (4.0, 0.80), (4.2, 0.85), (3.9, 0.75)]
labels = ["N1", "N2", "N3", "F1", "F2", "F3"]  # N = normal, F = fraud-like
# 🐍 PY: np.array turns the list into a 6x2 matrix we can do maths on
X = np.array(customers)

# 🧠 ML: step 1 - squared distance between every pair of customers
# 🐍 PY: broadcasting X[:, None, :] - X[None, :, :] makes a 6x6x2 grid of differences
D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
print("Squared distances:\n", D)

# 🧠 ML: step 2 - turn distance into 'probability I would pick you as my neighbour' (Gaussian)
sigma = 1.0  # real t-SNE tunes sigma per point using perplexity; we fix it to keep it simple
P_cond = np.exp(-D / (2 * sigma ** 2))
# 🐍 PY: np.fill_diagonal edits the matrix in place - a point is never its own neighbour
np.fill_diagonal(P_cond, 0.0)
# 🐍 PY: keepdims=True keeps shape (6,1) so each row divides by its own sum
P_cond = P_cond / P_cond.sum(axis=1, keepdims=True)
# 🧠 ML: step 3 - symmetrise so p_ij = p_ji, then normalise the whole matrix to sum to 1
P = (P_cond + P_cond.T) / (2 * len(X))
print("\nHigh-dim similarity P (rows sum ~1/6):\n", P)

# 🧠 ML: step 4 - a guess for the 2-D map; t-SNE uses a heavy-tailed Student-t here
rng = np.random.default_rng(42)
Y = rng.normal(0, 1, size=(6, 2))
DY = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
Q = 1.0 / (1.0 + DY)
np.fill_diagonal(Q, 0.0)
Q = Q / Q.sum()
# 🧠 ML: step 5 - KL divergence = how badly the 2-D map betrays the true neighbourhoods
mask = P > 0
kl = np.sum(P[mask] * np.log(P[mask] / Q[mask]))
print(f"\nKL(P||Q) for a random 2-D layout: {kl:.3f}")

# 🧠 ML: step 6 - a few gradient steps pull similar points together, push others apart
for step in range(1, 201):
    DY = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    W = 1.0 / (1.0 + DY)
    np.fill_diagonal(W, 0.0)
    Q = W / W.sum()
    # 🧠 ML: the classic t-SNE gradient: 4 * sum_j (p_ij - q_ij) * w_ij * (y_i - y_j)
    grad = 4 * (((P - Q) * W)[:, :, None] * (Y[:, None, :] - Y[None, :, :])).sum(axis=1)
    Y -= 2.0 * grad
    if step in (1, 50, 200):
        kl = np.sum(P[mask] * np.log(P[mask] / Q[mask]))
        print(f"step {step:3d}  KL={kl:.3f}")
print("Final 2-D map:", ", ".join(f"{n}=({y[0]:.2f},{y[1]:.2f})" for n, y in zip(labels, Y)))
# 🐍 PY: zip pairs each label with its row; argmax finds the closest neighbour
for name, row in zip(labels, P):
    best = labels[int(np.argmax(row))]
    print(f"{name}: strongest neighbour = {best}  (p={row.max():.3f})")

# ---
# t-SNE — CONCEPT lab · Financial Services: Visualising card-transaction behaviour to reveal fraud clusters
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
