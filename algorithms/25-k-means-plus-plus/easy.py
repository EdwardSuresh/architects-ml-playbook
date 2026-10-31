# k-means++ seeding by hand: pick 3 starting warehouse (DC) sites from 12 customer demand points
import numpy as np

# 🐍 PY: each row is a customer town's (x, y) position in miles on a simple map grid
customers = np.array([
    [2, 3], [3, 2], [1, 1], [2, 1],            # cluster near the west
    [20, 21], [22, 19], [21, 22], [19, 20],    # cluster in the north-east
    [30, 4], [31, 2], [29, 3], [32, 5],        # cluster in the south-east
], dtype=float)
K = 3

def dist2_to_nearest(points, centers):
    # 🐍 PY: broadcasting points[:, None, :] - centers[None, :, :] gives all point-center differences at once
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    # 🧠 ML: D(x)^2 = squared distance from each customer to its CLOSEST chosen center so far
    return d2.min(axis=1)

def kmeanspp_seed(points, k, rng, verbose=False):
    # 🧠 ML: step 1 - first center is a uniformly random customer
    centers = [points[rng.integers(len(points))]]
    while len(centers) < k:
        d2 = dist2_to_nearest(points, np.array(centers))
        # 🧠 ML: step 2 - probability of picking a point is proportional to D^2: far points are favoured
        probs = d2 / d2.sum()
        if verbose:
            # 🐍 PY: f"{p:.2f}" formats each probability with 2 decimals
            print("  pick probs:", " ".join(f"{p:.2f}" for p in probs))
        # 🐍 PY: rng.choice with p= draws an index using our custom probabilities
        centers.append(points[rng.choice(len(points), p=probs)])
        if verbose:
            print(f"  chose center #{len(centers)} at {centers[-1]}")
    return np.array(centers)

def inertia(points, centers):
    # 🧠 ML: inertia = total squared distance from customers to their nearest center (lower is better)
    return dist2_to_nearest(points, centers).sum()

rng = np.random.default_rng(42)
print("k-means++ seeding, step by step:")
pp = kmeanspp_seed(customers, K, rng, verbose=True)
print(f"k-means++ seeds:\n{pp}\ninertia = {inertia(customers, pp):.1f}\n")

# 🧠 ML: compare with plain random seeding over many trials - how often do 2 seeds land in one cluster?
rand_scores, pp_scores = [], []
for trial in range(200):
    # 🐍 PY: rng.choice(..., replace=False) picks K different row indexes
    rand_scores.append(inertia(customers, customers[rng.choice(len(customers), K, replace=False)]))
    pp_scores.append(inertia(customers, kmeanspp_seed(customers, K, rng)))
# 🐍 PY: converting a list to np.array lets us do vectorised comparisons like (arr > 100).mean()
rand_scores, pp_scores = np.array(rand_scores), np.array(pp_scores)
print("Seeding only (before any k-means iterations), 200 trials:")
print(f"  random    : mean inertia {rand_scores.mean():7.1f}, bad starts (>100) {(rand_scores > 100).mean():.0%}")
print(f"  k-means++ : mean inertia {pp_scores.mean():7.1f}, bad starts (>100) {(pp_scores > 100).mean():.0%}")

# ---
# k-Means++ — EASY lab · Supply Chain: Choosing distribution center locations
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
