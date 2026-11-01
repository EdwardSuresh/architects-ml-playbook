# AdaBoost by hand: 3 rounds of decision stumps on 10 drilling snapshots (1 = stuck pipe followed)
import numpy as np

# 🐍 PY: each row = [torque_kftlb, hookload_klb, rop_ft_hr]; a 2-D numpy array is like a small table
X = np.array([
    [12, 210, 85], [13, 215, 80], [18, 260, 40], [11, 205, 90], [19, 250, 35],
    [17, 240, 70], [14, 262, 72], [12, 255, 75], [15, 265, 30], [16, 220, 42],
])
# 🧠 ML: AdaBoost uses labels -1 / +1 (here +1 = stuck pipe within the next hour)
y = np.array([-1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
names = ["torque", "hookload", "rop"]
n = len(y)
# 🧠 ML: start with equal weight on every row - all mistakes count the same
w = np.full(n, 1 / n)


def best_stump(X, y, w):
    """Try every feature/threshold/direction; return the stump with the lowest WEIGHTED error."""
    best = None
    for f in range(X.shape[1]):
        # 🐍 PY: np.unique gives sorted distinct values to use as candidate thresholds
        for thr in np.unique(X[:, f]):
            for sign in (1, -1):
                # 🐍 PY: np.where(cond, a, b) is a vectorised if/else over the whole column
                pred = np.where(sign * (X[:, f] - thr) >= 0, 1, -1)
                err = w[pred != y].sum()
                if best is None or err < best[0]:
                    best = (err, f, thr, sign, pred)
    return best


ensemble_score = np.zeros(n)
for rnd in range(1, 4):
    err, f, thr, sign, pred = best_stump(X, y, w)
    # 🧠 ML: alpha = how much say this stump gets; lower error -> bigger vote
    alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
    rule = f"{names[f]} {'>=' if sign == 1 else '<='} {thr}"
    print(f"Round {rnd}: stump '{rule}'  weighted_error={err:.3f}  alpha={alpha:.3f}")
    # 🧠 ML: boost weights of rows this stump got WRONG so the next stump focuses on them
    w = w * np.exp(-alpha * y * pred)
    # 🐍 PY: dividing an array by its sum normalises it so weights add up to 1
    w = w / w.sum()
    print("   new weights:", " ".join(f"{v:.2f}" for v in w))
    ensemble_score += alpha * pred

# 🧠 ML: final prediction = sign of the weighted vote of all stumps
final = np.sign(ensemble_score)
print("\nrow | torque hookload rop | actual | vote   | predicted")
for i in range(n):
    print(f" {i:2d} | {X[i, 0]:5d} {X[i, 1]:8d} {X[i, 2]:4d} |   {y[i]:+d}   | {ensemble_score[i]:+.2f}  |   {int(final[i]):+d}")
print(f"\nTraining accuracy: {(final == y).mean():.0%}")

# ---
# AdaBoost — CONCEPT lab · Oil & Gas: Drilling stuck-pipe early warning
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
