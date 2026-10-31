# REINFORCE on a 3-arm bandit: which renewal discount (0%, 5%, 10%) keeps the most premium? (Easy)
import numpy as np

np.random.seed(42)
# 🐍 PY: parallel lists - index 0, 1, 2 line up across all three
discounts = [0.00, 0.05, 0.10]
renew_prob = [0.70, 0.82, 0.86]  # hidden truth: chance the customer renews at each discount
PREMIUM = 1200.0                  # annual premium in dollars

# 🧠 ML: expected reward for each arm = P(renew) * premium * (1 - discount)
for d, p in zip(discounts, renew_prob):
    print(f"discount {d:4.0%}: true expected retained premium = ${p * PREMIUM * (1 - d):7.2f}")

# 🧠 ML: the policy = softmax over 3 preference numbers (theta). Start indifferent
theta = np.zeros(3)
lr = 0.05

def softmax(z):
    # 🐍 PY: subtracting the max is a numerical-safety trick; it does not change the result
    e = np.exp(z - z.max())
    return e / e.sum()

for step in range(1, 3001):
    probs = softmax(theta)
    # 🧠 ML: sample an action FROM the policy (this is how policy-gradient explores)
    a = np.random.choice(3, p=probs)
    renewed = np.random.rand() < renew_prob[a]
    # 🐍 PY: a conditional expression - value_if_true if condition else value_if_false
    reward = PREMIUM * (1 - discounts[a]) if renewed else 0.0
    # 🧠 ML: grad of log pi(a) for softmax = one_hot(a) - probs
    grad_log_pi = -probs
    grad_log_pi[a] += 1
    # 🧠 ML: REINFORCE update: theta += lr * reward * grad log pi(a); reward scaled to $1000s
    theta += lr * (reward / 1000) * grad_log_pi
    if step in (1, 10, 100, 500, 1000, 3000):
        print(f"step {step:4d}  probs(0%,5%,10%) = {np.round(softmax(theta), 3)}")

best = int(np.argmax(softmax(theta)))
print(f"\nLearned policy favours a {discounts[best]:.0%} discount.")

# ---
# Policy Gradient (REINFORCE) — EASY lab · Insurance: Renewal retention offers: which discount for which policyholder
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
