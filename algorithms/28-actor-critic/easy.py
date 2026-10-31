# One actor-critic update by hand: should this month's portfolio be "Risk-On" or "Risk-Off"? (Easy)
import numpy as np

# 🐍 PY: a dict maps each action name to the monthly return we happened to observe after choosing it
ACTIONS = ["Risk-On (80% stocks)", "Risk-Off (20% stocks)"]
observed_reward = {0: 1.2, 1: 0.4}  # monthly return in percent: +1.2% vs +0.4%

# 🧠 ML: ACTOR = policy. Two 'preference' numbers turned into probabilities with softmax
theta = np.array([0.0, 0.0])
# 🧠 ML: CRITIC = value estimate V(s): 'how good is a typical month in this market state?'
V = 0.6  # critic currently expects +0.6% a month
alpha_actor, alpha_critic, gamma = 0.5, 0.5, 0.0  # gamma=0 -> one-month episode, no future term

# 🐍 PY: def with a default argument; np.exp works element-wise on the whole array
def softmax(z, temp=1.0):
    e = np.exp((z - z.max()) / temp)
    return e / e.sum()

for month in range(1, 4):
    probs = softmax(theta)
    # 🐍 PY: f-string with format spec :.2f and a list comprehension inside
    print(f"Month {month}: policy probs = {[f'{p:.2f}' for p in probs]}, critic V = {V:.3f}")
    # Pretend the actor sampled action 0 in month 1-2 and action 1 in month 3
    a = 0 if month < 3 else 1
    r = observed_reward[a]
    # 🧠 ML: TD error (advantage) = what happened - what the critic expected
    td_error = r + gamma * 0.0 - V
    print(f"   chose {ACTIONS[a]}, reward = {r:+.1f}%, TD error = {td_error:+.3f}")
    # 🧠 ML: gradient of log softmax = one_hot(a) - probs
    grad_log = -probs.copy()
    grad_log[a] += 1.0
    # 🧠 ML: actor step - push probability UP if advantage > 0, DOWN if < 0
    theta += alpha_actor * td_error * grad_log
    # 🧠 ML: critic step - nudge V toward the observed outcome
    V += alpha_critic * td_error
    print(f"   new theta = {np.round(theta, 3)}, new V = {V:.3f}")

# 🐍 PY: zip pairs names with probabilities; float() turns numpy scalars into plain Python numbers
print("\nFinal policy:", {name: round(float(p), 3) for name, p in zip(ACTIONS, softmax(theta))})
print("Positive surprise raised Risk-On; the below-average Risk-Off month lowered Risk-Off.")

# ---
# Actor-Critic — EASY lab · Financial Services: Monthly Stocks / Bonds / Cash allocation
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
