import math
from collections import Counter

# 8 short training complaints, each already labelled by a human agent
# 🐍 PY: a list of (text, label) tuples - a simple way to keep data and answers together
train = [
    ("charged twice on my credit card statement", "Credit Card"),
    ("credit card interest rate went up", "Credit Card"),
    ("mortgage escrow payment is wrong", "Mortgage"),
    ("loan servicer lost my mortgage payment", "Mortgage"),
    ("someone used my card without permission fraud", "Fraud"),
    ("unauthorized transfer from my account fraud", "Fraud"),
    ("overdraft fee charged again", "Fees"),
    ("monthly maintenance fee on my account", "Fees"),
]

# 🐍 PY: a set comprehension collects each label once
labels = sorted({label for _, label in train})
# 🐍 PY: Counter counts word occurrences; dict comprehension makes one Counter per class
word_counts = {c: Counter() for c in labels}
for text, label in train:
    word_counts[label].update(text.split())
vocab = {w for c in labels for w in word_counts[c]}
print("Vocabulary size:", len(vocab))
print("Top words in Fraud:", word_counts["Fraud"].most_common(3))

# 🧠 ML: prior = how common each class is before reading the text (here 2/8 each)
priors = {c: sum(1 for _, l in train if l == c) / len(train) for c in labels}

def class_score(text, c, alpha=1.0):
    # 🧠 ML: Bayes rule in log form: log P(class) + sum of log P(word | class)
    score = math.log(priors[c])
    total = sum(word_counts[c].values())
    for w in text.split():
        # 🧠 ML: Laplace smoothing (+alpha) stops an unseen word from making the probability zero
        p = (word_counts[c][w] + alpha) / (total + alpha * len(vocab))
        score += math.log(p)
    return score

new_complaint = "unauthorized charge on my credit card"
print(f"\nNew complaint: '{new_complaint}'")
# 🐍 PY: dict comprehension scores every class; max(..., key=...) picks the best one
scores = {c: class_score(new_complaint, c) for c in labels}
for c, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"  {c:<12} log-score {s:7.2f}")

# 🧠 ML: convert log-scores to probabilities that sum to 1 (softmax of the logs)
m = max(scores.values())
exp = {c: math.exp(s - m) for c, s in scores.items()}
probs = {c: v / sum(exp.values()) for c, v in exp.items()}
best = max(probs, key=probs.get)
print(f"Route to: {best} team (probability {probs[best]:.0%})")

# ---
# Naive Bayes — EASY lab · Financial Services: Customer complaint routing to the right product team
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
