import numpy as np

np.random.seed(42)
# 🐍 PY: a short claim note split into a list of words (tokens)
tokens = ["windshield", "cracked", "by", "rock"]
# 🧠 ML: each word becomes a small vector (embedding); real models learn 512+ numbers per word
# 🐍 PY: a dict maps each word to a hand-made 4-number vector
emb = {
    "windshield": [1.0, 0.9, 0.0, 0.1],
    "cracked":    [0.8, 1.0, 0.1, 0.0],
    "by":         [0.1, 0.0, 0.2, 0.1],
    "rock":       [0.6, 0.5, 0.0, 0.3],
}
# 🐍 PY: list comprehension pulls each token's vector in order -> 4x4 matrix
X = np.array([emb[t] for t in tokens])
d = X.shape[1]

# 🧠 ML: Q (query) = what am I looking for, K (key) = what do I offer, V (value) = what I pass on
# 🧠 ML: in a trained model W_q, W_k, W_v are learned; here they are small random matrices
W_q = np.eye(d) + 0.1 * np.random.randn(d, d)
W_k = np.eye(d) + 0.1 * np.random.randn(d, d)
W_v = np.eye(d)
# 🐍 PY: the @ operator is matrix multiplication
Q, K, V = X @ W_q, X @ W_k, X @ W_v

# 🧠 ML: score = how well each word's query matches every other word's key; divide by sqrt(d) to keep calm
scores = Q @ K.T / np.sqrt(d)
print("Raw attention scores (row = word looking, col = word looked at):")
# 🐍 PY: np.round + f-string padding prints a neat table
for t, row in zip(tokens, np.round(scores, 2)):
    print(f"  {t:>10s} {row}")

# 🧠 ML: softmax turns each row into percentages that sum to 1 (the attention weights)
def softmax(z):
    # 🐍 PY: keepdims=True keeps the shape (4,1) so broadcasting divides row by row
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

A = softmax(scores)
print("\nAttention weights (each row sums to 1):")
for t, row in zip(tokens, A):
    # 🐍 PY: nested f-string formatting inside a join builds one aligned line
    print(f"  {t:>10s} " + " ".join(f"{w:.2f}" for w in row))

# 🧠 ML: output = weighted blend of value vectors -> each word now carries context from the others
out = A @ V
print("\nContext-aware vector for 'cracked':", np.round(out[1], 2))
# 🧠 ML: crude routing - compare the sentence average to a 'glass' direction vector
glass_dir = np.array([1, 1, 0, 0]) / np.sqrt(2)
print(f"Similarity to Auto-Glass queue: {out.mean(axis=0) @ glass_dir:.2f}")
print("Word 'by' receives the least attention:", tokens[int(A.sum(axis=0).argmin())])

# ---
# Transformer — EASY lab · Insurance: Claim notes / first notice of loss text triage
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
