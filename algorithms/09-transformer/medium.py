import random
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
random.seed(42)
QUEUES = ["AutoGlass", "TotalLoss", "Injury", "FraudReview"]

# 🐍 PY: dict of lists - each queue has its own typical words; 'common' words appear everywhere
words = {
    "AutoGlass": ["windshield", "chip", "cracked", "glass", "rock", "window", "sunroof"],
    "TotalLoss": ["flooded", "fire", "rolled", "frame", "airbags", "towed", "destroyed"],
    "Injury": ["neck", "ambulance", "hospital", "pain", "whiplash", "er", "back"],
    "FraudReview": ["cash", "witness", "lapsed", "late", "receipts", "inconsistent", "prior"],
}
common = ["vehicle", "insured", "reported", "on", "highway", "at", "night", "the", "car", "damage"]


def make_note(q, length=8):
    # 🧠 ML: 2 queue words + 1-2 words from other queues (noise) + filler -> realistic messiness
    rivals = [x for x in QUEUES if x != q]
    others = [random.choice(words[random.choice(rivals)]) for _ in range(random.randint(1, 2))]
    toks = random.sample(words[q], 2) + others
    toks += random.choices(common, k=length - len(toks))
    random.shuffle(toks)
    return toks


# 🐍 PY: list comprehension of (tokens, label_index) tuples - 2000 synthetic FNOL notes
data = [(make_note(q), i) for _ in range(500) for i, q in enumerate(QUEUES)]
# 🧠 ML: 5% of labels flipped at random = adjusters sometimes mis-route; caps achievable accuracy
data = [(t, random.randrange(4) if random.random() < 0.05 else lab) for t, lab in data]
random.shuffle(data)
# 🐍 PY: set union of all words, then enumerate to give each a number; 0 is reserved for padding
vocab = {w: i + 1 for i, w in enumerate(sorted({w for toks, _ in data for w in toks}))}
X = torch.tensor([[vocab[w] for w in toks] for toks, _ in data])
y = torch.tensor([lab for _, lab in data])
X_tr, y_tr, X_te, y_te = X[:1600], y[:1600], X[1600:], y[1600:]
print(f"Vocab size={len(vocab)}  train={len(X_tr)}  test={len(X_te)}  example={data[0][0]}")


# 🐍 PY: a class that inherits nn.Module; __init__ builds layers, forward() runs them
class TinyClaimTransformer(nn.Module):
    def __init__(self, vocab_size, d=32, heads=2, n_cls=4):
        super().__init__()
        # 🧠 ML: embedding = lookup table turning word ids into learnable vectors
        self.emb = nn.Embedding(vocab_size + 1, d, padding_idx=0)
        # 🧠 ML: one encoder layer = multi-head self-attention + small feed-forward net
        self.enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=64, batch_first=True, dropout=0.1)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        h = self.enc(self.emb(x))
        # 🧠 ML: mean-pool all word vectors into one note vector, then score each queue
        return self.head(h.mean(dim=1))


model = TinyClaimTransformer(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(6):
    model.train()
    # 🐍 PY: torch.randperm shuffles indices; range(start, stop, step) walks mini-batches of 64
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()
    model.eval()
    # 🐍 PY: no_grad() turns off gradient tracking for faster, safer evaluation
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    print(f"epoch {epoch + 1}  last-batch loss={loss.item():.3f}  test acc={acc:.3f}")

with torch.no_grad():
    pred = model(X_te).argmax(1)
print("\nPer-queue test accuracy:")
for i, q in enumerate(QUEUES):
    m = y_te == i
    print(f"  {q:12s} {(pred[m] == i).float().mean().item():.2f}  (n={int(m.sum())})")

new = "insured reported neck pain after car rolled on highway".split()
# 🐍 PY: dict.get(w, 0) returns 0 for unknown words instead of raising KeyError
ids = torch.tensor([[vocab.get(w, 0) for w in new]])
with torch.no_grad():
    probs = torch.softmax(model(ids), dim=1)[0]
print("\nNew FNOL note:", " ".join(new))
for q, p in sorted(zip(QUEUES, probs.tolist()), key=lambda t: -t[1]):
    print(f"  {q:12s} {p:.2f} {'#' * int(p * 30)}")

# ---
# Transformer — MEDIUM lab · Insurance: Claim notes / first notice of loss text triage
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
