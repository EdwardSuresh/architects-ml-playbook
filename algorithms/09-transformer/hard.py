import os
import math
import random
import tempfile
import warnings
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
QUEUES = ["AutoGlass", "TotalLoss", "Injury", "FraudReview"]
PAD, UNK, MAXLEN = "<pad>", "<unk>", 16

KEYWORDS = {
    "AutoGlass": ["windshield", "chip", "cracked", "glass", "rock", "window", "sunroof", "wiper"],
    "TotalLoss": ["flooded", "fire", "rolled", "frame", "airbags", "towed", "destroyed", "submerged"],
    "Injury": ["neck", "ambulance", "hospital", "pain", "whiplash", "er", "back", "concussion"],
    "FraudReview": ["cash", "lapsed", "receipts", "inconsistent", "prior", "unwitnessed", "midnight", "new"],
}
FILLER = "vehicle insured reported on highway at the car damage driver called claimant after parked".split()

def make_note(q):
    # 🧠 ML: variable-length notes (6-16 words) force us to pad and mask - like real FNOL text
    n = random.randint(6, MAXLEN)
    rivals = [x for x in QUEUES if x != q]
    toks = random.sample(KEYWORDS[q], 2) + [random.choice(KEYWORDS[random.choice(rivals)])]
    toks += random.choices(FILLER, k=max(0, n - len(toks) - 2))
    random.shuffle(toks)
    # 🧠 ML: word ORDER matters for one phrase - 'not injured' must NOT be routed to Injury
    phrase = ["driver", "injured"] if q == "Injury" else (["not", "injured"] if random.random() < 0.4 else [])
    # 🐍 PY: slice assignment toks[i:i] = [...] inserts a list at position i without replacing
    i = random.randint(0, len(toks))
    toks[i:i] = phrase
    return toks[:MAXLEN]

# 🐍 PY: a nested comprehension - outer loop repeats n_per times, inner loop walks every queue
def build_data(n_per=600):
    data = [(make_note(q), i) for _ in range(n_per) for i, q in enumerate(QUEUES)]
    # 🧠 ML: 5% label noise = real-world mis-routing by adjusters
    data = [(t, random.randrange(4) if random.random() < 0.05 else y) for t, y in data]
    random.shuffle(data)
    return data

class Tokenizer:
    """Word-level vocab. Real systems use sub-word BPE/WordPiece like GPT/BERT."""
    def __init__(self, notes):
        words = sorted({w for toks in notes for w in toks})
        # 🧠 ML: id 0 = padding, id 1 = unknown word (handles words never seen in training)
        self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks):
        # 🐍 PY: [0] * k makes a list of k zeros - the padding to a fixed length
        ids = [self.stoi.get(w, 1) for w in toks][:MAXLEN]
        return ids + [0] * (MAXLEN - len(ids))

class PositionalEncoding(nn.Module):
    # 🧠 ML: attention alone ignores order; sine/cosine waves stamp each slot with its position
    def __init__(self, d, maxlen=MAXLEN):
        super().__init__()
        pos = torch.arange(maxlen).unsqueeze(1)
        # 🧠 ML: each pair of dimensions uses a different wave length, from fast to very slow
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(maxlen, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]

class ClaimRouter(nn.Module):
    def __init__(self, vocab, d=32, heads=4, layers=2, n_cls=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=64, dropout=0.1, batch_first=True)
        # 🧠 ML: stack of encoder layers; each = multi-head self-attention + feed-forward + residual
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        # 🧠 ML: attention pooling - one learned score per word; weights double as an explanation
        self.pool = nn.Linear(d, 1)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x, return_attn=False):
        # 🐍 PY: comparing a tensor to 0 gives a True/False tensor of the same shape
        mask = x == 0
        # 🧠 ML: src_key_padding_mask tells attention to ignore the <pad> slots
        h = self.encoder(self.pos(self.emb(x)), src_key_padding_mask=mask)
        # 🧠 ML: masked_fill sets padding scores to -1e9 so softmax gives them ~0 weight
        s = self.pool(h).squeeze(-1).masked_fill(mask, -1e9)
        w = torch.softmax(s, dim=1)
        logits = self.head((w.unsqueeze(-1) * h).sum(1))
        return (logits, w) if return_attn else logits

def train(model, Xtr, ytr, Xva, yva, epochs=10):
    # 🧠 ML: AdamW = Adam with weight decay, a standard regulariser for Transformers
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            # 🧠 ML: backward() computes gradients; step() nudges every weight to reduce the loss
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(yva, model(Xva).argmax(1), average="macro")
        # 🧠 ML: keep the best epoch on validation = simple early stopping against overfitting
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep + 1:2d}  val macro-F1={f1:.3f}")
    model.load_state_dict(best_state)
    return best_f1

data = build_data()
n = len(data)
tr, va, te = data[: int(.7 * n)], data[int(.7 * n): int(.85 * n)], data[int(.85 * n):]
# 🧠 ML: build the vocab from TRAINING notes only so test words cannot leak in
tok = Tokenizer([t for t, _ in tr])

def to_tensor(rows):
    return torch.tensor([tok.encode(t) for t, _ in rows]), torch.tensor([y for _, y in rows])

# 🐍 PY: map applies to_tensor to each split; tuple unpacking names all six results at once
(Xtr, ytr), (Xva, yva), (Xte, yte) = map(to_tensor, (tr, va, te))
print(f"notes train/val/test = {len(tr)}/{len(va)}/{len(te)}  vocab={len(tok.itos)}")

model = ClaimRouter(len(tok.itos))
print(f"params={sum(p.numel() for p in model.parameters()):,}")
train(model, Xtr, ytr, Xva, yva)
with torch.no_grad():
    pred = model(Xte).argmax(1)
tf_f1 = f1_score(yte, pred, average="macro")

# 🧠 ML: baseline - TF-IDF bag of words + logistic regression; bigrams give it some word order
vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\S+")
join = lambda rows: [" ".join(t) for t, _ in rows]
lr = LogisticRegression(max_iter=2000, C=5).fit(vec.fit_transform(join(tr)), ytr.numpy())
lr_f1 = f1_score(yte, lr.predict(vec.transform(join(te))), average="macro")
print(f"\nTest macro-F1  Transformer={tf_f1:.3f}   TF-IDF+LogReg baseline={lr_f1:.3f}")
print("Lesson: on short keyword notes a cheap baseline can match or beat it;"
      " Transformers pay off on long, messy text.")
print(classification_report(yte, pred, target_names=QUEUES, digits=2))

# 🧠 ML: explanation - which words got the most pooling attention for one routed claim
note = "claimant reported windshield cracked by rock on highway driver not injured".split()
with torch.no_grad():
    logits, w = model(torch.tensor([tok.encode(note)]), return_attn=True)
probs = torch.softmax(logits, 1)[0]
print("Note:", " ".join(note))
print("Route ->", QUEUES[int(probs.argmax())], f"(p={probs.max():.2f})  top words by attention:")
for word, a in sorted(zip(note, w[0][: len(note)].tolist()), key=lambda t: -t[1])[:4]:
    print(f"  {word:12s} {a:.2f} {'#' * int(a * 40)}")

# 🧠 ML: business rule - low confidence or FraudReview goes to a human adjuster (human-in-the-loop)
with torch.no_grad():
    conf = torch.softmax(model(Xte), 1).max(1).values
# 🧠 ML: the 0.85 threshold trades automation rate against accuracy; tune with claims ops
auto = conf >= 0.85
print(f"\nAuto-route {auto.float().mean():.0%} of claims at accuracy "
      f"{(pred[auto] == yte[auto]).float().mean():.3f}; rest go to triage desk.")
# Governance: LLMs (GPT-style decoder Transformers) can do this zero-shot, but insurers must
# document data lineage, bias tests and explainability under NAIC AI Model Bulletin + model-risk policy.
print("Governance: log model version, attention explanations and overrides for NAIC AI audit.")
# 🧠 ML: save vocab and queue names WITH the weights - the model is useless without them
path = os.path.join(tempfile.mkdtemp(), "claim_router.pt")
torch.save({"state": model.state_dict(), "vocab": tok.itos, "queues": QUEUES}, path)
print("Saved ->", os.path.basename(path))

# ---
# Transformer — HARD lab · Insurance: Claim notes / first notice of loss text triage
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
