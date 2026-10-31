import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)
SIZE = 16  # 16x16 pixel inspection patches


# --- 1. Generate synthetic pipe-wall inspection patches ---
def make_patch(defect):
    # 🧠 ML: normal steel = bright, slightly grainy texture with smooth shading
    # 🐍 PY: np.linspace + broadcasting builds a gentle left-to-right brightness gradient
    img = 0.7 + 0.1 * np.linspace(-1, 1, SIZE)[None, :] + rng.normal(0, 0.05, (SIZE, SIZE))
    if defect:
        if rng.random() < 0.5:
            # Corrosion pit: a dark round blob somewhere in the patch
            cy, cx, r = rng.integers(3, SIZE - 3, 2).tolist() + [rng.uniform(1.5, 3)]
            yy, xx = np.mgrid[:SIZE, :SIZE]
            img -= 0.35 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        else:
            # Crack: a thin dark line at a random angle
            angle, cx, cy = rng.uniform(0, np.pi), rng.uniform(5, 11), rng.uniform(5, 11)
            for t in np.linspace(-6, 6, 40):
                y, x = int(cy + t * np.sin(angle)), int(cx + t * np.cos(angle))
                if 0 <= y < SIZE and 0 <= x < SIZE:
                    img[y, x] -= 0.3
    return np.clip(img, 0, 1)

n = 1200
# 🐍 PY: rng.random(n) < 0.3 gives a True/False array -> about 30% defects
labels = (rng.random(n) < 0.3).astype(int)
# 🐍 PY: list comprehension builds each patch; np.stack turns the list into one (n, 16, 16) array
images = np.stack([make_patch(d) for d in labels])
print(f"Patches: {images.shape}, defect rate {labels.mean():.1%}")

# 🧠 ML: CNNs expect (batch, channels, height, width) - grayscale has 1 channel
X = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(labels, dtype=torch.float32)
split = int(0.8 * n)
X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

# --- 2. Tiny CNN ---
# 🧠 ML: Conv2d learns its own 3x3 filters (like the hand-made crack filter in the easy level)
# 🧠 ML: MaxPool halves the image size and keeps the strongest response in each 2x2 block
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16 -> 8x8
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8 -> 4x4
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 1),
)
# 🐍 PY: sum over a generator expression counts every trainable number in the model
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")
opt = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.BCEWithLogitsLoss()

# --- 3. Train with mini-batches ---
for epoch in range(1, 11):
    perm = torch.randperm(split)
    # 🐍 PY: .split(64) cuts the shuffled index list into chunks of 64
    for idx in perm.split(64):
        opt.zero_grad()
        loss = loss_fn(model(X_tr[idx]).squeeze(1), y_tr[idx])
        loss.backward()
        opt.step()
    if epoch % 2 == 0:
        print(f"epoch {epoch:2d}  last batch loss {loss.item():.3f}")

# --- 4. Evaluate ---
with torch.no_grad():
    prob = torch.sigmoid(model(X_te).squeeze(1)).numpy()
pred = (prob >= 0.5).astype(int)
truth = y_te.numpy().astype(int)
# 🐍 PY: boolean arithmetic - (pred == 1) & (truth == 1) is True only where both hold
tp = int(((pred == 1) & (truth == 1)).sum())
fp = int(((pred == 1) & (truth == 0)).sum())
fn = int(((pred == 0) & (truth == 1)).sum())
print(f"\nTest accuracy: {(pred == truth).mean():.3f}")
print(f"Defects caught: {tp}/{tp + fn}  (recall {tp / (tp + fn):.2f})")
print(f"False calls on clean steel: {fp}  (precision {tp / max(tp + fp, 1):.2f})")
print("\nInspection read-out: patches flagged as defects go to an integrity engineer for review;")
print("missed defects (false negatives) are the costly error, so recall is the key number.")

# ---
# Convolutional Neural Network (CNN) — MEDIUM lab · Oil & Gas: Pipeline inspection - detecting corrosion pits and cracks in image patches
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
