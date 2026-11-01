"""Pipeline defect detection CNN: augmentation, train/val/test, recall-first threshold,
confusion matrix, baseline comparison, gradient saliency map, model saving, PHMSA note."""
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

torch.set_num_threads(1)
torch.manual_seed(42)
rng = np.random.default_rng(42)
SIZE = 16


def make_patch(kind):
    """kind: 0 = clean, 1 = corrosion pit, 2 = crack. Adds weld seams and grain as nuisances."""
    img = 0.7 + 0.1 * np.linspace(-1, 1, SIZE)[None, :] + rng.normal(0, 0.05, (SIZE, SIZE))
    # 🧠 ML: nuisance features (a bright weld seam) on ANY patch stop the model using 'any edge = defect'
    if rng.random() < 0.3:
        img[rng.integers(2, SIZE - 2), :] += 0.15
    if kind == 1:
        cy, cx = rng.integers(3, SIZE - 3, 2)
        r = rng.uniform(1.2, 2.5)
        yy, xx = np.mgrid[:SIZE, :SIZE]
        img -= rng.uniform(0.1, 0.3) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    elif kind == 2:
        angle, cx, cy = rng.uniform(0, np.pi), rng.uniform(5, 11), rng.uniform(5, 11)
        for t in np.linspace(-5, 5, 30):
            y, x = int(cy + t * np.sin(angle)), int(cx + t * np.cos(angle))
            if 0 <= y < SIZE and 0 <= x < SIZE:
                img[y, x] -= rng.uniform(0.08, 0.25)
    return np.clip(img, 0, 1)


def make_dataset(n, defect_rate=0.15):
    kinds = np.where(rng.random(n) < defect_rate, rng.integers(1, 3, n), 0)
    X = np.stack([make_patch(k) for k in kinds]).astype(np.float32)
    return X, (kinds > 0).astype(np.float32), kinds


def augment(xb):
    # 🧠 ML: defects can appear at any orientation - random flips/rotations teach that for free
    if rng.random() < 0.5:
        xb = torch.flip(xb, dims=[3])
    if rng.random() < 0.5:
        xb = torch.flip(xb, dims=[2])
    return torch.rot90(xb, k=int(rng.integers(0, 4)), dims=[2, 3])


class DefectCNN(nn.Module):
    def __init__(self, c1=8, c2=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2))
        # 🧠 ML: global max pooling asks 'is a defect pattern present ANYWHERE?' - position-independent
        self.head = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(c2, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def train(model, Xtr, ytr, Xval, yval, epochs=15, use_aug=True):
    # 🧠 ML: pos_weight counters the 15% defect rate so missed defects cost more during training
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(ytr == 0).sum() / (ytr == 1).sum())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_state = np.inf, None
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr)).split(64):
            xb = augment(Xtr[idx]) if use_aug else Xtr[idx]
            opt.zero_grad()
            loss_fn(model(xb), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xval), yval).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def saliency(model, x):
    """Gradient of the defect score w.r.t. each pixel: which pixels would change the decision most."""
    model.eval()
    x = x.clone().requires_grad_(True)
    model(x).sum().backward()
    return x.grad.abs().squeeze().numpy()


def ascii_map(a):
    # 🐍 PY: index into a string of 'ink levels' to draw a heatmap as text
    shades = " .:-=+*#"
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return ["".join(shades[int(v * 7)] for v in row) for row in a]


if __name__ == "__main__":
    X, y, kinds = make_dataset(2400)
    Xt, yt = torch.tensor(X).unsqueeze(1), torch.tensor(y)
    tr, va, te = slice(0, 1600), slice(1600, 2000), slice(2000, 2400)
    print(f"Train/val/test: 1600/400/400   defect rate {y.mean():.1%}")

    # 🧠 ML: baseline - logistic regression on raw pixels (no notion of local shape)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr].reshape(1600, -1), y[tr])
    p_lr = lr.predict_proba(X[te].reshape(400, -1))[:, 1]

    results = {}
    for name, aug in [("CNN no-aug", False), ("CNN + aug", True)]:
        torch.manual_seed(42)
        m = train(DefectCNN(), Xt[tr], yt[tr], Xt[va], yt[va], use_aug=aug)
        results[name] = (m, predict(m, Xt[va]), predict(m, Xt[te]))

    # 🧠 ML: integrity work is recall-first - choose the HIGHEST threshold on VAL that keeps recall >= 95%
    print("\nModel        threshold  test recall  test precision  pits caught  cracks caught")
    rows = [("LogReg pix", None, 0.5, p_lr)]
    for name, (m, p_val, p_te) in results.items():
        cands = [t for t in np.arange(0.05, 0.96, 0.05) if recall_score(y[va], p_val >= t) >= 0.95]
        rows.append((name, m, max(cands) if cands else 0.5, p_te))
    for name, _, t, p in rows:
        pred = p >= t
        pits = pred[kinds[te] == 1].mean()
        cracks = pred[kinds[te] == 2].mean()
        print(f"{name:<12} {t:9.2f}  {recall_score(y[te], pred):11.2f}  "
              f"{precision_score(y[te], pred):14.2f}  {pits:11.0%}  {cracks:13.0%}")

    name, model, thr, p_te = rows[-1]
    cm = confusion_matrix(y[te], p_te >= thr)
    print(f"\nConfusion matrix for {name} (rows = truth clean/defect, cols = predicted):")
    print(f"               pred clean  pred defect\n  true clean   {cm[0, 0]:10d}  {cm[0, 1]:11d}"
          f"\n  true defect  {cm[1, 0]:10d}  {cm[1, 1]:11d}")

    # 🧠 ML: saliency on the most confident corrosion pit - bright marks should sit on the pit itself
    i = int(np.argmax(np.where(kinds[te] == 1, p_te, -1)))
    sal = saliency(model, Xt[te][i:i + 1])
    img_rows, sal_rows = ascii_map(-X[te][i]), ascii_map(sal)
    print(f"\nTest patch {i} ({['clean', 'pit', 'crack'][kinds[te][i]]}, p={p_te[i]:.2f})"
          f"   darkness map  |  saliency map")
    for a, b in zip(img_rows, sal_rows):
        print(f"   {a}  |  {b}")

    path = os.path.join(tempfile.mkdtemp(), "defect_cnn.pt")
    torch.save({"state_dict": model.state_dict(), "threshold": float(thr)}, path)
    print(f"\nSaved model and threshold -> {os.path.basename(path)}")
    print("PHMSA note: 49 CFR 192/195 integrity management - model flags support, not replace,")
    print("qualified ILI analysis; log every miss found in the field and revalidate the threshold.")

# ---
# Convolutional Neural Network (CNN) — PRODUCTION lab · Oil & Gas: Pipeline inspection - detecting corrosion pits and cracks in image patches
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
