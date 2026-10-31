import numpy as np

# A tiny 6x6 grayscale patch of pipe wall: 9 = bright clean steel, 1 = dark crack line
# 🐍 PY: a list of lists becomes a 2-D numpy array (rows x columns)
patch = np.array([
    [9, 9, 9, 1, 9, 9],
    [9, 9, 9, 1, 9, 9],
    [9, 9, 9, 1, 9, 9],
    [9, 9, 9, 1, 9, 9],
    [9, 9, 9, 1, 9, 9],
    [9, 9, 9, 1, 9, 9],
], dtype=float)
print("Inspection patch (dark column = crack):")
print(patch.astype(int))

# 🧠 ML: a 3x3 filter (kernel) - this one responds to dark vertical lines between bright pixels
# 🧠 ML: bright neighbours (+1) minus twice the centre (-2): a DARK centre column gives a BIG positive value
kernel = np.array([
    [1, -2, 1],
    [1, -2, 1],
    [1, -2, 1],
], dtype=float)
print("\nFilter (vertical dark-line detector):")
print(kernel.astype(int))

# 🧠 ML: convolution = slide the filter over the image, multiply overlapping numbers, add them up
# 🐍 PY: shape unpacking gives height and width in one line
h, w = patch.shape
out = np.zeros((h - 2, w - 2))
# 🐍 PY: nested loops visit every position where the 3x3 window fits
for i in range(h - 2):
    for j in range(w - 2):
        # 🐍 PY: slicing [i:i+3, j:j+3] cuts out the 3x3 window under the filter
        window = patch[i:i + 3, j:j + 3]
        # 🐍 PY: * is element-wise multiply for numpy arrays; .sum() adds all 9 products
        out[i, j] = (window * kernel).sum()

print("\nFeature map after convolution:")
print(out.astype(int))

# 🧠 ML: ReLU keeps positive evidence and zeroes the rest - same as in a real CNN layer
relu = np.maximum(out, 0)
# 🧠 ML: max pooling = 'is the pattern present anywhere?' -> one strong number
score = relu.max()
# 🐍 PY: np.unravel_index turns a flat position into a (row, col) pair
row, col = np.unravel_index(relu.argmax(), relu.shape)
print(f"\nStrongest crack response: {score:.0f} at feature-map column {col} (image column {col + 1})")
print("Verdict:", "DEFECT - send for dig verification" if score > 20 else "clean")

# Same filter on a clean patch with no crack
clean = np.full((6, 6), 9.0)
clean_score = max((clean[i:i + 3, j:j + 3] * kernel).sum() for i in range(4) for j in range(4))
print(f"Clean patch response: {clean_score:.0f} -> clean")

# ---
# Convolutional Neural Network (CNN) — EASY lab · Oil & Gas: Pipeline inspection - detecting corrosion pits and cracks in image patches
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
