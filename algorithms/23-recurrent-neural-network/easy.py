# RNN by hand: watch the hidden state "remember" 5 days of oil production for one well
import numpy as np

np.random.seed(42)
# 🐍 PY: a numpy array of 5 daily oil rates in barrels per day (bbl/d); the well is slowly declining
oil_bbl_per_day = np.array([520.0, 505.0, 498.0, 470.0, 462.0])
# 🧠 ML: neural nets learn better on small numbers, so scale rates to roughly 0..1
x = oil_bbl_per_day / 600.0

HIDDEN = 3
# 🧠 ML: the RNN has just three weight sets, REUSED at every day (that's what "recurrent" means)
# 🐍 PY: np.random.randn(rows, cols) creates a matrix of random normal numbers
W_xh = np.random.randn(HIDDEN, 1) * 0.5      # input -> hidden
W_hh = np.random.randn(HIDDEN, HIDDEN) * 0.5  # previous hidden -> hidden (the "memory" path)
W_hy = np.random.randn(1, HIDDEN) * 0.5      # hidden -> prediction
b_h = np.zeros((HIDDEN, 1))

# 🧠 ML: hidden state h starts empty - the network knows nothing before day 1
h = np.zeros((HIDDEN, 1))
print("day | rate_bbl/d | hidden state h (3 numbers)      | raw output")
# 🐍 PY: enumerate(..., start=1) numbers the days from 1 instead of 0
for day, xt in enumerate(x, start=1):
    # 🐍 PY: @ is matrix multiplication; np.array([[xt]]) makes a 1x1 column so shapes line up
    # 🧠 ML: new memory = tanh(today's input + yesterday's memory); tanh squashes into -1..1
    h = np.tanh(W_xh @ np.array([[xt]]) + W_hh @ h + b_h)
    y = (W_hy @ h).item()
    # 🐍 PY: .ravel() flattens the 3x1 column to a simple list for printing
    h_txt = " ".join(f"{v:+.3f}" for v in h.ravel())
    print(f" {day}  |   {oil_bbl_per_day[day - 1]:6.1f}   | [{h_txt}]          | {y:+.3f}")

# 🧠 ML: after day 5, h is a compressed summary of all 5 days - that's what we forecast from
print("\nFinal hidden state summarises the whole 5-day history:", np.round(h.ravel(), 3))

# 🧠 ML: the weights are untrained, so the raw output is meaningless until we fit W_hy
# Quick 'training' of only the output layer with least squares, to show the idea of learning:
H_all, h = [], np.zeros((HIDDEN, 1))
for xt in x[:-1]:
    h = np.tanh(W_xh @ np.array([[xt]]) + W_hh @ h + b_h)
    H_all.append(h.ravel())
# 🐍 PY: np.linalg.lstsq solves "find W so that H_all @ W ~= next-day targets"
W_fit, *_ = np.linalg.lstsq(np.array(H_all), x[1:], rcond=None)
pred = np.array(H_all) @ W_fit * 600
# 🧠 ML: caution - 3 weights fit to 4 points, in-sample; real training needs far more days (see Build)
for d, (p, a) in enumerate(zip(pred, oil_bbl_per_day[1:]), start=2):
    print(f"Day {d}: predicted {p:6.1f} bbl/d   actual {a:6.1f}")

# ---
# Recurrent Neural Network (RNN) — CONCEPT lab · Oil & Gas: Daily oil production rate forecasting for a well
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
