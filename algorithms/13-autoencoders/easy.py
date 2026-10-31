import numpy as np

# 8 healthy readings from a gas compressor, already scaled to 0-1:
# columns = [vibration, temperature, pressure, flow]
# 🐍 PY: on a healthy machine all 4 sensors rise and fall together with the load
healthy = np.array([
    [0.20, 0.22, 0.21, 0.19],
    [0.40, 0.41, 0.39, 0.42],
    [0.55, 0.53, 0.56, 0.54],
    [0.70, 0.72, 0.69, 0.71],
    [0.30, 0.29, 0.31, 0.30],
    [0.85, 0.84, 0.86, 0.85],
    [0.60, 0.62, 0.59, 0.61],
    [0.45, 0.44, 0.46, 0.45],
])

# 🧠 ML: an autoencoder squeezes inputs to a smaller 'code' and rebuilds them
# 🧠 ML: here the encoder is one weight vector: code = average of the 4 sensors (1 number = 'load')
encoder = np.array([0.25, 0.25, 0.25, 0.25])
# 🧠 ML: the decoder spreads the code back out: healthy machine -> every sensor near the load level
decoder = np.array([1.0, 1.0, 1.0, 1.0])

def reconstruct(x):
    # 🐍 PY: x @ encoder is a dot product -> one number; times decoder broadcasts back to 4 numbers
    code = x @ encoder
    return code, code * decoder

# 🐍 PY: list comprehension computes the reconstruction error for every healthy row
errors = [np.mean((row - reconstruct(row)[1]) ** 2) for row in healthy]
# 🧠 ML: alarm threshold = a bit above the worst error seen on healthy data
threshold = max(errors) * 3
print("Healthy reconstruction errors:", np.round(errors, 5))
print(f"Alarm threshold: {threshold:.5f}\n")

# New readings: one normal, one where vibration spikes but load is normal (bearing wear)
# 🐍 PY: a dict maps a description to the reading
new = {
    "normal load": np.array([0.50, 0.51, 0.49, 0.50]),
    "bearing wear": np.array([0.90, 0.50, 0.49, 0.50]),
}
# 🐍 PY: .items() yields (key, value) pairs we can unpack in the loop header
for name, x in new.items():
    code, rebuilt = reconstruct(x)
    err = np.mean((x - rebuilt) ** 2)
    # 🐍 PY: conditional expression picks a label in one line
    status = "ANOMALY" if err > threshold else "ok"
    print(f"{name:<13} code={code:.3f} rebuilt={np.round(rebuilt, 2)} error={err:.5f} -> {status}")
    # 🧠 ML: the sensor with the largest error tells the engineer where to look
    sensors = ["vibration", "temperature", "pressure", "flow"]
    print(f"{'':<13} biggest error on: {sensors[np.argmax((x - rebuilt) ** 2)]}")

# ---
# Autoencoders — EASY lab · Oil & Gas: Gas compressor sensor anomaly detection for predictive maintenance
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
