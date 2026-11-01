# Concept: estimate transit days for a new shipment by finding the most similar past shipments
import math

# 🐍 PY: a list of dicts - each dict is one past shipment with named fields
past = [
    {"id": "S1", "distance_mi": 300,  "weight_lb": 500,  "transit_days": 2},
    {"id": "S2", "distance_mi": 450,  "weight_lb": 1200, "transit_days": 3},
    {"id": "S3", "distance_mi": 900,  "weight_lb": 800,  "transit_days": 4},
    {"id": "S4", "distance_mi": 1200, "weight_lb": 2000, "transit_days": 5},
    {"id": "S5", "distance_mi": 1500, "weight_lb": 600,  "transit_days": 5},
    {"id": "S6", "distance_mi": 2000, "weight_lb": 1500, "transit_days": 7},
    {"id": "S7", "distance_mi": 2400, "weight_lb": 3000, "transit_days": 8},
    {"id": "S8", "distance_mi": 600,  "weight_lb": 2500, "transit_days": 3},
]
new = {"distance_mi": 1000, "weight_lb": 1000}

# 🧠 ML: miles and pounds have different ranges, so divide each by its max to put both on 0-1
# 🐍 PY: max() with a generator expression finds the biggest value without building a list
max_d = max(s["distance_mi"] for s in past)
max_w = max(s["weight_lb"] for s in past)

def distance(a, b):
    """Euclidean distance between two shipments on the scaled features."""
    dd = (a["distance_mi"] - b["distance_mi"]) / max_d
    dw = (a["weight_lb"] - b["weight_lb"]) / max_w
    # 🧠 ML: straight-line (Euclidean) distance = sqrt of the summed squared differences
    return math.sqrt(dd ** 2 + dw ** 2)

# 🐍 PY: add a new key to every dict in place
for s in past:
    s["dist"] = distance(new, s)

# 🐍 PY: sorted() with key=lambda sorts dicts by one field
ranked = sorted(past, key=lambda s: s["dist"])
print("Past shipments ranked by similarity to the new one:")
for s in ranked:
    print(f"  {s['id']}  {s['distance_mi']:>5} mi  {s['weight_lb']:>5} lb  "
          f"days={s['transit_days']}  distance={s['dist']:.3f}")

# 🧠 ML: k-NN prediction = average the target of the k closest neighbours
for k in (1, 3, 5):
    # 🐍 PY: slicing [:k] takes the first k items of the sorted list
    nearest = ranked[:k]
    avg = sum(s["transit_days"] for s in nearest) / k
    ids = ", ".join(s["id"] for s in nearest)
    print(f"k={k}: neighbours [{ids}] -> predicted transit {avg:.2f} days")

# 🧠 ML: distance-weighted version - closer neighbours get a bigger vote (weight = 1 / distance)
k = 3
weights = [1 / ranked[i]["dist"] for i in range(k)]
weighted = sum(w * ranked[i]["transit_days"] for i, w in enumerate(weights)) / sum(weights)
print(f"k=3 distance-weighted -> {weighted:.2f} days")

# ---
# k-Nearest Neighbors (k-NN) — CONCEPT lab · Supply Chain: Delivery transit-day estimation from similar past shipments
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
