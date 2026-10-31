# Easy: find hail-claim hotspots by hand with DBSCAN's two rules (eps, min_pts)
import numpy as np

# 🐍 PY: a dict mapping claim id -> (x_km, y_km) position east/north of the city centre
claims = {
    "C1": (1.0, 1.0), "C2": (1.5, 1.2), "C3": (1.2, 0.6), "C4": (0.8, 1.4),   # neighbourhood A
    "C5": (6.0, 6.0), "C6": (6.4, 6.3), "C7": (5.7, 6.5), "C8": (6.2, 5.6),   # neighbourhood B
    "C9": (3.5, 9.0),                                                          # lone claim far away
    "C10": (9.5, 1.0),                                                         # another lone claim
    "C11": (2.2, 1.2),                                                         # edge of A
}
EPS = 1.0      # km - "close enough to be the same hotspot"
MIN_PTS = 3    # a point plus at least 2 neighbours makes a dense core

# 🐍 PY: list() of dict keys gives a fixed order; np.array of values gives an N x 2 matrix
ids = list(claims)
pts = np.array(list(claims.values()))

# 🧠 ML: step 1 - for each claim, count how many claims (itself included) lie within eps
# 🐍 PY: broadcasting pts[:, None] - pts[None, :] gives every pairwise difference at once
dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
neighbours = {ids[i]: [ids[j] for j in range(len(ids)) if dist[i, j] <= EPS] for i in range(len(ids))}
for cid, nb in neighbours.items():
    # 🧠 ML: a CORE point has >= MIN_PTS points inside its eps circle
    kind = "CORE" if len(nb) >= MIN_PTS else "not core (border if a core point reaches it)"
    print(f"{cid:<4} neighbours={nb}  -> {kind}")

# 🧠 ML: step 2 - grow clusters from core points; anything reachable joins the cluster
label = {cid: None for cid in ids}
cluster_id = 0
for cid in ids:
    if label[cid] is not None or len(neighbours[cid]) < MIN_PTS:
        continue
    cluster_id += 1
    # 🐍 PY: a list used as a queue - pop() takes the next point to expand
    queue = [cid]
    while queue:
        p = queue.pop()
        if label[p] is not None:
            continue
        label[p] = cluster_id
        if len(neighbours[p]) >= MIN_PTS:
            queue.extend(neighbours[p])

# 🧠 ML: step 3 - anything never reached is NOISE (label -1): flag for individual review
# 🐍 PY: dict comprehension with a conditional value
label = {c: (l if l is not None else -1) for c, l in label.items()}
print("\nFinal labels:", label)
for k in sorted(set(label.values())):
    members = [c for c, l in label.items() if l == k]
    name = "NOISE - review individually" if k == -1 else f"Hotspot {k} - send an adjuster team"
    print(f"  {name}: {members}")

# ---
# DBSCAN — EASY lab · Insurance: Catastrophe claims hotspots after a hailstorm or tornado
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
