# One SARSA update by hand: a warehouse robot next to the forklift lane (Easy)
# 🐍 PY: a dict keyed by (state, action) tuples is a perfectly good tiny Q-table
Q = {
    ("aisle_B2", "east"): -3.0,    # step along the edge of the forklift lane (short route)
    ("aisle_B2", "north"): -4.0,   # step up to the safe aisle (one extra move)
    ("aisle_B3", "east"): -2.0,    # keep going along the edge
    ("aisle_B3", "north"): -3.5,   # move away from the lane
    ("aisle_B3", "south"): -20.0,  # step INTO the forklift lane: collision risk

}
alpha, gamma, epsilon = 0.5, 1.0, 0.1
print("Q-table before:")
# 🐍 PY: .items() gives (key, value) pairs; the key itself is a tuple we unpack
for (s, a), v in Q.items():
    print(f"  Q({s}, {a:5s}) = {v:6.2f}")

# The robot is in aisle_B2, takes 'east', pays -1 for the move, lands in aisle_B3.
s, a, r, s_next = "aisle_B2", "east", -1.0, "aisle_B3"
# 🧠 ML: SARSA is ON-POLICY - it uses the action the robot will ACTUALLY take next (maybe exploratory)
# Suppose epsilon-greedy explored (10% chance) and picked 'south' - into the forklift lane
a_next = "south"
print(f"\nExperience: S={s}, A={a}, R={r}, S'={s_next}, A'={a_next}  <- the 5 letters S-A-R-S-A")

# 🧠 ML: SARSA target uses Q(S', A') of the action really chosen
sarsa_target = r + gamma * Q[(s_next, a_next)]
# 🧠 ML: Q-learning target would instead use the BEST next action, max over a' of Q(S', a')
# 🐍 PY: a generator expression filtered by an if, fed straight into max()
q_target = r + gamma * max(v for (st, _), v in Q.items() if st == s_next)
print(f"SARSA target      = {r} + {gamma} * Q({s_next}, {a_next}) = {sarsa_target:.2f}")
print(f"Q-learning target = {r} + {gamma} * max Q({s_next}, .)  = {q_target:.2f}")

# 🧠 ML: move the old estimate a fraction alpha toward the target
td_error = sarsa_target - Q[(s, a)]
new_q = Q[(s, a)] + alpha * td_error
print(f"\nTD error = {sarsa_target:.2f} - ({Q[(s, a)]:.2f}) = {td_error:.2f}")
print(f"Q({s}, east): {Q[(s, a)]:.2f} -> {new_q:.2f}   (Q-learning would give {Q[(s, a)] + alpha * (q_target - Q[(s, a)]):.2f})")
Q[(s, a)] = new_q
print("\nTakeaway: because SARSA 'remembers' that its own exploration sometimes goes wrong,")
print("it rates risky moves lower than Q-learning does - that is why it learns safer routes.")

# ---
# SARSA — EASY lab · Supply Chain: Warehouse robot learning a safe route past a forklift lane
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
