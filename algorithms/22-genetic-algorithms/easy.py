# Genetic Algorithm by hand: find a short route for one truck visiting 5 customer stops
import random
import math
import itertools

random.seed(42)
# 🐍 PY: a dict maps each stop name to its (x, y) location in miles from the depot
stops = {"Depot": (0, 0), "A": (2, 6), "B": (5, 2), "C": (8, 7), "D": (6, 9), "E": (1, 3)}
customers = ["A", "B", "C", "D", "E"]


def route_miles(route):
    # 🐍 PY: list concatenation adds the depot at the start and end of the trip
    path = ["Depot"] + route + ["Depot"]
    # 🐍 PY: zip(path, path[1:]) pairs each stop with the next one
    return sum(math.dist(stops[a], stops[b]) for a, b in zip(path, path[1:]))


def fitness(route):
    # 🧠 ML: fitness = "how good" a route is; shorter route -> higher fitness
    return 1 / route_miles(route)


def crossover(p1, p2):
    # 🧠 ML: crossover - keep the first half of parent 1, fill the rest in parent 2's order
    child = p1[:2]
    # 🐍 PY: list comprehension keeps only stops not already in the child (no duplicate visits)
    child += [s for s in p2 if s not in child]
    return child


def mutate(route, rate=0.3):
    # 🧠 ML: mutation - occasionally swap two stops to explore new routes
    r = route.copy()
    if random.random() < rate:
        i, j = random.sample(range(len(r)), 2)
        # 🐍 PY: tuple unpacking swaps two list items in one line
        r[i], r[j] = r[j], r[i]
    return r


# 🧠 ML: population = a set of candidate routes; start with 8 random shuffles
# 🐍 PY: random.sample(list, k=len) returns a shuffled copy
population = [random.sample(customers, k=5) for _ in range(8)]
for gen in range(10):
    # 🐍 PY: sorted with key= orders routes by fitness, best first
    population = sorted(population, key=fitness, reverse=True)
    best = population[0]
    print(f"Gen {gen}: best={'-'.join(best)}  miles={route_miles(best):5.2f}  "
          f"worst={route_miles(population[-1]):5.2f}")
    # 🧠 ML: selection - the top 4 routes become parents; the rest are discarded
    parents = population[:4]
    children = [mutate(crossover(random.choice(parents), random.choice(parents))) for _ in range(4)]
    population = parents + children

print("\nGA route : Depot-" + "-".join(population[0]) + f"-Depot  {route_miles(population[0]):.2f} mi")
# 🧠 ML: with only 5 stops we can check all 120 orders; at 15+ stops brute force is impossible
best = min(itertools.permutations(customers), key=lambda r: route_miles(list(r)))
print("Best route: Depot-" + "-".join(best) + f"-Depot  {route_miles(list(best)):.2f} mi")

# ---
# Genetic Algorithms — EASY lab · Supply Chain: Delivery route optimization for one truck
# Part of 30 AI Algorithms: https://github.com/EdwardSuresh/30-ai-algorithms
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
