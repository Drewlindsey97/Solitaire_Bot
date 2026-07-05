import random
import time
from freecell_solver import solve, RANK_ORDER, SUITS

random.seed(42)
deck = [(r, s) for r in RANK_ORDER for s in SUITS]
random.shuffle(deck)

cols = [[] for _ in range(7)]
for i, card in enumerate(deck):
    cols[i % 7].append(card)

start = time.time()
path, explored, solved = solve(cols, time_limit=8.0)
elapsed = time.time() - start

if solved:
    print(f"FULLY SOLVED in {len(path)} moves (explored {explored} states, {elapsed:.1f}s)")
else:
    print(f"Time limit reached. Best partial plan: {len(path)} moves toward solution")
    print(f"(explored {explored} states, {elapsed:.1f}s)")
    print("\nSuggested next moves (best found so far):")
    for m in path[:10]:
        print(" ", m)
