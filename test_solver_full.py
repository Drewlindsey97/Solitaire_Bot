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
path, explored, solved, status = solve(cols)
elapsed = time.time() - start

if solved:
    print(f"FULLY SOLVED in {len(path)} moves (explored {explored} states, {elapsed:.1f}s)")
else:
    status_labels = {
        "exhausted": "PROVEN UNSOLVABLE",
        "capped": "MEMORY CAP HIT (not proven either way)",
        "timeout": "TIME LIMIT HIT (not proven either way)",
    }
    label = status_labels.get(status, status.upper())
    print(f"{label} (explored {explored} states, {elapsed:.1f}s)")
    print("\nBest partial line found:")
    for m in path[:10]:
        print(" ", m)
