import random
import time
from freecell_solver import solve_incrementally, generate_moves, State, RANK_ORDER, SUITS, UNKNOWN

random.seed(42)
deck = [(r, s) for r in RANK_ORDER for s in SUITS]
random.shuffle(deck)

# Classic Klondike deal: columns of 1,2,3,...,7 cards (28 dealt), only the
# last card in each column face-up - the rest is UNKNOWN (face-down) until
# the game reveals it. The remaining 24 cards start in stock.
cols = [[] for _ in range(7)]
deck_iter = iter(deck)
for col_idx in range(7):
    for card_idx in range(col_idx + 1):
        card = next(deck_iter)
        is_last = card_idx == col_idx
        cols[col_idx].append(card if is_last else UNKNOWN)
stock_total = 52 - sum(len(c) for c in cols)

# solve()'s best_path only records strict heuristic improvement, so a
# legal move that merely ties the heuristic (common with only one column
# move available before needing to draw) won't show up in path even
# though it was found and is perfectly legal - check generate_moves()
# directly on the initial state so this test isn't blind to that.
initial_state = State(cols, [], stock_total, stock_total, {})
initial_moves = generate_moves(initial_state)
print(f"Legal moves from the initial deal: {initial_moves}")

# solve_incrementally (not solve) is what the real bot calls, because a
# single solve() call fundamentally can't play through a whole Klondike
# game offline: once every column-top/waste move is exhausted, "draw" is
# the only move left, and a hypothetical draw within search can only ever
# reveal an UNKNOWN placeholder (see freecell_solver's module docstring) -
# real new information only ever enters via a live re-perceive after a
# real tap. So this test can only exercise what a *single* live round
# looks like: solved/exhausted here means "nothing more knowable without
# a live board read", not a proof about the real deal's winnability.
start = time.time()
path, explored, solved, status = solve_incrementally(
    cols, initial_stock_remaining=stock_total, initial_stock_total=stock_total)
elapsed = time.time() - start

if solved:
    print(f"FULLY SOLVED in {len(path)} moves (explored {explored} states, {elapsed:.1f}s)")
else:
    status_labels = {
        "exhausted": "PROVEN UNSOLVABLE FROM HERE",
        "stuck": "NO FURTHER PROGRESS WITHOUT A LIVE BOARD READ (expected offline - draw/redeal can't reveal real cards without one)",
        "capped": "MEMORY CAP HIT (not proven either way)",
        "timeout": "TIME LIMIT HIT (not proven either way)",
    }
    label = status_labels.get(status, status.upper())
    print(f"{label} (explored {explored} states, {elapsed:.1f}s)")
    print(f"\nCommitted {len(path)} move(s) before stopping:")
    for m in path[:15]:
        print(" ", m)
