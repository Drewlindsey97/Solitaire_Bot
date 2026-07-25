import json
from collections import Counter
from freecell_solver import solve_incrementally

with open("board_state.json") as f:
    board = json.load(f)

# Assign pseudo-suits based on color, since we don't have exact suit
# detection yet: the 1st BLACK read of a given rank is called S, the 2nd
# is called C (same idea for RED -> H/D). Each rank only has 2 real cards
# of a given color (e.g. only KS and KC exist), so a 3rd+ read of the same
# rank+color is impossible and must be an OCR misread - it's dropped
# rather than silently wrapping the counter back to the 1st suit, which
# would fabricate a duplicate card. A duplicate is worse than a dropped
# card: once one copy reaches that suit's foundation, the solver's found-
# tracking treats the suit as complete, so the leftover duplicate can
# never legally reach the foundation - permanently corrupting the board
# into something that looks solvable but provably isn't.
def assign_pseudo_suits(board):
    black_count = Counter()
    red_count = Counter()
    cols = []
    for key in sorted(board.keys()):
        if not key.startswith("col"):
            continue
        col = []
        for card in board[key]:
            rank = card["rank"]
            color = card["color"]
            if rank == "?" or color == "?":
                continue  # skip unreliable reads
            if color == "BLACK":
                if black_count[rank] >= 2:
                    print(f"  WARNING: dropping extra BLACK {rank} read - "
                          f"only 2 black suits exist per rank, this is "
                          f"almost certainly an OCR misread")
                    continue
                black_count[rank] += 1
                suit = "S" if black_count[rank] == 1 else "C"
            else:
                if red_count[rank] >= 2:
                    print(f"  WARNING: dropping extra RED {rank} read - "
                          f"only 2 red suits exist per rank, this is "
                          f"almost certainly an OCR misread")
                    continue
                red_count[rank] += 1
                suit = "H" if red_count[rank] == 1 else "D"
            col.append((rank, suit))
        cols.append(col)
    return cols

cols = assign_pseudo_suits(board)

print("Board loaded for solving:")
for i, c in enumerate(cols):
    print(f"  col{i}: {c}")

# Runs as several short, hard-bounded rounds (round_limit seconds each)
# instead of one long uncertain solve, so this always returns within
# total_budget seconds no matter how hard the board is - see
# solve_incrementally()'s docstring in freecell_solver.py.
path, explored, solved, status = solve_incrementally(
    cols, total_budget=150.0, round_limit=20.0)

label = {
    "solved": "FULLY SOLVED",
    "exhausted": "PROVEN UNSOLVABLE",
    "stuck": "STUCK - no further progress found, not proven either way",
    "timeout": "TIME BUDGET HIT - best partial line shown, not proven either way",
}[status]
print(f"\n{label}: {len(path)} moves (explored {explored} states)")
print("\nSuggested moves:")
for m in path[:15]:
    print(" ", m)
