import json
from freecell_solver import solve

with open("board_state.json") as f:
    board = json.load(f)

# assign pseudo-suits based on color, since we don't have exact suit detection yet
def assign_pseudo_suits(board):
    black_toggle = {}
    red_toggle = {}
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
                black_toggle[rank] = not black_toggle.get(rank, False)
                suit = "S" if black_toggle[rank] else "C"
            else:
                red_toggle[rank] = not red_toggle.get(rank, False)
                suit = "H" if red_toggle[rank] else "D"
            col.append((rank, suit))
        cols.append(col)
    return cols

cols = assign_pseudo_suits(board)

print("Board loaded for solving:")
for i, c in enumerate(cols):
    print(f"  col{i}: {c}")

path, explored, solved = solve(cols, time_limit=8.0)

print(f"\n{'FULLY SOLVED' if solved else 'Best plan found'}: {len(path)} moves (explored {explored} states)")
print("\nSuggested moves:")
for m in path[:15]:
    print(" ", m)
