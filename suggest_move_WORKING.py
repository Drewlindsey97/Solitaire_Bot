import json

with open("board_state.json") as f:
    board = json.load(f)

RANK_ORDER = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

def rank_value(r):
    return RANK_ORDER.index(r) if r in RANK_ORDER else -1

def opposite_color(c1, c2):
    return {"RED","BLACK"} == {c1, c2}

# free cells: from the screenshot, 4 total slots, currently 2 filled-visible + "2 Free" means 2 open
FREE_CELLS_OPEN = 2

columns = {k: v for k, v in board.items() if k.startswith("col")}

moves = []

for src_name, src_cards in columns.items():
    if not src_cards:
        continue
    src_card = src_cards[-1]  # exposed/bottom-most card = the one you can actually move
    src_rank = rank_value(src_card["rank"])
    src_color = src_card["color"]

    if src_rank == -1 or src_color == "?":
        continue  # unreliable read, skip

    # try placing on another column
    for dst_name, dst_cards in columns.items():
        if dst_name == src_name:
            continue

        if not dst_cards:
            moves.append({
                "from": src_name,
                "to": dst_name,
                "card": f"{src_card['rank']} ({src_color})",
                "reason": "move to empty column"
            })
            continue

        dst_card = dst_cards[-1]
        dst_rank = rank_value(dst_card["rank"])
        dst_color = dst_card["color"]

        if dst_rank == -1 or dst_color == "?":
            continue

        if dst_rank == src_rank + 1 and opposite_color(src_color, dst_color):
            moves.append({
                "from": src_name,
                "to": dst_name,
                "card": f"{src_card['rank']} ({src_color})",
                "onto": f"{dst_card['rank']} ({dst_color})",
                "reason": "valid tableau sequence move"
            })

# prefer real sequence moves over empty-column dumps
moves.sort(key=lambda m: 0 if "onto" in m else 1)

print(f"Free cells open: {FREE_CELLS_OPEN}\n")

if not moves:
    print("No immediate tableau moves found.")
    if FREE_CELLS_OPEN > 0:
        print("Consider moving a card to an open free cell to unblock a column.")
else:
    print("Suggested moves (best first):\n")
    for m in moves[:5]:
        if "onto" in m:
            print(f"  Move {m['card']} from {m['from']} -> {m['to']} (onto {m['onto']})")
        else:
            print(f"  Move {m['card']} from {m['from']} -> {m['to']} ({m['reason']})")
