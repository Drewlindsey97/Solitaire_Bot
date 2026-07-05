#!/usr/bin/env python3
import time
import sys
import os
import argparse
from board_reader_lib import (
    read_board, TABLEAU_X, TABLEAU_Y_TOP, COL_WIDTH,
    FREE_CELL_X, FOUNDATION_X, SLOT_Y, SLOT_W, SLOT_H,
    HIDDEN_CARD_H, STEP
)
from freecell_solver import solve, rank_val
import bridge

# ==============================================================================
# 1. CARD MAPPING & SUIT RESOLVER
# ==============================================================================
def assign_pseudo_suits(board):
    """
    Scans the board and assigns alternating suits (Spades/Clubs for Black,
    Hearts/Diamonds for Red) to each card. Modifies the card dictionaries in-place.
    """
    black_counts = {}
    red_counts = {}

    def process_card(card):
        if card is None or card.get("rank") == "?" or card.get("color") == "?":
            return
        rank = card["rank"]
        color = card["color"]
        
        if color == "BLACK":
            count = black_counts.get(rank, 0) + 1
            black_counts[rank] = count
            card["suit"] = "S" if count == 1 else "C"
        elif color == "RED":
            count = red_counts.get(rank, 0) + 1
            red_counts[rank] = count
            card["suit"] = "H" if count == 1 else "D"

    # Scan order: Foundation, Free Cells, then Tableau Columns (exposed cards first)
    for card in board.get("foundation", []):
        process_card(card)
        
    for card in board.get("free_cells", []):
        process_card(card)
        
    for col_idx in range(7):
        col_key = f"col{col_idx}"
        for card in board.get(col_key, []):
            process_card(card)

# ==============================================================================
# 2. COORDINATE RESOLUTION
# ==============================================================================
def get_element_coords(board, item_type, index):
    """
    Calculates the exact center coordinate (x, y) for target slots or top cards.
    """
    if item_type == "col":
        col_cards = board[f"col{index}"]
        num_cards = len(col_cards)
        x_center = TABLEAU_X[index] + COL_WIDTH / 2
        
        if num_cards == 0:
            # Empty column click target
            y_center = TABLEAU_Y_TOP + SLOT_H / 2
        else:
            # Find Y position of the bottom revealed card (exposed card)
            hidden_count = sum(1 for c in col_cards if c.get("rank") == "?")
            revealed_count = num_cards - hidden_count
            y_edge = TABLEAU_Y_TOP + hidden_count * HIDDEN_CARD_H + max(0, revealed_count - 1) * STEP
            y_center = y_edge + SLOT_H / 2
        return int(x_center), int(y_center)
        
    elif item_type == "free":
        x_center = FREE_CELL_X[index] + SLOT_W / 2
        y_center = SLOT_Y + SLOT_H / 2
        return int(x_center), int(y_center)
        
    elif item_type == "found":
        # We only have one foundation pile coordinates defined
        x_center = FOUNDATION_X[0] + SLOT_W / 2
        y_center = SLOT_Y + SLOT_H / 2
        return int(x_center), int(y_center)
        
    return None

# ==============================================================================
# 3. MOVE TRANSLATION TO PHYSICAL GESTURES
# ==============================================================================
def execute_move(board, move, sim_mode=False):
    """
    Translates a solver move into coordinates and triggers the swipe/tap.
    """
    kind = move[0]
    start_coords = None
    end_coords = None

    if kind == "col_to_found":
        _, ci, card = move
        start_coords = get_element_coords(board, "col", ci)
        end_coords = get_element_coords(board, "found", 0)
        
    elif kind == "free_to_found":
        _, card = move
        # Locate the free cell index carrying this card
        fi = None
        for idx, c in enumerate(board["free_cells"]):
            if c and c.get("rank") == card[0] and c.get("suit") == card[1]:
                fi = idx
                break
        if fi is not None:
            start_coords = get_element_coords(board, "free", fi)
            end_coords = get_element_coords(board, "found", 0)
            
    elif kind == "col_to_col":
        _, ci, cj, card = move
        start_coords = get_element_coords(board, "col", ci)
        end_coords = get_element_coords(board, "col", cj)
        
    elif kind == "col_to_free":
        _, ci, card = move
        # Locate first empty free cell
        fi = None
        for idx, c in enumerate(board["free_cells"]):
            if c is None:
                fi = idx
                break
        if fi is not None:
            start_coords = get_element_coords(board, "col", ci)
            end_coords = get_element_coords(board, "free", fi)
            
    elif kind == "free_to_col":
        _, cj, card = move
        fi = None
        for idx, c in enumerate(board["free_cells"]):
            if c and c.get("rank") == card[0] and c.get("suit") == card[1]:
                fi = idx
                break
        if fi is not None:
            start_coords = get_element_coords(board, "free", fi)
            end_coords = get_element_coords(board, "col", cj)

    if start_coords and end_coords:
        x1, y1 = start_coords
        x2, y2 = end_coords
        print(f"[*] Action: Move {kind.replace('_', ' ')}: {move[2] if len(move) > 2 else card} from ({x1}, {y1}) to ({x2}, {y2})")
        if sim_mode:
            print(f"   [Simulation] Would swipe: bridge.swipe({x1}, {y1}, {x2}, {y2})")
        else:
            bridge.swipe(x1, y1, x2, y2)
    else:
        print(f"[Error] Failed to resolve coordinates for move: {move}")

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Automated Solitaire Stash Bot")
    parser.add_argument(
        "--sim", 
        type=str, 
        help="Run in simulation/dry-run mode on a static screenshot file path instead of a live device."
    )
    args = parser.parse_args()

    sim_mode = args.sim is not None
    screenshot_file = args.sim if sim_mode else "live_screen.png"

    if sim_mode:
        print(f"[*] Running in SIMULATION mode on file: {screenshot_file}")
        if not os.path.exists(screenshot_file):
            print(f"[Error] Simulation file '{screenshot_file}' does not exist.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[*] Running in LIVE device mode (RUN_MODE: {bridge.RUN_MODE})")

    # Run loop
    while True:
        if not sim_mode:
            print("[*] Capturing screen...")
            img = bridge.screenshot()
            if img:
                img.save(screenshot_file)
                print(f"[*] Screen saved to {screenshot_file}")
            else:
                print("[Error] Failed to capture screenshot. Retrying in 3 seconds...")
                time.sleep(3.0)
                continue

        print("[*] Analyzing board state...")
        try:
            board = read_board(screenshot_file)
        except Exception as e:
            print(f"[Error] Failed to read board: {e}")
            if sim_mode:
                break
            time.sleep(3.0)
            continue

        # Print confidence or contents
        print("[*] Board Cards Detected:")
        for idx in range(7):
            col_key = f"col{idx}"
            col_info = [f"{c['rank']}({c['color']})" for c in board[col_key]]
            print(f"  col{idx}: {col_info}")
        
        free_info = [f"{c['rank']}({c['color']})" if c else "None" for c in board["free_cells"]]
        found_info = [f"{c['rank']}({c['color']})" if c else "None" for c in board["foundation"]]
        print(f"  free_cells: {free_info}")
        print(f"  foundation: {found_info}")

        # Resolve pseudo-suits
        assign_pseudo_suits(board)

        # Formulate initial solver inputs
        cols = []
        for idx in range(7):
            col = []
            for c in board[f"col{idx}"]:
                if c and "suit" in c:
                    col.append((c["rank"], c["suit"]))
            cols.append(col)

        free = []
        for c in board["free_cells"]:
            if c and "suit" in c:
                free.append((c["rank"], c["suit"]))

        found = {}
        for c in board["foundation"]:
            if c and "suit" in c:
                found[c["suit"]] = rank_val(c["rank"])

        print("[*] Formulated Solver State:")
        print(f"  Cols: {cols}")
        print(f"  Free: {free}")
        print(f"  Found: {found}")

        print("[*] Searching for a path...")
        path, explored, solved = solve(cols, initial_free=free, initial_found=found, time_limit=5.0)

        if solved:
            print(f"[+] FULLY SOLVED in {len(path)} moves (explored {explored} states)")
        else:
            print(f"[-] Time/State limit hit. Best partial path: {len(path)} moves (explored {explored} states)")

        if path:
            print("[*] Recommended move sequence:")
            for idx, mv in enumerate(path[:10]):
                print(f"  {idx+1}. {mv}")
            
            # Execute the first step
            execute_move(board, path[0], sim_mode=sim_mode)
        else:
            print("[*] No moves found. Board might already be solved or no path exists.")

        if sim_mode:
            # Only run once in simulation mode
            break
        
        # Interval wait between cycles
        print("[*] Waiting for UI update (3 seconds)...")
        time.sleep(3.0)

if __name__ == "__main__":
    main()
