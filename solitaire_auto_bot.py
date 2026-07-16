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
from freecell_solver import State, solve, rank_val

from monte_carlo_solver import (
    choose_move_monte_carlo,
    print_statistics,
)

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


def card_color(suit):
    return "RED" if suit in ("H", "D") else "BLACK"


def apply_move_to_board(board, move):
    """
    Mirrors freecell_solver.apply_move, but mutates the CV-read `board` dict
    (lists of card dicts) instead of the solver's compact (rank, suit)
    tuples. This keeps `board` in sync with the physical game after we
    execute a move without paying for a fresh screenshot + CV read, so a
    whole batch of moves can be run per screen-read cycle instead of one.
    """
    kind = move[0]

    def find_free(rank, suit):
        for idx, c in enumerate(board["free_cells"]):
            if c and c.get("rank") == rank and c.get("suit") == suit:
                return idx
        return None

    def first_empty_free():
        for idx, c in enumerate(board["free_cells"]):
            if c is None:
                return idx
        return None

    if kind == "col_to_found":
        _, ci, card = move
        board[f"col{ci}"].pop()
    elif kind == "free_to_found":
        _, card = move
        fi = find_free(card[0], card[1])
        if fi is not None:
            board["free_cells"][fi] = None
    elif kind == "col_to_col":
        _, ci, cj, card = move
        board[f"col{ci}"].pop()
        board[f"col{cj}"].append({"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0})
    elif kind == "col_to_free":
        _, ci, card = move
        board[f"col{ci}"].pop()
        fi = first_empty_free()
        if fi is not None:
            board["free_cells"][fi] = {"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0}
    elif kind == "free_to_col":
        _, cj, card = move
        fi = find_free(card[0], card[1])
        if fi is not None:
            board["free_cells"][fi] = None
        board[f"col{cj}"].append({"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0})


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

    # Sanity-check: for moves that pop the top of a tableau column, make sure
    # the physical top card actually matches what the solver believes is
    # there. Board state and solver state can diverge (e.g. a revealed card
    # whose suit couldn't be read gets dropped from the solver's view), and
    # blindly swiping in that case would drag the wrong real card.
    if kind in ("col_to_found", "col_to_col", "col_to_free"):
        ci = move[1]
        physical_col = board[f"col{ci}"]
        top = physical_col[-1] if physical_col else None
        card = move[-1]
        if not top or top.get("rank") != card[0] or top.get("suit") != card[1]:
            print(f"[Warn] Skipping move {move}: physical top of col{ci} "
                  f"({top}) does not match solver's expected card {card}. "
                  f"Board read is stale or ambiguous; will re-read next cycle.")
            return False

    if start_coords and end_coords:
        x1, y1 = start_coords
        x2, y2 = end_coords
        if kind in ("col_to_found", "free_to_found"):
            print(f"[*] Action: Tap to Foundation: {move[2] if len(move) > 2 else card} at ({x1}, {y1})")
            if sim_mode:
                print(f"   [Simulation] Would tap: bridge.tap({x1}, {y1})")
            else:
                bridge.tap(x1, y1)
        else:
            print(f"[*] Action: Move {kind.replace('_', ' ')}: {move[2] if len(move) > 2 else card} from ({x1}, {y1}) to ({x2}, {y2})")
            if sim_mode:
                print(f"   [Simulation] Would swipe: bridge.swipe({x1}, {y1}, {x2}, {y2})")
            else:
                bridge.swipe(x1, y1, x2, y2)
        return True
    else:
        print(f"[Error] Failed to resolve coordinates for move: {move}")
        return False

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
    parser.add_argument(
        "--moves-per-cycle",
        type=int,
        default=5,
        help="Max solved moves to execute before re-capturing the screen and re-solving "
             "(0 = execute the entire computed path in one go). Between moves in a batch "
             "we update our local board model instead of re-reading the screen, so higher "
             "values are faster but rely on the physical game matching our model exactly; "
             "lower values re-verify against a fresh screenshot more often. Default: 5.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Seconds to wait after a batch of moves for the UI to settle before the next "
             "screen capture. Default: 1.5.",
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
                if c and c.get("rank") == "?" and c.get("color") == "?":
                    # genuinely face-down card, always a leading run - not part
                    # of the playable stack yet, safe to skip
                    continue
                if c and "suit" in c:
                    col.append((c["rank"], c["suit"]))
                else:
                    # revealed card but rank/suit couldn't be resolved - we
                    # can't trust our read of this card or anything above it
                    # in the stack, so stop here instead of silently
                    # continuing past it (which would shift the rest of the
                    # column up and make the solver think a buried card is
                    # the exposed one)
                    print(f"[Warn] col{idx}: unresolved card {c!r}; truncating column read here")
                    break
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

        if args.solver == "monte-carlo":
            print("[*] Running Monte Carlo move search...")
            state = State(cols, free, found)
            move, statistics = choose_move_monte_carlo(state)
            print_statistics(statistics)

            if move:
                execute_move(board, move, sim_mode=sim_mode)
            else:
                print("[*] No moves found. Board might already be solved or no path exists.")
        else:
            print("[*] Searching for a path...")
            path, explored, solved = solve(cols, initial_free=free, initial_found=found, time_limit=5.0)

        if path:
            batch = path if args.moves_per_cycle <= 0 else path[:args.moves_per_cycle]
            print(f"[*] Executing {len(batch)} move(s) this cycle:")
            for idx, mv in enumerate(batch):
                print(f"  {idx+1}. {mv}")
                ok = execute_move(board, mv, sim_mode=sim_mode)
                if not ok:
                    print("[*] Stopping batch early; will re-read the board next cycle.")
                    break
                # Keep our local board model in sync so the next move's
                # coordinates can be computed without re-reading the screen.
                apply_move_to_board(board, mv)
        else:
            print("[*] No moves found. Board might already be solved or no path exists.")

        if sim_mode:
            # Only run once in simulation mode
            break
        
        # Interval wait between cycles
        print(f"[*] Waiting for UI update ({args.interval}s)...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
