#!/usr/bin/env python3
import time
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
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

from logcat_monitor import LogcatMonitor, default_logcat_path
from session_logger import SessionLogger, default_session_log_path

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
def execute_move(board, move, sim_mode=False, event_logger=None):
    """
    Translates a solver move into coordinates and triggers the swipe/tap.
    """
    kind = move[0]
    card = move[-1]
    emit = event_logger or (lambda event_name, **data: None)
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
            emit(
                "move_rejected",
                move=move,
                reason="physical_top_mismatch",
                physical_top=top,
                expected_card=card,
            )
            return False

    if start_coords and end_coords:
        x1, y1 = start_coords
        x2, y2 = end_coords
        if kind in ("col_to_found", "free_to_found"):
            print(f"[*] Action: Tap to Foundation: {card} at ({x1}, {y1})")
            emit(
                "gesture_planned",
                move=move,
                gesture="tap",
                start={"x": x1, "y": y1},
                simulation=sim_mode,
            )
            if sim_mode:
                print(f"   [Simulation] Would tap: bridge.tap({x1}, {y1})")
            else:
                bridge.tap(x1, y1)
        else:
            print(
                f"[*] Action: Move {kind.replace('_', ' ')}: {card} "
                f"from ({x1}, {y1}) to ({x2}, {y2})"
            )
            emit(
                "gesture_planned",
                move=move,
                gesture="swipe",
                start={"x": x1, "y": y1},
                end={"x": x2, "y": y2},
                simulation=sim_mode,
            )
            if sim_mode:
                print(f"   [Simulation] Would swipe: bridge.swipe({x1}, {y1}, {x2}, {y2})")
            else:
                bridge.swipe(x1, y1, x2, y2)
        emit("gesture_dispatched", move=move, simulation=sim_mode)
        return True
    else:
        print(f"[Error] Failed to resolve coordinates for move: {move}")
        emit("move_rejected", move=move, reason="coordinate_resolution_failed")
        return False

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def count_unresolved_cards(board):
    unresolved = 0
    for idx in range(7):
        for card in board.get(f"col{idx}", []):
            if card and (card.get("rank") == "?" or card.get("color") == "?"):
                unresolved += 1
    for area in ("free_cells", "foundation"):
        for card in board.get(area, []):
            if card and (card.get("rank") == "?" or card.get("color") == "?"):
                unresolved += 1
    return unresolved


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
    parser.add_argument(
        "--solver",
        choices=["search", "monte-carlo"],
        default="search",
        help="Choose the move-selection engine. Default: search."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write structured Solvitaire events as JSONL to this file."
    )
    parser.add_argument(
        "--logcat",
        action="store_true",
        help="Capture Android logcat in the background for this session."
    )
    parser.add_argument(
        "--logcat-file",
        type=str,
        help="Raw logcat output file. Default: logs/logcat_<timestamp>.log."
    )
    parser.add_argument(
        "--logcat-package",
        type=str,
        help="Android package to restrict logcat by process ID, when the app is running."
    )
    parser.add_argument(
        "--logcat-filter",
        action="append",
        default=[],
        help="Regex for logcat lines to keep. Repeat the option to add more filters."
    )
    parser.add_argument(
        "--clear-logcat",
        action="store_true",
        help="Clear the Android log buffer before capture starts."
    )
    args = parser.parse_args()

    sim_mode = args.sim is not None
    screenshot_file = args.sim if sim_mode else "live_screen.png"

    # Enabling logcat also enables a structured session log unless the user
    # already supplied an explicit JSONL path.
    session_log_path = Path(args.log_file) if args.log_file else None
    if args.logcat and session_log_path is None:
        session_log_path = default_session_log_path()

    session_logger = SessionLogger(session_log_path) if session_log_path else None

    def log_event(event_name, **data):
        if session_logger is not None:
            session_logger.event(event_name, **data)

    logcat_monitor = None
    logcat_path = None
    if args.logcat:
        logcat_path = Path(args.logcat_file) if args.logcat_file else default_logcat_path()
        logcat_monitor = LogcatMonitor(
            output_path=logcat_path,
            run_mode=bridge.RUN_MODE,
            package=args.logcat_package,
            include_patterns=args.logcat_filter,
            clear_first=args.clear_logcat,
        )

    if sim_mode:
        print(f"[*] Running in SIMULATION mode on file: {screenshot_file}")
        if not os.path.exists(screenshot_file):
            print(f"[Error] Simulation file '{screenshot_file}' does not exist.", file=sys.stderr)
            if session_logger is not None:
                session_logger.close()
            sys.exit(1)
    else:
        print(f"[*] Running in LIVE device mode (RUN_MODE: {bridge.RUN_MODE})")

    cycle_number = 0
    interrupted = False
    logcat_started = False

    try:
        if logcat_monitor is not None:
            if logcat_monitor.start():
                logcat_started = True
                pid_text = ", ".join(logcat_monitor.resolved_pids) or "all processes"
                print(f"[*] Logcat capture started: {logcat_path} ({pid_text})")
                log_event(
                    "logcat_started",
                    path=logcat_path,
                    package=args.logcat_package,
                    pids=logcat_monitor.resolved_pids,
                    filters=args.logcat_filter,
                )
            else:
                print(f"[Warn] Logcat capture could not start: {logcat_monitor.error}")
                log_event("logcat_start_failed", error=logcat_monitor.error)

        log_event(
            "session_started",
            mode="simulation" if sim_mode else "live",
            solver=args.solver,
            screenshot=screenshot_file,
            run_mode=bridge.RUN_MODE,
            moves_per_cycle=args.moves_per_cycle,
            interval_seconds=args.interval,
            started_local=datetime.now().isoformat(),
        )

        while True:
            cycle_number += 1
            cycle_started = time.perf_counter()
            log_event("cycle_started", cycle=cycle_number)

            if not sim_mode:
                print("[*] Capturing screen...")
                capture_started = time.perf_counter()
                img = bridge.screenshot()
                capture_seconds = time.perf_counter() - capture_started

                if img:
                    img.save(screenshot_file)
                    print(f"[*] Screen saved to {screenshot_file}")
                    log_event(
                        "screenshot_captured",
                        cycle=cycle_number,
                        path=screenshot_file,
                        duration_seconds=capture_seconds,
                    )
                else:
                    print("[Error] Failed to capture screenshot. Retrying in 3 seconds...")
                    log_event(
                        "screenshot_failed",
                        cycle=cycle_number,
                        duration_seconds=capture_seconds,
                    )
                    time.sleep(3.0)
                    continue

            print("[*] Analyzing board state...")
            board_started = time.perf_counter()
            try:
                board = read_board(screenshot_file)
            except Exception as exc:
                board_seconds = time.perf_counter() - board_started
                print(f"[Error] Failed to read board: {exc}")
                log_event(
                    "board_read_failed",
                    cycle=cycle_number,
                    error=str(exc),
                    duration_seconds=board_seconds,
                )
                if sim_mode:
                    break
                time.sleep(3.0)
                continue

            board_seconds = time.perf_counter() - board_started
            unresolved_cards = count_unresolved_cards(board)
            log_event(
                "board_read",
                cycle=cycle_number,
                duration_seconds=board_seconds,
                unresolved_cards=unresolved_cards,
                board=board,
            )

            print("[*] Board Cards Detected:")
            for idx in range(7):
                col_key = f"col{idx}"
                col_info = [f"{c['rank']}({c['color']})" for c in board[col_key]]
                print(f"  col{idx}: {col_info}")

            free_info = [f"{c['rank']}({c['color']})" if c else "None" for c in board["free_cells"]]
            found_info = [f"{c['rank']}({c['color']})" if c else "None" for c in board["foundation"]]
            print(f"  free_cells: {free_info}")
            print(f"  foundation: {found_info}")

            assign_pseudo_suits(board)

            cols = []
            truncated_columns = []
            for idx in range(7):
                col = []
                for card in board[f"col{idx}"]:
                    if card and card.get("rank") == "?" and card.get("color") == "?":
                        continue
                    if card and "suit" in card:
                        col.append((card["rank"], card["suit"]))
                    else:
                        print(f"[Warn] col{idx}: unresolved card {card!r}; truncating column read here")
                        truncated_columns.append(idx)
                        break
                cols.append(col)

            free = []
            for card in board["free_cells"]:
                if card and "suit" in card:
                    free.append((card["rank"], card["suit"]))

            found = {}
            for card in board["foundation"]:
                if card and "suit" in card:
                    found[card["suit"]] = rank_val(card["rank"])

            print("[*] Formulated Solver State:")
            print(f"  Cols: {cols}")
            print(f"  Free: {free}")
            print(f"  Found: {found}")
            log_event(
                "solver_state_built",
                cycle=cycle_number,
                columns=cols,
                free=free,
                foundations=found,
                truncated_columns=truncated_columns,
            )

            if args.solver == "monte-carlo":
                print("[*] Running Monte Carlo move search...")
                solver_started = time.perf_counter()
                state = State(cols, free, found)
                move, statistics = choose_move_monte_carlo(state)
                solver_seconds = time.perf_counter() - solver_started
                print_statistics(statistics)

                stats_payload = [
                    {
                        "move": stats.move,
                        "visits": stats.visits,
                        "wins": stats.wins,
                        "win_rate": stats.win_rate,
                        "average_score": stats.average_score,
                    }
                    for stats in statistics
                ]
                log_event(
                    "solver_finished",
                    cycle=cycle_number,
                    solver="monte-carlo",
                    duration_seconds=solver_seconds,
                    selected_move=move,
                    statistics=stats_payload,
                    simulations_completed=sum(stats.visits for stats in statistics),
                )

                if move:
                    ok = execute_move(
                        board,
                        move,
                        sim_mode=sim_mode,
                        event_logger=log_event,
                    )
                    log_event(
                        "move_result",
                        cycle=cycle_number,
                        move=move,
                        success=ok,
                    )
                else:
                    print("[*] No moves found. Board might already be solved or no path exists.")
                    log_event("no_move_selected", cycle=cycle_number, solver="monte-carlo")
            else:
                print("[*] Searching for a path...")
                solver_started = time.perf_counter()
                path, explored, solved = solve(
                    cols,
                    initial_free=free,
                    initial_found=found,
                    time_limit=5.0,
                )
                solver_seconds = time.perf_counter() - solver_started
                log_event(
                    "solver_finished",
                    cycle=cycle_number,
                    solver="search",
                    duration_seconds=solver_seconds,
                    explored_states=explored,
                    solved=solved,
                    path_length=len(path),
                    path=path,
                )

                if path:
                    batch = path if args.moves_per_cycle <= 0 else path[:args.moves_per_cycle]
                    print(f"[*] Executing {len(batch)} move(s) this cycle:")
                    for idx, move in enumerate(batch):
                        print(f"  {idx + 1}. {move}")
                        ok = execute_move(
                            board,
                            move,
                            sim_mode=sim_mode,
                            event_logger=log_event,
                        )
                        log_event(
                            "move_result",
                            cycle=cycle_number,
                            batch_index=idx,
                            move=move,
                            success=ok,
                        )
                        if not ok:
                            print("[*] Stopping batch early; will re-read the board next cycle.")
                            break
                        apply_move_to_board(board, move)
                else:
                    print("[*] No moves found. Board might already be solved or no path exists.")
                    log_event("no_move_selected", cycle=cycle_number, solver="search")

            cycle_seconds = time.perf_counter() - cycle_started
            log_event("cycle_finished", cycle=cycle_number, duration_seconds=cycle_seconds)

            if sim_mode:
                break

            print(f"[*] Waiting for UI update ({args.interval}s)...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[*] Interrupted by user. Shutting down cleanly...")
        log_event("session_interrupted", cycle=cycle_number)
    finally:
        if logcat_monitor is not None:
            logcat_monitor.stop()
            if logcat_started:
                log_event("logcat_stopped", path=logcat_path)
        log_event(
            "session_finished",
            cycles=cycle_number,
            interrupted=interrupted,
        )
        if session_logger is not None:
            print(f"[*] Structured session log: {session_logger.path}")
            session_logger.close()
        if logcat_started and logcat_path is not None:
            print(f"[*] Raw logcat file: {logcat_path}")


if __name__ == "__main__":
    main()
