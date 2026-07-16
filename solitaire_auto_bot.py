#!/usr/bin/env python3
import time
import sys
import os
import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from board_reader_lib import (
    read_board, TABLEAU_X, TABLEAU_Y_TOP, COL_WIDTH,
    FREE_CELL_X, FOUNDATION_X, SLOT_Y, SLOT_W, SLOT_H,
    HIDDEN_CARD_H, STEP
)
from freecell_solver import (
    State,
    apply_move,
    generate_complete_moves,
    solve,
    rank_val,
)

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
    These suits are constraint-derived, not directly observed from pixels.
    """
    black_counts = {}
    red_counts = {}

    def process_card(card):
        if card is None:
            return
        if card.get("suit") in ("S", "H", "D", "C"):
            card.setdefault("suit_source", "observed_exact")
            return
        if card.get("rank") == "?" or card.get("color") == "?":
            card["suit_source"] = "unresolved"
            return
        rank = card["rank"]
        color = card["color"]
        
        if color == "BLACK":
            count = black_counts.get(rank, 0) + 1
            black_counts[rank] = count
            if count == 1:
                card["suit"] = "S"
                card["suit_source"] = "resolved_by_constraints"
            elif count == 2:
                card["suit"] = "C"
                card["suit_source"] = "resolved_by_constraints"
            else:
                card["suit"] = "?"
                card["suit_source"] = "ambiguous"
        elif color == "RED":
            count = red_counts.get(rank, 0) + 1
            red_counts[rank] = count
            if count == 1:
                card["suit"] = "H"
                card["suit_source"] = "resolved_by_constraints"
            elif count == 2:
                card["suit"] = "D"
                card["suit_source"] = "resolved_by_constraints"
            else:
                card["suit"] = "?"
                card["suit_source"] = "ambiguous"
        else:
            card["suit_source"] = "ambiguous"

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
    Update the display board used for simulation-only geometry.

    Solver state and verification must use freecell_solver.apply_move().
    This helper exists only so dry-run batches can print plausible coordinates
    after a simulated move changes a visible stack.
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
    result = {
        "ok": False,
        "move": move,
        "gesture": None,
        "start": None,
        "end": None,
        "reason": None,
        "simulation": sim_mode,
    }

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
            result["reason"] = "physical_top_mismatch"
            result["physical_top"] = top
            result["expected_card"] = card
            return result

    if start_coords and end_coords:
        x1, y1 = start_coords
        x2, y2 = end_coords
        result["start"] = {"x": x1, "y": y1}
        result["end"] = {"x": x2, "y": y2}
        if kind in ("col_to_found", "free_to_found"):
            result["gesture"] = "tap"
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
            result["gesture"] = "swipe"
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
        result["ok"] = True
        return result
    else:
        print(f"[Error] Failed to resolve coordinates for move: {move}")
        emit("move_rejected", move=move, reason="coordinate_resolution_failed")
        result["reason"] = "coordinate_resolution_failed"
        return result

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


def state_to_data(state):
    return {
        "columns": [[list(card) for card in column] for column in state.cols],
        "free": [list(card) for card in state.free],
        "foundations": {suit: value for suit, value in state.found},
    }


def move_to_data(move):
    return list(move)


def build_normalized_state(board, allow_best_effort=False):
    """
    Split raw OCR observations from solver identities.

    The solver state contains only exact card identities. Each exact suit on
    the returned board is annotated with suit_source so live mode can refuse
    unsafe execution when the current state is not trustworthy.
    """
    resolved_board = deepcopy(board)
    assign_pseudo_suits(resolved_board)

    unresolved_cards = []
    ambiguous_cards = []
    truncated_columns = []
    cols = []

    def note_card(collection, index, card, reason):
        payload = {
            "area": collection,
            "index": index,
            "card": deepcopy(card),
            "reason": reason,
        }
        if reason == "ambiguous":
            ambiguous_cards.append(payload)
        else:
            unresolved_cards.append(payload)

    for idx in range(7):
        col = []
        for card_index, card in enumerate(resolved_board[f"col{idx}"]):
            source = card.get("suit_source")
            if card.get("rank") == "?" or card.get("color") == "?":
                note_card(f"col{idx}", card_index, card, "unresolved")
                if not allow_best_effort:
                    break
                continue
            if source == "ambiguous" or card.get("suit") not in ("S", "H", "D", "C"):
                note_card(f"col{idx}", card_index, card, "ambiguous")
                truncated_columns.append(idx)
                if not allow_best_effort:
                    break
                continue
            col.append((card["rank"], card["suit"]))
        cols.append(col)

    free = []
    for idx, card in enumerate(resolved_board["free_cells"]):
        if not card:
            continue
        source = card.get("suit_source")
        if card.get("rank") == "?" or card.get("color") == "?":
            note_card("free_cells", idx, card, "unresolved")
        elif source == "ambiguous" or card.get("suit") not in ("S", "H", "D", "C"):
            note_card("free_cells", idx, card, "ambiguous")
        else:
            free.append((card["rank"], card["suit"]))

    found = {}
    for idx, card in enumerate(resolved_board["foundation"]):
        if not card:
            continue
        source = card.get("suit_source")
        if card.get("rank") == "?" or card.get("color") == "?":
            note_card("foundation", idx, card, "unresolved")
        elif source == "ambiguous" or card.get("suit") not in ("S", "H", "D", "C"):
            note_card("foundation", idx, card, "ambiguous")
        else:
            found[card["suit"]] = rank_val(card["rank"])

    state = State(cols, free, found)
    trustworthy = not unresolved_cards and not ambiguous_cards
    trust_status = "trustworthy" if trustworthy else "untrusted"

    return {
        "board": resolved_board,
        "state": state,
        "state_data": state_to_data(state),
        "columns": cols,
        "free": free,
        "foundations": found,
        "unresolved_cards": unresolved_cards,
        "ambiguous_cards": ambiguous_cards,
        "truncated_columns": truncated_columns,
        "trustworthy": trustworthy,
        "trust_status": trust_status,
        "allow_best_effort": allow_best_effort,
    }


def compare_normalized_state(expected_state, actual_normalized):
    expected = state_to_data(expected_state)
    actual = actual_normalized["state_data"]
    differences = []

    for key in ("columns", "free", "foundations"):
        if expected[key] != actual[key]:
            differences.append({
                "field": key,
                "expected": expected[key],
                "actual": actual[key],
            })

    if actual_normalized["unresolved_cards"]:
        differences.append({
            "field": "unresolved_cards",
            "expected": [],
            "actual": actual_normalized["unresolved_cards"],
        })

    if actual_normalized["ambiguous_cards"]:
        differences.append({
            "field": "ambiguous_cards",
            "expected": [],
            "actual": actual_normalized["ambiguous_cards"],
        })

    if not actual_normalized["trustworthy"]:
        differences.append({
            "field": "trust_status",
            "expected": "trustworthy",
            "actual": actual_normalized["trust_status"],
        })

    return {
        "matches": not differences,
        "differences": differences,
        "expected_state": expected,
        "actual_state": actual,
        "actual_unresolved_cards": actual_normalized["unresolved_cards"],
        "actual_ambiguous_cards": actual_normalized["ambiguous_cards"],
        "actual_trust_status": actual_normalized["trust_status"],
    }


def read_and_normalize_board(screenshot_path, allow_best_effort=False):
    board = read_board(str(screenshot_path))
    return build_normalized_state(board, allow_best_effort=allow_best_effort)


def capture_live_screenshot(path):
    img = bridge.screenshot()
    if img is None:
        return False
    img.save(path)
    return True


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def failure_artifact_dir():
    stamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    path = Path("logs") / "failures" / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_failure_artifacts(
    *,
    before_screenshot,
    verification_screenshots,
    previous_normalized,
    expected_state,
    selected_move,
    final_actual_normalized,
    mismatch_report,
    session_log_path=None,
    logcat_path=None,
):
    directory = failure_artifact_dir()
    directory.mkdir(parents=True, exist_ok=True)
    before_dest = directory / "before.png"
    if before_screenshot and Path(before_screenshot).exists():
        shutil.copy2(before_screenshot, before_dest)

    copied_verifications = []
    for idx, screenshot in enumerate(verification_screenshots, start=1):
        src = Path(screenshot)
        if src.exists():
            dest = directory / f"verification_{idx}.png"
            shutil.copy2(src, dest)
            copied_verifications.append(str(dest))

    write_json(directory / "previous_state.json", {
        "state": previous_normalized["state_data"],
        "trust_status": previous_normalized["trust_status"],
        "unresolved_cards": previous_normalized["unresolved_cards"],
        "ambiguous_cards": previous_normalized["ambiguous_cards"],
    })
    write_json(directory / "expected_state.json", state_to_data(expected_state))
    write_json(directory / "final_actual_state.json", {
        "state": final_actual_normalized["state_data"] if final_actual_normalized else None,
        "trust_status": final_actual_normalized["trust_status"] if final_actual_normalized else "unavailable",
        "unresolved_cards": final_actual_normalized["unresolved_cards"] if final_actual_normalized else [],
        "ambiguous_cards": final_actual_normalized["ambiguous_cards"] if final_actual_normalized else [],
    })
    write_json(directory / "selected_move.json", {"move": move_to_data(selected_move)})
    write_json(directory / "mismatch_report.json", mismatch_report)
    write_json(directory / "references.json", {
        "before_screenshot": str(before_dest) if before_screenshot else None,
        "verification_screenshots": copied_verifications,
        "session_log": str(session_log_path) if session_log_path else None,
        "logcat": str(logcat_path) if logcat_path else None,
    })
    return directory


def verify_expected_state(
    *,
    expected_state,
    allow_best_effort,
    attempts,
    delay,
    screenshot_prefix,
    event_logger=None,
):
    emit = event_logger or (lambda event_name, **data: None)
    screenshots = []
    final_actual = None
    final_report = {
        "matches": False,
        "differences": [{"field": "verification", "actual": "not_attempted"}],
    }

    for attempt in range(1, attempts + 1):
        if delay > 0:
            time.sleep(delay)

        screenshot_path = Path(f"{screenshot_prefix}_verify_{attempt}.png")
        if not capture_live_screenshot(screenshot_path):
            final_report = {
                "matches": False,
                "differences": [{"field": "screenshot", "attempt": attempt, "actual": "capture_failed"}],
            }
            emit("verification_attempt_failed", attempt=attempt, reason="screenshot_failed")
            continue

        screenshots.append(screenshot_path)
        try:
            final_actual = read_and_normalize_board(
                screenshot_path,
                allow_best_effort=allow_best_effort,
            )
            final_report = compare_normalized_state(expected_state, final_actual)
        except Exception as exc:
            final_report = {
                "matches": False,
                "differences": [{"field": "board_read", "attempt": attempt, "actual": str(exc)}],
            }
            emit("verification_attempt_failed", attempt=attempt, reason="board_read_failed", error=str(exc))
            continue

        emit(
            "verification_attempt",
            attempt=attempt,
            screenshot=screenshot_path,
            matches=final_report["matches"],
            mismatch=final_report,
        )
        if final_report["matches"]:
            return {
                "ok": True,
                "attempts": attempt,
                "screenshots": screenshots,
                "actual_normalized": final_actual,
                "mismatch_report": final_report,
            }

    return {
        "ok": False,
        "attempts": attempts,
        "screenshots": screenshots,
        "actual_normalized": final_actual,
        "mismatch_report": final_report,
    }


def choose_next_move(args, current_state, legal_moves, log_event):
    if not legal_moves:
        return None, {"reason": "no_legal_moves"}

    if args.solver == "monte-carlo":
        print("[*] Running Monte Carlo move ranking...")
        solver_started = time.perf_counter()
        move, statistics = choose_move_monte_carlo(
            current_state,
            legal_moves=legal_moves,
        )
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
            solver="monte-carlo",
            duration_seconds=solver_seconds,
            candidate_moves=legal_moves,
            selected_move=move,
            selection_reason="highest_monte_carlo_rank",
            statistics=stats_payload,
            simulations_completed=sum(stats.visits for stats in statistics),
        )
        return move, {
            "reason": "highest_monte_carlo_rank",
            "statistics": stats_payload,
            "duration_seconds": solver_seconds,
        }

    print("[*] Searching for a path...")
    solver_started = time.perf_counter()
    path, explored, solved = solve(
        current_state.cols,
        initial_free=list(current_state.free),
        initial_found=current_state.found_dict(),
        time_limit=5.0,
    )
    solver_seconds = time.perf_counter() - solver_started
    move = path[0] if path else None
    if move is not None and move not in legal_moves:
        raise RuntimeError(f"Search selected a move outside complete legal moves: {move!r}")

    log_event(
        "solver_finished",
        solver="search",
        duration_seconds=solver_seconds,
        explored_states=explored,
        solved=solved,
        path_length=len(path),
        path=path,
        candidate_moves=legal_moves,
        selected_move=move,
        selection_reason="first_move_from_search_path",
    )
    return move, {
        "reason": "first_move_from_search_path",
        "path": path,
        "explored_states": explored,
        "solved": solved,
        "duration_seconds": solver_seconds,
    }


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
    parser.add_argument(
        "--verify-attempts",
        type=int,
        default=3,
        help="Live mode verification screenshot attempts after each gesture. Default: 3.",
    )
    parser.add_argument(
        "--verify-delay",
        type=float,
        default=0.75,
        help="Seconds to wait before each live verification attempt. Default: 0.75.",
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

            before_screenshot = Path(screenshot_file)
            if not sim_mode:
                before_screenshot = Path(f"live_screen_cycle_{cycle_number}_before.png")
                print("[*] Capturing screen...")
                capture_started = time.perf_counter()
                captured = capture_live_screenshot(before_screenshot)
                capture_seconds = time.perf_counter() - capture_started

                if captured:
                    print(f"[*] Screen saved to {before_screenshot}")
                    log_event(
                        "screenshot_captured",
                        cycle=cycle_number,
                        path=before_screenshot,
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
                normalized = read_and_normalize_board(
                    before_screenshot,
                    allow_best_effort=sim_mode,
                )
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
            board = normalized["board"]
            state = normalized["state"]
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
                col_info = [
                    f"{c['rank']}({c['color']},{c.get('suit', '?')},{c.get('suit_source', '?')})"
                    for c in board[col_key]
                ]
                print(f"  col{idx}: {col_info}")

            free_info = [
                f"{c['rank']}({c['color']},{c.get('suit', '?')},{c.get('suit_source', '?')})"
                if c else "None"
                for c in board["free_cells"]
            ]
            found_info = [
                f"{c['rank']}({c['color']},{c.get('suit', '?')},{c.get('suit_source', '?')})"
                if c else "None"
                for c in board["foundation"]
            ]
            print(f"  free_cells: {free_info}")
            print(f"  foundation: {found_info}")

            print("[*] Normalized Solver State:")
            print(f"  Cols: {normalized['columns']}")
            print(f"  Free: {normalized['free']}")
            print(f"  Found: {normalized['foundations']}")
            print(f"  Trust: {normalized['trust_status']}")
            log_event(
                "solver_state_built",
                cycle=cycle_number,
                columns=normalized["columns"],
                free=normalized["free"],
                foundations=normalized["foundations"],
                unresolved_cards=normalized["unresolved_cards"],
                ambiguous_cards=normalized["ambiguous_cards"],
                truncated_columns=normalized["truncated_columns"],
                trust_status=normalized["trust_status"],
            )

            if not sim_mode and not normalized["trustworthy"]:
                print("[Error] Current state is not trustworthy enough for live execution. Stopping.")
                log_event(
                    "unsafe_state_stopped",
                    cycle=cycle_number,
                    unresolved_cards=normalized["unresolved_cards"],
                    ambiguous_cards=normalized["ambiguous_cards"],
                )
                break

            legal_moves = generate_complete_moves(state)
            log_event("legal_moves_generated", cycle=cycle_number, legal_moves=legal_moves)
            print(f"[*] Legal moves: {len(legal_moves)}")

            try:
                move, selection = choose_next_move(
                    args,
                    state,
                    legal_moves,
                    lambda event_name, **data: log_event(event_name, cycle=cycle_number, **data),
                )
            except Exception as exc:
                print(f"[Error] Move selection failed: {exc}")
                log_event("move_selection_failed", cycle=cycle_number, error=str(exc))
                break

            if not move:
                print("[*] No legal move selected. Stopping cleanly.")
                log_event("no_move_selected", cycle=cycle_number, solver=args.solver, legal_moves=legal_moves)
                break

            if move not in generate_complete_moves(state):
                print(f"[Error] Selected move is no longer legal: {move}")
                log_event("move_rejected", cycle=cycle_number, move=move, reason="selected_move_not_legal")
                break

            expected_state = apply_move(state, move)
            log_event(
                "move_selected",
                cycle=cycle_number,
                move=move,
                selection=selection,
                expected_state=state_to_data(expected_state),
            )

            if sim_mode and args.solver == "search" and selection.get("path"):
                batch = selection["path"] if args.moves_per_cycle <= 0 else selection["path"][:args.moves_per_cycle]
            else:
                batch = [move]

            print(f"[*] Executing {len(batch)} move(s) this cycle:")
            current_state = state
            gesture_result = {"ok": False, "reason": "not_dispatched"}
            for idx, batch_move in enumerate(batch):
                if batch_move not in generate_complete_moves(current_state):
                    print(f"[Error] Batch move is not legal: {batch_move}")
                    log_event(
                        "move_rejected",
                        cycle=cycle_number,
                        batch_index=idx,
                        move=batch_move,
                        reason="batch_move_not_legal",
                    )
                    break
                print(f"  {idx + 1}. {batch_move}")
                gesture_result = execute_move(
                    board,
                    batch_move,
                    sim_mode=sim_mode,
                    event_logger=lambda event_name, **data: log_event(event_name, cycle=cycle_number, **data),
                )
                log_event(
                    "move_result",
                    cycle=cycle_number,
                    batch_index=idx,
                    move=batch_move,
                    success=gesture_result["ok"],
                    gesture_result=gesture_result,
                )
                if not gesture_result["ok"]:
                    print("[*] Stopping; gesture could not be safely dispatched.")
                    break
                current_state = apply_move(current_state, batch_move)
                if sim_mode:
                    apply_move_to_board(board, batch_move)

            if sim_mode:
                cycle_seconds = time.perf_counter() - cycle_started
                log_event("cycle_finished", cycle=cycle_number, duration_seconds=cycle_seconds)
                break

            if not gesture_result["ok"]:
                break

            verification = verify_expected_state(
                expected_state=expected_state,
                allow_best_effort=False,
                attempts=max(1, args.verify_attempts),
                delay=max(0.0, args.verify_delay),
                screenshot_prefix=f"live_screen_cycle_{cycle_number}",
                event_logger=lambda event_name, **data: log_event(event_name, cycle=cycle_number, **data),
            )
            if verification["ok"]:
                log_event(
                    "move_verified",
                    cycle=cycle_number,
                    move=move,
                    attempts=verification["attempts"],
                    actual_state=verification["actual_normalized"]["state_data"],
                )
            else:
                failure_dir = save_failure_artifacts(
                    before_screenshot=before_screenshot,
                    verification_screenshots=verification["screenshots"],
                    previous_normalized=normalized,
                    expected_state=expected_state,
                    selected_move=move,
                    final_actual_normalized=verification["actual_normalized"],
                    mismatch_report=verification["mismatch_report"],
                    session_log_path=session_logger.path if session_logger is not None else None,
                    logcat_path=logcat_path,
                )
                print(f"[Error] Verification failed. Debug artifacts saved to {failure_dir}")
                log_event(
                    "verification_failed",
                    cycle=cycle_number,
                    move=move,
                    failure_dir=failure_dir,
                    mismatch=verification["mismatch_report"],
                )
                break

            cycle_seconds = time.perf_counter() - cycle_started
            log_event("cycle_finished", cycle=cycle_number, duration_seconds=cycle_seconds)

            print("[*] Move verified. Continuing with a fresh capture.")

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
