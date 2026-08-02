#!/usr/bin/env python3
import time
import sys
import os
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
from board_reader_lib import (
    read_board, TABLEAU_X, TABLEAU_Y_TOP, COL_WIDTH,
    FOUNDATION_X, SLOT_Y, SLOT_W, SLOT_H,
    HIDDEN_CARD_H, STEP, STOCK_TOTAL, STOCK_TAP_X, STOCK_TAP_Y,
)
from freecell_solver import State, solve, apply_move, rank_val, UNKNOWN
import race_policy
from solver_state import build_solver_state

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
        # board_reader_lib now reads the real suit from the card's suit pip;
        # only fall back to alternating pseudo-suits when that read is
        # missing. Pseudo-suits are assigned by scan-order encounter, which
        # isn't stable across board-read cycles (a same-rank same-color
        # card can flip S<->C or H<->D between reads), causing solver
        # desync - a real suit read doesn't have that problem.
        if card.get("suit") in ("S", "H", "D", "C"):
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

    # Scan order: Foundation, Waste, then Tableau Columns (exposed cards first)
    for card in board.get("foundation", []):
        process_card(card)

    for card in board.get("waste", []):
        process_card(card)

    for col_idx in range(7):
        col_key = f"col{col_idx}"
        for card in board.get(col_key, []):
            process_card(card)

# ==============================================================================
# 2. COORDINATE RESOLUTION
# ==============================================================================
def find_foundation_slot(board, suit):
    """
    Foundation piles fill whichever of the 4 physical boxes is available -
    they aren't fixed to a suit ahead of time - so targeting one means
    finding whichever box already holds this suit, or (for that suit's
    Ace, its first card) the first empty box. Returns a FOUNDATION_X index,
    or None if no matching or empty box exists.
    """
    for i, c in enumerate(board["foundation"]):
        if c and c.get("suit") == suit:
            return i
    for i, c in enumerate(board["foundation"]):
        if c is None:
            return i
    return None


def get_element_coords(board, item_type, index, depth_from_bottom=0):
    """
    Calculates the exact center coordinate (x, y) for target slots or top cards.

    depth_from_bottom: for "col", how many cards up from the exposed bottom
    card to grab - 0 (default) is the bottom/last card, as for a
    single-card move. A multi-card run drag touches the card at the TOP
    of the run being moved (further up the column), so its depth is
    run_length - 1.
    """
    if item_type == "col":
        col_cards = board[f"col{index}"]
        num_cards = len(col_cards)
        x_center = TABLEAU_X[index] + COL_WIDTH / 2

        if num_cards == 0:
            # Empty column click target
            y_center = TABLEAU_Y_TOP + SLOT_H / 2
        else:
            # Find Y position of the card `depth_from_bottom` cards up from
            # the bottom-most revealed (exposed) card.
            hidden_count = sum(1 for c in col_cards if c.get("rank") == "?")
            revealed_count = num_cards - hidden_count
            y_edge = TABLEAU_Y_TOP + hidden_count * HIDDEN_CARD_H \
                + max(0, revealed_count - 1 - depth_from_bottom) * STEP
            # Only the bottom-most card (depth 0) renders at its full
            # height - every card above it in the stack is overlapped by
            # the one below it, so only a STEP-tall sliver is actually
            # visible/tappable there. Centering with the full SLOT_H
            # offset overshoots past that sliver into the next card's
            # territory - confirmed live: every multi-card run source
            # (depth_from_bottom > 0) was grabbing the card one position
            # below the intended one, and only single-card moves
            # (depth_from_bottom == 0, always the fully-visible bottom
            # card) were unaffected.
            band_height = SLOT_H if depth_from_bottom == 0 else STEP
            y_center = y_edge + band_height / 2
        return int(x_center), int(y_center)

    elif item_type == "waste":
        if not board["waste"]:
            return None
        top = board["waste"][-1]
        x_center = top["x"] + SLOT_W / 2
        y_center = SLOT_Y + SLOT_H / 2
        return int(x_center), int(y_center)

    elif item_type == "found":
        # index is the target suit here, not a physical slot number
        slot_idx = find_foundation_slot(board, index)
        if slot_idx is None:
            return None
        x_center = FOUNDATION_X[slot_idx] + SLOT_W / 2
        y_center = SLOT_Y + SLOT_H / 2
        return int(x_center), int(y_center)

    elif item_type == "stock":
        return STOCK_TAP_X, STOCK_TAP_Y

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

    "draw" and "redeal" reveal/rearrange real physical cards this local
    model has no way to know ahead of a fresh screen read - the main loop
    breaks the batch immediately after executing either (see main()), so
    this function is never asked to simulate their effect.
    """
    kind = move[0]

    def record_foundation(card):
        rank, suit = card
        slot_idx = find_foundation_slot(board, suit)
        if slot_idx is not None:
            board["foundation"][slot_idx] = {
                "rank": rank, "suit": suit, "color": card_color(suit), "score": 1.0,
            }

    if kind == "col_to_found":
        _, ci, card = move
        board[f"col{ci}"].pop()
        record_foundation(card)
    elif kind == "waste_to_found":
        _, card = move
        board["waste"].pop()
        record_foundation(card)
    elif kind == "col_to_col":
        _, ci, cj, card, run_length = move
        moved = board[f"col{ci}"][-run_length:]
        del board[f"col{ci}"][-run_length:]
        board[f"col{cj}"].extend(moved)
    elif kind == "waste_to_col":
        _, cj, card = move
        board["waste"].pop()
        board[f"col{cj}"].append({"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0})


# ==============================================================================
# 3. MOVE TRANSLATION TO PHYSICAL GESTURES
# ==============================================================================
def execute_move(board, move, sim_mode=False, event_logger=None):
    """
    Translates a solver move into coordinates and triggers the swipe/tap.
    """
    kind = move[0]
    card = None
    emit = event_logger or (lambda event_name, **data: None)
    start_coords = None
    end_coords = None

    if kind == "col_to_found":
        _, ci, card = move
        start_coords = get_element_coords(board, "col", ci)
        end_coords = get_element_coords(board, "found", card[1])

    elif kind == "waste_to_found":
        _, card = move
        start_coords = get_element_coords(board, "waste", None)
        end_coords = get_element_coords(board, "found", card[1])

    elif kind == "col_to_col":
        _, ci, cj, card, run_length = move
        start_coords = get_element_coords(board, "col", ci, depth_from_bottom=run_length - 1)
        end_coords = get_element_coords(board, "col", cj)

    elif kind == "waste_to_col":
        _, cj, card = move
        start_coords = get_element_coords(board, "waste", None)
        end_coords = get_element_coords(board, "col", cj)

    elif kind in ("draw", "redeal"):
        # Same physical button either way - the game itself decides
        # whether the tap draws the next 3 cards or triggers the redeal.
        start_coords = end_coords = get_element_coords(board, "stock", None)

    # Sanity-check: for moves that pop the top of a tableau column or the
    # top of waste, make sure the physical top card actually matches what
    # the solver believes is there. Board state and solver state can
    # diverge (e.g. a revealed card whose suit couldn't be read gets
    # dropped from the solver's view), and blindly swiping in that case
    # would drag the wrong real card.
    if kind in ("col_to_found", "col_to_col"):
        ci = move[1]
        run_length = move[4] if kind == "col_to_col" else 1
        physical_col = board[f"col{ci}"]
        top = physical_col[-run_length] if len(physical_col) >= run_length else None
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
    elif kind in ("waste_to_found", "waste_to_col"):
        top = board["waste"][-1] if board["waste"] else None
        if not top or top.get("rank") != card[0] or top.get("suit") != card[1]:
            print(f"[Warn] Skipping move {move}: physical top of waste "
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
        if kind in ("col_to_found", "waste_to_found", "draw", "redeal"):
            label = card if card is not None else kind
            print(f"[*] Action: Tap ({kind}): {label} at ({x1}, {y1})")
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
            run_note = f" (+{move[4] - 1} card(s) under it)" if kind == "col_to_col" and move[4] > 1 else ""
            print(
                f"[*] Action: Move {kind.replace('_', ' ')}: {card}{run_note} "
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
# How long a history-contradicting foundation value must persist before we
# believe it over the history. Two quick sightings are NOT independent
# evidence - the unreliable-frame retry path re-reads just 1.5s apart, so an
# animation overlay lasting ~3s would "confirm" itself under a count-based
# rule. Real board changes persist indefinitely; overlays don't.
FOUNDATION_CONFIRM_SECONDS = 5.0


def reconcile_foundation_reads(board, accepted, pending, max_jump, emit, now):
    """Validate this frame's foundation reads against foundation physics.

    Score thresholds alone cannot separate animation garbage from genuine
    reads (corpus-measured: garbage scores up to 0.72, genuine reads as low
    as 0.32) - but physics can: within a run a foundation slot's suit never
    changes, its pile never shrinks, and it can only grow by as many cards
    as could have been played since the last read. Reads that violate this
    are rejected and the last accepted value is carried forward, so a
    transient popup can't corrupt the solver state OR flap the board
    signature the stuck-move detector depends on. A rejected value that
    keeps being read for FOUNDATION_CONFIRM_SECONDS is accepted anyway (a
    legitimate big jump, or the user started a new game) so a wrong
    rejection can't stick forever.

    accepted: slot_idx -> last trusted card dict, or None for observed-empty;
              a slot missing from the dict has never been observed, and its
              first reliable read is accepted as baseline.
    pending:  slot_idx -> [candidate, first_seen_monotonic] for reads
              awaiting persistence confirmation.
    Mutates board["foundation"] in place; returns a list of trust notes
    (empty when every slot was consistent).
    """
    notes = []
    slots = board.get("foundation", [])
    for i, card in enumerate(slots):
        if card is not None and not card.get("reliable", False):
            # Untrusted read (obscured/ambiguous crop): substitute the last
            # accepted value; don't touch pending - unreliable frames are
            # transient and shouldn't break a confirmation streak.
            prev = accepted.get(i)
            slots[i] = dict(prev, carried=True) if prev else None
            if prev:
                emit("foundation_read_carried", slot=i, untrusted=card, using=prev)
            else:
                notes.append(f"foundation slot {i}: untrusted read {card!r} with no history")
                emit("foundation_read_untrusted_no_history", slot=i, untrusted=card)
            continue

        read_key = (card["suit"], rank_val(card["rank"])) if card else None
        if i not in accepted:
            accepted[i] = dict(card) if card else None
            continue

        prev = accepted[i]
        prev_rv = rank_val(prev["rank"]) if prev else -1
        if read_key is None:
            compatible = prev is None  # an established pile never empties mid-run
        else:
            suit, rv = read_key
            same_suit = prev is None or prev["suit"] == suit
            compatible = same_suit and 0 <= rv - prev_rv <= max_jump

        if compatible:
            accepted[i] = dict(card) if card else None
            pending.pop(i, None)
            continue

        entry = pending.get(i)
        if entry and entry[0] == read_key:
            if now - entry[1] >= FOUNDATION_CONFIRM_SECONDS:
                accepted[i] = dict(card) if card else None
                pending.pop(i, None)
                emit("foundation_read_accepted_after_persistence", slot=i, value=read_key)
                continue
        else:
            pending[i] = [read_key, now]

        slots[i] = dict(prev, carried=True) if prev else None
        prev_key = (prev["suit"], rank_val(prev["rank"])) if prev else None
        notes.append(
            f"foundation slot {i}: read {read_key} contradicts history "
            f"{prev_key}; keeping last known value"
        )
        emit("foundation_read_rejected", slot=i, rejected=read_key, keeping=prev_key)
    return notes


def count_unresolved_cards(board):
    unresolved = 0
    for idx in range(7):
        for card in board.get(f"col{idx}", []):
            if card and (card.get("rank") == "?" or card.get("color") == "?"):
                unresolved += 1
    for area in ("waste", "foundation"):
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
        default=None,
        help="Seconds to wait after a batch of moves for the UI to settle before the next "
             "screen capture. Default: 1.5 (0.5 with --fast).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster pacing profile: ~500ms swipes, 0.85-1.0s inter-gesture pauses, no "
             "scan pauses, and a 0.5s --interval default. The pause floor is the game's "
             "card-settle animation (see bridge.wait_human_delay) - pauses cut below it "
             "made batched moves silently fail in live runs, so --fast shaves pacing "
             "without undercutting it. If moves start silently failing anyway (repeated "
             "'Board unchanged after attempting' warnings), drop this flag or raise "
             "--swipe-ms / --gesture-delay.",
    )
    parser.add_argument(
        "--swipe-ms",
        type=int,
        default=None,
        help="Swipe gesture duration in milliseconds. Default: ~800 (~500 with --fast). "
             "Swipes much faster than ~700ms were observed not registering as drags; "
             "500 needs on-device verification.",
    )
    parser.add_argument(
        "--gesture-delay",
        type=float,
        default=None,
        help="Fixed pause in seconds after each tap/swipe, overriding the profile's "
             "range. Use to bisect the device's real card-settle floor (0.2-0.8s was "
             "observed too short; 0.9-1.5s is known good).",
    )
    parser.add_argument(
        "--solver",
        choices=["search", "monte-carlo", "race"],
        default="search",
        help="Choose the move-selection engine. 'race' is the timed-round "
             "policy: instant greedy selection (no search budget), strict "
             "foundation > reveal > empty-column priority, refuses moves that "
             "neither reveal nor found (the in-place shuffle), and plays "
             "several moves per screen read. Use it when a countdown is "
             "running - see race_policy.py. Default: search."
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

    if args.interval is None:
        args.interval = 0.5 if args.fast else 1.5
    if args.fast:
        bridge.configure_timing(
            swipe_ms=500, delay_min=0.85, delay_max=1.0, scan_pause_chance=0.0,
        )
    if args.swipe_ms is not None:
        bridge.configure_timing(swipe_ms=args.swipe_ms)
    if args.gesture_delay is not None:
        bridge.configure_timing(delay_min=args.gesture_delay, delay_max=args.gesture_delay)

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
    previous_solver_signature = None
    previous_first_move = None
    excluded_first_moves = set()
    # A move that fails once might just be a one-off flaky gesture, worth
    # retrying later - but a move that keeps failing repeatedly from the
    # SAME board state (some specific source/destination pairing the
    # device consistently rejects at that column depth/coordinates) will
    # otherwise get proposed again every time some unrelated move
    # elsewhere succeeds and clears the one-retry exclusion above, wasting
    # most of the run stuck on it. Keyed on (board signature, move), not
    # just the move alone: the same move tuple can mean a completely
    # different physical drag once column heights/hidden-card counts
    # change, so a failure at one layout must not blacklist it forever at
    # every future layout too. Only meaningful when each cycle executes
    # and verifies exactly one move - with a multi-move batch (the
    # default), a failure can only be pinned on "something in the batch
    # didn't happen ", not specifically on its first move.
    MOVE_PERMANENT_EXCLUDE_THRESHOLD = 3
    track_move_failures = args.moves_per_cycle == 1
    move_failure_counts = {}
    permanently_excluded_by_state = set()
    # Real board states from the last several cycles - not just the
    # immediately previous one. Two occupied columns can score identically
    # under the heuristic (e.g. a red Queen sitting on either of two black
    # Kings), so the search can end up "preferring" to shuffle a card
    # back and forth between them across cycles: each individual move
    # genuinely succeeds (so the single-previous-cycle stuck-move check
    # above never fires), but it just walks the board back to a state
    # already visited a few cycles ago instead of making progress.
    recent_board_signatures = deque(maxlen=10)
    # Temporal foundation tracking (see reconcile_foundation_reads): the
    # jump cap is how many foundation plays could legitimately happen since
    # the previous read - the *_to_found moves actually executed last cycle
    # plus a couple of game-side auto-completes. Deriving it from executed
    # moves (rather than the --moves-per-cycle setting) keeps the physics
    # filter tight even with batch 0, where a whole winning path can run in
    # one cycle.
    foundation_accepted = {}
    foundation_pending = {}
    last_cycle_found_plays = 0
    consecutive_unreliable_frames = 0
    consecutive_impossible_frames = 0
    previous_issue_signature = None

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

            waste_info = [f"{c['rank']}({c['color']})" for c in board["waste"]]
            found_info = [f"{c['rank']}({c['color']})" if c else "None" for c in board["foundation"]]
            print(f"  waste: {waste_info}")
            print(f"  foundation: {found_info}")

            assign_pseudo_suits(board)

            def emit_trust(event_name, **data):
                log_event(event_name, cycle=cycle_number, **data)

            trust_notes = reconcile_foundation_reads(
                board, foundation_accepted, foundation_pending,
                last_cycle_found_plays + 2, emit_trust, time.monotonic(),
            )

            state = build_solver_state(board, stock_total=STOCK_TOTAL)
            cols = state["cols"]
            waste = state["waste"]
            found = state["found"]
            stock_remaining = state["stock_remaining"]
            truncated_columns = state["truncated_columns"]
            issues = trust_notes + state["issues"]
            for msg in issues:
                print(f"[Warn] {msg}")

            # A single 52-card deck can never over- or under-account for its
            # cards - stock_remaining outside [0, STOCK_TOTAL] means the CV
            # read itself is corrupted (e.g. a garbled screenshot right
            # after a device reconnect made detect_column_height()
            # hallucinate a column dozens of cards deep, or a whole
            # foundation pile was dropped from the read), not that the board
            # is in a strange but real state. Solving/executing against a
            # hallucinated board computes tap coordinates for cards that
            # don't exist at those positions - confirmed live: this silently
            # sent taps outside the tableau entirely and backed out of the
            # game. Treat it like a failed capture/read and retry. Lesser
            # trust problems (a truncated column, an unresolved waste card)
            # get a couple of quick re-reads - animations settle in about a
            # second - before proceeding best-effort so a persistent quirk
            # can't stall the loop forever.
            # A CHANGED issue set means a new problem, not the same stable
            # quirk - re-arm the retry budget so a fresh transient still
            # gets its re-reads even after a permanent oddity (e.g. a
            # stably ambiguous straddled crop) exhausted them.
            issue_signature = tuple(issues)
            if issues and issue_signature != previous_issue_signature:
                consecutive_unreliable_frames = 0
            previous_issue_signature = issue_signature

            stock_impossible = stock_remaining < 0 or stock_remaining > STOCK_TOTAL
            consecutive_impossible_frames = \
                consecutive_impossible_frames + 1 if stock_impossible else 0
            if not sim_mode and (
                stock_impossible or (issues and consecutive_unreliable_frames < 2)
            ):
                consecutive_unreliable_frames += 1
                print(f"[Warn] Unreliable board read "
                      f"(stock_remaining={stock_remaining}); "
                      f"discarding this cycle and re-reading.")
                log_event(
                    "board_read_unreliable",
                    cycle=cycle_number,
                    issues=issues,
                    stock_remaining=stock_remaining,
                    stock_impossible=stock_impossible,
                    consecutive=consecutive_unreliable_frames,
                )
                # Impossible stock retries indefinitely by design (executing
                # against a hallucinated board physically backed out of the
                # game once), but after a while it's clearly not a transient
                # animation - a dialog or overlay needs a human - so back
                # off to spare the device and the log.
                if consecutive_impossible_frames > 5:
                    print("[Warn] Board has read as impossible for "
                          f"{consecutive_impossible_frames} consecutive reads; "
                          "an overlay or dialog may need attention.")
                    time.sleep(10.0)
                else:
                    time.sleep(3.0 if stock_impossible else 1.5)
                continue
            if not issues:
                # only a clean frame re-arms the retry budget - a stable
                # board quirk gets its two re-reads once, then proceeds
                # best-effort every cycle instead of paying the retry tax
                # forever (see the issue-signature re-arm above for new
                # problems)
                consecutive_unreliable_frames = 0
            last_cycle_found_plays = 0

            print("[*] Formulated Solver State:")
            print(f"  Cols: {cols}")
            print(f"  Waste: {waste}")
            print(f"  Found: {found}")
            print(f"  Stock remaining: {stock_remaining}")
            log_event(
                "solver_state_built",
                cycle=cycle_number,
                columns=cols,
                waste=waste,
                foundations=found,
                stock_remaining=stock_remaining,
                truncated_columns=truncated_columns,
                issues=issues,
            )

            # execute_move() only confirms a gesture was dispatched, not that
            # the physical game actually applied it - a swipe can be issued
            # cleanly and still not register on the device. Detect that by
            # comparing this fresh read against the read from before the
            # previous cycle's moves: if nothing changed despite having
            # attempted a real move, that move is a no-op on the real board
            # and re-suggesting it as-is would just repeat forever.
            current_solver_signature = (
                tuple(tuple(c) for c in cols), tuple(waste),
                tuple(sorted(found.items())), stock_remaining,
            )
            if (args.solver == "search" and previous_first_move is not None
                    and current_solver_signature == previous_solver_signature):
                print(f"[Warn] Board unchanged after attempting {previous_first_move}; "
                      f"excluding it and re-solving.")
                log_event("move_stuck_excluding", cycle=cycle_number, move=previous_first_move)
                excluded_first_moves.add(previous_first_move)

                if track_move_failures:
                    failure_key = (previous_solver_signature, previous_first_move)
                    fail_count = move_failure_counts.get(failure_key, 0) + 1
                    move_failure_counts[failure_key] = fail_count
                    if fail_count >= MOVE_PERMANENT_EXCLUDE_THRESHOLD:
                        print(f"[Warn] {previous_first_move} has now failed {fail_count} times "
                              f"from this exact board state; excluding it for this state.")
                        log_event(
                            "move_permanently_excluding",
                            cycle=cycle_number,
                            move=previous_first_move,
                            fail_count=fail_count,
                        )
                        permanently_excluded_by_state.add(failure_key)
            else:
                excluded_first_moves = set()
                if track_move_failures and previous_first_move is not None:
                    # The move attempted from the previous state either
                    # succeeded or this is a fresh state either way - any
                    # accumulated failure count for that specific
                    # (state, move) pairing no longer reflects reality.
                    move_failure_counts.pop((previous_solver_signature, previous_first_move), None)
            previous_solver_signature = current_solver_signature
            previous_first_move = None
            recent_board_signatures.append(current_solver_signature)

            solved = False  # monte-carlo doesn't track full-game-solved status; only "search" sets this

            if args.solver == "race":
                # Timed round: pick instantly and play a batch per screen
                # read. No search budget - the clock is the scarce resource,
                # and with face-down cards the search couldn't plan past the
                # first one anyway (it reported "exhausted" every cycle).
                solver_started = time.perf_counter()
                state = State(cols, waste, stock_remaining, STOCK_TOTAL, found)
                batch_cap = args.moves_per_cycle if args.moves_per_cycle > 0 else 8
                batch = race_policy.plan_batch(
                    state, max_moves=batch_cap, exclude=excluded_first_moves,
                )
                solver_seconds = time.perf_counter() - solver_started
                log_event(
                    "solver_finished",
                    cycle=cycle_number,
                    solver="race",
                    duration_seconds=solver_seconds,
                    path_length=len(batch),
                    path=batch,
                )
                if batch:
                    previous_first_move = batch[0]
                    print(f"[*] Race: executing {len(batch)} move(s) this cycle:")
                    for idx, move in enumerate(batch):
                        print(f"  {idx + 1}. {move}")
                        ok = execute_move(
                            board, move, sim_mode=sim_mode, event_logger=log_event,
                        )
                        if ok and move[0] in ("col_to_found", "waste_to_found"):
                            last_cycle_found_plays += 1
                        log_event(
                            "move_result", cycle=cycle_number, batch_index=idx,
                            move=move, success=ok,
                        )
                        if not ok:
                            print("[*] Stopping batch early; re-reading next cycle.")
                            break
                        if move[0] in ("draw", "redeal"):
                            break
                        apply_move_to_board(board, move)
                else:
                    print("[*] Race: no productive move available (board stuck).")
                    log_event("no_move_selected", cycle=cycle_number, solver="race")
            elif args.solver == "monte-carlo":
                print("[*] Running Monte Carlo move search...")
                solver_started = time.perf_counter()
                state = State(cols, waste, stock_remaining, STOCK_TOTAL, found)
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
                    if ok and move[0] in ("col_to_found", "waste_to_found"):
                        last_cycle_found_plays += 1
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
                cycle_guard_excluded = set()
                state_permanent_exclusions = {
                    move for (signature, move) in permanently_excluded_by_state
                    if signature == current_solver_signature
                }
                for _attempt in range(4):
                    path, explored, solved, status = solve(
                        cols,
                        initial_waste=waste,
                        initial_stock_remaining=stock_remaining,
                        initial_stock_total=STOCK_TOTAL,
                        initial_found=found,
                        time_limit=5.0,
                        excluded_first_moves=(
                            excluded_first_moves | cycle_guard_excluded | state_permanent_exclusions
                        ),
                    )
                    if not path:
                        break
                    # Would committing this move just walk the board back into
                    # a state we've already visited in the last several
                    # cycles? Two occupied columns can score identically
                    # under the heuristic (e.g. a red Queen sitting on either
                    # of two black Kings), so the search can end up
                    # "preferring" to shuffle a card between them forever -
                    # each individual move genuinely succeeds (the
                    # single-previous-cycle stuck-move check above never
                    # fires), it just never goes anywhere.
                    start_state = State(cols, waste, stock_remaining, STOCK_TOTAL, found)
                    next_state = apply_move(start_state, path[0])
                    next_signature = (
                        next_state.cols, next_state.waste,
                        next_state.found, next_state.stock_remaining,
                    )
                    if next_signature not in recent_board_signatures:
                        break
                    print(f"[Warn] {path[0]} would revisit a recently seen board "
                          f"state; excluding it and re-solving.")
                    log_event("move_cycle_excluding", cycle=cycle_number, move=path[0])
                    cycle_guard_excluded.add(path[0])

                    # This exclusion only lasts for the rest of *this*
                    # cycle's retry loop (cycle_guard_excluded is rebuilt
                    # fresh next cycle) - when the only two productive-
                    # looking moves are symmetric under the heuristic (e.g.
                    # a red 6 that stacks equally well on either of two
                    # black 7s), both get excluded here, attempts run out,
                    # and the "proceeding with the best one found anyway"
                    # fallback below just re-executes the same oscillation
                    # next cycle. Track repeat offenders the same way the
                    # stuck-move guard above does, so a move caught cycling
                    # from this exact state several times gets permanently
                    # excluded from it instead of oscillating forever.
                    cycle_failure_key = (current_solver_signature, path[0])
                    cycle_fail_count = move_failure_counts.get(cycle_failure_key, 0) + 1
                    move_failure_counts[cycle_failure_key] = cycle_fail_count
                    if cycle_fail_count >= MOVE_PERMANENT_EXCLUDE_THRESHOLD:
                        print(f"[Warn] {path[0]} has now been caught cycling "
                              f"{cycle_fail_count} times from this exact board "
                              f"state; excluding it for this state.")
                        log_event(
                            "move_permanently_excluding",
                            cycle=cycle_number,
                            move=path[0],
                            fail_count=cycle_fail_count,
                            reason="cycling",
                        )
                        permanently_excluded_by_state.add(cycle_failure_key)
                else:
                    print("[Warn] Could not find a non-cycling move after several "
                          "attempts; proceeding with the best one found anyway.")
                solver_seconds = time.perf_counter() - solver_started
                log_event(
                    "solver_finished",
                    cycle=cycle_number,
                    solver="search",
                    duration_seconds=solver_seconds,
                    explored_states=explored,
                    solved=solved,
                    status=status,
                    path_length=len(path),
                    path=path,
                )

                if path:
                    batch = path if args.moves_per_cycle <= 0 else path[:args.moves_per_cycle]
                    previous_first_move = batch[0]
                    print(f"[*] Executing {len(batch)} move(s) this cycle:")
                    batch_completed = True
                    for idx, move in enumerate(batch):
                        print(f"  {idx + 1}. {move}")
                        ok = execute_move(
                            board,
                            move,
                            sim_mode=sim_mode,
                            event_logger=log_event,
                        )
                        if ok and move[0] in ("col_to_found", "waste_to_found"):
                            last_cycle_found_plays += 1
                        log_event(
                            "move_result",
                            cycle=cycle_number,
                            batch_index=idx,
                            move=move,
                            success=ok,
                        )
                        if not ok:
                            print("[*] Stopping batch early; will re-read the board next cycle.")
                            batch_completed = False
                            break
                        if move[0] in ("draw", "redeal"):
                            # Reveals/rearranges real physical cards this
                            # local model can't simulate (see
                            # apply_move_to_board's docstring) - stop the
                            # batch here so the next cycle re-reads the
                            # screen instead of planning on stale info.
                            print("[*] Stopping batch after draw/redeal; will re-read the board next cycle.")
                            batch_completed = False
                            break
                        apply_move_to_board(board, move)

                    # solve()'s `solved` flag only certifies a winning path exists
                    # from the analyzed position, not that this cycle executed all
                    # of it - only treat the game as finished once the full path
                    # has actually landed on the board.
                    solved = solved and batch_completed and len(batch) == len(path)
                    if solved:
                        print("[*] GAME SOLVED! All cards are on foundation.")
                        log_event("game_solved", cycle=cycle_number)
                else:
                    if solved:
                        print("[*] GAME SOLVED! All cards are on foundation.")
                        log_event("game_solved", cycle=cycle_number)
                    else:
                        print("[*] No moves found. Board might already be solved or no path exists.")
                        log_event("no_move_selected", cycle=cycle_number, solver="search")

            cycle_seconds = time.perf_counter() - cycle_started
            log_event("cycle_finished", cycle=cycle_number, duration_seconds=cycle_seconds)

            if sim_mode or solved:
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
