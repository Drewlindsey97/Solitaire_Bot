#!/usr/bin/env python3
import time
import sys
import os
import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
from board_reader_lib import (
    read_board, BoardLayout, save_calibration_artifacts,
    save_exposed_card_diagnostics,
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
        if card.get("face_down"):
            card["suit_source"] = "hidden_face_down"
            return
        if card.get("suit_source") == "ambiguous":
            card["suit"] = "?"
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
def _round_point(point):
    return {"x": int(round(point[0])), "y": int(round(point[1]))}


def _bbox_data(x, y, w, h):
    return {
        "x": int(round(x)),
        "y": int(round(y)),
        "w": int(round(w)),
        "h": int(round(h)),
    }


def _slot_bbox(layout, item_type, index):
    if item_type == "free":
        x = layout.free_cell_x[index]
    elif item_type == "found":
        foundation_index = min(index, len(layout.foundation_x) - 1)
        x = layout.foundation_x[foundation_index]
    else:
        x = layout.tableau_x[index]
    return _bbox_data(x, layout.slot_y, layout.slot_width, layout.slot_height)


def _bbox_center(bbox):
    return (
        bbox["x"] + bbox["w"] / 2,
        bbox["y"] + bbox["h"] / 2,
    )


def _original_point(normalized_point, transform):
    if transform is None:
        return normalized_point
    return transform.normalized_to_original(*normalized_point)


def _transform_log(transform):
    if transform is None:
        return None
    return transform.to_json_data()


def column_hidden_count(board, index):
    observed = board.get("observed_columns")
    if observed:
        return observed[index]["hidden_count"]
    return sum(1 for c in board.get(f"col{index}", []) if c.get("face_down"))


def column_visible_cards(board, index):
    observed = board.get("observed_columns")
    if observed:
        return observed[index]["visible_cards"]
    return [c for c in board.get(f"col{index}", []) if not c.get("face_down")]


def get_element_point(board, item_type, index, transform=None, layout=None, role="destination"):
    """
    Calculates normalized and original center coordinates for slots and cards.
    """
    layout = layout or BoardLayout()
    bbox = None
    source = None
    if item_type == "col":
        visible_cards = column_visible_cards(board, index)
        top_bbox = visible_cards[-1].get("bbox") if visible_cards else None
        if role == "source" and top_bbox:
            bbox = dict(top_bbox)
            bbox["h"] = min(bbox.get("h", layout.full_card_height), layout.slot_height)
            source = "observed_card_bbox"
        elif role == "destination" and top_bbox:
            bbox = _bbox_data(
                top_bbox["x"],
                top_bbox["y"],
                top_bbox["w"],
                min(top_bbox.get("h", layout.full_card_height), layout.slot_height),
            )
            source = "observed_destination_card_bbox"
        else:
            # No observed bbox for the top card (e.g. a card appended by
            # apply_move_to_board's simulation-only bookkeeping) -- fall back
            # to the calibrated stack position instead of assuming the top of
            # the column, so cards stacked under hidden/revealed cards still
            # get a plausible drop point.
            hidden_count = column_hidden_count(board, index)
            revealed_count = len(visible_cards)
            if hidden_count + revealed_count == 0:
                y = layout.tableau_y_top
                source = "calibrated_empty_column_slot"
            else:
                y = (
                    layout.tableau_y_top
                    + hidden_count * layout.hidden_card_step
                    + max(0, revealed_count - 1) * layout.revealed_card_step
                )
                source = "calibrated_column_fallback"
            bbox = _bbox_data(
                layout.tableau_x[index],
                y,
                layout.column_width,
                layout.slot_height,
            )

    elif item_type == "free":
        bbox = _slot_bbox(layout, "free", index)
        source = "calibrated_free_cell_slot"

    elif item_type == "found":
        bbox = _slot_bbox(layout, "found", index)
        source = "calibrated_foundation_slot"
    else:
        return None

    normalized = _bbox_center(bbox)
    original = _original_point(normalized, transform)
    return {
        "normalized": _round_point(normalized),
        "original": _round_point(original),
        "bbox": bbox,
        "source": source,
    }


def get_element_coords(board, item_type, index, transform=None, layout=None):
    point = get_element_point(board, item_type, index, transform=transform, layout=layout)
    if point is None:
        return None
    original = point["original"]
    return original["x"], original["y"]


def find_free_cell_index(board, rank, suit):
    for idx, c in enumerate(board["free_cells"]):
        if c and c.get("rank") == rank and c.get("suit") == suit:
            return idx
    return None


def find_empty_free_cell_index(board):
    for idx, c in enumerate(board["free_cells"]):
        if c is None:
            return idx
    return None


def plan_gesture(board, move, transform=None, layout=None):
    layout = layout or BoardLayout()
    kind = move[0]
    card = move[-1]
    start = None
    end = None

    if kind == "col_to_found":
        _, ci, card = move
        start = get_element_point(board, "col", ci, transform=transform, layout=layout, role="source")
        end = get_element_point(board, "found", 0, transform=transform, layout=layout)
    elif kind == "free_to_found":
        _, card = move
        fi = find_free_cell_index(board, card[0], card[1])
        if fi is not None:
            start = get_element_point(board, "free", fi, transform=transform, layout=layout, role="source")
            end = get_element_point(board, "found", 0, transform=transform, layout=layout)
    elif kind == "col_to_col":
        _, ci, cj, card = move
        start = get_element_point(board, "col", ci, transform=transform, layout=layout, role="source")
        end = get_element_point(board, "col", cj, transform=transform, layout=layout)
    elif kind == "col_to_free":
        _, ci, card = move
        fi = find_empty_free_cell_index(board)
        if fi is not None:
            start = get_element_point(board, "col", ci, transform=transform, layout=layout, role="source")
            end = get_element_point(board, "free", fi, transform=transform, layout=layout)
    elif kind == "free_to_col":
        _, cj, card = move
        fi = find_free_cell_index(board, card[0], card[1])
        if fi is not None:
            start = get_element_point(board, "free", fi, transform=transform, layout=layout, role="source")
            end = get_element_point(board, "col", cj, transform=transform, layout=layout)

    if start is None or end is None:
        return None

    gesture = "tap" if kind in ("col_to_found", "free_to_found") else "swipe"
    return {
        "move": move,
        "card": card,
        "gesture": gesture,
        "source_normalized": start["normalized"],
        "source_original": start["original"],
        "destination_normalized": end["normalized"],
        "destination_original": end["original"],
        "layout_profile": layout.to_json_data(),
        "transform": _transform_log(transform),
        "observed_source_bounding_box": start["bbox"],
        "source_target": start["source"],
        "destination_bounding_box_or_slot": end["bbox"],
        "destination_target": end["source"],
    }


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

    if kind == "col_to_found":
        _, ci, card = move
        board[f"col{ci}"].pop()
    elif kind == "free_to_found":
        _, card = move
        fi = find_free_cell_index(board, card[0], card[1])
        if fi is not None:
            board["free_cells"][fi] = None
    elif kind == "col_to_col":
        _, ci, cj, card = move
        board[f"col{ci}"].pop()
        board[f"col{cj}"].append({"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0})
    elif kind == "col_to_free":
        _, ci, card = move
        board[f"col{ci}"].pop()
        fi = find_empty_free_cell_index(board)
        if fi is not None:
            board["free_cells"][fi] = {"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0}
    elif kind == "free_to_col":
        _, cj, card = move
        fi = find_free_cell_index(board, card[0], card[1])
        if fi is not None:
            board["free_cells"][fi] = None
        board[f"col{cj}"].append({"rank": card[0], "suit": card[1], "color": card_color(card[1]), "score": 1.0})


# ==============================================================================
# 3. MOVE TRANSLATION TO PHYSICAL GESTURES
# ==============================================================================
def execute_move(board, move, sim_mode=False, event_logger=None, transform=None, layout=None):
    """
    Translates a solver move into coordinates and triggers the swipe/tap.
    """
    kind = move[0]
    card = move[-1]
    emit = event_logger or (lambda event_name, **data: None)
    result = {
        "ok": False,
        "move": move,
        "gesture": None,
        "start": None,
        "end": None,
        "reason": None,
        "simulation": sim_mode,
    }

    # Sanity-check: for moves that pop the top of a tableau column, make sure
    # the physical top card actually matches what the solver believes is
    # there. Board state and solver state can diverge (e.g. a revealed card
    # whose suit couldn't be read gets dropped from the solver's view), and
    # blindly swiping in that case would drag the wrong real card.
    if kind in ("col_to_found", "col_to_col", "col_to_free"):
        ci = move[1]
        physical_col = column_visible_cards(board, ci)
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

    plan = plan_gesture(board, move, transform=transform, layout=layout)
    if plan:
        x1 = plan["source_original"]["x"]
        y1 = plan["source_original"]["y"]
        x2 = plan["destination_original"]["x"]
        y2 = plan["destination_original"]["y"]
        result["gesture"] = plan["gesture"]
        result["start"] = plan["source_original"]
        result["end"] = plan["destination_original"]
        result["gesture_plan"] = plan
        if plan["gesture"] == "tap":
            print(f"[*] Action: Tap to Foundation: {card} at ({x1}, {y1})")
            emit("gesture_planned", **plan, start=plan["source_original"], simulation=sim_mode)
            if sim_mode:
                print(f"   [Simulation] Would tap: bridge.tap({x1}, {y1})")
            else:
                bridge.tap(x1, y1)
        else:
            print(
                f"[*] Action: Move {kind.replace('_', ' ')}: {card} "
                f"from ({x1}, {y1}) to ({x2}, {y2})"
            )
            emit("gesture_planned", **plan, start=plan["source_original"], end=plan["destination_original"], simulation=sim_mode)
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
            if card and not card.get("face_down") and (card.get("rank") == "?" or card.get("color") == "?"):
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


@dataclass(frozen=True)
class ExpectedTransition:
    kind: str
    expected_state: State
    hidden_counts_before: list[int]
    hidden_counts_after: list[int]
    reveal_column: int | None = None

    def to_json_data(self):
        return {
            "kind": self.kind,
            "expected_state": state_to_data(self.expected_state),
            "hidden_counts_before": self.hidden_counts_before,
            "hidden_counts_after": self.hidden_counts_after,
            "reveal_column": self.reveal_column,
        }


def observation_to_data(observation):
    if observation is None:
        return None
    if hasattr(observation, "to_json_data"):
        return observation.to_json_data()
    return observation


def hidden_counts_from_normalized(normalized):
    observed = normalized.get("observed_columns")
    if observed is not None:
        return [column["hidden_count"] for column in observed]
    return [
        sum(1 for card in normalized.get("board", {}).get(f"col{idx}", []) if card.get("face_down"))
        for idx in range(7)
    ]


def build_expected_transition(previous_normalized, move):
    expected_state = apply_move(previous_normalized["state"], move)
    before_hidden = hidden_counts_from_normalized(previous_normalized)
    after_hidden = list(before_hidden)
    reveal_column = None
    if move[0] in ("col_to_found", "col_to_col", "col_to_free"):
        source_col = move[1]
        observed = previous_normalized.get("observed_columns")
        if observed is not None:
            visible = observed[source_col]["visible_cards"]
        else:
            visible = previous_normalized["board"].get(f"col{source_col}", [])
        if len(visible) == 1 and before_hidden[source_col] > 0:
            after_hidden[source_col] -= 1
            reveal_column = source_col
    kind = "reveal" if reveal_column is not None else "deterministic"
    return ExpectedTransition(kind, expected_state, before_hidden, after_hidden, reveal_column)


def _identity_status(card):
    if card is None:
        return "empty"
    rank_status = _rank_trust_status(card)
    if rank_status in ("unresolved", "ambiguous"):
        return rank_status
    if card.get("rank") == "?" or card.get("color") == "?" or card.get("suit_source") == "unresolved":
        return "unresolved"
    if card.get("suit_source") == "ambiguous" or card.get("suit") not in ("S", "H", "D", "C"):
        return "ambiguous"
    return "known"


def _rank_trust_status(card):
    recognition = card.get("recognition") if card else None
    if not recognition:
        return "known"
    provenance = recognition.get("rank_provenance")
    if provenance in ("template_confirmed", "corner_glyph_confirmed", "cross_source_corroborated"):
        return "known"
    if provenance in ("conflicting_recognizers",):
        return "ambiguous"
    if provenance in ("shape_heuristic_only", "unresolved"):
        return "unresolved"
    return "unresolved"


def _top_visible_card(normalized, column_index):
    observed = normalized.get("observed_columns", [])
    if not observed:
        return None
    cards = observed[column_index]["visible_cards"]
    return cards[-1] if cards else None


def _column_structurally_empty(normalized, column_index):
    observed = normalized.get("observed_columns", [])
    if not observed:
        return not normalized["state"].cols[column_index]
    return observed[column_index]["hidden_count"] == 0 and not observed[column_index]["visible_cards"]


def assess_move_safety(move, normalized):
    reasons = []
    kind = move[0]

    def check_source_col(column_index):
        status = _identity_status(_top_visible_card(normalized, column_index))
        if status == "unresolved":
            reasons.append("blocked_by_unresolved_source")
        elif status == "ambiguous":
            reasons.append("blocked_by_ambiguous_identity")

    def check_dest_col(column_index):
        if _column_structurally_empty(normalized, column_index):
            return
        status = _identity_status(_top_visible_card(normalized, column_index))
        if status == "unresolved":
            reasons.append("blocked_by_unresolved_destination")
        elif status == "ambiguous":
            reasons.append("blocked_by_ambiguous_identity")

    def check_free_card(card_tuple):
        for card in normalized["board"].get("free_cells", []):
            if card and card.get("rank") == card_tuple[0] and card.get("suit") == card_tuple[1]:
                status = _identity_status(card)
                if status == "unresolved":
                    reasons.append("blocked_by_unresolved_source")
                elif status == "ambiguous":
                    reasons.append("blocked_by_ambiguous_identity")
                return
        reasons.append("blocked_by_unresolved_source")

    if kind in ("col_to_found", "col_to_free"):
        check_source_col(move[1])
    elif kind == "col_to_col":
        check_source_col(move[1])
        check_dest_col(move[2])
    elif kind == "free_to_found":
        check_free_card(move[1])
    elif kind == "free_to_col":
        check_free_card(move[2])
        check_dest_col(move[1])

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "move": move,
        "safe": not unique_reasons,
        "blocked_by_unresolved_source": "blocked_by_unresolved_source" in unique_reasons,
        "blocked_by_unresolved_destination": "blocked_by_unresolved_destination" in unique_reasons,
        "blocked_by_ambiguous_identity": "blocked_by_ambiguous_identity" in unique_reasons,
        "reasons": unique_reasons,
    }


def filter_safe_legal_moves(moves, normalized):
    assessments = [assess_move_safety(move, normalized) for move in moves]
    generated_sources = {
        move[1]
        for move in moves
        if move[0] in ("col_to_found", "col_to_col", "col_to_free")
    }
    for column_index, column in enumerate(normalized.get("observed_columns", [])):
        if column_index in generated_sources or not column["visible_cards"]:
            continue
        status = _identity_status(column["visible_cards"][-1])
        if status == "unresolved":
            assessments.append({
                "move": ("blocked_col_source", column_index),
                "safe": False,
                "blocked_by_unresolved_source": True,
                "blocked_by_unresolved_destination": False,
                "blocked_by_ambiguous_identity": False,
                "reasons": ["blocked_by_unresolved_source"],
            })
        elif status == "ambiguous":
            assessments.append({
                "move": ("blocked_col_source", column_index),
                "safe": False,
                "blocked_by_unresolved_source": False,
                "blocked_by_unresolved_destination": False,
                "blocked_by_ambiguous_identity": True,
                "reasons": ["blocked_by_ambiguous_identity"],
            })
    safe_moves = [item["move"] for item in assessments if item["safe"]]
    excluded = [item for item in assessments if not item["safe"]]
    return safe_moves, assessments, excluded


def compare_expected_transition(expected_transition, actual_normalized):
    if expected_transition.kind == "deterministic":
        return compare_normalized_state(expected_transition.expected_state, actual_normalized)

    differences = []
    actual_hidden = hidden_counts_from_normalized(actual_normalized)
    if actual_hidden != expected_transition.hidden_counts_after:
        differences.append({
            "field": "hidden_counts",
            "expected": expected_transition.hidden_counts_after,
            "actual": actual_hidden,
        })
    col = expected_transition.reveal_column
    if col is not None:
        actual_visible = actual_normalized["observed_columns"][col]["visible_cards"]
        expected_visible_len = len(expected_transition.expected_state.cols[col]) + 1
        if len(actual_visible) != expected_visible_len:
            differences.append({
                "field": f"col{col}_visible_count",
                "expected": expected_visible_len,
                "actual": len(actual_visible),
            })
        elif actual_visible and (
            actual_visible[-1].get("rank") == "?" or
            actual_visible[-1].get("color") == "?" or
            actual_visible[-1].get("suit_source") in ("ambiguous", "unresolved")
        ):
            differences.append({
                "field": f"col{col}_revealed_card",
                "expected": "new readable exposed card",
                "actual": actual_visible[-1],
            })
    if actual_normalized["ambiguous_cards"] or actual_normalized["unresolved_cards"]:
        differences.append({
            "field": "trust_reasons",
            "expected": [],
            "actual": {
                "unresolved_exposed_cards": actual_normalized["unresolved_cards"],
                "ambiguous_cards": actual_normalized["ambiguous_cards"],
            },
        })
    return {
        "matches": not differences,
        "differences": differences,
        "expected_transition": expected_transition.to_json_data(),
        "actual_state": actual_normalized["state_data"],
        "actual_hidden_counts": actual_hidden,
        "actual_unresolved_cards": actual_normalized["unresolved_cards"],
        "actual_ambiguous_cards": actual_normalized["ambiguous_cards"],
        "actual_trust_status": actual_normalized["trust_status"],
    }


def build_normalized_state(board, allow_best_effort=False):
    """
    Split raw OCR observations from solver identities.

    The solver state contains only exact card identities. Each exact suit on
    the returned board is annotated with suit_source so live mode can refuse
    unsafe execution when the current state is not trustworthy.
    """
    resolved_board = deepcopy(board)
    assign_pseudo_suits(resolved_board)

    unresolved_exposed_cards = []
    ambiguous_exposed_cards = []
    truncated_columns = []
    cols = []
    observed_columns = []
    untrusted_reasons = []

    def note_card(collection, index, card, reason):
        payload = {
            "area": collection,
            "index": index,
            "card": deepcopy(card),
            "reason": reason,
        }
        if reason == "ambiguous":
            ambiguous_exposed_cards.append(payload)
        else:
            unresolved_exposed_cards.append(payload)

    for idx in range(7):
        col = []
        hidden_count = 0
        visible_cards = []
        for card_index, card in enumerate(resolved_board[f"col{idx}"]):
            if card.get("face_down"):
                hidden_count += 1
                continue
            visible_cards.append(card)
            source = card.get("suit_source")
            rank_status = _rank_trust_status(card)
            if card.get("rank") == "?" or card.get("color") == "?":
                note_card(f"col{idx}", card_index, card, "unresolved")
                if not allow_best_effort:
                    break
                continue
            if rank_status == "unresolved":
                note_card(f"col{idx}", card_index, card, "unresolved")
                if not allow_best_effort:
                    break
                continue
            if rank_status == "ambiguous":
                note_card(f"col{idx}", card_index, card, "ambiguous")
                truncated_columns.append(idx)
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
        observed_columns.append({
            "hidden_count": hidden_count,
            "visible_cards": visible_cards,
        })

    free = []
    for idx, card in enumerate(resolved_board["free_cells"]):
        if not card:
            continue
        source = card.get("suit_source")
        rank_status = _rank_trust_status(card)
        if card.get("rank") == "?" or card.get("color") == "?" or rank_status == "unresolved":
            note_card("free_cells", idx, card, "unresolved")
        elif source == "ambiguous" or rank_status == "ambiguous" or card.get("suit") not in ("S", "H", "D", "C"):
            note_card("free_cells", idx, card, "ambiguous")
        else:
            free.append((card["rank"], card["suit"]))

    found = {}
    for idx, card in enumerate(resolved_board["foundation"]):
        if not card:
            continue
        source = card.get("suit_source")
        rank_status = _rank_trust_status(card)
        if card.get("rank") == "?" or card.get("color") == "?" or rank_status == "unresolved":
            note_card("foundation", idx, card, "unresolved")
        elif source == "ambiguous" or rank_status == "ambiguous" or card.get("suit") not in ("S", "H", "D", "C"):
            note_card("foundation", idx, card, "ambiguous")
        else:
            found[card["suit"]] = rank_val(card["rank"])

    state = State(cols, free, found)
    if unresolved_exposed_cards:
        untrusted_reasons.append("unresolved_exposed_cards")
    if ambiguous_exposed_cards:
        untrusted_reasons.append("ambiguous_exposed_cards")
    trustworthy = not untrusted_reasons
    trust_status = "trustworthy" if trustworthy else "untrusted"

    return {
        "board": resolved_board,
        "observed_columns": observed_columns,
        "hidden_counts": [column["hidden_count"] for column in observed_columns],
        "visible_confidence": [
            [card.get("score", 0.0) for card in column["visible_cards"]]
            for column in observed_columns
        ],
        "state": state,
        "state_data": state_to_data(state),
        "columns": cols,
        "free": free,
        "foundations": found,
        "unresolved_cards": unresolved_exposed_cards,
        "ambiguous_cards": ambiguous_exposed_cards,
        "unresolved_exposed_cards": unresolved_exposed_cards,
        "ambiguous_exposed_cards": ambiguous_exposed_cards,
        "unresolved_exposed_count": len(unresolved_exposed_cards),
        "truncated_columns": truncated_columns,
        "trustworthy": trustworthy,
        "trust_status": trust_status,
        "untrusted_reasons": untrusted_reasons,
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


def classify_verification_failure(expected_state, actual_normalized, previous_normalized=None):
    if actual_normalized is None:
        return "board_read_failed"
    expected = state_to_data(expected_state)
    actual = actual_normalized["state_data"]
    previous = previous_normalized["state_data"] if previous_normalized else None
    if previous is not None and actual == previous and actual != expected:
        return "gesture_not_applied"
    if actual == expected:
        return "move_applied_but_verification_failed"
    return "unrecognized_divergence"


def read_and_normalize_board(screenshot_path, allow_best_effort=False, layout=None):
    read_result = read_board(str(screenshot_path), layout=layout, include_metadata=True)
    normalized = build_normalized_state(read_result["board"], allow_best_effort=allow_best_effort)
    normalized["observation"] = observation_to_data(read_result.get("observation"))
    normalized["layout"] = read_result["layout"]
    normalized["transform"] = read_result["transform"]
    normalized["detection_report"] = read_result["detection_report"]
    return normalized


def capture_live_screenshot(path):
    img = bridge.screenshot()
    if img is None:
        return False
    img.save(path)
    return True


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def draw_gesture_preview(screenshot_path, gesture_plan, output_path):
    image = cv2.imread(str(screenshot_path))
    if image is None:
        raise FileNotFoundError(screenshot_path)
    source = gesture_plan["source_original"]
    dest = gesture_plan["destination_original"]
    start = (int(source["x"]), int(source["y"]))
    end = (int(dest["x"]), int(dest["y"]))
    cv2.arrowedLine(image, start, end, (0, 255, 255), 5, tipLength=0.08)
    cv2.circle(image, start, 14, (0, 0, 255), -1)
    cv2.circle(image, end, 14, (255, 0, 0), -1)
    cv2.putText(image, "source", (start[0] + 16, start[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "destination", (end[0] + 16, end[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise RuntimeError(f"failed to write preview image: {out}")
    return out


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
        "hidden_counts": previous_normalized.get("hidden_counts", []),
        "unresolved_exposed_cards": previous_normalized.get("unresolved_exposed_cards", previous_normalized["unresolved_cards"]),
        "ambiguous_exposed_cards": previous_normalized.get("ambiguous_exposed_cards", previous_normalized["ambiguous_cards"]),
    })
    write_json(directory / "expected_state.json", state_to_data(expected_state))
    write_json(directory / "final_actual_state.json", {
        "state": final_actual_normalized["state_data"] if final_actual_normalized else None,
        "trust_status": final_actual_normalized["trust_status"] if final_actual_normalized else "unavailable",
        "unresolved_cards": final_actual_normalized["unresolved_cards"] if final_actual_normalized else [],
        "ambiguous_cards": final_actual_normalized["ambiguous_cards"] if final_actual_normalized else [],
        "hidden_counts": final_actual_normalized.get("hidden_counts", []) if final_actual_normalized else [],
        "unresolved_exposed_cards": final_actual_normalized.get("unresolved_exposed_cards", final_actual_normalized["unresolved_cards"]) if final_actual_normalized else [],
        "ambiguous_exposed_cards": final_actual_normalized.get("ambiguous_exposed_cards", final_actual_normalized["ambiguous_cards"]) if final_actual_normalized else [],
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
    expected_transition=None,
    previous_normalized=None,
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
            if expected_transition is not None:
                final_report = compare_expected_transition(expected_transition, final_actual)
            else:
                final_report = compare_normalized_state(expected_state, final_actual)
            if not final_report["matches"]:
                final_report["classification"] = classify_verification_failure(
                    expected_state,
                    final_actual,
                    previous_normalized=previous_normalized,
                )
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

    # The search solver already computes a multi-move path for the cycle;
    # short-circuiting here would silently discard it and drop batching down
    # to a single move (main()'s batch selection only uses selection["path"]
    # when present). Only bypass the solver for solvers that never batch, so
    # an obviously-safe ace move skips a solver call without losing a path.
    if args.solver != "search":
        ace_foundation_moves = [
            move for move in legal_moves
            if move[0] in ("col_to_found", "free_to_found") and move[-1][0] == "A"
        ]
        if ace_foundation_moves:
            move = ace_foundation_moves[0]
            log_event(
                "solver_finished",
                solver=args.solver,
                duration_seconds=0.0,
                candidate_moves=legal_moves,
                selected_move=move,
                selection_reason="safe_ace_foundation_priority",
            )
            return move, {
                "reason": "safe_ace_foundation_priority",
                "candidate_moves": legal_moves,
            }

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
    parser.add_argument(
        "--layout-profile",
        type=str,
        help="Optional JSON BoardLayout profile to use instead of the built-in reference layout.",
    )
    parser.add_argument(
        "--calibrate-layout",
        action="store_true",
        help="Write board-reader calibration artifacts for the selected screenshot and exit without gestures.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="logs/calibration/latest",
        help="Directory for --calibrate-layout artifacts. Default: logs/calibration/latest.",
    )
    parser.add_argument(
        "--debug-exposed-card",
        action="append",
        default=[],
        metavar="COLUMN,INDEX",
        help="Write targeted diagnostics for an exposed tableau card and exit without gestures. Repeatable.",
    )
    parser.add_argument(
        "--preview-gesture",
        type=str,
        help="Draw the selected gesture over the original screenshot and exit without executing it.",
    )
    args = parser.parse_args()

    sim_mode = args.sim is not None
    screenshot_file = args.sim if sim_mode else "live_screen.png"
    layout = BoardLayout.from_json(args.layout_profile) if args.layout_profile else BoardLayout()

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

    if args.calibrate_layout:
        if not sim_mode:
            print("[Error] --calibrate-layout requires --sim and never captures or gestures on a live device.", file=sys.stderr)
            if session_logger is not None:
                session_logger.close()
            sys.exit(2)
        print(f"[*] Writing calibration artifacts to: {args.calibration_dir}")
        report = save_calibration_artifacts(screenshot_file, args.calibration_dir, layout=layout)
        print("[*] Calibration complete.")
        for idx, column in enumerate(report["columns"]):
            rows = column.get("rows", [])
            rejections = column.get("rejections", [])
            print(f"  col{idx}: rows={len(rows)} rejections={rejections}")
        if session_logger is not None:
            session_logger.close()
        return

    if args.debug_exposed_card:
        if not sim_mode:
            print("[Error] --debug-exposed-card requires --sim and never captures or gestures on a live device.", file=sys.stderr)
            if session_logger is not None:
                session_logger.close()
            sys.exit(2)
        for spec in args.debug_exposed_card:
            try:
                column_text, index_text = spec.split(",", 1)
                column = int(column_text)
                index = int(index_text)
            except ValueError:
                print(f"[Error] Invalid --debug-exposed-card value: {spec!r}. Expected COLUMN,INDEX.", file=sys.stderr)
                if session_logger is not None:
                    session_logger.close()
                sys.exit(2)
            output_dir = Path(args.calibration_dir)
            if len(args.debug_exposed_card) > 1:
                output_dir = output_dir / f"col{column}_{index}"
            print(f"[*] Writing exposed-card diagnostics for col{column}[{index}] to: {output_dir}")
            diagnostic = save_exposed_card_diagnostics(
                screenshot_file,
                column,
                index,
                output_dir,
                layout=layout,
            )
            result = diagnostic["recognition"]["current_recognition_result"]
            rank_result = diagnostic["recognition"]["independent_rank_result"]
            print(f"  current={result}")
            print(f"  rank={rank_result['rank']} score={rank_result['score']} source={rank_result['selected_source']}")
        if session_logger is not None:
            session_logger.close()
        return

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
                    layout=layout,
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
            unresolved_cards = normalized.get("unresolved_exposed_count", len(normalized["unresolved_cards"]))
            log_event(
                "board_read",
                cycle=cycle_number,
                duration_seconds=board_seconds,
                hidden_counts=normalized.get("hidden_counts", []),
                unresolved_exposed_count=normalized.get("unresolved_exposed_count", len(normalized["unresolved_cards"])),
                unresolved_exposed_cards=normalized.get("unresolved_exposed_cards", normalized["unresolved_cards"]),
                ambiguous_exposed_cards=normalized.get("ambiguous_exposed_cards", normalized["ambiguous_cards"]),
                visible_confidence=normalized.get("visible_confidence", []),
                trust_status=normalized["trust_status"],
                untrusted_reasons=normalized.get("untrusted_reasons", []),
                board=board,
                observation=normalized.get("observation"),
            )

            print("[*] Board Cards Detected:")
            for idx in range(7):
                col_key = f"col{idx}"
                observed = normalized.get("observed_columns", [{"hidden_count": 0, "visible_cards": board.get(f"col{i}", [])} for i in range(7)])[idx]
                col_info = [
                    f"{c['rank']}({c['color']},{c.get('suit', '?')},{c.get('suit_source', '?')},score={c.get('score', 0.0)})"
                    for c in observed["visible_cards"]
                ]
                print(f"  col{idx}: hidden={observed['hidden_count']} visible={col_info}")

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
            print(f"  Hidden: {normalized.get('hidden_counts', [])}")
            print(f"  Unresolved exposed: {len(normalized.get('unresolved_exposed_cards', normalized['unresolved_cards']))}")
            print(f"  Ambiguous exposed: {len(normalized.get('ambiguous_exposed_cards', normalized['ambiguous_cards']))}")
            print(f"  Trust: {normalized['trust_status']}")
            log_event(
                "solver_state_built",
                cycle=cycle_number,
                columns=normalized["columns"],
                free=normalized["free"],
                foundations=normalized["foundations"],
                hidden_counts=normalized.get("hidden_counts", []),
                unresolved_exposed_cards=normalized.get("unresolved_exposed_cards", normalized["unresolved_cards"]),
                ambiguous_exposed_cards=normalized.get("ambiguous_exposed_cards", normalized["ambiguous_cards"]),
                visible_confidence=normalized.get("visible_confidence", []),
                truncated_columns=normalized["truncated_columns"],
                trust_status=normalized["trust_status"],
                untrusted_reasons=normalized.get("untrusted_reasons", []),
            )

            if not sim_mode and not normalized["trustworthy"]:
                print("[Error] Current state is not trustworthy enough for live execution. Stopping.")
                log_event(
                    "unsafe_state_stopped",
                    cycle=cycle_number,
                    unresolved_exposed_cards=normalized.get("unresolved_exposed_cards", normalized["unresolved_cards"]),
                    ambiguous_exposed_cards=normalized.get("ambiguous_exposed_cards", normalized["ambiguous_cards"]),
                    untrusted_reasons=normalized.get("untrusted_reasons", []),
                )
                break

            raw_legal_moves = generate_complete_moves(state)
            legal_moves, candidate_safety, excluded_candidates = filter_safe_legal_moves(raw_legal_moves, normalized)
            log_event(
                "legal_moves_generated",
                cycle=cycle_number,
                legal_moves=legal_moves,
                raw_legal_moves=raw_legal_moves,
                candidate_safety=candidate_safety,
                excluded_candidates=excluded_candidates,
            )
            print(f"[*] Legal moves: {len(legal_moves)}")
            if excluded_candidates:
                print(f"[*] Excluded unsafe candidate moves: {len(excluded_candidates)}")

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

            current_legal_moves, _, _ = filter_safe_legal_moves(generate_complete_moves(state), normalized)
            if move not in current_legal_moves:
                print(f"[Error] Selected move is no longer legal: {move}")
                log_event("move_rejected", cycle=cycle_number, move=move, reason="selected_move_not_legal")
                break

            expected_transition = build_expected_transition(normalized, move)
            expected_state = expected_transition.expected_state
            log_event(
                "move_selected",
                cycle=cycle_number,
                move=move,
                selection=selection,
                expected_state=state_to_data(expected_state),
                expected_transition=expected_transition.to_json_data(),
            )

            if args.preview_gesture:
                plan = plan_gesture(
                    board,
                    move,
                    transform=normalized.get("transform"),
                    layout=normalized.get("layout"),
                )
                if plan is None:
                    print(f"[Error] Failed to resolve preview coordinates for move: {move}", file=sys.stderr)
                    log_event("move_rejected", cycle=cycle_number, move=move, reason="coordinate_resolution_failed")
                    break
                preview_path = draw_gesture_preview(before_screenshot, plan, args.preview_gesture)
                log_event("gesture_preview_written", cycle=cycle_number, path=preview_path, gesture_plan=plan)
                print(f"[*] Gesture preview written to: {preview_path}")
                print(f"[*] Preview move: {move}")
                print(f"[*] Source normalized: {plan['source_normalized']} original: {plan['source_original']}")
                print(f"[*] Destination normalized: {plan['destination_normalized']} original: {plan['destination_original']}")
                break

            if sim_mode and args.solver == "search" and selection.get("path"):
                batch = selection["path"] if args.moves_per_cycle <= 0 else selection["path"][:args.moves_per_cycle]
            else:
                batch = [move]

            print(f"[*] Executing {len(batch)} move(s) this cycle:")
            current_state = state
            gesture_result = {"ok": False, "reason": "not_dispatched"}
            for idx, batch_move in enumerate(batch):
                current_batch_moves, _, _ = filter_safe_legal_moves(generate_complete_moves(current_state), normalized)
                if batch_move not in current_batch_moves:
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
                    transform=normalized.get("transform"),
                    layout=normalized.get("layout"),
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
                expected_transition=expected_transition,
                previous_normalized=normalized,
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
