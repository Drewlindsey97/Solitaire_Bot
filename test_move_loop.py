#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from freecell_solver import (
    State,
    apply_move,
    generate_complete_moves,
    generate_moves,
)
from monte_carlo_solver import choose_move_monte_carlo
from solitaire_auto_bot import (
    build_expected_transition,
    build_normalized_state,
    classify_verification_failure,
    compare_expected_transition,
    compare_normalized_state,
    choose_next_move,
    filter_safe_legal_moves,
    save_failure_artifacts,
    state_to_data,
    verify_expected_state,
    main,
)


def empty_board():
    board = {f"col{idx}": [] for idx in range(7)}
    board["free_cells"] = [None, None, None, None]
    board["foundation"] = [None]
    return board


def card(rank, suit=None, color=None):
    color = color or ("RED" if suit in ("H", "D") else "BLACK")
    payload = {"rank": rank, "color": color, "score": 1.0}
    if suit:
        payload["suit"] = suit
    return payload


def recognized_card(rank, suit, provenance, selected_source="rank_corner", score=0.9):
    payload = card(rank, suit)
    payload["recognition"] = {
        "rank_provenance": provenance,
        "selected_source": selected_source,
        "attempts": [
            {
                "source": selected_source,
                "rank": rank,
                "score": score,
                "margin": 0.2,
                "candidates": [{"rank": rank, "score": score, "variant": "test"}],
            }
        ],
    }
    return payload


def hidden_card():
    return {
        "rank": "?",
        "color": "?",
        "score": 0.0,
        "face_down": True,
        "provenance": "hidden_face_down",
    }


class MoveLoopTests(unittest.TestCase):
    def test_complete_legal_moves_are_separate_from_pruned_search_moves(self):
        state = State(
            cols=[[("A", "S")], [("K", "H")], [], [], [], [], []],
            free=[],
            found={},
        )

        complete = generate_complete_moves(state)
        pruned = generate_moves(state)

        self.assertIn(("col_to_found", 0, ("A", "S")), complete)
        self.assertIn(("col_to_free", 1, ("K", "H")), complete)
        self.assertEqual(pruned, [("col_to_found", 0, ("A", "S"))])

    def test_apply_move_returns_new_state_without_mutating_input(self):
        state = State(
            cols=[[("A", "S")], [], [], [], [], [], []],
            free=[],
            found={},
        )

        new_state = apply_move(state, ("col_to_found", 0, ("A", "S")))

        self.assertEqual(state.cols[0], (("A", "S"),))
        self.assertEqual(state.found_dict(), {})
        self.assertEqual(new_state.cols[0], ())
        self.assertEqual(new_state.found_dict(), {"S": 0})

    def test_apply_move_supports_every_move_type(self):
        cases = [
            (
                State([[("A", "S")], [], [], [], [], [], []], [], {}),
                ("col_to_found", 0, ("A", "S")),
                State([[], [], [], [], [], [], []], [], {"S": 0}),
            ),
            (
                State([[], [], [], [], [], [], []], [("A", "S")], {}),
                ("free_to_found", ("A", "S")),
                State([[], [], [], [], [], [], []], [], {"S": 0}),
            ),
            (
                State([[("Q", "H")], [("K", "S")], [], [], [], [], []], [], {}),
                ("col_to_col", 0, 1, ("Q", "H")),
                State([[], [("K", "S"), ("Q", "H")], [], [], [], [], []], [], {}),
            ),
            (
                State([[("Q", "H")], [], [], [], [], [], []], [], {}),
                ("col_to_free", 0, ("Q", "H")),
                State([[], [], [], [], [], [], []], [("Q", "H")], {}),
            ),
            (
                State([[("K", "S")], [], [], [], [], [], []], [("Q", "H")], {}),
                ("free_to_col", 0, ("Q", "H")),
                State([[("K", "S"), ("Q", "H")], [], [], [], [], [], []], [], {}),
            ),
        ]

        for initial, move, expected in cases:
            with self.subTest(move=move):
                self.assertEqual(apply_move(initial, move).key(), expected.key())

    def test_monte_carlo_obeys_supplied_legal_move_whitelist(self):
        state = State(
            cols=[[("A", "S")], [("K", "H")], [], [], [], [], []],
            free=[],
            found={},
        )
        whitelisted = [("col_to_free", 1, ("K", "H"))]

        move, statistics = choose_move_monte_carlo(
            state,
            legal_moves=whitelisted,
            simulations=10,
            time_limit=1.0,
            seed=1,
        )

        self.assertEqual(move, whitelisted[0])
        self.assertEqual([stats.move for stats in statistics], whitelisted)

    def test_monte_carlo_rejects_illegal_supplied_candidates(self):
        state = State([[("A", "S")], [], [], [], [], [], []], [], {})

        with self.assertRaises(ValueError):
            choose_move_monte_carlo(
                state,
                legal_moves=[("free_to_found", ("A", "S"))],
                simulations=1,
            )

    def test_verification_success(self):
        expected = State([[("K", "S")], [], [], [], [], [], []], [], {})
        actual = {
            "state": expected,
            "state_data": state_to_data(expected),
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "trustworthy": True,
            "trust_status": "trustworthy",
        }

        report = compare_normalized_state(expected, actual)

        self.assertTrue(report["matches"])
        self.assertEqual(report["differences"], [])

    def test_verification_mismatch_details(self):
        expected = State([[("K", "S")], [], [], [], [], [], []], [], {})
        actual_state = State([[("Q", "H")], [], [], [], [], [], []], [], {})
        actual = {
            "state": actual_state,
            "state_data": state_to_data(actual_state),
            "unresolved_cards": [{"area": "col0", "index": 1}],
            "ambiguous_cards": [],
            "trustworthy": False,
            "trust_status": "untrusted",
        }

        report = compare_normalized_state(expected, actual)

        fields = [diff["field"] for diff in report["differences"]]
        self.assertFalse(report["matches"])
        self.assertIn("columns", fields)
        self.assertIn("unresolved_cards", fields)
        self.assertIn("trust_status", fields)

    def test_verification_retry_succeeds_on_later_attempt(self):
        expected = State([[("K", "S")], [], [], [], [], [], []], [], {})
        wrong_state = State([[("Q", "H")], [], [], [], [], [], []], [], {})
        wrong = {
            "state": wrong_state,
            "state_data": state_to_data(wrong_state),
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "trustworthy": True,
            "trust_status": "trustworthy",
        }
        right = {
            "state": expected,
            "state_data": state_to_data(expected),
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "trustworthy": True,
            "trust_status": "trustworthy",
        }

        with patch("solitaire_auto_bot.capture_live_screenshot", return_value=True), \
                patch("solitaire_auto_bot.read_and_normalize_board", side_effect=[wrong, right]):
            result = verify_expected_state(
                expected_state=expected,
                allow_best_effort=False,
                attempts=3,
                delay=0,
                screenshot_prefix="verify_test",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)

    def test_verification_failure_saves_artifacts(self):
        expected = State([[("K", "S")], [], [], [], [], [], []], [], {})
        actual_state = State([[("Q", "H")], [], [], [], [], [], []], [], {})
        previous = {
            "state_data": state_to_data(State([[("Q", "H")], [], [], [], [], [], []], [], {})),
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "trust_status": "trustworthy",
        }
        actual = {
            "state": actual_state,
            "state_data": state_to_data(actual_state),
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "trustworthy": True,
            "trust_status": "trustworthy",
        }
        mismatch = compare_normalized_state(expected, actual)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            before = temp_path / "before.png"
            verify = temp_path / "verify.png"
            before.write_bytes(b"before")
            verify.write_bytes(b"verify")
            failure_dir = temp_path / "failure"

            with patch("solitaire_auto_bot.failure_artifact_dir", return_value=failure_dir):
                saved = save_failure_artifacts(
                    before_screenshot=before,
                    verification_screenshots=[verify],
                    previous_normalized=previous,
                    expected_state=expected,
                    selected_move=("col_to_col", 0, 1, ("Q", "H")),
                    final_actual_normalized=actual,
                    mismatch_report=mismatch,
                    session_log_path=Path("logs/session.jsonl"),
                    logcat_path=Path("logs/logcat.log"),
                )

            self.assertEqual(saved, failure_dir)
            self.assertTrue((failure_dir / "before.png").exists())
            self.assertTrue((failure_dir / "verification_1.png").exists())
            self.assertTrue((failure_dir / "mismatch_report.json").exists())

    def test_live_mode_executes_only_one_gesture_per_captured_state(self):
        initial = State([[("A", "S")], [], [], [], [], [], []], [], {})
        verified = apply_move(initial, ("col_to_found", 0, ("A", "S")))
        empty = State([[], [], [], [], [], [], []], [], {})
        normalized_initial = {
            "board": empty_board() | {"col0": [card("A", "S")]},
            "state": initial,
            "state_data": state_to_data(initial),
            "columns": [[("A", "S")], [], [], [], [], [], []],
            "free": [],
            "foundations": {},
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "truncated_columns": [],
            "trustworthy": True,
            "trust_status": "trustworthy",
        }
        normalized_verified = dict(normalized_initial)
        normalized_verified.update({
            "state": verified,
            "state_data": state_to_data(verified),
            "columns": [[], [], [], [], [], [], []],
            "foundations": {"S": 0},
        })
        normalized_empty = dict(normalized_verified)
        normalized_empty.update({
            "state": empty,
            "state_data": state_to_data(empty),
            "foundations": {},
        })
        execute = Mock(return_value={"ok": True, "gesture": "tap", "reason": None})

        with patch("sys.argv", ["solitaire_auto_bot.py", "--verify-delay", "0", "--verify-attempts", "1"]), \
                patch("solitaire_auto_bot.capture_live_screenshot", return_value=True), \
                patch("solitaire_auto_bot.read_and_normalize_board", side_effect=[
                    normalized_initial,
                    normalized_verified,
                    normalized_empty,
                ]), \
                patch("solitaire_auto_bot.execute_move", execute):
            main()

        self.assertEqual(execute.call_count, 1)

    def test_unresolved_identity_prevents_unsafe_live_execution(self):
        unsafe = {
            "board": empty_board() | {"col0": [{"rank": "?", "color": "?", "score": 0.0}]},
            "state": State([[], [], [], [], [], [], []], [], {}),
            "state_data": state_to_data(State([[], [], [], [], [], [], []], [], {})),
            "columns": [[], [], [], [], [], [], []],
            "free": [],
            "foundations": {},
            "unresolved_cards": [{"area": "col0", "index": 0}],
            "ambiguous_cards": [],
            "truncated_columns": [],
            "trustworthy": False,
            "trust_status": "untrusted",
        }
        execute = Mock(return_value={"ok": True})

        with patch("sys.argv", ["solitaire_auto_bot.py"]), \
                patch("solitaire_auto_bot.capture_live_screenshot", return_value=True), \
                patch("solitaire_auto_bot.read_and_normalize_board", return_value=unsafe), \
                patch("solitaire_auto_bot.execute_move", execute):
            main()

        execute.assert_not_called()

    def test_simulation_mode_still_runs(self):
        state = State([[("A", "S")], [], [], [], [], [], []], [], {})
        normalized = {
            "board": empty_board() | {"col0": [card("A", "S")]},
            "state": state,
            "state_data": state_to_data(state),
            "columns": [[("A", "S")], [], [], [], [], [], []],
            "free": [],
            "foundations": {},
            "unresolved_cards": [],
            "ambiguous_cards": [],
            "truncated_columns": [],
            "trustworthy": True,
            "trust_status": "trustworthy",
        }
        execute = Mock(return_value={"ok": True, "gesture": "tap", "reason": None})

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file, \
                patch("sys.argv", ["solitaire_auto_bot.py", "--sim", temp_file.name]), \
                patch("solitaire_auto_bot.read_and_normalize_board", return_value=normalized), \
                patch("solitaire_auto_bot.execute_move", execute):
            main()

        self.assertGreaterEqual(execute.call_count, 1)

    def test_existing_logging_behavior_still_writes_jsonl(self):
        from session_logger import SessionLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.jsonl"
            logger = SessionLogger(path, session_id="move-loop")
            logger.event("verification", move=("col_to_found", 0, ("A", "S")))
            logger.close()

            text = path.read_text(encoding="utf-8")
            self.assertIn('"event": "verification"', text)
            self.assertIn('"move-loop"', text)

    def test_build_normalized_state_tracks_uncertainty(self):
        board = empty_board()
        board["col0"] = [
            {"rank": "?", "color": "?", "score": 0.0},
            card("K", color="BLACK"),
            card("K", color="BLACK"),
            card("K", color="BLACK"),
        ]

        normalized = build_normalized_state(board, allow_best_effort=True)

        self.assertFalse(normalized["trustworthy"])
        self.assertEqual(normalized["board"]["col0"][1]["suit_source"], "resolved_by_constraints")
        self.assertEqual(normalized["board"]["col0"][3]["suit_source"], "ambiguous")

    def test_heuristic_only_rank_is_not_live_trustworthy(self):
        board = empty_board()
        board["col0"] = [recognized_card("Q", "S", "shape_heuristic_only", selected_source="live_shape_heuristic")]

        normalized = build_normalized_state(board)

        self.assertFalse(normalized["trustworthy"])
        self.assertEqual(normalized["unresolved_cards"][0]["reason"], "unresolved")
        self.assertEqual(normalized["columns"][0], [])

    def test_disagreement_between_rank_recognizers_makes_card_ambiguous(self):
        board = empty_board()
        board["col0"] = [recognized_card("5", "S", "conflicting_recognizers")]

        normalized = build_normalized_state(board)

        self.assertFalse(normalized["trustworthy"])
        self.assertEqual(normalized["ambiguous_cards"][0]["reason"], "ambiguous")
        self.assertEqual(normalized["columns"][0], [])

    def test_incorrect_high_heuristic_score_does_not_produce_trustworthy_state(self):
        board = empty_board()
        bad = recognized_card("Q", "S", "shape_heuristic_only", selected_source="live_shape_heuristic", score=0.99)
        board["col0"] = [bad]

        normalized = build_normalized_state(board)

        self.assertFalse(normalized["trustworthy"])
        self.assertEqual(normalized["unresolved_cards"][0]["card"]["recognition"]["rank_provenance"], "shape_heuristic_only")

    def test_visually_confirmed_ace_produces_foundation_move(self):
        board = empty_board()
        board["col0"] = [recognized_card("A", "S", "corner_glyph_confirmed", selected_source="ace_corner_detector")]

        normalized = build_normalized_state(board)
        legal, _, _ = filter_safe_legal_moves(generate_complete_moves(normalized["state"]), normalized)

        self.assertTrue(normalized["trustworthy"])
        self.assertIn(("col_to_found", 0, ("A", "S")), legal)

    def test_safe_ace_to_foundation_is_selected_before_col_to_free(self):
        state = State([[("A", "S")], [("K", "H")], [], [], [], [], []], [], {})
        legal_moves = generate_complete_moves(state)

        selected, selection = choose_next_move(
            SimpleNamespace(solver="monte-carlo"),
            state,
            legal_moves,
            lambda *args, **kwargs: None,
        )

        self.assertEqual(selected, ("col_to_found", 0, ("A", "S")))
        self.assertEqual(selection["reason"], "safe_ace_foundation_priority")

    def test_hidden_cards_do_not_make_state_untrusted(self):
        board = empty_board()
        board["col0"] = [hidden_card(), hidden_card(), card("K", "S")]

        normalized = build_normalized_state(board)

        self.assertTrue(normalized["trustworthy"])
        self.assertEqual(normalized["hidden_counts"], [2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(normalized["unresolved_exposed_count"], 0)
        self.assertEqual(normalized["columns"][0], [("K", "S")])

    def test_verification_classifies_unchanged_board_as_gesture_not_applied(self):
        previous_state = State([[("5", "S")], [], [], [], [], [], []], [], {})
        expected_state = apply_move(previous_state, ("col_to_free", 0, ("5", "S")))
        previous = {
            "state_data": state_to_data(previous_state),
        }
        actual = {
            "state_data": state_to_data(previous_state),
        }

        self.assertEqual(
            classify_verification_failure(expected_state, actual, previous_normalized=previous),
            "gesture_not_applied",
        )

    def test_unresolved_exposed_card_makes_state_untrusted(self):
        board = empty_board()
        board["col0"] = [hidden_card(), {"rank": "?", "color": "?", "score": 0.0, "face_down": False}]

        normalized = build_normalized_state(board, allow_best_effort=True)

        self.assertFalse(normalized["trustworthy"])
        self.assertEqual(normalized["hidden_counts"][0], 1)
        self.assertEqual(len(normalized["unresolved_exposed_cards"]), 1)

    def test_ambiguous_free_cell_makes_state_untrusted(self):
        board = empty_board()
        board["free_cells"][0] = {"rank": "K", "color": "BLACK", "score": 0.9, "suit": "?", "suit_source": "ambiguous"}

        normalized = build_normalized_state(board)

        self.assertFalse(normalized["trustworthy"])
        self.assertEqual(normalized["ambiguous_exposed_cards"][0]["area"], "free_cells")

    def test_solver_state_contains_only_visible_cards_and_preserves_hidden_counts(self):
        board = empty_board()
        board["col0"] = [hidden_card(), hidden_card(), card("Q", "H"), card("J", "S")]

        normalized = build_normalized_state(board)

        self.assertEqual(normalized["columns"][0], [("Q", "H"), ("J", "S")])
        self.assertEqual(normalized["state"].cols[0], (("Q", "H"), ("J", "S")))
        self.assertEqual(normalized["observed_columns"][0]["hidden_count"], 2)
        self.assertEqual(len(normalized["observed_columns"][0]["visible_cards"]), 2)

    def test_moving_last_visible_card_can_reveal_unknown_hidden_card(self):
        board = empty_board()
        board["col0"] = [hidden_card(), card("A", "S")]
        normalized = build_normalized_state(board)

        transition = build_expected_transition(normalized, ("col_to_found", 0, ("A", "S")))

        self.assertEqual(transition.kind, "reveal")
        self.assertEqual(transition.reveal_column, 0)
        self.assertEqual(transition.hidden_counts_before[0], 1)
        self.assertEqual(transition.hidden_counts_after[0], 0)
        self.assertEqual(transition.expected_state.cols[0], ())

    def test_reveal_transition_verification_succeeds_structurally(self):
        board = empty_board()
        board["col0"] = [hidden_card(), card("A", "S")]
        before = build_normalized_state(board)
        transition = build_expected_transition(before, ("col_to_found", 0, ("A", "S")))

        after_board = empty_board()
        after_board["col0"] = [card("K", "H")]
        after = build_normalized_state(after_board)

        report = compare_expected_transition(transition, after)

        self.assertTrue(report["matches"])

    def test_reveal_transition_verification_fails_when_hidden_count_does_not_decrease(self):
        board = empty_board()
        board["col0"] = [hidden_card(), card("A", "S")]
        before = build_normalized_state(board)
        transition = build_expected_transition(before, ("col_to_found", 0, ("A", "S")))

        after_board = empty_board()
        after_board["col0"] = [hidden_card(), card("K", "H")]
        after = build_normalized_state(after_board)

        report = compare_expected_transition(transition, after)

        self.assertFalse(report["matches"])
        self.assertIn("hidden_counts", [diff["field"] for diff in report["differences"]])

    def test_hidden_card_identity_is_not_invented(self):
        board = empty_board()
        board["col0"] = [hidden_card(), card("A", "S")]
        normalized = build_normalized_state(board)
        transition = build_expected_transition(normalized, ("col_to_found", 0, ("A", "S")))

        self.assertEqual(normalized["columns"][0], [("A", "S")])
        self.assertEqual(transition.expected_state.cols[0], ())
        self.assertNotIn(("?", "?"), transition.expected_state.cols[0])

    def test_unresolved_exposed_destination_is_not_treated_as_empty(self):
        board = empty_board()
        board["col0"] = [card("8", "S")]
        board["col1"] = [{"rank": "?", "color": "?", "score": 0.0, "face_down": False}]
        normalized = build_normalized_state(board, allow_best_effort=True)

        safe_moves, _, excluded = filter_safe_legal_moves(generate_complete_moves(normalized["state"]), normalized)

        self.assertNotIn(("col_to_col", 0, 1, ("8", "S")), safe_moves)
        self.assertTrue(any(item["blocked_by_unresolved_destination"] for item in excluded))

    def test_no_move_may_target_unresolved_exposed_destination(self):
        board = empty_board()
        board["col0"] = [card("8", "S")]
        board["col1"] = [{"rank": "?", "color": "?", "score": 0.0, "face_down": False}]
        board["col2"] = [card("9", "H")]
        normalized = build_normalized_state(board, allow_best_effort=True)

        safe_moves, _, _ = filter_safe_legal_moves(generate_complete_moves(normalized["state"]), normalized)

        self.assertFalse(any(move[0] == "col_to_col" and move[2] == 1 for move in safe_moves))

    def test_no_move_may_originate_from_unresolved_exposed_source(self):
        board = empty_board()
        board["col0"] = [{"rank": "?", "color": "?", "score": 0.0, "face_down": False}]
        board["col1"] = [card("9", "H")]
        normalized = build_normalized_state(board, allow_best_effort=True)

        safe_moves, _, excluded = filter_safe_legal_moves(generate_complete_moves(normalized["state"]), normalized)

        self.assertFalse(any(move[0].startswith("col_to") and move[1] == 0 for move in safe_moves))
        self.assertFalse(any(move == ("col_to_col", 0, 1, ("?", "?")) for move in safe_moves))
        self.assertTrue(any(item["blocked_by_unresolved_source"] for item in excluded))

    def test_fully_known_columns_still_generate_legal_moves(self):
        board = empty_board()
        board["col0"] = [card("8", "S")]
        board["col1"] = [card("9", "H")]
        normalized = build_normalized_state(board)

        safe_moves, _, excluded = filter_safe_legal_moves(generate_complete_moves(normalized["state"]), normalized)

        self.assertIn(("col_to_col", 0, 1, ("8", "S")), safe_moves)
        self.assertEqual(excluded, [])

    def test_hidden_face_down_cards_do_not_block_unrelated_moves(self):
        board = empty_board()
        board["col0"] = [hidden_card(), card("K", "S")]
        board["col1"] = [card("8", "S")]
        board["col2"] = [card("9", "H")]
        normalized = build_normalized_state(board)

        safe_moves, _, _ = filter_safe_legal_moves(generate_complete_moves(normalized["state"]), normalized)

        self.assertIn(("col_to_col", 1, 2, ("8", "S")), safe_moves)


if __name__ == "__main__":
    unittest.main()
