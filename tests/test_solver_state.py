import unittest

from freecell_solver import UNKNOWN
from solver_state import build_solver_state


def card(rank="A", suit="H", **extra):
    return dict(rank=rank, suit=suit, score=0.9, **extra)


class SolverStateTests(unittest.TestCase):
    def test_invalid_tableau_rank_preserves_slots(self):
        state = build_solver_state({"col0": [card("bogus"), card("2")]})
        self.assertEqual(state["cols"][0], [UNKNOWN, UNKNOWN])
        self.assertEqual(state["truncated_columns"], [0])

    def test_invalid_waste_rank_becomes_unknown(self):
        state = build_solver_state({"waste": [card("bogus")]})
        self.assertEqual(state["waste"], [UNKNOWN])

    def test_reliable_flag_cannot_override_invalid_identity(self):
        for bad in (card("bogus", reliable=True),
                    card(suit="?", reliable=True),
                    {"rank": "A", "score": 0.9, "reliable": True},
                    {"rank": "A", "suit": "H", "reliable": True}):
            with self.subTest(card=bad):
                state = build_solver_state({"foundation": [bad]})
                self.assertEqual(state["found"], {})
                self.assertEqual(state["foundation_reads"], ["unreliable"])

    def test_explicit_unreliable_flag_is_respected(self):
        state = build_solver_state({"foundation": [card(reliable=False)]})
        self.assertEqual(state["found"], {})

    def test_valid_foundation_without_reliable_flag_is_accepted(self):
        state = build_solver_state({"foundation": [card()]})
        self.assertEqual(state["found"], {"H": 0})

    def test_visible_card_contradiction_drops_foundation(self):
        state = build_solver_state({"col0": [card()], "foundation": [card("2")]})
        self.assertEqual(state["found"], {})
        self.assertEqual(state["foundation_reads"], ["unreliable"])

    def test_tableau_sequence_break_truncates_to_unknown(self):
        # K(H) Q(S) J(H) 10(S) 9(H) 8(S) 7(H) 6(S) 5(H) 4(S) is a legal
        # alternating-descending run; an 8 can never legally follow a 4.
        cards = [card("K", "H"), card("Q", "S"), card("J", "H"), card("10", "S"),
                 card("9", "H"), card("8", "S"), card("7", "H"), card("6", "S"),
                 card("5", "H"), card("4", "S"), card("8", "D")]
        state = build_solver_state({"col0": cards})
        self.assertEqual(state["cols"][0][:10], [
            ("K", "H"), ("Q", "S"), ("J", "H"), ("10", "S"), ("9", "H"),
            ("8", "S"), ("7", "H"), ("6", "S"), ("5", "H"), ("4", "S"),
        ])
        self.assertEqual(state["cols"][0][10], UNKNOWN)
        self.assertEqual(state["truncated_columns"], [0])

    def test_legal_tableau_run_is_kept_whole(self):
        cards = [card("K", "H"), card("Q", "S"), card("J", "H")]
        state = build_solver_state({"col0": cards})
        self.assertEqual(state["cols"][0], [("K", "H"), ("Q", "S"), ("J", "H")])
        self.assertEqual(state["truncated_columns"], [])
