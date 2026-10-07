from pathlib import Path
import cv2
import numpy as np
from solitaire_auto_bot import reconcile_foundation_reads
from board_reader_lib import gameplay_screen_reason

ROOT = Path(__file__).resolve().parents[1]


def reconcile(card, accepted, pending, now):
    board = {"foundation": [card, None, None, None]}
    notes = reconcile_foundation_reads(board, accepted, pending, 2, lambda *a, **k: None, now)
    return board, notes


def ace(suit):
    return {"rank": "A", "suit": suit, "score": .9, "reliable": True}


def test_landing_frame_cannot_lock_in_wrong_ace_suit():
    accepted, pending = {}, {}
    board, notes = reconcile(ace("D"), accepted, pending, 0)
    assert notes and board["foundation"][0] is None
    board, notes = reconcile(ace("H"), accepted, pending, .3)
    assert notes and accepted[0] is None
    board, notes = reconcile(ace("H"), accepted, pending, .6)
    assert not notes and accepted[0]["suit"] == "H"
    assert board["foundation"][0]["suit"] == "H"


def test_established_pile_is_carried_through_obscured_read():
    accepted, pending = {0: ace("H")}, {}
    board, _ = reconcile({"rank": "J", "suit": "?", "reliable": False}, accepted, pending, 1)
    assert board["foundation"][0]["suit"] == "H"
    assert board["foundation"][0]["carried"]


def test_compatible_growth_does_not_need_extra_confirmation():
    accepted, pending = {0: ace("H")}, {}
    card = dict(ace("H"), rank="2")
    board, notes = reconcile(card, accepted, pending, 1)
    assert not notes and accepted[0]["rank"] == "2"


def test_results_panel_is_not_gameplay():
    assert gameplay_screen_reason(cv2.imread(str(ROOT / "tests/fixtures/completed-round.png")))


def test_calibrated_gameplay_remains_accepted():
    for name in ("frame_0016.png", "frame_0028.png", "frame_0100.png"):
        assert gameplay_screen_reason(cv2.imread(str(ROOT / "Gameplay" / name))) is None


def test_wrong_resolution_is_rejected_before_card_reading():
    assert gameplay_screen_reason(np.zeros((800, 360, 3), dtype=np.uint8))


def test_empty_foundation_slots_do_not_read_printed_ace_as_card():
    from board_reader_lib import read_board
    board = read_board(str(ROOT / "Gameplay" / "frame_0016.png"))
    image = cv2.imread(str(ROOT / "Gameplay" / "frame_0016.png"))
    # A clean, empty slot is green felt, not a white card face.
    from board_reader_lib import FOUNDATION_X, SLOT_Y, SLOT_W, SLOT_H
    for i, x in enumerate(FOUNDATION_X):
        patch = image[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        if ((hsv[..., 1] < 80) & (hsv[..., 2] > 200)).mean() < .25:
            assert board["foundation"][i] is None


def test_matching_source_play_confirms_foundation_without_extra_read():
    board = {"foundation": [ace("H"), None, None, None]}
    accepted, pending = {}, {}
    notes = reconcile_foundation_reads(board, accepted, pending, 2,
        lambda *a, **k: None, 0, expected_cards=[("A", "H")])
    assert not notes and accepted[0]["suit"] == "H"


def test_wrong_suit_is_not_confirmed_by_a_different_source_play():
    board = {"foundation": [ace("D"), None, None, None]}
    accepted, pending = {}, {}
    notes = reconcile_foundation_reads(board, accepted, pending, 2,
        lambda *a, **k: None, 0, expected_cards=[("A", "H")])
    assert notes and accepted[0] is None
