from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from board_reader_lib import read_board, read_board_image
from solver_state import build_solver_state
from freecell_solver import UNKNOWN, State
from race_policy import plan_batch

ROOT = Path(__file__).resolve().parents[1]
ACE_FRAME = ROOT / "tests/fixtures/misread-heart-ace.png"


def test_live_heart_ace_read_and_foundation_priority():
    board = read_board(str(ACE_FRAME))
    assert (board["waste"][-1]["rank"], board["waste"][-1]["suit"]) == ("A", "H")
    state = build_solver_state(board)
    assert all(card == UNKNOWN for card in state["waste"][:-1])
    solver = State(state["cols"], state["waste"], state["stock_remaining"], 24, state["found"])
    assert plan_batch(solver, max_moves=5)[0] == ("waste_to_found", ("A", "H"))


def test_waste_fan_has_no_duplicate_peaks():
    board = read_board(str(ACE_FRAME))
    xs = [card["x"] for card in board["waste"]]
    assert len(xs) == 3
    assert all(b - a >= 20 for a, b in zip(xs, xs[1:]))


def test_covered_waste_does_not_contradict_foundation():
    covered = {"rank": "3", "suit": "D", "score": .6, "covered": True}
    foundation = {"rank": "3", "suit": "D", "score": .9}
    state = build_solver_state({"waste": [covered], "foundation": [foundation]}, stock_total=52)
    assert state["waste"] == [UNKNOWN]
    assert state["found"] == {"D": 2}
    assert state["issues"] == []


def test_memory_reader_matches_lossless_saved_pixels():
    for name in ("frame_0016.png", "frame_0028.png", "frame_0031.png"):
        path = ROOT / "Gameplay" / name
        rgb = np.asarray(Image.open(path).convert("RGB"))
        assert read_board_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) == read_board(str(path))
