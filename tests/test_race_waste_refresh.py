from freecell_solver import State
from race_policy import plan_batch


def test_exposed_heart_ace_is_banked_before_draw():
    state = State([[] for _ in range(7)], [("A", "H")], 20, 24, {})
    assert plan_batch(state, max_moves=5) == [("waste_to_found", ("A", "H"))]


def test_refresh_after_waste_move_before_drawing_past_next_card():
    # Covered waste cards reach the solver as UNKNOWN unless stock history
    # has confirmed them, so the batch must stop and re-read.
    unknown = ("?", "?")
    state = State([[] for _ in range(7)], [unknown, ("A", "S")], 20, 24, {})
    assert plan_batch(state, max_moves=5) == [("waste_to_found", ("A", "S"))]


def test_confirmed_waste_history_keeps_the_batch_going():
    # Stock history confirmed the 2S under the AS, so both bank in one read.
    state = State([[] for _ in range(7)], [("2", "S"), ("A", "S")], 20, 24, {})
    assert plan_batch(state, max_moves=5)[:2] == [
        ("waste_to_found", ("A", "S")), ("waste_to_found", ("2", "S"))]


def test_confirmed_stock_lets_the_batch_draw_and_play():
    # Empty waste, next draw known to turn up the AH on top: draw, then bank.
    unknown = ("?", "?")
    stock = [("9", "C"), ("7", "D"), ("A", "H")] + [unknown] * 17
    state = State([[] for _ in range(7)], [], 20, 24, {}, stock=stock)
    assert plan_batch(state, max_moves=5)[:2] == [("draw",), ("waste_to_found", ("A", "H"))]


def test_unknown_draw_still_ends_the_batch():
    state = State([[] for _ in range(7)], [], 20, 24, {})
    assert plan_batch(state, max_moves=5) == [("draw",)]


def test_known_tableau_moves_still_batch():
    cols = [[("A", "H")], [("A", "S")]] + [[] for _ in range(5)]
    state = State(cols, [], 20, 24, {})
    # endgame=False: with every card face up the bot would otherwise leave
    # docking to the game's auto-complete (tests/test_auto_complete.py).
    batch = plan_batch(state, max_moves=5, endgame=False)
    assert batch[:2] == [("col_to_found", 0, ("A", "H")),
                         ("col_to_found", 1, ("A", "S"))]


def test_rejected_waste_move_is_not_repeated():
    cols = [[("5", "S")]] + [[] for _ in range(6)]
    state = State(cols, [("4", "H")], 20, 24, {})
    rejected = ("waste_to_col", 0, ("4", "H"))
    batch = plan_batch(state, max_moves=5, exclude={rejected})
    assert rejected not in batch
    assert batch == [("draw",)]
