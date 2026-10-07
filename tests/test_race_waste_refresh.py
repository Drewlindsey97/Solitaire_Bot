from freecell_solver import State
from race_policy import plan_batch


def test_exposed_heart_ace_is_banked_before_draw():
    state = State([[] for _ in range(7)], [("A", "H")], 20, 24, {})
    assert plan_batch(state, max_moves=5) == [("waste_to_found", ("A", "H"))]


def test_refresh_after_waste_move_before_drawing_past_next_card():
    state = State([[] for _ in range(7)], [("Q", "H"), ("A", "S")], 20, 24, {})
    assert plan_batch(state, max_moves=5) == [("waste_to_found", ("A", "S"))]


def test_known_tableau_moves_still_batch():
    cols = [[("A", "H")], [("A", "S")]] + [[] for _ in range(5)]
    state = State(cols, [], 20, 24, {})
    batch = plan_batch(state, max_moves=5)
    assert batch[:2] == [("col_to_found", 0, ("A", "H")),
                         ("col_to_found", 1, ("A", "S"))]


def test_rejected_waste_move_is_not_repeated():
    cols = [[("5", "S")]] + [[] for _ in range(6)]
    state = State(cols, [("4", "H")], 20, 24, {})
    rejected = ("waste_to_col", 0, ("4", "H"))
    batch = plan_batch(state, max_moves=5, exclude={rejected})
    assert rejected not in batch
    assert batch == [("draw",)]
