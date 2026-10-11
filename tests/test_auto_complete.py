"""Auto-complete endgame: with every card face up the game finishes the round
itself once the stock and waste are played out, so the bot plays those out
and stops instead of docking cards one by one."""
from freecell_solver import State
from race_policy import plan_batch, auto_complete_ready, choose_move

U = ("?", "?")


def test_ready_only_when_everything_is_face_up_and_the_pile_is_empty():
    cols = [[("K", "D"), ("Q", "S")]] + [[] for _ in range(6)]
    assert auto_complete_ready(State(cols, [], 0, 24, {}))
    assert not auto_complete_ready(State(cols, [("2", "H")], 0, 24, {}))
    assert not auto_complete_ready(State(cols, [], 3, 24, {}))
    hidden = [[U, ("K", "D")]] + [[] for _ in range(6)]
    assert not auto_complete_ready(State(hidden, [], 0, 24, {}))


def test_no_moves_once_ready():
    cols = [[("K", "D"), ("Q", "S")], [("A", "H")]] + [[] for _ in range(5)]
    assert plan_batch(State(cols, [], 0, 24, {}), max_moves=10) == []


def test_endgame_plays_the_waste_instead_of_docking():
    # AH could dock, but the 6C on the waste is what still needs to come out.
    cols = [[("7", "D")], [("A", "H")]] + [[] for _ in range(5)]
    batch = plan_batch(State(cols, [U, ("6", "C")], 5, 24, {}), max_moves=10)
    assert batch[0] == ("waste_to_col", 0, ("6", "C"))
    assert ("col_to_found", 1, ("A", "H")) not in batch


def test_endgame_draws_rather_than_docking():
    cols = [[("7", "D")], [("A", "H")]] + [[] for _ in range(5)]
    assert plan_batch(State(cols, [U, ("Q", "C")], 5, 24, {}), max_moves=10) == [("draw",)]


def test_endgame_docks_when_that_frees_a_spot_for_the_waste_card():
    # 2H can only go up once the AH is docked.
    cols = [[("A", "H")]] + [[] for _ in range(6)]
    batch = plan_batch(State(cols, [U, ("2", "H")], 5, 24, {}), max_moves=10)
    assert batch[:2] == [("col_to_found", 0, ("A", "H")), ("waste_to_found", ("2", "H"))]


def test_stalled_endgame_docks_again():
    cols = [[("7", "D")], [("A", "H")]] + [[] for _ in range(5)]
    state = State(cols, [U, ("Q", "C")], 5, 24, {})
    assert plan_batch(state, max_moves=10, endgame=False)[0] == ("col_to_found", 1, ("A", "H"))


def test_hidden_cards_keep_normal_play():
    cols = [[U, ("7", "D")], [("A", "H")]] + [[] for _ in range(5)]
    assert choose_move(State(cols, [U, ("Q", "C")], 5, 24, {})) == ("col_to_found", 1, ("A", "H"))


def test_endgame_docks_a_whole_line_when_the_waste_card_needs_it():
    # The live endgame screenshot: QC on the waste only goes up after
    # 9D, 10C, 10D, JD and JC are docked, so the bot must start docking
    # instead of redealing.
    cols = [[("K", "D"), ("Q", "S"), ("J", "D"), ("10", "C"), ("9", "D")], [("K", "S")],
            [("K", "C"), ("Q", "D"), ("J", "C"), ("10", "D")], [], [], [], []]
    state = State(cols, [("Q", "C")], 0, 24, {"S": 10, "H": 12, "D": 7, "C": 8})
    assert plan_batch(state, max_moves=10)[0] == ("col_to_found", 0, ("9", "D"))
