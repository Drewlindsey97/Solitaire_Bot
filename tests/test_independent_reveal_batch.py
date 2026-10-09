from freecell_solver import UNKNOWN, State, apply_move
from race_policy import plan_batch


def test_independent_reveals_share_one_read_without_drawing_past_them():
    state = State([[UNKNOWN, ('A', 'H')], [UNKNOWN, ('A', 'S')]] + [[]] * 5,
                  [], 24, 24, {})
    batch = plan_batch(state, max_moves=10)
    assert batch == [('col_to_found', 0, ('A', 'H')),
                     ('col_to_found', 1, ('A', 'S'))]
    for move in batch:
        state = apply_move(state, move)
    assert state.cols[0][-1] == UNKNOWN
    assert state.cols[1][-1] == UNKNOWN


def test_independent_tableau_moves_continue_after_a_reveal():
    state = State([[UNKNOWN, ('6', 'C')], [('7', 'H')],
                   [UNKNOWN, ('4', 'S')], [('5', 'D')], [], [], []],
                  [], 24, 24, {})
    batch = plan_batch(state, max_moves=10)
    assert len(batch) >= 2
    assert {move[1] for move in batch[:2]} == {0, 2}
    assert all(move[0] == 'col_to_col' for move in batch)
    assert all(move[1] not in (0, 2) and move[2] not in (0, 2)
               for move in batch[2:])


def test_batch_limit_still_applies_to_independent_reveals():
    state = State([[UNKNOWN, ('A', 'H')], [UNKNOWN, ('A', 'S')]] + [[]] * 5,
                  [], 24, 24, {})
    assert len(plan_batch(state, max_moves=1)) == 1
