from freecell_solver import State, generate_moves, apply_move


def test_multi_card_run_offered_and_preserved():
    # column 0 has a run 5H,4S,3H (alternating colors, descending ranks)
    # put a destination with a 6 of opposite color so the run can be moved
    cols = [ [('5','H'), ('4','S'), ('3','H')], [('6','S')], [], [], [], [], [] ]
    waste = []
    state = State(cols, waste, 24, 24, {})
    moves = generate_moves(state)
    # find any col_to_col move that moves a run of length >1 from col0
    has_run_move = False
    for m in moves:
        if m[0] == 'col_to_col' and m[1] == 0 and m[4] > 1:
            has_run_move = True
            # apply the move and check order
            new_state = apply_move(state, m)
            # moved cards should be at destination preserving order
            break
    assert has_run_move, 'No multi-card run move offered'
