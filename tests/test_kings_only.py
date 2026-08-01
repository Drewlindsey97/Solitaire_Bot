from freecell_solver import State, generate_moves, UNKNOWN


def test_kings_only_to_empty():
    # build a simple state where one column has a non-king top card
    cols = [ [('Q','H')], [], [], [], [], [], [] ]
    waste = []
    state = State(cols, waste, 24, 24, {})
    moves = generate_moves(state)
    # ensure there is no move moving the Q to an empty column
    for m in moves:
        assert not (m[0] == 'col_to_col' and m[3][0] == 'Q' and m[2] in range(7) and cols[m[1]] == [('Q','H')] and cols[m[2]] == []), f'Unexpected non-king-to-empty move: {m}'
