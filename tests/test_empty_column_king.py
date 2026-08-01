from freecell_solver import State, generate_moves, UNKNOWN

# Test: only King may be moved to an empty column

def make_card(rank, suit):
    return (rank, suit)


def test_empty_column_only_king():
    cols = [[], [], [], [], [], [], []]
    # Source column has a Queen (movable single)
    cols[0] = [("Q","S")]
    waste = []
    stock_remaining = 24
    stock_total = 24
    found = {}
    state = State(cols, waste, stock_remaining, stock_total, found)
    moves = generate_moves(state)
    # There should be no move placing the Queen onto an empty column
    assert not any(m[0]=='col_to_col' and m[1]==0 and m[2]!=0 and (m[3][0] == 'Q' or (isinstance(m[3], tuple) and m[3][0]=='Q')) for m in moves), f"moves={moves}"

if __name__=='__main__':
    test_empty_column_only_king()
    print('ok')
