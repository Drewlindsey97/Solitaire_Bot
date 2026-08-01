from freecell_solver import State, generate_moves

# Test: multi-card run moves are offered

def test_run_moves_offered():
    # column 0 has K,Q,J (in order bottom->top), column 1 has Q of opposite color so the J can be placed on it
    cols = [ [("K","S"), ("Q","H"), ("J","S")], [("Q","D")], [], [], [], [], [] ]
    waste = []
    state = State(cols, waste, 24, 24, {})
    moves = generate_moves(state)
    # expect at least one col_to_col moving run from col0 to col1
    assert any(m[0]=='col_to_col' and m[1]==0 and m[2]==1 for m in moves), f"moves={moves}"

if __name__=='__main__':
    test_run_moves_offered()
    print('ok')
