from freecell_solver import State, generate_moves, apply_move, solve

# Trivial solve: Ace in col0 should be moved to foundation by solve()

def test_trivial_ace_solve():
    cols = [ [("A","H")], [], [], [], [], [], [] ]
    waste = []
    state_cols = cols
    path, explored, solved, status = solve(cols, waste, 24, 24, {})
    # path should include a col_to_found move
    assert path and any(m[0]=='col_to_found' for m in path), f"path={path}"

if __name__=='__main__':
    test_trivial_ace_solve()
    print('ok')
