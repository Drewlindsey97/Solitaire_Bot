from klondike_solver import KlondikeState, solve, generate_moves, apply_move

# Simple test: one tableau pile with Ace up on first column -> should move to foundation

def make_card(rank,suit):
    return (rank,suit)


def test_ace_to_foundation():
    tableau = [ ([], []), ([], []), ([], []), ([], []), ([], []), ([], []), ([], []) ]
    # Put Ace of Hearts faceUp in col 0
    tableau[0] = ([], [make_card('A','H')])
    state = KlondikeState(tableau)
    moves = generate_moves(state)
    assert any(m[0]=='col_to_found' for m in moves), f"moves={moves}"
    # solver returns full solution only when game is won; here verify move generation and application
    assert any(m[0]=='col_to_found' for m in moves), f"no col_to_found in moves {moves}"
    # apply the move and ensure foundation updated
    from klondike_solver import apply_move
    m = [m for m in moves if m[0]=='col_to_found'][0]
    new = apply_move(state, m)
    assert new.foundations['H'] == 0, f"foundation not updated: {new.foundations}"


if __name__=='__main__':
    test_ace_to_foundation()
    print('ok')
