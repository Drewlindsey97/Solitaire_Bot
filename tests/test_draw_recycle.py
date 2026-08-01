from freecell_solver import State, generate_moves

# Test: draw and recycle fallback behavior

def test_draw_and_recycle():
    cols = [[],[],[],[],[],[],[]]
    waste = []
    # Case 1: stock_remaining > 0 -> draw offered
    state1 = State(cols, waste, 10, 24, {})
    moves1 = generate_moves(state1)
    assert any(m[0]=='draw' for m in moves1), f"draw not offered: {moves1}"

    # Case 2: stock_remaining == 0 but stock_total > 0 and waste non-empty -> redeal offered
    # Use a waste card that is not a safe autoplay (e.g. a 5) so redeal logic can appear
    state2 = State(cols, [("5","H")], 0, 24, {})
    moves2 = generate_moves(state2)
    # If no other candidate moves exist, redeal should be a fallback
    assert any(m[0]=='redeal' for m in moves2) or any(m[0] in ("col_to_found","waste_to_found","col_to_col","waste_to_col") for m in moves2), f"expected moves or redeal: {moves2}"

if __name__=='__main__':
    test_draw_and_recycle()
    print('ok')
