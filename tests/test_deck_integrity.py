from freecell_solver import State, UNKNOWN, apply_move, generate_moves, solve
from monte_carlo_solver import total_card_count
from solver_state import build_solver_state


def card(rank, suit, score=.9):
    return dict(rank=rank, suit=suit, score=score)


def test_duplicate_tableau_reads_block_both_runs_and_preserve_slots():
    reading = build_solver_state({
        "col0": [card("K", "H"), card("Q", "S")],
        "col1": [card("K", "H", .99)],
    })
    assert reading["cols"][:2] == [[UNKNOWN, UNKNOWN], [UNKNOWN]]
    assert reading["truncated_columns"] == [0, 1]
    assert any("duplicate card KH" in issue for issue in reading["issues"])


def test_duplicate_between_tableau_and_exposed_waste_is_untrusted():
    reading = build_solver_state({"col0": [card("A", "H")], "waste": [card("A", "H")]})
    assert reading["cols"][0] == [UNKNOWN]
    assert reading["waste"] == [UNKNOWN]


def test_same_rank_different_suits_are_not_duplicates():
    reading = build_solver_state({"col0": [card("K", "H")], "col1": [card("K", "D")]})
    assert reading["truncated_columns"] == []
    assert not any("duplicate card" in issue for issue in reading["issues"])


def test_covered_waste_identity_is_not_trusted_as_a_duplicate():
    covered = dict(card("A", "H"), covered=True)
    reading = build_solver_state({"col0": [card("A", "H")], "waste": [covered]})
    assert reading["cols"][0] == [("A", "H")]
    assert reading["waste"] == [UNKNOWN]
    assert not any("duplicate card" in issue for issue in reading["issues"])


def test_redeal_conserves_52_cards_after_seven_stock_cards_were_founded():
    state = State([[UNKNOWN] * 4 for _ in range(7)], [UNKNOWN] * 17, 0, 24, {"H": 6})
    after = apply_move(state, ("redeal",))
    assert total_card_count(state) == total_card_count(after) == 52
    assert after.stock_remaining == 17
    assert not after.waste


def test_repeated_draws_and_redeals_preserve_known_order_and_count():
    cards = [("3", "H"), UNKNOWN, ("5", "S"), ("7", "C")]
    state = State([[] for _ in range(7)], cards, 0, 24, {})
    for _ in range(3):
        state = apply_move(state, ("redeal",))
        assert list(state.stock) == cards
        state = apply_move(state, ("draw",))
        assert list(state.waste) == cards[:3]
        state = apply_move(state, ("draw",))
        assert list(state.waste) == cards
        assert total_card_count(state) == len(cards)


def test_empty_waste_cannot_create_a_redeal():
    state = State([[UNKNOWN]] + [[] for _ in range(6)], [], 0, 24, {})
    assert generate_moves(state) == []
    after = apply_move(state, ("redeal",))
    assert total_card_count(after) == total_card_count(state)


def test_stock_identity_order_is_part_of_state_key():
    a = State([[]] * 7, [], 2, 24, {}, stock=[("A", "H"), ("2", "H")])
    b = State([[]] * 7, [], 2, 24, {}, stock=[("2", "H"), ("A", "H")])
    assert a.key() != b.key()


def test_search_can_solve_using_confirmed_stock_identity():
    path, _, solved, _ = solve([[]] * 7, [], 1, 24, {}, time_limit=.1,
                              initial_stock=[("A", "H")])
    assert solved
    assert path == [("draw",), ("waste_to_found", ("A", "H"))]
