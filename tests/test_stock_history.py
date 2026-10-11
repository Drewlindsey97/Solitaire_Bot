from freecell_solver import UNKNOWN
from stock_history import StockHistory


def reading(waste=(), cols=None, found=None, issues=()):
    cols = cols or [[UNKNOWN] * 4 for _ in range(7)]
    found = found or {}
    return dict(cols=cols, waste=list(waste), found=found,
                stock_remaining=52 - sum(map(len, cols)) - len(waste)
                - sum(v + 1 for v in found.values()), issues=list(issues))


def fan(top, size=3):
    return [UNKNOWN] * (size - 1) + [top]


def test_mid_round_attachment_does_not_invent_hidden_waste_history():
    tracker = StockHistory()
    state = reading(fan(("4", "H")))
    tracker.observe(state)
    assert not state["stock_history_synced"]
    assert state["stock_cards"] is None
    assert len(state["waste"]) == 3


def test_confirmed_draws_retain_buried_waste_and_exact_stock_count():
    tracker = StockHistory()
    tracker.observe(reading())
    tracker.record_move(("draw",))
    first = reading(fan(("4", "H")))
    tracker.observe(first)
    assert first["stock_remaining"] == 21
    tracker.record_move(("draw",))
    second = reading(fan(("5", "S")))
    tracker.observe(second)
    assert second["stock_remaining"] == 18
    assert len(second["waste"]) == 6
    assert second["waste"][2] == ("4", "H")
    assert second["stock_history_synced"]


def test_failed_draw_does_not_advance_stock_or_waste():
    tracker = StockHistory()
    tracker.observe(reading())
    tracker.record_move(("draw",))
    tracker.observe(reading(fan(("4", "H"))))
    tracker.record_move(("draw",))
    unchanged = reading(fan(("4", "H")))
    tracker.observe(unchanged)
    assert unchanged["stock_history_synced"]
    assert unchanged["stock_remaining"] == 21
    assert len(unchanged["waste"]) == 3


def test_failed_first_draw_leaves_stock_undrawn():
    tracker = StockHistory()
    tracker.observe(reading())
    tracker.record_move(("draw",))
    unchanged = reading()
    tracker.observe(unchanged)
    assert unchanged["stock_remaining"] == 24
    assert not unchanged["waste"]


def test_redeal_remembers_only_observed_identities_in_draw_order():
    tracker = StockHistory()
    tracker.observe(reading())
    tops = [("4", "H"), ("5", "S"), ("6", "H"), ("7", "S"),
            ("8", "H"), ("9", "S"), ("10", "H"), ("J", "S")]
    for top in tops:
        tracker.record_move(("draw",))
        latest = reading(fan(top))
        tracker.observe(latest)
    assert latest["stock_remaining"] == 0
    tracker.record_move(("redeal",))
    redealt = reading()
    tracker.observe(redealt)
    assert redealt["stock_cards"][2::3] == tuple(tops)
    assert redealt["stock_cards"][0] == UNKNOWN
    tracker.record_move(("draw",))
    replay = reading(fan(tops[0]))
    tracker.observe(replay)
    assert replay["stock_remaining"] == 21
    assert replay["stock_history_synced"]


def test_confirmed_waste_play_updates_size_and_learns_newly_exposed_card():
    tracker = StockHistory()
    cols = [[UNKNOWN] * 4 for _ in range(7)]
    cols[0][-1] = ("5", "S")
    tracker.observe(reading(cols=cols))
    tracker.record_move(("draw",))
    tracker.observe(reading(fan(("4", "H")), cols=cols))
    tracker.record_move(("waste_to_col", 0, ("4", "H")))
    changed = [list(c) for c in cols]
    changed[0].append(("4", "H"))
    state = reading(fan(("3", "D"), 2), cols=changed)
    tracker.observe(state)
    assert state["stock_history_synced"]
    assert state["waste"] == [UNKNOWN, ("3", "D")]
    assert state["stock_remaining"] == 21


def test_mismatched_known_top_discards_history():
    tracker = StockHistory()
    tracker.observe(reading())
    tracker.record_move(("draw",))
    tracker.observe(reading(fan(("4", "H"))))
    mismatch = reading(fan(("4", "D")))
    notes = tracker.observe(mismatch)
    assert not mismatch["stock_history_synced"]
    assert notes


def test_untrusted_capture_discards_history():
    tracker = StockHistory()
    tracker.observe(reading())
    state = reading(issues=["duplicate card"])
    tracker.observe(state)
    assert not state["stock_history_synced"]


def test_duplicate_against_buried_history_discards_it():
    tracker = StockHistory()
    tracker.observe(reading())
    tracker.record_move(("draw",))
    tracker.observe(reading(fan(("4", "H"))))
    tracker.record_move(("draw",))
    tracker.observe(reading(fan(("5", "S"))))
    tracker.record_move(("draw",))
    duplicate = reading(fan(("4", "H")))
    tracker.observe(duplicate)
    assert not duplicate["stock_history_synced"]


def test_attach_after_redeal_seeds_unknown_order_without_guesses():
    tracker = StockHistory()
    tracker.observe(reading(fan(("4", "H"))))
    tracker.record_move(("redeal",))
    empty = reading()
    tracker.observe(empty)
    assert empty["stock_history_synced"]
    assert empty["stock_cards"] == (UNKNOWN,) * 24


def test_partial_fan_after_waste_play_keeps_older_covered_history():
    tracker = StockHistory()
    cols = [[UNKNOWN] * 4 for _ in range(7)]
    cols[0][-1] = ("6", "S")
    tracker.observe(reading(cols=cols))
    for top in [("4", "H"), ("5", "H")]:
        tracker.record_move(("draw",))
        tracker.observe(reading(fan(top), cols=cols))
    tracker.record_move(("waste_to_col", 0, ("5", "H")))
    changed = [list(c) for c in cols]
    changed[0].append(("5", "H"))
    latest = reading(fan(("4", "S"), 2), cols=changed)
    tracker.observe(latest)
    assert latest["stock_history_synced"]
    assert len(latest["waste"]) == 5
    assert latest["waste"][2] == ("4", "H")
    assert latest["stock_remaining"] == 18
