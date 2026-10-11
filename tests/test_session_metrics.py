from session_metrics import summarize


def test_dispatches_are_not_reported_as_confirmed_moves():
    result = summarize([
        {"event": "session_started", "timestamp": "2026-10-08T00:00:00+00:00"},
        *[{"event": "gesture_dispatched"} for _ in range(5)],
        {"event": "move_verification", "dispatched": 2, "confirmed": 2},
        {"event": "move_verification", "dispatched": 1, "confirmed": 0},
        {"event": "move_verification", "dispatched": 1, "confirmed": None},
        {"event": "solver_state_built", "foundations": {"H": 0}},
        {"event": "solver_state_built", "foundations": {"H": 2}},
        {"event": "solver_finished", "duration_seconds": .02},
        {"event": "solver_finished", "duration_seconds": .04},
        {"event": "session_finished", "timestamp": "2026-10-08T00:00:10+00:00"},
    ])
    assert result["moves_confirmed"] == 2
    assert result["moves_confirmed_unchanged"] == 1
    assert result["moves_unverified"] == 1
    assert result["moves_awaiting_capture"] == 1
    assert result["confirmed_moves_per_second"] == .2
    assert result["foundation_gain_observed"] == 2
    assert result["median_seconds"]["solver_finished"] == .03


def test_missing_boundaries_do_not_invent_an_elapsed_time_or_rate():
    result = summarize([{"event": "gesture_dispatched"}])
    assert result["elapsed_seconds"] is None
    assert result["confirmed_moves_per_second"] is None
