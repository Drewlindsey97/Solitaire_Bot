import bridge


def test_explicit_swipe_duration_is_fixed_in_competitive_mode(monkeypatch):
    monkeypatch.setattr(bridge, "HUMAN_MODE", False)
    monkeypatch.setattr(bridge, "SWIPE_MS_MIN", 250)
    monkeypatch.setattr(bridge, "SWIPE_MS_MAX", 250)

    bridge.configure_timing(swipe_ms=200)

    assert bridge.SWIPE_MS_MIN == 200
    assert bridge.SWIPE_MS_MAX == 200


def test_human_mode_keeps_duration_variation(monkeypatch):
    monkeypatch.setattr(bridge, "HUMAN_MODE", True)

    bridge.configure_timing(swipe_ms=800)

    assert bridge.SWIPE_MS_MIN == 700
    assert bridge.SWIPE_MS_MAX == 900
