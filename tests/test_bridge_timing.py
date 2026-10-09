import bridge


def test_explicit_swipe_duration_keeps_live_verified_floor(monkeypatch):
    monkeypatch.setattr(bridge, "SWIPE_MS_MIN", 720)
    monkeypatch.setattr(bridge, "SWIPE_MS_MAX", 860)

    bridge.configure_timing(swipe_ms=650)

    assert bridge.SWIPE_MS_MIN == 720
    assert bridge.SWIPE_MS_MAX == 731
