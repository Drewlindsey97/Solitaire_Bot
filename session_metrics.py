"""Summarize session timings and capture-confirmed moves from a JSONL log."""

import argparse
from datetime import datetime
import json
from statistics import median


def summarize(records):
    times = {}
    dispatched = confirmed = checked = unchanged = unverified = 0
    founded = []
    started = finished = None
    for record in records:
        event = record.get("event")
        if event in ("screenshot_captured", "board_read", "solver_finished", "cycle_finished"):
            duration = record.get("duration_seconds")
            if isinstance(duration, (int, float)):
                times.setdefault(event, []).append(duration)
        if event == "gesture_dispatched":
            dispatched += 1
        if event == "move_verification":
            count = record["dispatched"]
            if record["confirmed"] is None:
                unverified += count
            else:
                checked += count
                confirmed += record["confirmed"]
                unchanged += count - record["confirmed"]
        if event == "solver_state_built":
            founded.append(sum(v + 1 for v in record["foundations"].values()))
        if event == "session_started":
            started = datetime.fromisoformat(record["timestamp"])
        if event == "session_finished":
            finished = datetime.fromisoformat(record["timestamp"])
    elapsed = (finished - started).total_seconds() if started and finished else None
    return {
        "elapsed_seconds": elapsed,
        "gestures_dispatched": dispatched,
        "moves_checked": checked,
        "moves_confirmed": confirmed,
        "moves_confirmed_unchanged": unchanged,
        "moves_unverified": unverified,
        "moves_awaiting_capture": max(0, dispatched - checked - unverified),
        "confirmed_moves_per_second": confirmed / elapsed if elapsed and elapsed > 0 else None,
        "foundation_gain_observed": founded[-1] - founded[0] if founded else None,
        "median_seconds": {event: median(values) for event, values in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Path to a single Solvitaire JSONL session")
    args = parser.parse_args()
    with open(args.log, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    sessions = {r.get("session_id") for r in records if r.get("session_id")}
    if len(sessions) > 1:
        parser.error("use a log containing one session, rather than combining runs")
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
