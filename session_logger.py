#!/usr/bin/env python3
"""Structured JSON-lines logging for Solvitaire runs."""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_session_log_path() -> Path:
    return Path("logs") / f"session_{timestamp_slug()}.jsonl"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


class SessionLogger:
    """Append one structured event per line to a JSONL file."""

    def __init__(self, path: str | Path, session_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._closed = False

    def event(self, event_name: str, **data: Any) -> None:
        if self._closed:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event_name,
            **_json_safe(data),
        }

        with self._lock:
            self._file.write(json.dumps(record, sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
