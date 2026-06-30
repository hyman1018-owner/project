"""Structured JSONL telemetry for hardware bring-up sessions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from time import monotonic, time
from typing import Any


class TelemetryLogger:
    """Append timestamped events that can be correlated across desktop/iPhone."""

    def __init__(self, log_dir: Path, session_name: str | None = None) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = session_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"autocamtracker-telemetry-{stamp}.jsonl"
        self._lock = Lock()

    def log(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "timestamp_ms": int(time() * 1000),
            "monotonic_s": monotonic(),
            **fields,
        }
        line = json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
