"""Persistent metadata-only vehicle identity store for AutoCamTracker V1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from time import time
from typing import Any

from autocamtracker.vision.detector import TrackedDetection


@dataclass
class StoredVehicleIdentity:
    vehicle_id: int
    display_name: str
    class_name: str
    last_track_id: int | None
    last_frame_index: int
    last_seen_timestamp: float
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    metadata: dict[str, Any]


@dataclass
class VehicleIdentitySummary:
    vehicle_id: int
    display_name: str
    class_name: str
    last_track_id: int | None
    last_frame_index: int
    confidence: float
    master_feature_count: int = 0
    pending_feature_count: int = 0
    candidate_feature_count: int = 0
    updated_at: float = 0.0


@dataclass
class IdentityStoreSummary:
    vehicle_count: int
    master_feature_count: int
    pending_feature_count: int
    candidate_feature_count: int
    vehicles: list[VehicleIdentitySummary]


class VehicleIdentityStore:
    """SQLite-backed store for GID, last bbox, and basic vehicle metadata.

    V1.3 intentionally keeps embeddings out of this class. ReID features live
    in FeatureGallery and are written only by explicit Add Feature actions.
    """

    def __init__(self, db_path: Path | str, commit_interval_seconds: float = 0.5) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.commit_interval_seconds = max(0.0, float(commit_interval_seconds))
        self._last_commit_at = time()
        self._pending_updates: dict[int, tuple[Any, ...]] = {}
        self._ensure_schema()

    def close(self) -> None:
        self.flush()
        self.connection.close()

    def flush(self) -> None:
        if not self._pending_updates:
            return
        for parameters in self._pending_updates.values():
            self.connection.execute(
                """
                UPDATE vehicles
                SET updated_at = ?,
                    class_name = ?,
                    last_track_id = ?,
                    last_frame_index = ?,
                    last_seen_timestamp = ?,
                    confidence = ?,
                    bbox_json = ?,
                    center_json = ?,
                    metadata_json = COALESCE(?, metadata_json)
                WHERE id = ?
                """,
                parameters,
            )
        self.connection.commit()
        self._last_commit_at = time()
        self._pending_updates.clear()

    def create_vehicle(self, detection: TrackedDetection, metadata: dict[str, Any] | None = None) -> int:
        now = time()
        payload = self._detection_payload(detection)
        cursor = self.connection.execute(
            """
            INSERT INTO vehicles (
                created_at,
                updated_at,
                class_name,
                last_track_id,
                last_frame_index,
                last_seen_timestamp,
                confidence,
                bbox_json,
                center_json,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                now,
                detection.class_name,
                detection.track_id,
                detection.frame_index,
                detection.timestamp,
                detection.confidence,
                payload["bbox_json"],
                payload["center_json"],
                self._metadata_json(metadata),
            ),
        )
        self._commit_now()
        return int(cursor.lastrowid)

    def update_vehicle(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        payload = self._detection_payload(detection)
        exists = self.connection.execute(
            "SELECT 1 FROM vehicles WHERE id = ?",
            (vehicle_id,),
        ).fetchone() is not None
        if not exists:
            return False
        self._pending_updates[vehicle_id] = (
            time(),
            detection.class_name,
            detection.track_id,
            detection.frame_index,
            detection.timestamp,
            detection.confidence,
            payload["bbox_json"],
            payload["center_json"],
            self._metadata_json(metadata),
            vehicle_id,
        )
        self._commit_if_due()
        return True

    def get_vehicle(self, vehicle_id: int) -> StoredVehicleIdentity | None:
        row = self.connection.execute(
            """
            SELECT
                id,
                display_name,
                class_name,
                last_track_id,
                last_frame_index,
                last_seen_timestamp,
                confidence,
                bbox_json,
                center_json,
                metadata_json
            FROM vehicles
            WHERE id = ?
            """,
            (vehicle_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredVehicleIdentity(
            vehicle_id=int(row["id"]),
            display_name=self._display_name(row),
            class_name=str(row["class_name"]),
            last_track_id=row["last_track_id"],
            last_frame_index=int(row["last_frame_index"]),
            last_seen_timestamp=float(row["last_seen_timestamp"]),
            confidence=float(row["confidence"]),
            bbox=self._bbox_from_json(row["bbox_json"]),
            center=self._center_from_json(row["center_json"]),
            metadata=self._metadata_from_json(row["metadata_json"] if "metadata_json" in row.keys() else None),
        )

    def display_label(self, vehicle_id: int) -> str:
        row = self.connection.execute(
            "SELECT id, display_name FROM vehicles WHERE id = ?",
            (vehicle_id,),
        ).fetchone()
        if row is None:
            return str(vehicle_id)
        return self._display_name(row)

    def update_display_name(self, vehicle_id: int, display_name: str) -> bool:
        value = display_name.strip() or None
        cursor = self.connection.execute(
            "UPDATE vehicles SET display_name = ?, updated_at = ? WHERE id = ?",
            (value, time(), vehicle_id),
        )
        self._commit_now()
        return cursor.rowcount > 0

    def delete_vehicle(self, vehicle_id: int) -> bool:
        cursor = self.connection.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
        deleted = cursor.rowcount > 0
        self._commit_now()
        return deleted

    def clear_track_link(self, vehicle_id: int, track_id: int | None) -> bool:
        if track_id is None:
            return False
        cursor = self.connection.execute(
            "UPDATE vehicles SET last_track_id = NULL, updated_at = ? WHERE id = ? AND last_track_id = ?",
            (time(), vehicle_id, track_id),
        )
        if cursor.rowcount > 0:
            self._commit_now()
            return True
        return False

    def _commit_if_due(self) -> None:
        if self._pending_updates and time() - self._last_commit_at >= self.commit_interval_seconds:
            self.flush()

    def _commit_now(self) -> None:
        self.connection.commit()
        self._last_commit_at = time()

    def summary(self, feature_counts: dict[int, dict[str, int]] | None = None, limit: int = 50) -> IdentityStoreSummary:
        feature_counts = feature_counts or {}
        total = self.connection.execute("SELECT COUNT(*) AS vehicle_count FROM vehicles").fetchone()
        rows = self.connection.execute(
            """
            SELECT
                id,
                display_name,
                class_name,
                last_track_id,
                last_frame_index,
                confidence,
                updated_at
            FROM vehicles
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        vehicles: list[VehicleIdentitySummary] = []
        for row in rows:
            counts = feature_counts.get(int(row["id"]), {})
            vehicles.append(
                VehicleIdentitySummary(
                    vehicle_id=int(row["id"]),
                    display_name=self._display_name(row),
                    class_name=str(row["class_name"]),
                    last_track_id=row["last_track_id"],
                    last_frame_index=int(row["last_frame_index"]),
                    confidence=float(row["confidence"]),
                    master_feature_count=int(counts.get("master", 0)),
                    pending_feature_count=int(counts.get("pending", 0)),
                    candidate_feature_count=int(counts.get("candidate", 0)),
                    updated_at=float(row["updated_at"]),
                )
            )
        return IdentityStoreSummary(
            vehicle_count=int(total["vehicle_count"] or 0),
            master_feature_count=sum(int(counts.get("master", 0)) for counts in feature_counts.values()),
            pending_feature_count=sum(int(counts.get("pending", 0)) for counts in feature_counts.values()),
            candidate_feature_count=sum(int(counts.get("candidate", 0)) for counts in feature_counts.values()),
            vehicles=vehicles,
        )

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                class_name TEXT NOT NULL,
                last_track_id INTEGER,
                last_frame_index INTEGER NOT NULL,
                last_seen_timestamp REAL NOT NULL,
                confidence REAL NOT NULL,
                bbox_json TEXT NOT NULL,
                center_json TEXT NOT NULL,
                metadata_json TEXT
            )
            """
        )
        self._ensure_column("vehicles", "display_name", "TEXT")
        self._ensure_column("vehicles", "metadata_json", "TEXT")
        self.connection.execute("DROP TABLE IF EXISTS observations")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicles_updated_at ON vehicles(updated_at)"
        )
        self.connection.commit()

    @staticmethod
    def _detection_payload(detection: TrackedDetection) -> dict[str, str]:
        return {
            "bbox_json": json.dumps(list(detection.bbox)),
            "center_json": json.dumps(list(detection.center)),
        }

    @staticmethod
    def _metadata_json(metadata: dict[str, Any] | None) -> str | None:
        if metadata is None:
            return None
        return json.dumps(metadata, sort_keys=True)

    @staticmethod
    def _metadata_from_json(value: str | None) -> dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _bbox_from_json(value: str) -> tuple[float, float, float, float]:
        items = json.loads(value)
        return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))

    @staticmethod
    def _center_from_json(value: str) -> tuple[float, float]:
        items = json.loads(value)
        return (float(items[0]), float(items[1]))

    @staticmethod
    def _display_name(row: sqlite3.Row) -> str:
        value = row["display_name"] if "display_name" in row.keys() else None
        return str(value).strip() if value else str(row["id"])

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        if any(row["name"] == column_name for row in rows):
            return
        self.connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
