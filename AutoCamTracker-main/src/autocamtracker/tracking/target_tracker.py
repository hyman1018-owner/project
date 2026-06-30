"""Target Tracking module for AutoCamTracker V1.

Responsibilities:
- Manage selected track IDs.
- Support single-target selection.
- Support auto-selecting one vehicle candidate.
- Detect target lost state and reset to selectable mode when needed.

This module uses track_id as the identity source. It should not run YOLO or
crop frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autocamtracker.tracking.detection_store import DetectionStore, VehicleCandidate


TrackingStatus = Literal["idle", "tracking", "coasting", "searching", "camera_cut", "lost", "failed"]


@dataclass
class SelectedTarget:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    center: tuple[float, float]
    status: TrackingStatus = "tracking"
    lost_frame_count: int = 0


@dataclass
class TrackingConfig:
    max_lost_frames: int = 45
    reacquire_enabled: bool = True


class TargetTracker:
    """Owns target selection and target lost state."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self.selected_track_ids: list[int] = []
        self.selected_targets: list[SelectedTarget] = []
        self.status: TrackingStatus = "idle"
        self.lost_alert: str | None = None

    def select_track(self, track_id: int) -> None:
        self.selected_track_ids = [track_id]
        self.lost_alert = None
        self.status = "tracking" if self.selected_track_ids else "idle"

    def clear_selection(self) -> None:
        self.selected_track_ids = []
        self.selected_targets = []
        self.status = "idle"
        self.lost_alert = None

    def auto_select_one(self, candidates: list[VehicleCandidate]) -> None:
        if not candidates:
            self.clear_selection()
            return
        self.selected_track_ids = [candidates[0].track_id]
        self.status = "tracking"
        self.lost_alert = None

    def update_from_store(self, store: DetectionStore) -> list[SelectedTarget]:
        updated_targets: list[SelectedTarget] = []
        failed_ids: list[int] = []

        for track_id in self.selected_track_ids:
            track = store.get_track(track_id)
            if track is None:
                failed_ids.append(track_id)
                continue

            status: TrackingStatus = "tracking"
            if track.lost_frame_count > 0:
                status = "lost"
            if track.lost_frame_count > self.config.max_lost_frames:
                status = "failed"
                failed_ids.append(track_id)

            updated_targets.append(
                SelectedTarget(
                    track_id=track_id,
                    bbox=track.latest_bbox,
                    class_name=track.class_name,
                    confidence=track.latest_confidence,
                    center=track.latest_center,
                    status=status,
                    lost_frame_count=track.lost_frame_count,
                )
            )

        if failed_ids:
            self.lost_alert = (
                "Tracking failed. Target lost and selection was reset. "
                "Please select a target again."
            )
            self.clear_selection()
            self.status = "failed"
            return []

        self.selected_targets = updated_targets
        if not self.selected_track_ids:
            self.status = "idle"
        elif any(target.status == "lost" for target in updated_targets):
            self.status = "lost"
        else:
            self.status = "tracking"
        return self.selected_targets

    def get_state(self) -> dict[str, object]:
        return {
            "selected_track_ids": list(self.selected_track_ids),
            "selected_count": len(self.selected_track_ids),
            "status": self.status,
            "lost_alert": self.lost_alert,
        }
