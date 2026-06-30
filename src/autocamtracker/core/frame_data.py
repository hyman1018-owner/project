"""Structured per-frame pipeline data for AutoCamTracker V1.

The UI can still render the familiar Before / After view, but status, logging,
recording, and later worker-thread handoff should consume this typed snapshot
instead of rebuilding state from several objects and status strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autocamtracker.tracking.detection_store import VehicleCandidate
from autocamtracker.vision.reframer import FramingStatus
from autocamtracker.tracking.target_tracker import SelectedTarget
from autocamtracker.vision.detector import TrackedDetection


@dataclass
class FrameData:
    raw_frame: Any
    before_frame: Any
    after_frame: Any
    detections: list[TrackedDetection]
    candidates: list[VehicleCandidate]
    selected_targets: list[SelectedTarget]
    framing_status: FramingStatus
    tracking_status: str
    selected_global_vehicle_id: int | None = None
    selected_local_track_id: int | None = None
    camera_cut_detected: bool = False
    lost_frames: int = 0
    reacquire_score: float = 0.0
    display_fps: float = 0.0
    source_fps: float | None = None
    inference_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    receive_latency_ms: float | None = None
    pipeline_time_ms: float = 0.0
    identity_time_ms: float = 0.0
    reframe_time_ms: float = 0.0
    preview_time_ms: float = 0.0
    skipped_frames: int = 0
    notes: list[str] = field(default_factory=list)
