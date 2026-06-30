"""YOLO Data module for AutoCamTracker V1.

Responsibilities:
- Store current tracked detections.
- Maintain vehicle tracks keyed by track_id.
- Keep detection history for debug and simple reacquire logic.
- Rank vehicle candidates for auto-select.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Literal

from autocamtracker.vision.detector import TrackedDetection


RankStrategy = Literal["stable", "largest", "center", "confidence"]


@dataclass
class VehicleCandidate:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    center: tuple[float, float]
    area: float
    distance_to_frame_center: float
    frame_index: int
    age_frames: int
    lost_frame_count: int
    stability_score: float


@dataclass
class VehicleTrack:
    track_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    latest_bbox: tuple[float, float, float, float]
    latest_confidence: float
    latest_center: tuple[float, float]
    center_history: list[tuple[float, float]] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    lost_frame_count: int = 0
    tracker_name: str = "botsort"

    def update(self, detection: TrackedDetection) -> None:
        self.last_seen_frame = detection.frame_index
        self.latest_bbox = detection.bbox
        self.latest_confidence = detection.confidence
        self.latest_center = detection.center
        self.center_history.append(detection.center)
        self.confidence_history.append(detection.confidence)
        self.lost_frame_count = 0
        self.tracker_name = detection.tracker_name


class DetectionStore:
    """Stores YOLO tracking output and exposes ranked vehicle candidates."""

    def __init__(self, history_size: int = 90) -> None:
        self.history_size = history_size
        self.current_detections: list[TrackedDetection] = []
        self.vehicle_tracks: dict[int, VehicleTrack] = {}
        self.detection_history: deque[list[TrackedDetection]] = deque(maxlen=history_size)
        self.current_frame_index = 0

    def reset(self) -> None:
        self.current_detections = []
        self.vehicle_tracks = {}
        self.detection_history.clear()
        self.current_frame_index = 0

    def update(
        self,
        detections: list[TrackedDetection],
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    ) -> list[VehicleCandidate]:
        self.current_detections = [d for d in detections if d.track_id is not None]
        self.detection_history.append(self.current_detections)

        if detections:
            self.current_frame_index = max(d.frame_index for d in detections)
        else:
            self.current_frame_index += 1

        seen_track_ids: set[int] = set()
        for detection in self.current_detections:
            assert detection.track_id is not None
            seen_track_ids.add(detection.track_id)
            if detection.track_id not in self.vehicle_tracks:
                self.vehicle_tracks[detection.track_id] = VehicleTrack(
                    track_id=detection.track_id,
                    class_name=detection.class_name,
                    first_seen_frame=detection.frame_index,
                    last_seen_frame=detection.frame_index,
                    latest_bbox=detection.bbox,
                    latest_confidence=detection.confidence,
                    latest_center=detection.center,
                    center_history=[detection.center],
                    confidence_history=[detection.confidence],
                    tracker_name=detection.tracker_name,
                )
            else:
                self.vehicle_tracks[detection.track_id].update(detection)

        for track_id, track in self.vehicle_tracks.items():
            if track_id not in seen_track_ids:
                track.lost_frame_count += 1

        return self.get_candidates(frame_shape)

    def get_candidates(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    ) -> list[VehicleCandidate]:
        frame_center = self._frame_center(frame_shape)
        candidates: list[VehicleCandidate] = []

        for detection in self.current_detections:
            assert detection.track_id is not None
            x1, y1, x2, y2 = detection.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            distance = hypot(
                detection.center[0] - frame_center[0],
                detection.center[1] - frame_center[1],
            )
            candidates.append(
                VehicleCandidate(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    center=detection.center,
                    area=area,
                    distance_to_frame_center=distance,
                    frame_index=detection.frame_index,
                    age_frames=max(1, detection.frame_index - self.vehicle_tracks[detection.track_id].first_seen_frame + 1),
                    lost_frame_count=self.vehicle_tracks[detection.track_id].lost_frame_count,
                    stability_score=self._stability_score(
                        confidence=detection.confidence,
                        area=area,
                        frame_shape=frame_shape,
                        distance_to_frame_center=distance,
                        age_frames=max(
                            1,
                            detection.frame_index - self.vehicle_tracks[detection.track_id].first_seen_frame + 1,
                        ),
                    ),
                )
            )
        return candidates

    def rank_candidates(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
        strategy: RankStrategy = "largest",
    ) -> list[VehicleCandidate]:
        candidates = self.get_candidates(frame_shape)
        if strategy == "stable":
            return sorted(candidates, key=lambda item: item.stability_score, reverse=True)
        if strategy == "largest":
            return sorted(candidates, key=lambda item: item.area, reverse=True)
        if strategy == "center":
            return sorted(candidates, key=lambda item: item.distance_to_frame_center)
        if strategy == "confidence":
            return sorted(candidates, key=lambda item: item.confidence, reverse=True)
        raise ValueError(f"Unsupported rank strategy: {strategy}")

    def get_track(self, track_id: int) -> VehicleTrack | None:
        return self.vehicle_tracks.get(track_id)

    def get_candidate_at_point(
        self,
        x: float,
        y: float,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
        padding_ratio: float = 0.08,
    ) -> VehicleCandidate | None:
        candidates = self.get_candidates(frame_shape)
        hits: list[tuple[float, float, VehicleCandidate]] = []
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bbox
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            pad = max(8.0, min(width, height) * padding_ratio)
            if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad:
                center_distance = hypot(x - candidate.center[0], y - candidate.center[1])
                hits.append((candidate.area, center_distance, candidate))

        if not hits:
            return None
        _, _, candidate = min(hits, key=lambda item: (item[0], item[1]))
        return candidate

    @staticmethod
    def _frame_center(
        frame_shape: tuple[int, int, int] | tuple[int, int] | None,
    ) -> tuple[float, float]:
        if frame_shape is None:
            return (0.0, 0.0)
        height, width = frame_shape[:2]
        return (width / 2.0, height / 2.0)

    @staticmethod
    def _stability_score(
        confidence: float,
        area: float,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None,
        distance_to_frame_center: float,
        age_frames: int,
    ) -> float:
        if frame_shape is None:
            frame_area = max(1.0, area)
            frame_diagonal = 1.0
        else:
            height, width = frame_shape[:2]
            frame_area = max(1.0, float(width * height))
            frame_diagonal = max(1.0, hypot(width, height))

        area_score = min(1.0, area / frame_area * 12.0)
        center_score = max(0.0, 1.0 - distance_to_frame_center / frame_diagonal)
        age_score = min(1.0, age_frames / 12.0)
        return confidence * 0.45 + area_score * 0.25 + center_score * 0.15 + age_score * 0.15
