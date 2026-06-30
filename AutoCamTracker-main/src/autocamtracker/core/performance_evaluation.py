"""Performance evaluation helpers for detector and tracker quality.

The desktop UI can compute live runtime metrics immediately. Dataset metrics
such as precision, recall, and mAP still require labelled evaluation counts or
AP samples, so this module keeps those calculations explicit and testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean

from autocamtracker.core.frame_data import FrameData


@dataclass(frozen=True)
class ConfusionMatrixStats:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @property
    def precision(self) -> float | None:
        return _safe_divide(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> float | None:
        return _safe_divide(self.true_positive, self.true_positive + self.false_negative)

    @property
    def accuracy(self) -> float | None:
        total = self.true_positive + self.false_positive + self.false_negative + self.true_negative
        return _safe_divide(self.true_positive + self.true_negative, total)


@dataclass(frozen=True)
class RuntimePerformanceSnapshot:
    frame_count: int
    average_fps: float | None
    latest_fps: float | None
    source_fps: float | None
    average_inference_ms: float | None
    latest_inference_ms: float | None
    average_pipeline_ms: float | None
    latest_pipeline_ms: float | None
    average_confidence: float | None
    latest_confidence: float | None
    tracking_stability: float | None
    id_switches: int
    locked_frames: int
    lost_frames: int
    detection_count: int
    candidate_count: int
    skipped_frames: int
    tracking_status: str
    selected_local_track_id: int | None
    selected_global_vehicle_id: int | None


class PerformanceEvaluationTracker:
    """Collects a rolling window of frame-level metrics for the UI."""

    def __init__(self, window_size: int = 300) -> None:
        self.window_size = max(1, int(window_size))
        self._fps_values: deque[float] = deque(maxlen=self.window_size)
        self._inference_ms_values: deque[float] = deque(maxlen=self.window_size)
        self._pipeline_ms_values: deque[float] = deque(maxlen=self.window_size)
        self._confidence_values: deque[float] = deque(maxlen=self.window_size)
        self._locked_values: deque[bool] = deque(maxlen=self.window_size)
        self._last_selected_local_track_id: int | None = None
        self._id_switches = 0
        self._latest: RuntimePerformanceSnapshot | None = None

    def reset(self) -> None:
        self._fps_values.clear()
        self._inference_ms_values.clear()
        self._pipeline_ms_values.clear()
        self._confidence_values.clear()
        self._locked_values.clear()
        self._last_selected_local_track_id = None
        self._id_switches = 0
        self._latest = None

    def record_frame(self, frame_data: FrameData) -> RuntimePerformanceSnapshot:
        fps = _positive_or_none(frame_data.display_fps)
        inference_ms = _positive_or_none(frame_data.inference_time_ms)
        pipeline_ms = _positive_or_none(frame_data.pipeline_time_ms)
        selected_target = frame_data.selected_targets[0] if frame_data.selected_targets else None
        confidence = _positive_or_none(selected_target.confidence if selected_target is not None else None)
        target_locked = bool(
            frame_data.tracking_status == "tracking"
            and selected_target is not None
            and selected_target.lost_frame_count == 0
            and selected_target.status == "tracking"
        )

        if fps is not None:
            self._fps_values.append(fps)
        if inference_ms is not None:
            self._inference_ms_values.append(inference_ms)
        if pipeline_ms is not None:
            self._pipeline_ms_values.append(pipeline_ms)
        if confidence is not None:
            self._confidence_values.append(confidence)
        self._locked_values.append(target_locked)

        selected_lid = frame_data.selected_local_track_id
        if (
            selected_lid is not None
            and self._last_selected_local_track_id is not None
            and selected_lid != self._last_selected_local_track_id
        ):
            self._id_switches += 1
        if selected_lid is not None:
            self._last_selected_local_track_id = selected_lid

        locked_frames = sum(1 for locked in self._locked_values if locked)
        frame_count = len(self._locked_values)
        self._latest = RuntimePerformanceSnapshot(
            frame_count=frame_count,
            average_fps=_mean(self._fps_values),
            latest_fps=fps,
            source_fps=frame_data.source_fps,
            average_inference_ms=_mean(self._inference_ms_values),
            latest_inference_ms=inference_ms,
            average_pipeline_ms=_mean(self._pipeline_ms_values),
            latest_pipeline_ms=pipeline_ms,
            average_confidence=_mean(self._confidence_values),
            latest_confidence=confidence,
            tracking_stability=_safe_divide(locked_frames, frame_count),
            id_switches=self._id_switches,
            locked_frames=locked_frames,
            lost_frames=max(0, frame_count - locked_frames),
            detection_count=len(frame_data.detections),
            candidate_count=len(frame_data.candidates),
            skipped_frames=frame_data.skipped_frames,
            tracking_status=frame_data.tracking_status,
            selected_local_track_id=selected_lid,
            selected_global_vehicle_id=frame_data.selected_global_vehicle_id,
        )
        return self._latest

    def snapshot(self) -> RuntimePerformanceSnapshot:
        if self._latest is not None:
            return self._latest
        return RuntimePerformanceSnapshot(
            frame_count=0,
            average_fps=None,
            latest_fps=None,
            source_fps=None,
            average_inference_ms=None,
            latest_inference_ms=None,
            average_pipeline_ms=None,
            latest_pipeline_ms=None,
            average_confidence=None,
            latest_confidence=None,
            tracking_stability=None,
            id_switches=0,
            locked_frames=0,
            lost_frames=0,
            detection_count=0,
            candidate_count=0,
            skipped_frames=0,
            tracking_status="idle",
            selected_local_track_id=None,
            selected_global_vehicle_id=None,
        )


def mean_average_precision(ap_values: list[float]) -> float | None:
    cleaned = [max(0.0, min(1.0, value)) for value in ap_values]
    if not cleaned:
        return None
    return fmean(cleaned)


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: deque[float]) -> float | None:
    if not values:
        return None
    return fmean(values)


def _positive_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0.0:
        return None
    return value
