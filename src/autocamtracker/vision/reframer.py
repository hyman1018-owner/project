"""Reframe / Digital Zoom / Output Frame module for AutoCamTracker V1.

Responsibilities:
- Build a crop window from selected target bboxes.
- Support single-target and multi-target group framing.
- Apply wide, medium, and close-up presets.
- Apply smooth movement and dead zone.
- Produce the tracking output frame and framing status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autocamtracker.tracking.target_tracker import SelectedTarget


FramingMode = Literal["wide", "medium", "close"]


@dataclass
class FramingConfig:
    output_width: int = 640
    output_height: int = 360
    framing_mode: FramingMode = "medium"
    smooth_factor: float = 0.18
    dead_zone_ratio: float = 0.08
    fallback_to_original: bool = True


@dataclass
class FramingStatus:
    crop_window: tuple[int, int, int, int]
    framing_mode: FramingMode
    target_center: tuple[float, float] | None
    frame_center: tuple[float, float]
    error_x: float
    error_y: float


class Reframer:
    """Creates the after-view frame from selected tracked targets."""

    TARGET_WIDTH_RATIOS: dict[FramingMode, float] = {
        "wide": 0.30,
        "medium": 0.48,
        "close": 0.68,
    }

    def __init__(self, config: FramingConfig | None = None) -> None:
        self.config = config or FramingConfig()
        self.current_center: tuple[float, float] | None = None

    def set_framing_mode(self, mode: FramingMode) -> None:
        self.config.framing_mode = mode

    def render(self, frame, selected_targets: list[SelectedTarget]):
        import cv2

        status = self.status(frame, selected_targets)
        x, y, width, height = status.crop_window
        if not selected_targets:
            output = cv2.resize(frame, (self.config.output_width, self.config.output_height))
            return output, status

        cropped = frame[y : y + height, x : x + width]
        output = cv2.resize(cropped, (self.config.output_width, self.config.output_height))
        return output, status

    def status(self, frame, selected_targets: list[SelectedTarget]) -> FramingStatus:
        frame_h, frame_w = frame.shape[:2]
        if not selected_targets:
            return FramingStatus(
                crop_window=(0, 0, frame_w, frame_h),
                framing_mode=self.config.framing_mode,
                target_center=None,
                frame_center=(frame_w / 2.0, frame_h / 2.0),
                error_x=0.0,
                error_y=0.0,
            )

        group_bbox = self._union_bbox([target.bbox for target in selected_targets])
        target_center = self._bbox_center(group_bbox)
        frame_center = (frame_w / 2.0, frame_h / 2.0)
        desired_center = self._apply_dead_zone(target_center, frame_center, frame_w, frame_h)
        smooth_center = self._smooth_center(desired_center)
        crop_window = self._compute_crop_window(group_bbox, smooth_center, frame_w, frame_h)

        return FramingStatus(
            crop_window=crop_window,
            framing_mode=self.config.framing_mode,
            target_center=target_center,
            frame_center=frame_center,
            error_x=target_center[0] - frame_center[0],
            error_y=target_center[1] - frame_center[1],
        )

    def make_comparison_frame(self, before_frame, after_frame):
        import cv2
        import numpy as np

        before = cv2.resize(before_frame, (self.config.output_width, self.config.output_height))
        after = cv2.resize(after_frame, (self.config.output_width, self.config.output_height))
        return np.hstack([before, after])

    def reset(self) -> None:
        self.current_center = None

    def _compute_crop_window(
        self,
        bbox: tuple[float, float, float, float],
        center: tuple[float, float],
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int]:
        x1, _, x2, _ = bbox
        target_width = max(1.0, x2 - x1)
        target_ratio = self.TARGET_WIDTH_RATIOS[self.config.framing_mode]
        crop_w = min(float(frame_w), max(target_width / target_ratio, 1.0))
        crop_h = crop_w * (self.config.output_height / self.config.output_width)
        if crop_h > frame_h:
            crop_h = float(frame_h)
            crop_w = crop_h * (self.config.output_width / self.config.output_height)

        x = int(round(center[0] - crop_w / 2.0))
        y = int(round(center[1] - crop_h / 2.0))
        width = int(round(crop_w))
        height = int(round(crop_h))

        x = max(0, min(frame_w - width, x))
        y = max(0, min(frame_h - height, y))
        width = max(1, min(width, frame_w))
        height = max(1, min(height, frame_h))
        return (x, y, width, height)

    def _apply_dead_zone(
        self,
        target_center: tuple[float, float],
        frame_center: tuple[float, float],
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        dead_x = frame_w * self.config.dead_zone_ratio
        dead_y = frame_h * self.config.dead_zone_ratio
        error_x = target_center[0] - frame_center[0]
        error_y = target_center[1] - frame_center[1]

        adjusted_x = frame_center[0] if abs(error_x) <= dead_x else target_center[0]
        adjusted_y = frame_center[1] if abs(error_y) <= dead_y else target_center[1]
        return (adjusted_x, adjusted_y)

    def _smooth_center(self, target_center: tuple[float, float]) -> tuple[float, float]:
        if self.current_center is None:
            self.current_center = target_center
            return target_center

        alpha = max(0.0, min(1.0, self.config.smooth_factor))
        current_x, current_y = self.current_center
        target_x, target_y = target_center
        smoothed = (
            current_x + (target_x - current_x) * alpha,
            current_y + (target_y - current_y) * alpha,
        )
        self.current_center = smoothed
        return smoothed

    @staticmethod
    def _union_bbox(
        bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float]:
        x1 = min(bbox[0] for bbox in bboxes)
        y1 = min(bbox[1] for bbox in bboxes)
        x2 = max(bbox[2] for bbox in bboxes)
        y2 = max(bbox[3] for bbox in bboxes)
        return (x1, y1, x2, y2)

    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
