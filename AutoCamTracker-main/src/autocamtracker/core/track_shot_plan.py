"""Racetrack camera shot modes and normalized In/Out trigger zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrackShotMode = Literal["AI Tracking", "Fixed Cut", "In/Out Auto"]
TrackShotState = Literal["tracking", "fixed_cut", "armed", "active", "complete"]


@dataclass(frozen=True)
class TrackZone:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Track zone coordinates must be normalized from 0 to 1")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("Track zone must have positive width and height")

    def contains(self, point: tuple[float, float], frame_shape) -> bool:
        frame_h, frame_w = frame_shape[:2]
        x = point[0] / max(1.0, float(frame_w))
        y = point[1] / max(1.0, float(frame_h))
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    @classmethod
    def parse(cls, value: str) -> "TrackZone":
        try:
            parts = [float(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError("Use left,top,right,bottom values from 0 to 1") from exc
        if len(parts) != 4:
            raise ValueError("Use exactly four values: left,top,right,bottom")
        return cls(*parts)

    def text(self) -> str:
        return ",".join(f"{value:.2f}" for value in (self.left, self.top, self.right, self.bottom))


@dataclass(frozen=True)
class TrackShotDecision:
    publish_tracking: bool
    state: TrackShotState
    reason: str


def should_publish_motor_tracking(
    source_type: str,
    motor_armed: bool,
    motor_ready: bool,
    decision: TrackShotDecision,
) -> bool:
    """Allow physical motor commands only for an explicitly armed iPhone feed."""

    return source_type == "iphone" and motor_armed and motor_ready and decision.publish_tracking


class TrackShotController:
    """Gates motor tracking for fixed cuts and operator-defined In/Out shots."""

    def __init__(
        self,
        mode: TrackShotMode = "AI Tracking",
        in_zone: TrackZone | None = None,
        out_zone: TrackZone | None = None,
    ) -> None:
        self.mode = mode
        self.in_zone = in_zone or TrackZone(0.0, 0.0, 0.2, 1.0)
        self.out_zone = out_zone or TrackZone(0.8, 0.0, 1.0, 1.0)
        self.state: TrackShotState = "tracking"
        self.set_mode(mode)

    def set_mode(self, mode: TrackShotMode) -> None:
        if mode not in {"AI Tracking", "Fixed Cut", "In/Out Auto"}:
            raise ValueError(f"Unsupported track shot mode: {mode}")
        self.mode = mode
        if mode == "Fixed Cut":
            self.state = "fixed_cut"
        elif mode == "In/Out Auto":
            self.state = "armed"
        else:
            self.state = "tracking"

    def configure_zones(self, in_zone: TrackZone, out_zone: TrackZone) -> None:
        self.in_zone = in_zone
        self.out_zone = out_zone
        if self.mode == "In/Out Auto":
            self.state = "armed"

    def rearm(self) -> None:
        self.state = "armed" if self.mode == "In/Out Auto" else self.state

    def evaluate(self, frame_data, frame_shape) -> TrackShotDecision:
        target = self._fresh_target(frame_data)
        if self.mode == "Fixed Cut":
            self.state = "fixed_cut"
            return TrackShotDecision(False, self.state, "warm-up fixed cut")

        if self.mode == "AI Tracking":
            self.state = "tracking"
            return TrackShotDecision(target is not None, self.state, "target locked" if target else "target unavailable")

        if self.state == "complete":
            return TrackShotDecision(False, self.state, "shot complete; press Rearm")
        if self.state == "armed":
            if target is None or not self.in_zone.contains(target.center, frame_shape):
                return TrackShotDecision(False, self.state, "waiting for target in In zone")
            self.state = "active"
        if target is None:
            return TrackShotDecision(False, self.state, "target temporarily lost")
        if self.out_zone.contains(target.center, frame_shape):
            self.state = "complete"
            return TrackShotDecision(False, self.state, "target reached Out zone")
        return TrackShotDecision(True, self.state, "target inside active shot")

    @staticmethod
    def _fresh_target(frame_data):
        if frame_data.tracking_status != "tracking":
            return None
        return next(
            (
                target
                for target in frame_data.selected_targets
                if (
                    (target.status == "tracking" and target.lost_frame_count == 0)
                    or (target.status == "coasting" and target.lost_frame_count <= 3)
                )
            ),
            None,
        )
