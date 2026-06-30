"""Global vehicle identity and manual ReID reacquire logic for AutoCamTracker V1."""

from __future__ import annotations

from dataclasses import dataclass, field

from autocamtracker.tracking.feature_gallery import FeatureGallery
from autocamtracker.tracking.target_tracker import SelectedTarget
from autocamtracker.tracking.vehicle_identity_store import VehicleIdentityStore
from autocamtracker.vision.detector import TrackedDetection


@dataclass
class VehicleIdentity:
    global_vehicle_id: int | None
    last_track_id: int | None
    class_name: str
    confidence: float
    last_bbox: tuple[float, float, float, float]
    last_center: tuple[float, float]
    last_frame_index: int
    last_seen_timestamp: float
    color_signature: object | None = None
    lost_frames: int = 0
    status: str = "tracking"
    track_aliases: list[int] = field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)


class ReacquireEngine:
    """Scores current detections against the selected in-memory identity."""

    def __init__(self, min_score: float = 0.62, margin: float = 0.08, confirm_frames: int = 2) -> None:
        self.min_score = min_score
        self.margin = margin
        self.confirm_frames = confirm_frames
        self._pending_key: int | None = None
        self._pending_count = 0

    def reset_pending(self) -> None:
        self._pending_key = None
        self._pending_count = 0

    def color_signature(self, frame, bbox: tuple[float, float, float, float]):
        import cv2

        x1, y1, x2, y2 = self._clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype("float32")

    def choose(
        self,
        identity: VehicleIdentity,
        detections: list[TrackedDetection],
        frame,
    ) -> tuple[TrackedDetection | None, float]:
        if not detections:
            self.reset_pending()
            return None, 0.0

        scored = [(self._score(identity, detection, frame), detection) for detection in detections]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        if best_score < self.min_score or best_score - second_score < self.margin:
            self.reset_pending()
            return None, best_score

        pending_key = best.track_id if best.track_id is not None else best.frame_index
        if pending_key == self._pending_key:
            self._pending_count += 1
        else:
            self._pending_key = pending_key
            self._pending_count = 1

        if self._pending_count >= self.confirm_frames:
            self.reset_pending()
            return best, best_score
        return None, best_score

    def _score(self, identity: VehicleIdentity, detection: TrackedDetection, frame) -> float:
        tracker_match = (
            1.0
            if detection.track_id is not None and detection.track_id == identity.last_track_id
            else 0.0
        )
        color = self._color_similarity(identity, detection, frame)
        size = self._size_similarity(identity.last_bbox, detection.bbox)
        motion = self._motion_similarity(identity.last_center, detection.center, frame.shape[1], frame.shape[0])
        confidence = max(0.0, min(1.0, detection.confidence))
        class_match = 1.0 if detection.class_name == identity.class_name else 0.0
        return (
            0.34 * tracker_match
            + 0.24 * color
            + 0.14 * size
            + 0.12 * motion
            + 0.10 * confidence
            + 0.06 * class_match
        )

    def _color_similarity(self, identity: VehicleIdentity, detection: TrackedDetection, frame) -> float:
        import cv2

        if identity.color_signature is None:
            return 0.0
        signature = self.color_signature(frame, detection.bbox)
        if signature is None:
            return 0.0
        score = cv2.compareHist(identity.color_signature, signature, cv2.HISTCMP_CORREL)
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _size_similarity(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        first_w = max(1.0, first[2] - first[0])
        first_h = max(1.0, first[3] - first[1])
        second_w = max(1.0, second[2] - second[0])
        second_h = max(1.0, second[3] - second[1])
        first_area = first_w * first_h
        second_area = second_w * second_h
        area = min(first_area, second_area) / max(first_area, second_area)
        first_aspect = first_w / first_h
        second_aspect = second_w / second_h
        aspect = min(first_aspect, second_aspect) / max(first_aspect, second_aspect)
        return float(0.7 * area + 0.3 * aspect)

    @staticmethod
    def _motion_similarity(
        previous: tuple[float, float],
        current: tuple[float, float],
        frame_w: int,
        frame_h: int,
    ) -> float:
        diagonal = max(1.0, (frame_w**2 + frame_h**2) ** 0.5)
        distance = ((previous[0] - current[0]) ** 2 + (previous[1] - current[1]) ** 2) ** 0.5
        return float(max(0.0, 1.0 - distance / (0.6 * diagonal)))

    @staticmethod
    def _clamp_bbox(
        bbox: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        left = max(0, min(frame_w - 1, int(round(x1))))
        top = max(0, min(frame_h - 1, int(round(y1))))
        right = max(left + 1, min(frame_w, int(round(x2))))
        bottom = max(top + 1, min(frame_h, int(round(y2))))
        return left, top, right, bottom


class GlobalIdentityManager:
    """Keeps selected GID independent from local tracker IDs."""

    def __init__(
        self,
        max_lost_frames: int = 150,
        searching_after_frames: int = 5,
        predictive_coast_frames: int = 12,
        coasting_min_confidence: float = 0.24,
        identity_store: VehicleIdentityStore | None = None,
        feature_gallery: FeatureGallery | None = None,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.searching_after_frames = searching_after_frames
        self.predictive_coast_frames = predictive_coast_frames
        self.coasting_min_confidence = max(0.20, min(0.50, coasting_min_confidence))
        self.next_global_vehicle_id = 1
        self.identity_store = identity_store
        self.feature_gallery = feature_gallery
        self.selected_identity: VehicleIdentity | None = None
        self.reacquire = ReacquireEngine()
        self.status = "idle"
        self.last_reacquire_score = 0.0
        self.camera_cut_seen = False
        self.auto_reid_min_score = 0.72
        self.auto_reid_margin = 0.08
        self.auto_reid_confirm_frames = 3
        self._auto_reid_pending_track_id: int | None = None
        self._auto_reid_pending_count = 0

    @property
    def selected_global_vehicle_id(self) -> int | None:
        return self.selected_identity.global_vehicle_id if self.selected_identity is not None else None

    @property
    def selected_local_track_id(self) -> int | None:
        return self.selected_identity.last_track_id if self.selected_identity is not None else None

    @property
    def lost_frames(self) -> int:
        return self.selected_identity.lost_frames if self.selected_identity is not None else 0

    def reset(self) -> None:
        self.selected_identity = None
        self.status = "idle"
        self.last_reacquire_score = 0.0
        self.camera_cut_seen = False
        self.reacquire.reset_pending()
        self._reset_auto_reid_pending()

    def set_auto_reid_threshold(self, min_score: float) -> None:
        self.auto_reid_min_score = max(0.0, min(1.0, float(min_score)))
        self._reset_auto_reid_pending()

    def select_detection(self, detection: TrackedDetection, frame, persist: bool = True) -> VehicleIdentity:
        color_signature = self.reacquire.color_signature(frame, detection.bbox)
        global_vehicle_id = self._resolve_global_vehicle_id(detection) if persist else None
        identity = self._identity_from_detection(global_vehicle_id, detection, color_signature)
        self.selected_identity = identity
        self.status = "tracking"
        self.last_reacquire_score = 1.0
        self.camera_cut_seen = False
        self.reacquire.reset_pending()
        self._reset_auto_reid_pending()
        return identity

    def link_detection(self, vehicle_id: int, detection: TrackedDetection, frame) -> VehicleIdentity | None:
        if self.identity_store is not None and self.identity_store.get_vehicle(vehicle_id) is None:
            return None
        color_signature = self.reacquire.color_signature(frame, detection.bbox)
        if self.identity_store is not None:
            self.identity_store.update_vehicle(vehicle_id, detection, {"linked_manually": True})
        identity = self._identity_from_detection(vehicle_id, detection, color_signature)
        self.selected_identity = identity
        self.status = "tracking"
        self.last_reacquire_score = 1.0
        self.camera_cut_seen = False
        self.reacquire.reset_pending()
        self._reset_auto_reid_pending()
        return identity

    def select_stored_vehicle(
        self,
        vehicle_id: int,
        detections: list[TrackedDetection],
        frame,
        min_score: float = 0.72,
    ) -> tuple[VehicleIdentity | None, float]:
        if self.identity_store is None:
            return None, 0.0

        stored = self.identity_store.get_vehicle(vehicle_id)
        if stored is None:
            return None, 0.0

        ranked = (
            self.feature_gallery.rank_detections_for_vehicle(vehicle_id, detections, frame)
            if self.feature_gallery is not None
            else []
        )
        best = ranked[0] if ranked else None
        if best is not None and best.score >= min_score:
            color_signature = self.reacquire.color_signature(frame, best.detection.bbox)
            identity = self._identity_from_detection(vehicle_id, best.detection, color_signature)
            self.selected_identity = identity
            self.status = "tracking"
            self.last_reacquire_score = best.score
            self.camera_cut_seen = False
            self.reacquire.reset_pending()
            self._reset_auto_reid_pending()
            if self.identity_store is not None:
                self.identity_store.update_vehicle(vehicle_id, best.detection, {"matched_by": "master_feature_gallery"})
            return identity, best.score

        if self._should_preserve_selected_vehicle(vehicle_id):
            self.last_reacquire_score = best.score if best is not None else self.last_reacquire_score
            self._reset_auto_reid_pending()
            return self.selected_identity, self.last_reacquire_score

        identity = VehicleIdentity(
            global_vehicle_id=vehicle_id,
            last_track_id=None,
            class_name=stored.class_name,
            confidence=stored.confidence,
            last_bbox=stored.bbox,
            last_center=stored.center,
            last_frame_index=stored.last_frame_index,
            last_seen_timestamp=stored.last_seen_timestamp,
            color_signature=None,
            lost_frames=self.searching_after_frames,
            status="searching",
            track_aliases=[],
        )
        self.selected_identity = identity
        self.status = "searching"
        self.last_reacquire_score = best.score if best is not None else 0.0
        self.camera_cut_seen = False
        self.reacquire.reset_pending()
        self._reset_auto_reid_pending()
        return identity, self.last_reacquire_score

    def _should_preserve_selected_vehicle(self, vehicle_id: int) -> bool:
        identity = self.selected_identity
        return bool(
            identity is not None
            and identity.global_vehicle_id == vehicle_id
            and identity.last_track_id is not None
            and self.status == "tracking"
            and identity.status in {"tracking", "coasting"}
            and not self.camera_cut_seen
            and identity.lost_frames <= self.predictive_coast_frames
        )

    def handle_camera_cut(self) -> None:
        if self.selected_identity is None:
            return
        self.selected_identity.last_track_id = None
        self.selected_identity.status = "camera_cut"
        self.status = "camera_cut"
        self.camera_cut_seen = True
        self.reacquire.reset_pending()
        self._reset_auto_reid_pending()

    def update(self, detections: list[TrackedDetection], frame) -> list[SelectedTarget]:
        if self.selected_identity is None:
            self.status = "idle"
            self.last_reacquire_score = 0.0
            return []

        target = self._find_by_current_track(detections)
        if target is None:
            target, score = self._choose_auto_reid_target(detections, frame)
            if target is None and not self._selected_gid_has_master_features():
                target, score = self.reacquire.choose(self.selected_identity, detections, frame)
            self.last_reacquire_score = score
        else:
            self.last_reacquire_score = 1.0
            self._reset_auto_reid_pending()

        if target is not None:
            self._update_identity(target, frame)
            self.status = "tracking"
            self.selected_identity.status = "tracking"
            self.camera_cut_seen = False
            return [self._selected_target_from_detection(target, "tracking")]

        identity = self.selected_identity
        identity.lost_frames += 1
        if (
            not self.camera_cut_seen
            and identity.lost_frames <= self.predictive_coast_frames
            and self._can_predict_safely(identity, frame.shape)
        ):
            self.status = "tracking"
            identity.status = "coasting"
            return [self._coasted_selected_target(frame.shape)]
        if self.camera_cut_seen:
            self.status = "camera_cut"
        elif identity.lost_frames > self.max_lost_frames:
            self.status = "lost"
        elif identity.lost_frames >= self.searching_after_frames:
            self.status = "searching"
        else:
            self.status = "tracking"
        identity.status = self.status
        return [self._selected_target_from_identity()]

    def _choose_auto_reid_target(
        self,
        detections: list[TrackedDetection],
        frame,
    ) -> tuple[TrackedDetection | None, float]:
        identity = self.selected_identity
        if (
            identity is None
            or identity.global_vehicle_id is None
            or self.feature_gallery is None
            or not detections
        ):
            self._reset_auto_reid_pending()
            return None, 0.0

        ranked = self.feature_gallery.rank_detections_for_vehicle(
            identity.global_vehicle_id,
            self._spatial_reid_candidates(identity, detections, frame.shape),
            frame,
        )
        if not ranked:
            return None, 0.0

        best = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else 0.0
        if (
            best.score < self.auto_reid_min_score
            or best.score - second_score < self.auto_reid_margin
        ):
            self._reset_auto_reid_pending()
            return None, best.score

        pending_key = self._reid_pending_key(best.detection)
        if pending_key == self._auto_reid_pending_track_id:
            self._auto_reid_pending_count += 1
        else:
            self._auto_reid_pending_track_id = pending_key
            self._auto_reid_pending_count = 1

        if self._auto_reid_pending_count >= self.auto_reid_confirm_frames:
            self._reset_auto_reid_pending()
            return best.detection, best.score
        return None, best.score

    def _selected_gid_has_master_features(self) -> bool:
        identity = self.selected_identity
        return bool(
            identity is not None
            and identity.global_vehicle_id is not None
            and self.feature_gallery is not None
            and self.feature_gallery.has_master_features(identity.global_vehicle_id)
        )

    def _reset_auto_reid_pending(self) -> None:
        self._auto_reid_pending_track_id = None
        self._auto_reid_pending_count = 0

    def is_selected_detection(self, detection: TrackedDetection) -> bool:
        identity = self.selected_identity
        if identity is None:
            return False
        if detection.track_id is not None and identity.last_track_id is not None:
            return detection.track_id == identity.last_track_id
        return (
            identity.global_vehicle_id is not None
            and detection.frame_index == identity.last_frame_index
            and self._bbox_iou(detection.bbox, identity.last_bbox) >= 0.80
        )

    def global_id_for_detection(self, detection: TrackedDetection) -> int | None:
        if self.is_selected_detection(detection):
            return self.selected_global_vehicle_id
        return None

    def _resolve_global_vehicle_id(self, detection: TrackedDetection) -> int:
        if (
            self.selected_identity is not None
            and self.selected_identity.global_vehicle_id is not None
            and detection.track_id is not None
            and detection.track_id == self.selected_identity.last_track_id
        ):
            if self.identity_store is not None:
                self.identity_store.update_vehicle(self.selected_identity.global_vehicle_id, detection)
            return self.selected_identity.global_vehicle_id

        if self.identity_store is None:
            global_vehicle_id = self.next_global_vehicle_id
            self.next_global_vehicle_id += 1
            return global_vehicle_id

        return self.identity_store.create_vehicle(detection)

    def _find_by_current_track(self, detections: list[TrackedDetection]) -> TrackedDetection | None:
        identity = self.selected_identity
        if identity is None or identity.last_track_id is None:
            return None
        for detection in detections:
            if detection.track_id == identity.last_track_id:
                return detection
        return None

    @staticmethod
    def _reid_pending_key(detection: TrackedDetection) -> int:
        if detection.track_id is not None:
            return int(detection.track_id)
        x1, y1, x2, y2 = detection.bbox
        center_x = int(round((x1 + x2) / 20.0))
        center_y = int(round((y1 + y2) / 20.0))
        return hash((center_x, center_y))

    @staticmethod
    def _bbox_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _update_identity(self, detection: TrackedDetection, frame) -> None:
        if self.selected_identity is None:
            return
        identity = self.selected_identity
        frame_delta = max(1, detection.frame_index - identity.last_frame_index)
        measured_velocity = (
            (detection.center[0] - identity.last_center[0]) / frame_delta,
            (detection.center[1] - identity.last_center[1]) / frame_delta,
        )
        identity.velocity = (
            identity.velocity[0] * 0.65 + measured_velocity[0] * 0.35,
            identity.velocity[1] * 0.65 + measured_velocity[1] * 0.35,
        )
        identity.last_track_id = detection.track_id
        identity.class_name = detection.class_name
        identity.confidence = detection.confidence
        identity.last_bbox = detection.bbox
        identity.last_center = detection.center
        identity.last_frame_index = detection.frame_index
        identity.last_seen_timestamp = detection.timestamp
        identity.lost_frames = 0
        signature = self.reacquire.color_signature(frame, detection.bbox)
        if signature is not None:
            identity.color_signature = signature
        if self.identity_store is not None and identity.global_vehicle_id is not None:
            self.identity_store.update_vehicle(identity.global_vehicle_id, detection)
        if detection.track_id is not None and detection.track_id not in identity.track_aliases:
            identity.track_aliases.append(detection.track_id)
            identity.track_aliases = identity.track_aliases[-12:]

    def _spatial_reid_candidates(
        self,
        identity: VehicleIdentity,
        detections: list[TrackedDetection],
        frame_shape,
    ) -> list[TrackedDetection]:
        if not detections:
            return []
        frame_h, frame_w = frame_shape[:2]
        diagonal = max(1.0, float((frame_w**2 + frame_h**2) ** 0.5))
        predicted = (
            identity.last_center[0] + identity.velocity[0] * max(1, identity.lost_frames + 1),
            identity.last_center[1] + identity.velocity[1] * max(1, identity.lost_frames + 1),
        )
        radius = diagonal * min(0.75, 0.25 + identity.lost_frames * 0.035)
        ranked = sorted(
            detections,
            key=lambda detection: (
                (detection.center[0] - predicted[0]) ** 2
                + (detection.center[1] - predicted[1]) ** 2
            ),
        )
        nearby = [
            detection
            for detection in ranked
            if (
                (detection.center[0] - predicted[0]) ** 2
                + (detection.center[1] - predicted[1]) ** 2
            )
            <= radius**2
        ]
        if nearby:
            return nearby[:6]
        if identity.lost_frames >= 5:
            return ranked[:8]
        return []

    def _can_predict_safely(self, identity: VehicleIdentity, frame_shape) -> bool:
        frame_h, frame_w = frame_shape[:2]
        margin_x = frame_w * 0.08
        margin_y = frame_h * 0.08
        x, y = identity.last_center
        if x <= margin_x or x >= frame_w - margin_x or y <= margin_y or y >= frame_h - margin_y:
            return False
        speed = (identity.velocity[0] ** 2 + identity.velocity[1] ** 2) ** 0.5
        max_speed = max(frame_w, frame_h) * (0.08 if identity.lost_frames <= 3 else 0.12)
        return speed <= max_speed

    def _coasted_selected_target(self, frame_shape) -> SelectedTarget:
        assert self.selected_identity is not None
        identity = self.selected_identity
        frame_h, frame_w = frame_shape[:2]
        lost = max(1, identity.lost_frames)
        dx = identity.velocity[0] * lost
        dy = identity.velocity[1] * lost
        x1, y1, x2, y2 = identity.last_bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        center_x = max(width / 2.0, min(frame_w - width / 2.0, identity.last_center[0] + dx))
        center_y = max(height / 2.0, min(frame_h - height / 2.0, identity.last_center[1] + dy))
        bbox = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        if lost <= 3:
            confidence = identity.confidence
        else:
            decay_progress = min(1.0, (lost - 3) / max(1, self.predictive_coast_frames - 3))
            confidence = identity.confidence * (1.0 - 0.70 * decay_progress)
        return SelectedTarget(
            track_id=identity.last_track_id if identity.last_track_id is not None else -1,
            bbox=bbox,
            class_name=identity.class_name,
            confidence=max(self.coasting_min_confidence, confidence),
            center=(center_x, center_y),
            status="coasting",
            lost_frame_count=lost,
        )

    def _identity_from_detection(
        self,
        vehicle_id: int | None,
        detection: TrackedDetection,
        color_signature: object | None,
    ) -> VehicleIdentity:
        return VehicleIdentity(
            global_vehicle_id=vehicle_id,
            last_track_id=detection.track_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            last_bbox=detection.bbox,
            last_center=detection.center,
            last_frame_index=detection.frame_index,
            last_seen_timestamp=detection.timestamp,
            color_signature=color_signature,
            track_aliases=[] if detection.track_id is None else [detection.track_id],
        )

    @staticmethod
    def _selected_target_from_detection(
        detection: TrackedDetection,
        status: str,
    ) -> SelectedTarget:
        return SelectedTarget(
            track_id=detection.track_id if detection.track_id is not None else -1,
            bbox=detection.bbox,
            class_name=detection.class_name,
            confidence=detection.confidence,
            center=detection.center,
            status=status,  # type: ignore[arg-type]
            lost_frame_count=0,
        )

    def _selected_target_from_identity(self) -> SelectedTarget:
        assert self.selected_identity is not None
        identity = self.selected_identity
        status = "lost" if self.status in {"searching", "camera_cut", "lost"} else "tracking"
        return SelectedTarget(
            track_id=identity.last_track_id if identity.last_track_id is not None else -1,
            bbox=identity.last_bbox,
            class_name=identity.class_name,
            confidence=identity.confidence,
            center=identity.last_center,
            status=status,  # type: ignore[arg-type]
            lost_frame_count=identity.lost_frames,
        )
