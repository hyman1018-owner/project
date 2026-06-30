"""Adapters for external multi-object trackers bundled under code/model.

The V1 app owns camera input, YOLO detection, selection, and reframing. These
adapters only convert YOLO detections into stable track IDs and return them in a
small neutral format that VideoDetector can map back to TrackedDetection.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


@dataclass
class TrackerInputDetection:
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


@dataclass
class TrackerOutputDetection:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


class DeepOcSortAdapter:
    """Wraps code/model/Deep-OC-SORT-main OCSort for live V1 frames."""

    def __init__(
        self,
        model_dir: Path,
        det_thresh: float = 0.25,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ) -> None:
        repo_root = model_dir / "Deep-OC-SORT-main"
        if not repo_root.exists():
            raise FileNotFoundError(f"Deep-OC-SORT repo not found: {repo_root}")
        repo_root_text = str(repo_root)
        if repo_root_text not in sys.path:
            sys.path.insert(0, repo_root_text)

        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_byte = True
        cache_root = Path(tempfile.gettempdir()) / "autocamtracker-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

        from trackers.ocsort_tracker.ocsort import OCSort

        self._tracker_cls = OCSort
        self.tracker = self._create_tracker()

    def reset(self) -> None:
        self.tracker = self._create_tracker()

    def _create_tracker(self):
        return self._tracker_cls(
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            use_byte=self.use_byte,
        )

    def update(self, detections: list[TrackerInputDetection]) -> list[TrackerOutputDetection]:
        if not detections:
            tracks = self.tracker.update_public(
                np.empty((0, 4), dtype=float),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=float),
            )
            return self._format_tracks(tracks, detections)

        boxes = np.asarray([detection.bbox for detection in detections], dtype=float)
        class_ids = np.asarray([detection.class_id for detection in detections], dtype=int)
        scores = np.asarray([detection.confidence for detection in detections], dtype=float)
        tracks = self.tracker.update_public(boxes, class_ids, scores)
        return self._format_tracks(tracks, detections)

    def _format_tracks(
        self,
        tracks: Any,
        detections: list[TrackerInputDetection],
    ) -> list[TrackerOutputDetection]:
        if tracks is None or len(tracks) == 0:
            return []

        outputs: list[TrackerOutputDetection] = []
        for row in np.asarray(tracks):
            x1, y1, x2, y2 = [float(value) for value in row[:4]]
            track_id = int(row[4])
            class_id = int(row[5]) if len(row) > 5 else -1
            matched = self._match_detection((x1, y1, x2, y2), class_id, detections)
            class_name = matched.class_name if matched else str(class_id)
            confidence = matched.confidence if matched else 0.0
            outputs.append(
                TrackerOutputDetection(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
            )
        return outputs

    def _match_detection(
        self,
        bbox: tuple[float, float, float, float],
        class_id: int,
        detections: list[TrackerInputDetection],
    ) -> TrackerInputDetection | None:
        same_class = [detection for detection in detections if detection.class_id == class_id]
        candidates = same_class or detections
        if not candidates:
            return None
        return max(candidates, key=lambda detection: self._iou(bbox, detection.bbox))

    @staticmethod
    def _iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
