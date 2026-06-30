"""Input Video + YOLO Detection module for AutoCamTracker V1.

Responsibilities:
- Open webcam, local video file, or screen-region sources.
- Load an Ultralytics YOLO model.
- Run YOLO tracking with BoT-SORT or Deep OC-SORT.
- Return raw frames plus tracked detections.

This module should not manage target selection, UI layout, reframing, or
recording file output.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import time
from typing import Any, Callable, Iterable, Literal
from urllib.parse import urlparse


SourceType = Literal["webcam", "video_file", "video_url", "screen_region", "iphone"]
TrackerName = Literal["bytetrack", "botsort", "deepocsort"]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "code" / "model"
CACHE_ROOT = Path(tempfile.gettempdir()) / "autocamtracker-cache"


TRACKER_CONFIGS: dict[TrackerName, str] = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "deepocsort": "deepocsort.yaml",
}

VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}


from autocamtracker.tracking.tracker_adapter import DeepOcSortAdapter, TrackerInputDetection


@dataclass
class InputConfig:
    source_type: SourceType = "webcam"
    camera_index: int = 0
    video_path: str | None = None
    video_url: str | None = None
    screen_region: tuple[int, int, int, int] | None = None
    model_path: str = "yolo26s.pt"
    tracker_name: TrackerName = "botsort"
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.65
    vehicle_classes_only: bool = True
    tracker_buffer_seconds: float = 5.0
    target_source_fps: float = 30.0
    detector_imgsz: int | None = 640
    tracker_reid_enabled: bool = False


@dataclass
class TrackedDetection:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    center: tuple[float, float]
    frame_index: int
    timestamp: float
    tracker_name: TrackerName


class VideoDetector:
    """Reads frames and runs Ultralytics YOLO track mode."""

    def __init__(self, config: InputConfig, frame_provider: Callable[[], Any | None] | None = None) -> None:
        self.config = config
        self.model: Any | None = None
        self.tracker_adapter: DeepOcSortAdapter | None = None
        self.capture: Any | None = None
        self.screen_capture: Any | None = None
        self.source_fps: float | None = None
        self.source_frame_count: int | None = None
        self.frame_index = 0
        self._cv2 = None
        self._tracker_config_path: Path | None = None
        self.frame_provider = frame_provider

    def load_model(self) -> None:
        cache_root = CACHE_ROOT
        cache_root.mkdir(parents=True, exist_ok=True)
        for name in ("ultralytics", "matplotlib", "xdg"):
            (cache_root / name).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

        from ultralytics import YOLO

        resolved_model_path = self._resolve_model_path(self.config.model_path)
        self.config.model_path = str(resolved_model_path)
        self.model = YOLO(str(resolved_model_path))
        if self.config.tracker_name == "deepocsort":
            self.tracker_adapter = DeepOcSortAdapter(
                model_dir=MODEL_DIR,
                det_thresh=self.config.confidence_threshold,
                max_age=self._tracker_buffer_frames(),
                iou_threshold=0.3,
            )
        else:
            self.tracker_adapter = None

    def open_source(self) -> None:
        if self.config.source_type in {"webcam", "video_file", "video_url"}:
            import cv2

            self._cv2 = cv2
            source: int | str
            if self.config.source_type == "webcam":
                source = self.config.camera_index
                backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
                self.capture = self._open_camera_capture(cv2, backend)
                if self.capture is None:
                    raise RuntimeError(self._camera_error_message(self.config.camera_index))
                self._configure_tracker_buffer()
                return
            elif self.config.source_type == "video_file":
                if not self.config.video_path:
                    raise ValueError("video_path is required for video_file input")
                source = str(self._resolve_input_path(self.config.video_path))
                backend = cv2.CAP_ANY
            elif self.config.source_type == "video_url":
                if not self.config.video_url:
                    raise ValueError("video_url is required for video_url input")
                source = self._resolve_video_url(self.config.video_url)
                backend = cv2.CAP_ANY

            self.capture = cv2.VideoCapture(source, backend)
            if not self.capture.isOpened():
                raise RuntimeError(f"Unable to open video source: {source}")
            self._configure_capture(cv2, self.capture)
            self.source_fps = self._read_capture_fps(cv2)
            self.source_frame_count = self._read_capture_frame_count(cv2)
            self._configure_tracker_buffer()

        elif self.config.source_type == "screen_region":
            if self.config.screen_region is None:
                raise ValueError("screen_region is required for screen_region input")
            import mss

            self.screen_capture = mss.mss()
            self.source_fps = None
            self.source_frame_count = None
            self._configure_tracker_buffer()

        elif self.config.source_type == "iphone":
            if self.frame_provider is None:
                raise ValueError("frame_provider is required for iphone input")
            self.source_fps = max(1.0, float(self.config.target_source_fps))
            self.source_frame_count = None
            self._configure_tracker_buffer()

        else:
            raise ValueError(f"Unsupported source_type: {self.config.source_type}")

    def read_frame(self) -> Any | None:
        if self.config.source_type in {"webcam", "video_file", "video_url"}:
            if self.capture is None:
                raise RuntimeError("Input source is not open")
            ok, frame = self.capture.read()
            if not ok:
                return None
            self.frame_index += 1
            return frame

        if self.config.source_type == "screen_region":
            if self.screen_capture is None:
                raise RuntimeError("Screen capture source is not open")
            import cv2
            import numpy as np

            x, y, width, height = self.config.screen_region or (0, 0, 0, 0)
            image = self.screen_capture.grab(
                {"left": x, "top": y, "width": width, "height": height}
            )
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.frame_index += 1
            return frame

        if self.config.source_type == "iphone":
            if self.frame_provider is None:
                raise RuntimeError("iPhone frame provider is unavailable")
            frame = self.frame_provider()
            if frame is not None:
                self.frame_index += 1
            return frame

        return None

    def track_frame(self, frame: Any) -> list[TrackedDetection]:
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded")

        if self.config.tracker_name == "deepocsort":
            results = self.model.predict(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.detector_imgsz,
                verbose=False,
            )
            return self._track_with_deepocsort(results)

        tracker_config = str(self._tracker_config_path or TRACKER_CONFIGS[self.config.tracker_name])
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker_config,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.detector_imgsz,
            verbose=False,
        )
        return self._parse_results(results)

    def _track_with_deepocsort(self, results: Iterable[Any]) -> list[TrackedDetection]:
        if self.tracker_adapter is None:
            raise RuntimeError("Deep OC-SORT tracker is not initialized")

        raw_detections = self._parse_prediction_results(results)
        tracked = self.tracker_adapter.update(raw_detections)
        timestamp = time()
        return [
            TrackedDetection(
                track_id=detection.track_id,
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                center=(
                    (detection.bbox[0] + detection.bbox[2]) / 2.0,
                    (detection.bbox[1] + detection.bbox[3]) / 2.0,
                ),
                frame_index=self.frame_index,
                timestamp=timestamp,
                tracker_name=self.config.tracker_name,
            )
            for detection in tracked
        ]

    def read_and_track(self) -> tuple[Any | None, list[TrackedDetection]]:
        frame = self.read_frame()
        if frame is None:
            return None, []
        detections = self.track_frame(frame)
        return frame, detections

    def close(self, clear_temp_cache: bool = False) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.screen_capture is not None:
            self.screen_capture.close()
        self.capture = None
        self.screen_capture = None
        self.source_fps = None
        self.source_frame_count = None
        self.frame_index = 0
        if clear_temp_cache:
            self.clear_temp_cache()

    @staticmethod
    def clear_temp_cache() -> None:
        if CACHE_ROOT.exists():
            shutil.rmtree(CACHE_ROOT, ignore_errors=True)

    def get_source_fps(self) -> float | None:
        return self.source_fps

    def get_source_frame_count(self) -> int | None:
        return self.source_frame_count

    def get_current_frame_index(self) -> int:
        return self.frame_index

    def seek_video_frame(self, frame_index: int) -> bool:
        if self.config.source_type not in {"video_file", "video_url"} or self.capture is None or self._cv2 is None:
            return False
        target_frame = max(0, int(frame_index))
        if self.source_frame_count is not None:
            target_frame = min(max(0, self.source_frame_count - 1), target_frame)
        self.reset_tracker_state()
        ok = self.capture.set(self._cv2.CAP_PROP_POS_FRAMES, target_frame)
        if ok:
            self.frame_index = target_frame
        return bool(ok)

    def reset_tracker_state(self) -> None:
        if self.tracker_adapter is not None:
            reset_adapter = getattr(self.tracker_adapter, "reset", None)
            if callable(reset_adapter):
                reset_adapter()
        trackers = getattr(self.model, "trackers", None)
        if not trackers:
            return
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def _configure_tracker_buffer(self) -> None:
        buffer_frames = self._tracker_buffer_frames()
        if self.tracker_adapter is not None:
            self.tracker_adapter.max_age = buffer_frames
            self.tracker_adapter.reset()

        if self.config.tracker_name == "botsort":
            self._tracker_config_path = self._write_botsort_config(buffer_frames)
        elif self.config.tracker_name == "bytetrack":
            self._tracker_config_path = self._write_bytetrack_config(buffer_frames)
        else:
            self._tracker_config_path = None

    def _tracker_buffer_frames(self) -> int:
        fps = self.source_fps if self.source_fps and self.source_fps > 1.0 else 30.0
        return max(1, int(round(float(self.config.tracker_buffer_seconds) * fps)))

    def _write_botsort_config(self, track_buffer: int) -> Path:
        config_dir = CACHE_ROOT / "trackers"
        config_dir.mkdir(parents=True, exist_ok=True)
        reid_enabled = bool(self.config.tracker_reid_enabled)
        reid_suffix = "reid" if reid_enabled else "motion"
        path = config_dir / f"botsort_{reid_suffix}_buffer_{track_buffer}.yaml"
        path.write_text(
            "\n".join(
                [
                    "tracker_type: botsort",
                    "track_high_thresh: 0.25",
                    "track_low_thresh: 0.1",
                    "new_track_thresh: 0.25",
                    f"track_buffer: {track_buffer}",
                    "match_thresh: 0.8",
                    "fuse_score: True",
                    "gmc_method: sparseOptFlow",
                    "proximity_thresh: 0.5",
                    "appearance_thresh: 0.8",
                    f"with_reid: {str(reid_enabled)}",
                    f"model: {(MODEL_DIR / 'yolo26s-reid.onnx').as_posix()}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _write_bytetrack_config(track_buffer: int) -> Path:
        config_dir = CACHE_ROOT / "trackers"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / f"bytetrack_buffer_{track_buffer}.yaml"
        path.write_text(
            "\n".join(
                [
                    "tracker_type: bytetrack",
                    "track_high_thresh: 0.25",
                    "track_low_thresh: 0.1",
                    "new_track_thresh: 0.25",
                    f"track_buffer: {track_buffer}",
                    "match_thresh: 0.8",
                    "fuse_score: True",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def skip_video_frames(self, frame_count: int) -> int:
        if self.config.source_type not in {"video_file", "video_url"} or self.capture is None:
            return 0

        skipped = 0
        for _ in range(max(0, frame_count)):
            if not self.capture.grab():
                break
            self.frame_index += 1
            skipped += 1
        return skipped

    def _parse_results(self, results: Iterable[Any]) -> list[TrackedDetection]:
        parsed: list[TrackedDetection] = []
        timestamp = time()

        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = self._to_list(getattr(boxes, "xyxy", []))
            cls_values = self._to_list(getattr(boxes, "cls", []))
            conf_values = self._to_list(getattr(boxes, "conf", []))
            id_values = self._to_list(getattr(boxes, "id", []))

            for index, bbox_values in enumerate(xyxy):
                class_id = int(cls_values[index]) if index < len(cls_values) else -1
                class_name = str(names.get(class_id, class_id))
                confidence = float(conf_values[index]) if index < len(conf_values) else 0.0

                if self.config.vehicle_classes_only and class_name not in VEHICLE_CLASS_NAMES:
                    continue
                if confidence < self.config.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [float(value) for value in bbox_values]
                track_id = int(id_values[index]) if index < len(id_values) else None
                parsed.append(
                    TrackedDetection(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        frame_index=self.frame_index,
                        timestamp=timestamp,
                        tracker_name=self.config.tracker_name,
                    )
                )

        return parsed

    def _parse_prediction_results(self, results: Iterable[Any]) -> list[TrackerInputDetection]:
        parsed: list[TrackerInputDetection] = []

        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = self._to_list(getattr(boxes, "xyxy", []))
            cls_values = self._to_list(getattr(boxes, "cls", []))
            conf_values = self._to_list(getattr(boxes, "conf", []))

            for index, bbox_values in enumerate(xyxy):
                class_id = int(cls_values[index]) if index < len(cls_values) else -1
                class_name = str(names.get(class_id, class_id))
                confidence = float(conf_values[index]) if index < len(conf_values) else 0.0

                if self.config.vehicle_classes_only and class_name not in VEHICLE_CLASS_NAMES:
                    continue
                if confidence < self.config.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [float(value) for value in bbox_values]
                parsed.append(
                    TrackerInputDetection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                )

        return parsed

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        path = Path(model_path).expanduser()
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(
                [
                    MODEL_DIR / path,
                    PROJECT_ROOT / path,
                    Path.cwd() / path,
                    path,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        searched = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Put the model under code/model and click Refresh Models.\n"
            f"Searched:\n{searched}"
        )

    @staticmethod
    def _resolve_input_path(input_path: str) -> Path:
        path = Path(input_path).expanduser()
        if path.is_absolute():
            return path
        candidates = [PROJECT_ROOT / path, Path.cwd() / path, path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return PROJECT_ROOT / path

    @staticmethod
    def _validate_video_url(video_url: str) -> str:
        value = video_url.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https", "rtsp", "rtmp"} or not parsed.netloc:
            raise ValueError(
                "Video URL must start with http://, https://, rtsp://, or rtmp://"
            )
        return value

    @classmethod
    def _resolve_video_url(cls, video_url: str) -> str:
        value = cls._validate_video_url(video_url)
        parsed = urlparse(value)
        if parsed.scheme in {"rtsp", "rtmp"}:
            return value
        return cls._extract_stream_url(value)

    @staticmethod
    def _extract_stream_url(video_url: str) -> str:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError(
                "Network video URLs require yt-dlp. Install dependencies with "
                "`.venv/bin/python -m pip install -r requirements.txt`."
            ) from exc

        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "format": "best[protocol^=http][vcodec!=none]/best[protocol^=m3u8][vcodec!=none]/best[vcodec!=none]/best",
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(video_url, download=False)
        except Exception as exc:
            raise RuntimeError(f"Unable to resolve video URL: {video_url}") from exc

        if not isinstance(info, dict):
            raise RuntimeError(f"Unable to resolve video URL: {video_url}")

        stream_url = info.get("url")
        if stream_url:
            return str(stream_url)

        formats = info.get("formats") or []
        for item in reversed(formats):
            if not isinstance(item, dict):
                continue
            candidate = item.get("url")
            vcodec = item.get("vcodec")
            protocol = str(item.get("protocol") or "")
            if candidate and vcodec != "none" and protocol.startswith(("http", "m3u8")):
                return str(candidate)

        raise RuntimeError(f"No playable video stream found for URL: {video_url}")

    @staticmethod
    def _camera_error_message(camera_index: int) -> str:
        mac_hint = ""
        if sys.platform == "darwin":
            mac_hint = (
                "\n\nmacOS permission fix:\n"
                "1. Open System Settings > Privacy & Security > Camera.\n"
                "2. Enable Camera permission for Visual Studio Code, Terminal, "
                "or the app that launched Python.\n"
                "3. Quit and reopen VSCode/Terminal, then run again."
            )
        return (
            "Unable to open MacBook camera. "
            f"Camera index {camera_index} and fallback indexes 0-4 are not available, blocked by permission, "
            "or currently used by another app."
            f"{mac_hint}"
        )

    def _open_camera_capture(self, cv2: Any, backend: int) -> Any | None:
        indexes = [self.config.camera_index]
        indexes.extend(index for index in range(5) if index != self.config.camera_index)

        for index in indexes:
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                self.config.camera_index = index
                self._configure_capture(cv2, capture)
                self.source_fps = self._read_capture_fps(cv2, capture)
                return capture
            capture.release()
        return None

    def _read_capture_fps(self, cv2: Any, capture: Any | None = None) -> float | None:
        target = capture or self.capture
        if target is None:
            return None
        fps = float(target.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0 or fps > 240.0:
            return None
        return fps

    @staticmethod
    def _configure_capture(cv2: Any, capture: Any) -> None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _read_capture_frame_count(self, cv2: Any, capture: Any | None = None) -> int | None:
        target = capture or self.capture
        if target is None:
            return None
        frame_count = int(target.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return None
        return frame_count
