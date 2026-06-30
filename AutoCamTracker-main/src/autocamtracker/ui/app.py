"""Tkinter UI + Recording + Debug Log module for AutoCamTracker V1.

Responsibilities:
- Create the Tkinter desktop UI.
- Wire together input, YOLO tracking, data store, target tracking, and reframe.
- Show before and after views.
- Expose controls for source, tracker, framing mode, and recording.

This file is intentionally a V1 integration scaffold. The core logic lives in
the other four modules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
from threading import Thread
from time import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    from PIL import Image, ImageGrab, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageGrab = None
    ImageTk = None

from autocamtracker.tracking.auto_feature_sampler import AutoFeatureMode, AutoFeatureSampler
from autocamtracker.vision.detector import InputConfig, VideoDetector
from autocamtracker.tracking.detection_store import DetectionStore
from autocamtracker.core.desktop_state import IdentitySessionLinks
from autocamtracker.tracking.feature_gallery import FeatureGallery
from autocamtracker.core.frame_data import FrameData
from autocamtracker.tracking.identity_manager import GlobalIdentityManager
from autocamtracker.core.pipeline_processor import PipelineProcessor
from autocamtracker.core.telemetry_logger import TelemetryLogger
from autocamtracker.core.performance_evaluation import PerformanceEvaluationTracker
from autocamtracker.core.pipeline_worker import TrackingWorker
from autocamtracker.vision.reframer import FramingConfig, Reframer
from autocamtracker.vision.scene_cut import SceneCutDetector
from autocamtracker.server.websocket_server import TrackingWebSocketServer
from autocamtracker.core.track_shot_plan import TrackShotController, TrackZone, should_publish_motor_tracking
from autocamtracker.tracking.vehicle_identity_store import VehicleIdentityStore


@dataclass
class AppConfig:
    window_title: str = "AutoCamTracker V1.75"
    update_interval_ms: int = 15
    output_width: int = 640
    output_height: int = 360
    log_dir: Path = Path("outputs")
    telemetry_dir: Path = Path("outputs") / "telemetry"
    identity_db_path: Path = Path("outputs") / "vehicle_identity.sqlite3"
    model_dir: Path = Path(__file__).resolve().parents[3] / "code" / "model"
    default_model: str = "yolo26n.pt"
    default_reid_model: str = "yolo26s-reid.onnx"



from autocamtracker.ui.mixins.ui_builder import UIBuilderMixin
from autocamtracker.ui.mixins.identity_panel import IdentityPanelMixin
from autocamtracker.ui.mixins.video_pipeline import VideoPipelineMixin
from autocamtracker.ui.mixins.commands import CommandsMixin
from autocamtracker.ui.mixins.performance_panel import PerformancePanelMixin

class AutoCamTrackerApp(UIBuilderMixin, IdentityPanelMixin, VideoPipelineMixin, CommandsMixin, PerformancePanelMixin):
    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.config = config or AppConfig()
        self.root.title(self.config.window_title)
        self.root.minsize(1120, 720)

        self.input_config = InputConfig()
        self.detector: VideoDetector | None = None
        self.tracking_worker: TrackingWorker | None = None
        self.store = DetectionStore()
        self.identity_store = VehicleIdentityStore(self.config.identity_db_path)
        self.feature_gallery = FeatureGallery(
            self.config.identity_db_path,
            reid_model_path=str(self.config.model_dir / self.config.default_reid_model),
        )
        self.identity_manager = GlobalIdentityManager(
            identity_store=self.identity_store,
            feature_gallery=self.feature_gallery,
        )
        self.auto_feature_sampler = AutoFeatureSampler(self.feature_gallery)
        self.scene_cut_detector = SceneCutDetector()
        self.telemetry_logger = TelemetryLogger(self.config.telemetry_dir)
        self.performance_evaluator = PerformanceEvaluationTracker()
        self.reframer = Reframer(
            FramingConfig(
                output_width=self.config.output_width,
                output_height=self.config.output_height,
            )
        )
        self.pipeline = PipelineProcessor(
            store=self.store,
            identity_manager=self.identity_manager,
            scene_cut_detector=self.scene_cut_detector,
            reframer=self.reframer,
        )
        self.iphone_status_queue: SimpleQueue[str] = SimpleQueue()
        self.iphone_control_queue: SimpleQueue[dict] = SimpleQueue()
        self.tracking_server = TrackingWebSocketServer(
            on_status=self._queue_iphone_status,
            on_control=self._queue_iphone_control,
            telemetry_logger=self.telemetry_logger,
        )
        self.telemetry_logger.log(
            "app_started",
            version=self.config.window_title,
            telemetry_path=self.telemetry_logger.path,
        )
        self.track_shot_controller = TrackShotController()
        # Physical motor output is explicitly armed by Auto Track or Find GID.
        # A selected target can therefore still drive digital reframing without
        # unexpectedly moving the DockKit accessory.
        self.iphone_motor_tracking_enabled = False

        self.running = False
        self.recording = False
        self.last_frame_time = time()
        self.loop_started_at = time()
        self.fps = 0.0
        self.skipped_frames = 0
        self.last_inference_time_ms = 0.0
        self.model_options: dict[str, str] = {}
        self.reid_model_options: dict[str, str] = {}
        self.active_input_signature: tuple[object, ...] | None = None
        self.last_frame_shape: tuple[int, int, int] | tuple[int, int] | None = None
        self.last_raw_frame = None
        self.current_frame_data: FrameData | None = None
        self.display_width = self.config.output_width
        self.display_height = self.config.output_height
        self.preview_width_limit = self.display_width
        self.preview_height_limit = self.display_height
        self.rendered_image_width = self.display_width
        self.rendered_image_height = self.display_height
        self.timeline_dragging = False
        self.refreshing_identity_panel = False
        self.selected_identity_tree_ids: set[int] = set()
        self.identity_session_links = IdentitySessionLinks()
        self.last_identity_panel_refresh_at = 0.0
        self.identity_preview_window: tk.Toplevel | None = None
        self.performance_window: tk.Toplevel | None = None
        self.identity_preview_label: ttk.Label | None = None
        self.identity_preview_photo = None
        self.identity_preview_vehicle_id: int | None = None
        self.auto_feature_status_message = ""
        self.last_desktop_state_publish_at = 0.0
        self.last_frame_telemetry_at = 0.0
        self.last_preview_render_at = 0.0
        self.preview_render_interval_seconds = 0.10

        self.before_image_ref = None
        self.after_image_ref = None
        self._build_ui()
        self.root.after(100, self._drain_iphone_status)
        self.root.after(100, self._drain_iphone_control)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_model_options()
        self.refresh_reid_model_options()
        self.root.after_idle(self.on_source_selected)
        self.root.after_idle(self._preload_reid_model)

    """Tkinter integration shell for the five V1 modules."""
