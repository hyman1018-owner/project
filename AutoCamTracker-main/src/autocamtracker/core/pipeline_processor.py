"""Frame pipeline boundary for AutoCamTracker V1.

This is the UI-safe stepping stone toward a threaded PipelineWorker: the Tk app
still owns scheduling, playback, and image rendering, while this class owns the
per-frame data flow from detections to identity, reframing, and FrameData.
"""

from __future__ import annotations

from time import time
from typing import Callable

from autocamtracker.tracking.detection_store import DetectionStore
from autocamtracker.core.frame_data import FrameData
from autocamtracker.tracking.identity_manager import GlobalIdentityManager
from autocamtracker.vision.reframer import Reframer
from autocamtracker.vision.scene_cut import SceneCutDetector
from autocamtracker.vision.detector import TrackedDetection


class PipelineProcessor:
    def __init__(
        self,
        store: DetectionStore,
        identity_manager: GlobalIdentityManager,
        scene_cut_detector: SceneCutDetector,
        reframer: Reframer,
    ) -> None:
        self.store = store
        self.identity_manager = identity_manager
        self.scene_cut_detector = scene_cut_detector
        self.reframer = reframer

    def reset(self) -> None:
        self.store.reset()
        self.identity_manager.reset()
        self.scene_cut_detector.reset()
        self.reframer.reset()

    def process(
        self,
        frame,
        detections: list[TrackedDetection],
        draw_detections: Callable[[object, list[TrackedDetection]], object],
        reset_tracker_state: Callable[[], None] | None = None,
        inference_time_ms: float = 0.0,
        source_fps: float | None = None,
        skipped_frames: int = 0,
        render_preview: bool = True,
        decode_time_ms: float = 0.0,
        receive_latency_ms: float | None = None,
    ) -> FrameData:
        pipeline_started_at = time()
        identity_started_at = time()
        camera_cut = self.scene_cut_detector.update(frame)
        if camera_cut:
            if reset_tracker_state is not None:
                reset_tracker_state()
            self.store.reset()
            self.identity_manager.handle_camera_cut()
            detections = []

        candidates = self.store.update(detections, frame.shape)
        selected_targets = self.identity_manager.update(detections, frame)
        identity_time_ms = (time() - identity_started_at) * 1000.0

        reframe_started_at = time()
        if render_preview:
            after_frame, framing_status = self.reframer.render(frame, selected_targets)
        else:
            framing_status = self.reframer.status(frame, selected_targets)
            after_frame = frame
        reframe_time_ms = (time() - reframe_started_at) * 1000.0
        preview_started_at = time()
        before_frame = draw_detections(frame, detections) if render_preview else frame
        preview_time_ms = (time() - preview_started_at) * 1000.0

        return FrameData(
            raw_frame=frame,
            before_frame=before_frame,
            after_frame=after_frame,
            detections=detections,
            candidates=candidates,
            selected_targets=selected_targets,
            framing_status=framing_status,
            tracking_status=self.identity_manager.status,
            selected_global_vehicle_id=self.identity_manager.selected_global_vehicle_id,
            selected_local_track_id=self.identity_manager.selected_local_track_id,
            camera_cut_detected=camera_cut,
            lost_frames=self.identity_manager.lost_frames,
            reacquire_score=self.identity_manager.last_reacquire_score,
            source_fps=source_fps,
            inference_time_ms=inference_time_ms,
            decode_time_ms=decode_time_ms,
            receive_latency_ms=receive_latency_ms,
            pipeline_time_ms=(time() - pipeline_started_at) * 1000.0,
            identity_time_ms=identity_time_ms,
            reframe_time_ms=reframe_time_ms,
            preview_time_ms=preview_time_ms,
            skipped_frames=skipped_frames,
        )
