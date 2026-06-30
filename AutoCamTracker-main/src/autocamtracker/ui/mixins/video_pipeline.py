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
except ImportError:
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
from autocamtracker.core.pipeline_worker import TrackingWorker
from autocamtracker.vision.reframer import FramingConfig, Reframer
from autocamtracker.vision.scene_cut import SceneCutDetector
from autocamtracker.server.websocket_server import (
    CENTER_ZOOM_FACTOR,
    SOURCE_VERSION,
    TrackingWebSocketServer,
    zoom_factor_for_framing,
)
from autocamtracker.core.track_shot_plan import TrackShotController, TrackZone, should_publish_motor_tracking
from autocamtracker.tracking.vehicle_identity_store import VehicleIdentityStore

class VideoPipelineMixin:
    def _request_worker_frame(self) -> None:
        if not self.running or self.tracking_worker is None:
            return
        if self.tracking_worker.request_frame():
            self.loop_started_at = time()
        self.root.after(5, self._loop)

    def _loop(self) -> None:
        if not self.running or self.detector is None or self.tracking_worker is None:
            return

        result = self.tracking_worker.poll()
        if result is None:
            self.root.after(5, self._loop)
            return
        if result.error is not None:
            error = result.error
            self.stop()
            messagebox.showerror("Detection failed", str(error))
            return

        frame, frame_data = result.raw_frame, result.frame_data
        self.last_inference_time_ms = result.inference_time_ms
        if frame is None or frame_data is None:
            if self.input_config.source_type == "iphone":
                self.status_var.set("Status: waiting for iPhone camera frames")
                self.root.after(30, self._request_worker_frame)
                return
            self.stop()
            return

        self._process_frame_data(frame_data, frame)
        now = time()
        elapsed = max(1e-6, now - self.last_frame_time)
        self.fps = 1.0 / elapsed
        self.last_frame_time = now
        frame_data.display_fps = self.fps
        frame_data.source_fps = self.detector.get_source_fps()
        frame_data.skipped_frames = self.skipped_frames
        self.performance_evaluator.record_frame(frame_data)

        auto_feature_note = (
            f" | AutoFeat: {self.auto_feature_status_message}"
            if self.auto_feature_status_message
            else ""
        )
        self.status_var.set(
            "Status: "
            f"{frame_data.tracking_status} | Display FPS: {self.fps:.1f} | "
            f"Source: {self._source_fps_label()} | Speed: {self.playback_speed_var.get()} | "
            f"Skipped: {self.skipped_frames} | "
            f"Candidates: {len(frame_data.candidates)} | "
            f"GID: {frame_data.selected_global_vehicle_id or '--'} | "
            f"LID: {frame_data.selected_local_track_id if frame_data.selected_local_track_id is not None else '--'} | "
            f"Lost: {frame_data.lost_frames} | ReID: {frame_data.reacquire_score:.2f} | "
            f"Cut: {'yes' if frame_data.camera_cut_detected else 'no'} | "
            f"Motor: {self._motor_state_label()} | "
            f"Crop: {frame_data.framing_status.crop_window}"
            f"{auto_feature_note}"
        )

        self._drop_late_video_frames()
        self._sync_timeline_from_detector()
        self.root.after(self._next_loop_delay_ms(), self._request_worker_frame)

    def _process_frame_data(self, frame_data: FrameData, frame) -> None:
        self.last_frame_shape = frame.shape
        self.last_raw_frame = frame
        self._sync_reframer_to_source_size(frame.shape)
        
        if frame_data.camera_cut_detected:
            self._stop_auto_feature_capture_for_scene_change()
        self.current_frame_data = frame_data
        
        shot_decision = self.track_shot_controller.evaluate(frame_data, frame.shape)
        self.track_shot_state_var.set(
            f"Shot: {self.track_shot_controller.mode} · {shot_decision.state} · {shot_decision.reason}"
        )
        motor_output_active = should_publish_motor_tracking(
            self.input_config.source_type,
            self.iphone_motor_tracking_enabled,
            self.tracking_server.motor_ready,
            shot_decision,
        )
        if motor_output_active:
            self.tracking_server.publish_frame(frame_data, frame.shape)
        elif (
            self.input_config.source_type == "iphone"
            and self.iphone_motor_tracking_enabled
            and shot_decision.reason in {"target unavailable", "target temporarily lost"}
        ):
            self._disable_iphone_motor_tracking(shot_decision.reason)
        else:
            self.tracking_server.publish_stop(CENTER_ZOOM_FACTOR)
        self._log_frame_telemetry(frame_data, frame.shape, shot_decision, motor_output_active)
        if frame_data.before_frame is not frame:
            after_view_frame = frame_data.raw_frame if self.input_config.source_type == "iphone" else frame_data.after_frame
            self._update_images(frame_data.before_frame, after_view_frame)
        self._run_auto_feature_sampling(frame)
        self.refresh_identity_db_panel(force=False)
        self.publish_desktop_state(force=False)

    def _log_frame_telemetry(self, frame_data: FrameData, frame_shape, shot_decision, motor_output_active: bool) -> None:
        now = time()
        if now - self.last_frame_telemetry_at < 0.5:
            return
        self.last_frame_telemetry_at = now
        selected_target = frame_data.selected_targets[0] if frame_data.selected_targets else None
        frame_h, frame_w = frame_shape[:2]
        target_locked = (
            frame_data.tracking_status == "tracking"
            and selected_target is not None
            and (
                (selected_target.status == "tracking" and selected_target.lost_frame_count == 0)
                or (selected_target.status == "coasting" and selected_target.lost_frame_count <= 3)
            )
        )
        fresh_target = selected_target if target_locked else None
        motor_status = self.tracking_server.motor_status
        self.telemetry_logger.log(
            "frame_state",
            tracking_status=frame_data.tracking_status,
            target_locked=target_locked,
            predicted_target=bool(fresh_target is not None and fresh_target.status == "coasting"),
            selected_gid=frame_data.selected_global_vehicle_id,
            selected_lid=frame_data.selected_local_track_id,
            lost_frames=frame_data.lost_frames,
            candidate_count=len(frame_data.candidates),
            confidence=float(fresh_target.confidence) if fresh_target is not None else 0.0,
            bbox=fresh_target.bbox if fresh_target is not None else None,
            target_center=fresh_target.center if fresh_target is not None else None,
            frame_width=frame_w,
            frame_height=frame_h,
            framing_mode=self.framing_var.get(),
            crop_window=frame_data.framing_status.crop_window,
            error_x=frame_data.framing_status.error_x,
            error_y=frame_data.framing_status.error_y,
            shot_state=shot_decision.state,
            shot_reason=shot_decision.reason,
            motor_armed=self.iphone_motor_tracking_enabled,
            motor_ready=self.tracking_server.motor_ready,
            motor_output_active=motor_output_active,
            requested_zoom_factor=self._frame_zoom_factor(frame_data, fresh_target, frame_shape),
            phone_camera_zoom_factor=motor_status.camera_zoom_factor if motor_status is not None else None,
            phone_camera_display_zoom_factor=(
                motor_status.camera_display_zoom_factor if motor_status is not None else None
            ),
            phone_last_command_zoom_factor=(
                motor_status.last_command.get("zoom_factor")
                if motor_status is not None and isinstance(motor_status.last_command, dict)
                else None
            ),
            phone_last_command_source_version=(
                motor_status.last_command.get("source_version")
                if motor_status is not None and isinstance(motor_status.last_command, dict)
                else None
            ),
            receive_latency_ms=frame_data.receive_latency_ms,
            decode_time_ms=frame_data.decode_time_ms,
            inference_time_ms=frame_data.inference_time_ms,
            pipeline_time_ms=frame_data.pipeline_time_ms,
            identity_time_ms=frame_data.identity_time_ms,
            reframe_time_ms=frame_data.reframe_time_ms,
            preview_time_ms=frame_data.preview_time_ms,
        )

    def _render_current_video_frame(self) -> None:
        if self.detector is None:
            return
        
        def read():
            return self.detector.read_and_track()
            
        frame, detections = (
            self.tracking_worker.run_locked(read)
            if self.tracking_worker is not None
            else read()
        )
        if frame is None:
            return
            
        frame_data = self.pipeline.process(
            frame=frame,
            detections=detections,
            draw_detections=self._draw_detections,
            reset_tracker_state=self.detector.reset_tracker_state if self.detector is not None else None,
            inference_time_ms=self.last_inference_time_ms,
            source_fps=self.detector.get_source_fps() if self.detector is not None else None,
            skipped_frames=self.skipped_frames,
            render_preview=True,
        )
        self._process_frame_data(frame_data, frame)
        self._sync_timeline_from_detector()

    def _next_loop_delay_ms(self) -> int:
        if not self._is_video_source_active():
            return self.config.update_interval_ms

        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return self.config.update_interval_ms

        target_interval_ms = 1000.0 / max(1.0, source_fps) / self._playback_speed()
        elapsed_ms = (time() - self.loop_started_at) * 1000.0
        return max(1, int(round(target_interval_ms - elapsed_ms)))

    def _drop_late_video_frames(self) -> None:
        if not self._is_video_source_active():
            return

        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return

        target_interval_ms = 1000.0 / max(1.0, source_fps) / self._playback_speed()
        elapsed_ms = (time() - self.loop_started_at) * 1000.0
        frames_due = int(elapsed_ms // max(1.0, target_interval_ms))
        frames_to_skip = min(30, max(0, frames_due - 1))
        if frames_to_skip <= 0:
            return

        skip = lambda: self.detector.skip_video_frames(frames_to_skip)
        skipped = (
            self.tracking_worker.run_locked(skip)
            if self.tracking_worker is not None
            else skip()
        )
        self.skipped_frames += skipped

    def _playback_speed(self) -> float:
        value = self.playback_speed_var.get().strip().lower().replace("x", "")
        try:
            speed = float(value)
        except ValueError:
            return 1.0
        return max(0.05, speed)

    def _source_fps_label(self) -> str:
        if self.detector is None:
            return "--"
        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return "--"
        return f"{source_fps:.1f}"

    def _sync_timeline_from_detector(self) -> None:
        if not self._is_video_source_active():
            self.timeline_scale.configure(to=0)
            self.timeline_var.set(0)
            self.timeline_label_var.set("00:00 / 00:00")
            return

        frame_count = self.detector.get_source_frame_count() or 0
        current_frame = self.detector.get_current_frame_index()
        self.timeline_scale.configure(to=max(0, frame_count - 1))
        if not self.timeline_dragging:
            self.timeline_var.set(min(max(0, current_frame), max(0, frame_count - 1)))
            self._update_timeline_label(current_frame)

    def _update_timeline_label(self, frame_index: int) -> None:
        frame_count = self.detector.get_source_frame_count() if self.detector is not None else None
        fps = self.detector.get_source_fps() if self.detector is not None else None
        if not frame_count or not fps:
            self.timeline_label_var.set(f"{frame_index}")
            return
        current_seconds = max(0.0, frame_index / fps)
        total_seconds = max(0.0, frame_count / fps)
        self.timeline_label_var.set(
            f"{self._format_time(current_seconds)} / {self._format_time(total_seconds)}"
        )

    def _reset_runtime_state(self) -> None:
        self.pipeline.reset()
        self.feature_gallery.reset_runtime_cache()
        self.identity_session_links.clear()
        self.last_frame_shape = None
        self.last_raw_frame = None
        self.current_frame_data = None
        self.last_preview_render_at = 0.0
        self.skipped_frames = 0
        self.performance_evaluator.reset()
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        self._set_identity_mode("click bbox to select a local track")
        self._refresh_selection_panel()

    def _clear_screen_region_selection(self) -> None:
        self.input_config.screen_region = None
        self.screen_region_var.set("No screen region selected")
        if self.detector is not None and self.detector.config.source_type == "screen_region":
            self._close_detector()
            self.detector = None
            self.active_input_signature = None
        self._reset_runtime_state()

    def _is_video_source_active(self) -> bool:
        return (
            self.detector is not None
            and self.input_config.source_type in {"video_file", "video_url"}
        )

    def _should_render_preview_frame(self) -> bool:
        if self.input_config.source_type != "iphone":
            return True
        now = time()
        if now - self.last_preview_render_at < self.preview_render_interval_seconds:
            return False
        self.last_preview_render_at = now
        return True

    def _set_display_size(self, width: int, height: int) -> bool:
        width = max(160, min(3840, int(width)))
        height = max(90, min(2160, int(height)))
        if width == self.display_width and height == self.display_height:
            return False

        self.display_width = width
        self.display_height = height
        if hasattr(self, "before_canvas"):
            self.before_canvas.configure(width=width, height=height)
            self.after_canvas.configure(width=width, height=height)
        return True

    def _sync_reframer_to_source_size(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int],
    ) -> None:
        frame_h, frame_w = frame_shape[:2]
        self.config.output_width = frame_w
        self.config.output_height = frame_h
        self.reframer.config.output_width = frame_w
        self.reframer.config.output_height = frame_h
        width, height = self._fit_size_to_source_aspect(
            self.preview_width_limit,
            self.preview_height_limit,
        )
        self._set_display_size(width, height)

    def _close_detector(self) -> None:
        if self.tracking_worker is not None:
            self.tracking_worker.close()
            self.tracking_worker = None
        if self.detector is None:
            return
        clear_temp_cache = self.detector.config.source_type in {"video_file", "video_url"}
        self.detector.close(clear_temp_cache=clear_temp_cache)

    def _discover_model_files(self) -> list[Path]:
        suffixes = {".pt", ".pth", ".onnx", ".engine", ".mlpackage", ".torchscript"}
        if not self.config.model_dir.exists():
            return []
        return sorted(
            path
            for path in self.config.model_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )

    def _model_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.model_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _short_label(value: str, max_length: int = 38) -> str:
        if len(value) <= max_length:
            return value
        keep = max(8, max_length - 3)
        return f"{value[:keep]}..."

    def _normalized_video_url(self) -> str | None:
        value = self.video_url_var.get().strip()
        return value or None

    def _detection_for_track(self, track_id: int):
        for detection in self.store.current_detections:
            if detection.track_id == track_id:
                return detection
        return None

    def _detection_for_vehicle_id(self, vehicle_id: int):
        identity = self.identity_manager.selected_identity
        if (
            identity is not None
            and identity.global_vehicle_id == vehicle_id
            and identity.last_track_id is not None
        ):
            detection = self._detection_for_track(identity.last_track_id)
            if detection is not None:
                return detection

        stored = self.identity_store.get_vehicle(vehicle_id)
        if stored is not None and stored.last_track_id is not None:
            return self._detection_for_track(stored.last_track_id)
        return None

    def _draw_detections(self, frame, detections):
        import cv2

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection.bbox]
            global_id = self.identity_session_links.vehicle_for_track(detection.track_id)
            if global_id is None:
                global_id = self.identity_manager.global_id_for_detection(detection)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            is_selected = self.identity_manager.is_selected_detection(detection)
            box_color = (0, 0, 255) if is_selected else (80, 220, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 4 if is_selected else 3)

            if is_selected:
                gid_height = 52
                gid_scale = cv2.getFontScaleFromHeight(font_face, gid_height, 3)
                text_x = max(0, x1)
                gid_y = max(gid_height + 4, y1 - 8)
                cv2.putText(
                    annotated,
                    f"GID {global_id if global_id is not None else '--'}",
                    (text_x, gid_y),
                    font_face,
                    gid_scale,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
        return annotated

    def _motor_state_label(self) -> str:
        if not self.iphone_motor_tracking_enabled:
            return "OFF"
        if self.tracking_server.client_count == 0:
            return "WAITING IPHONE"
        if not self.tracking_server.motor_ready:
            return "WAITING DOCKKIT"
        return "ON"

    def publish_desktop_state(self, force: bool = False) -> None:
        if self.tracking_server.client_count == 0:
            return
        now = time()
        if not force and now - self.last_desktop_state_publish_at < 0.5:
            return
        self.last_desktop_state_publish_at = now
        self.tracking_server.publish(self._desktop_state_message())

    def _desktop_state_message(self) -> dict:
        frame_data = self.current_frame_data
        motor_status = self.tracking_server.motor_status
        selected_target = None
        if frame_data is not None and frame_data.selected_targets:
            selected_target = frame_data.selected_targets[0]

        error_x = 0.0
        error_y = 0.0
        target_locked = False
        if frame_data is not None:
            target_locked = (
                frame_data.tracking_status == "tracking"
                and selected_target is not None
                and (
                    (selected_target.status == "tracking" and selected_target.lost_frame_count == 0)
                    or (selected_target.status == "coasting" and selected_target.lost_frame_count <= 3)
                )
            )
            if target_locked and self.last_frame_shape is not None:
                frame_h, frame_w = self.last_frame_shape[:2]
                error_x = frame_data.framing_status.error_x / max(1.0, frame_w / 2.0)
                error_y = frame_data.framing_status.error_y / max(1.0, frame_h / 2.0)
        fresh_target = selected_target if target_locked else None
        zoom_factor = self._frame_zoom_factor(frame_data, fresh_target, self.last_frame_shape)

        return {
            "type": "desktop_state",
            "version": "1.0",
            "source_version": SOURCE_VERSION,
            "timestamp_ms": int(time() * 1000),
            "source": self.source_var.get(),
            "running": bool(self.running),
            "tracking": {
                "status": frame_data.tracking_status if frame_data is not None else "idle",
                "target_locked": target_locked,
                "target_id": (
                    frame_data.selected_global_vehicle_id
                    if frame_data is not None and frame_data.selected_global_vehicle_id is not None
                    else frame_data.selected_local_track_id
                    if frame_data is not None
                    else None
                ),
                "selected_gid": (
                    frame_data.selected_global_vehicle_id
                    if frame_data is not None and frame_data.selected_global_vehicle_id is not None
                    else self.identity_manager.selected_global_vehicle_id
                ),
                "selected_lid": frame_data.selected_local_track_id if frame_data is not None else None,
                "error_x": max(-1.0, min(1.0, float(error_x))),
                "error_y": max(-1.0, min(1.0, float(error_y))),
                "confidence": float(fresh_target.confidence) if fresh_target is not None else 0.0,
                "predicted_target": bool(fresh_target is not None and fresh_target.status == "coasting"),
                "lost_frames": int(frame_data.lost_frames) if frame_data is not None else 0,
                "candidate_count": len(frame_data.candidates) if frame_data is not None else 0,
                "bbox": fresh_target.bbox if fresh_target is not None else None,
                "target_center": fresh_target.center if fresh_target is not None else None,
            },
            "motor": {
                "armed": bool(self.iphone_motor_tracking_enabled),
                "ready": bool(self.tracking_server.motor_ready),
                "client_count": int(self.tracking_server.client_count),
                "docked": bool(motor_status.docked) if motor_status is not None else False,
                "manual_ready": bool(motor_status.manual_ready) if motor_status is not None else False,
                "system_tracking_enabled": (
                    motor_status.system_tracking_enabled if motor_status is not None else None
                ),
                "last_error": motor_status.last_error if motor_status is not None else None,
                "current_velocity": motor_status.current_velocity if motor_status is not None else None,
                "last_command": motor_status.last_command if motor_status is not None else None,
                "last_stop_reason": motor_status.last_stop_reason if motor_status is not None else None,
                "camera_zoom_factor": motor_status.camera_zoom_factor if motor_status is not None else None,
                "camera_display_zoom_factor": (
                    motor_status.camera_display_zoom_factor if motor_status is not None else None
                ),
            },
            "framing": {
                "mode": self.framing_var.get(),
                "crop_window": frame_data.framing_status.crop_window if frame_data is not None else None,
                "error_x": frame_data.framing_status.error_x if frame_data is not None else 0.0,
                "error_y": frame_data.framing_status.error_y if frame_data is not None else 0.0,
                "zoom_factor": zoom_factor,
            },
            "diagnostics": {
                "desktop_version": f"AutoCamTracker V{SOURCE_VERSION}",
                "receive_latency_ms": frame_data.receive_latency_ms if frame_data is not None else None,
                "decode_time_ms": frame_data.decode_time_ms if frame_data is not None else 0.0,
                "inference_time_ms": frame_data.inference_time_ms if frame_data is not None else 0.0,
                "pipeline_time_ms": frame_data.pipeline_time_ms if frame_data is not None else 0.0,
                "phone_last_command_source_version": (
                    motor_status.last_command.get("source_version")
                    if motor_status is not None and isinstance(motor_status.last_command, dict)
                    else None
                ),
                "phone_last_command_zoom_factor": (
                    motor_status.last_command.get("zoom_factor")
                    if motor_status is not None and isinstance(motor_status.last_command, dict)
                    else None
                ),
                "zoom_command_matches_phone": (
                    abs(float(zoom_factor) - float(motor_status.last_command.get("zoom_factor", -999))) < 0.05
                    if zoom_factor is not None
                    and motor_status is not None
                    and isinstance(motor_status.last_command, dict)
                    and motor_status.last_command.get("zoom_factor") is not None
                    else False
                ),
            },
            "gids": self._desktop_state_gids(),
        }

    def _frame_zoom_factor(self, frame_data, target, frame_shape) -> float | None:
        if frame_data is None or target is None or frame_shape is None:
            return None
        return zoom_factor_for_framing(self.framing_var.get())

    def _desktop_state_gids(self) -> list[dict]:
        summary = self.identity_store.summary(feature_counts=self.feature_gallery.summary_by_vehicle())
        selected_gid = self.identity_manager.selected_global_vehicle_id
        selected_tree_ids = set(self.selected_identity_tree_ids)
        visible_track_ids = {detection.track_id for detection in self.store.current_detections}
        return [
            {
                "gid": vehicle.vehicle_id,
                "display_name": vehicle.display_name,
                "class_name": vehicle.class_name,
                "last_track_id": vehicle.last_track_id,
                "last_frame_index": vehicle.last_frame_index,
                "confidence": vehicle.confidence,
                "master_feature_count": vehicle.master_feature_count,
                "pending_feature_count": vehicle.pending_feature_count,
                "candidate_feature_count": vehicle.candidate_feature_count,
                "trackable": vehicle.master_feature_count > 0,
                "visible": vehicle.last_track_id in visible_track_ids,
                "selected": vehicle.vehicle_id == selected_gid or vehicle.vehicle_id in selected_tree_ids,
            }
            for vehicle in summary.vehicles
        ]

    def _update_images(self, before_frame, after_frame) -> None:
        if Image is None or ImageTk is None:
            return
        import cv2

        before_rgb = cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB)

        size = (self.display_width, self.display_height)
        before_image = Image.fromarray(before_rgb).resize(size)
        after_image = Image.fromarray(after_rgb).resize(size)
        self.rendered_image_width, self.rendered_image_height = size

        self.before_image_ref = ImageTk.PhotoImage(before_image)
        self.after_image_ref = ImageTk.PhotoImage(after_image)
        
        if not hasattr(self, "before_image_id"):
            self.before_image_id = self.before_canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self.before_image_ref,
            )
            self.after_image_id = self.after_canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self.after_image_ref,
            )
        else:
            self.before_canvas.coords(self.before_image_id, 0, 0)
            self.before_canvas.itemconfig(self.before_image_id, image=self.before_image_ref)
            self.after_canvas.coords(self.after_image_id, 0, 0)
            self.after_canvas.itemconfig(self.after_image_id, image=self.after_image_ref)


def main() -> None:
    root = tk.Tk()
    app = AutoCamTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
