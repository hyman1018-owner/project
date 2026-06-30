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
from autocamtracker.server.websocket_server import TrackingWebSocketServer
from autocamtracker.core.track_shot_plan import TrackShotController, TrackZone, should_publish_motor_tracking
from autocamtracker.tracking.vehicle_identity_store import VehicleIdentityStore

class CommandsMixin:
    def start(self) -> None:
        try:
            self.apply_ui_config()
            if self.input_config.source_type == "iphone":
                self._start_iphone_link()
            desired_signature = self._input_signature(self.input_config)
            can_resume_current_source = (
                self.detector is not None
                and not self.running
                and self.active_input_signature == desired_signature
            )
            if can_resume_current_source:
                self.running = True
                self.last_frame_time = time()
                self.tracking_worker = TrackingWorker(
                    self.detector,
                    self.pipeline,
                    self._draw_detections,
                    lambda: getattr(self, "skipped_frames", 0),
                    self._should_render_preview_frame,
                    self.tracking_server.latest_frame_timing,
                )
                self.tracking_worker.discard_results()
                self._update_transport_actions()
                self._request_worker_frame()
                return

            if self.detector is not None:
                self._close_detector()
            self._reset_runtime_state()
            frame_provider = (
                self.tracking_server.read_latest_frame
                if self.input_config.source_type == "iphone"
                else None
            )
            self.detector = VideoDetector(replace(self.input_config), frame_provider=frame_provider)
            self.detector.load_model()
            self.detector.open_source()
            self.tracking_worker = TrackingWorker(
                self.detector,
                self.pipeline,
                self._draw_detections,
                lambda: getattr(self, "skipped_frames", 0),
                self._should_render_preview_frame,
                self.tracking_server.latest_frame_timing,
            )
            self.active_input_signature = desired_signature
            self.running = True
            self.last_frame_time = time()
            self.skipped_frames = 0
            self._update_transport_actions()
            self._request_worker_frame()
        except Exception as exc:
            self.running = False
            if self.detector is not None:
                self._close_detector()
                self.detector = None
            self._update_transport_actions()
            messagebox.showerror("Start failed", str(exc))

    def pause(self) -> None:
        self.running = False
        self._disable_iphone_motor_tracking("tracking paused")
        self._update_transport_actions()

    def stop(self) -> None:
        self.running = False
        self._disable_iphone_motor_tracking("tracking stopped")
        if self.detector is not None:
            self._close_detector()
        self.detector = None
        self.active_input_signature = None
        self._reset_runtime_state()
        self._update_transport_actions()

    def reset_tracking(self) -> None:
        self._disable_iphone_motor_tracking("tracking reset")
        self._reset_runtime_state()
        self.refresh_identity_db_panel()

    def clear_selection(self) -> None:
        self._disable_iphone_motor_tracking("selection cleared")
        self.identity_manager.reset()
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        if hasattr(self, "identity_tree"):
            self.identity_tree.selection_remove(*self.identity_tree.selection())
        self.selected_identity_tree_ids.clear()
        self._set_identity_mode("click bbox to select a local track")
        self.refresh_identity_db_panel()

    def choose_video_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.source_var.set("video_file")
            self._update_source_controls()
            self.input_config.video_path = path
            self.video_path_var.set(f"Video: {self._short_label(Path(path).name)}")

    def on_source_selected(self, _event=None) -> None:
        self._disable_iphone_motor_tracking("input source changed")
        if self.detector is not None:
            self.stop()
        self._update_source_controls()
        if self.source_var.get() == "iphone":
            self._start_iphone_link()
            self.root.after_idle(self.start)

    def on_tracking_configuration_changed(self, _event=None) -> None:
        if self.detector is not None:
            self.stop()
        self.apply_ui_config()
        self.status_var.set("Status: tracking configuration changed; press Start")

    def _start_iphone_link(self) -> None:
        self._refresh_iphone_url()
        self.tracking_server.start()
        if self.tracking_server.client_count == 0:
            self.iphone_connection_var.set("iPhone link: starting…")

    def _preload_reid_model(self) -> None:
        Thread(target=self.feature_gallery.preload_embedding, name="reid-preload", daemon=True).start()

    def _enable_iphone_motor_tracking(self, action: str) -> str:
        """Arm physical gimbal output for a target-selection action."""

        if self.source_var.get() != "iphone":
            self.iphone_motor_tracking_enabled = False
            self.tracking_server.publish_stop()
            self.telemetry_logger.log(
                "motor_arm_blocked",
                action=action,
                reason="motor requires iPhone input",
                source=self.source_var.get(),
            )
            return "digital tracking only (motor requires iPhone input)"

        self._start_iphone_link()
        self.track_shot_mode_var.set("AI Tracking")
        self.track_shot_controller.set_mode("AI Tracking")
        self.track_shot_state_var.set("Shot: AI Tracking · tracking · motor armed")
        self.iphone_motor_tracking_enabled = True
        self.telemetry_logger.log(
            "motor_armed",
            action=action,
            client_count=self.tracking_server.client_count,
            motor_ready=self.tracking_server.motor_ready,
        )
        self.publish_desktop_state(force=True)
        if self.tracking_server.client_count == 0:
            return f"{action} motor armed; waiting for iPhone connection"
        if not self.tracking_server.motor_ready:
            return f"{action} motor armed; waiting for DockKit Manual Mode"
        return f"{action} motor tracking ON"

    def _disable_iphone_motor_tracking(self, reason: str) -> None:
        """Disarm motor output and immediately send an explicit safety stop."""

        was_enabled = bool(self.iphone_motor_tracking_enabled)
        self.iphone_motor_tracking_enabled = False
        self.tracking_server.publish_stop()
        self.telemetry_logger.log("motor_disarmed", reason=reason, was_enabled=was_enabled)
        if hasattr(self, "track_shot_state_var"):
            self.track_shot_state_var.set(
                f"Shot: {self.track_shot_controller.mode} · motor OFF · {reason}"
            )
        self.publish_desktop_state(force=True)

    def _refresh_iphone_url(self) -> str:
        url = self.tracking_server.preferred_url
        self.iphone_url_var.set(url)
        return url

    def copy_iphone_url(self) -> None:
        url = self._refresh_iphone_url()
        self.root.clipboard_clear()
        self.root.clipboard_append(url)
        self.root.update_idletasks()
        self.status_var.set(f"Status: copied iPhone URL: {url}")

    def _queue_iphone_status(self, message: str) -> None:
        self.iphone_status_queue.put(message)

    def _queue_iphone_control(self, payload: dict) -> None:
        self.iphone_control_queue.put(payload)

    def _drain_iphone_status(self) -> None:
        did_update = False
        try:
            while True:
                message = self.iphone_status_queue.get_nowait()
                self.iphone_connection_var.set(f"iPhone link: {message}")
                did_update = True
        except Empty:
            pass
        if did_update:
            self.publish_desktop_state(force=True)
        try:
            self.root.after(100, self._drain_iphone_status)
        except tk.TclError:
            pass

    def _drain_iphone_control(self) -> None:
        try:
            while True:
                self._handle_iphone_control(self.iphone_control_queue.get_nowait())
        except Empty:
            pass
        try:
            self.root.after(100, self._drain_iphone_control)
        except tk.TclError:
            pass

    def _handle_iphone_control(self, payload: dict) -> None:
        action = str(payload.get("action") or "").strip()
        if not action:
            return

        if action == "select_source":
            source = str(payload.get("source") or "").strip()
            self.command_select_source(source, actor="iPhone")
        elif action == "auto_track":
            self.command_auto_track(actor="iPhone", ensure_iphone_source=True, start_if_needed=True)
        elif action == "select_gid":
            gid = self._control_gid(payload)
            if gid is not None:
                self.command_select_gid(gid, actor="iPhone")
        elif action == "find_gid":
            gid = self._control_gid(payload)
            self.command_find_gid(gid, actor="iPhone")
        elif action == "set_framing":
            framing = str(payload.get("framing") or payload.get("mode") or "").strip().lower()
            self.command_set_framing(framing, actor="iPhone")
        elif action == "stop_motor":
            self.command_stop_motor(actor="iPhone", reason="iPhone STOP")
        elif action == "request_state":
            pass
        else:
            self.status_var.set(f"Status: ignored unknown iPhone control: {action}")

        self.publish_desktop_state(force=True)

    @staticmethod
    def _control_gid(payload: dict) -> int | None:
        try:
            return int(payload.get("gid"))
        except (TypeError, ValueError):
            return None

    def command_select_source(self, source: str, *, actor: str = "Desktop") -> bool:
        self.telemetry_logger.log("control_select_source", actor=actor, source=source)
        if source not in {"webcam", "video_file", "video_url", "screen_region", "iphone"}:
            self.status_var.set(f"Status: {actor} requested unsupported source: {source or '--'}")
            self.publish_desktop_state(force=True)
            return False
        self.source_var.set(source)
        self.on_source_selected()
        self.status_var.set(f"Status: {actor} selected input source: {source}")
        self.publish_desktop_state(force=True)
        return True

    def command_auto_track(
        self,
        *,
        actor: str = "Desktop",
        ensure_iphone_source: bool = False,
        start_if_needed: bool = False,
    ) -> None:
        self.telemetry_logger.log("control_auto_track", actor=actor)
        if ensure_iphone_source and self.source_var.get() != "iphone":
            self.command_select_source("iphone", actor=actor)
        if start_if_needed and not self.running:
            self.start()
        self._run_auto_track_command(actor=actor)
        self.publish_desktop_state(force=True)

    def command_select_gid(self, vehicle_id: int, *, actor: str = "Desktop") -> bool:
        self.telemetry_logger.log("control_select_gid", actor=actor, gid=vehicle_id)
        if self.identity_store.get_vehicle(vehicle_id) is None:
            self.status_var.set(f"Status: {actor} requested missing GID {vehicle_id}")
            self.publish_desktop_state(force=True)
            return False
        self.selected_identity_tree_ids = {vehicle_id}
        if hasattr(self, "identity_tree") and str(vehicle_id) in self.identity_tree.get_children():
            self.identity_tree.selection_set(str(vehicle_id))
            self.identity_tree.see(str(vehicle_id))
            self.on_identity_tree_select()
        label = self.identity_store.display_label(vehicle_id)
        self._set_identity_mode(f"Selected GID {label} from {actor}")
        self.status_var.set(f"Status: {actor} selected GID {label}")
        self.publish_desktop_state(force=True)
        return True

    def command_find_gid(self, vehicle_id: int | None = None, *, actor: str = "Desktop") -> str:
        self.telemetry_logger.log("control_find_gid", actor=actor, gid=vehicle_id)
        if vehicle_id is not None and not self.command_select_gid(vehicle_id, actor=actor):
            return "break"
        result = self._run_find_gid_command(actor=actor)
        self.telemetry_logger.log("control_find_gid_result", actor=actor, gid=vehicle_id, result=result)
        self.publish_desktop_state(force=True)
        return result

    def command_set_framing(self, mode: str, *, actor: str = "Desktop") -> bool:
        if mode not in {"wide", "medium", "close"}:
            self.telemetry_logger.log("control_set_framing_rejected", actor=actor, mode=mode)
            self.status_var.set(f"Status: {actor} requested unsupported framing: {mode or '--'}")
            self.publish_desktop_state(force=True)
            return False
        self.framing_var.set(mode)
        self.reframer.set_framing_mode(mode)
        self.telemetry_logger.log("control_set_framing", actor=actor, mode=mode)
        self.status_var.set(f"Status: {actor} set framing: {mode}")
        self.publish_desktop_state(force=True)
        return True

    def command_stop_motor(self, *, actor: str = "Desktop", reason: str | None = None) -> None:
        self.telemetry_logger.log("control_stop_motor", actor=actor, reason=reason)
        self._disable_iphone_motor_tracking(reason or f"{actor} STOP")
        self.status_var.set(f"Status: {actor} requested motor stop")
        self.publish_desktop_state(force=True)

    def send_iphone_test_pulse(self) -> None:
        self._start_iphone_link()
        if self.tracking_server.client_count == 0:
            self.status_var.set("Status: waiting for iPhone before sending test pulse")
            return

        # A short, low-speed rightward pulse followed by an explicit safety stop.
        for delay_ms in range(0, 600, 100):
            self.root.after(delay_ms, self.tracking_server.publish_test_pulse)
        self.root.after(650, self.tracking_server.publish_stop)
        self.status_var.set("Status: sending 650 ms iPhone tracking test pulse")

    def send_iphone_recenter(self) -> None:
        self._start_iphone_link()
        if self.tracking_server.client_count == 0:
            self.status_var.set("Status: waiting for iPhone before recenter")
            return
        self.tracking_server.publish_stop()
        self.tracking_server.publish_control("recenter")
        self.telemetry_logger.log("desktop_recenter_requested")
        self.status_var.set("Status: requested iPhone gimbal recenter")

    def apply_video_url(self, _event=None) -> None:
        video_url = self._normalized_video_url()
        if video_url is None:
            self.input_config.video_url = None
            self.video_url_status_var.set("No video URL selected")
            return
        self.source_var.set("video_url")
        self._update_source_controls()
        self.input_config.video_url = video_url
        self.video_url_status_var.set(f"URL: {self._short_label(video_url)}")

    def auto_select_one(self) -> None:
        self.command_auto_track(actor="Desktop")

    def _run_auto_track_command(self, *, actor: str = "Desktop") -> None:
        candidates = self.store.rank_candidates(self.last_frame_shape, strategy="stable")
        if not candidates or self.last_raw_frame is None:
            self._disable_iphone_motor_tracking("Auto Track found no vehicle")
            self.identity_manager.reset()
            self.refresh_identity_db_panel()
            self.status_var.set(f"Status: {actor} Auto Track found no visible vehicle; motor stopped")
            return
        detection = self._detection_for_track(candidates[0].track_id)
        if detection is None:
            self._disable_iphone_motor_tracking("Auto Track target disappeared")
            self.identity_manager.reset()
            self.refresh_identity_db_panel()
            self.status_var.set(f"Status: {actor} Auto Track target disappeared; motor stopped")
            return
        identity = self.identity_manager.select_detection(detection, self.last_raw_frame, persist=False)
        motor_note = self._enable_iphone_motor_tracking("Auto Track")
        self.refresh_identity_db_panel()
        self.status_var.set(
            f"Status: {actor} auto tracking local track "
            f"{identity.last_track_id if identity.last_track_id is not None else '--'} without writing Identity DB; "
            f"{motor_note}"
        )

    def on_timeline_press(self, _event) -> None:
        self.timeline_dragging = True

    def on_timeline_drag(self, value: str) -> None:
        if self.timeline_dragging:
            self._update_timeline_label(int(float(value)))

    def on_timeline_release(self, _event) -> None:
        self.timeline_dragging = False
        if not self._is_video_source_active():
            return

        was_running = self.running
        self.running = False
        if self.tracking_worker is not None:
            self.tracking_worker.discard_results()
        target_frame = int(self.timeline_var.get())
        seek = lambda: self.detector.seek_video_frame(target_frame)
        seek_succeeded = (
            self.tracking_worker.run_locked(seek)
            if self.tracking_worker is not None
            else seek()
        )
        if not seek_succeeded:
            self.running = was_running
            if was_running:
                self._request_worker_frame()
            return

        self.store.reset()
        self.identity_manager.reset()
        self.scene_cut_detector.reset()
        self.reframer.reset()
        self.identity_session_links.clear()
        self.skipped_frames = 0
        self._render_current_video_frame()
        self.running = was_running
        if was_running:
            self._request_worker_frame()

    def on_before_click(self, event) -> None:
        if self.last_frame_shape is None:
            return

        frame_height, frame_width = self.last_frame_shape[:2]
        image_width = max(1, self.rendered_image_width)
        image_height = max(1, self.rendered_image_height)
        image_x = event.x
        image_y = event.y
        if image_x < 0 or image_y < 0 or image_x > image_width or image_y > image_height:
            return
        frame_x = image_x * frame_width / image_width
        frame_y = image_y * frame_height / image_height


        candidate = self.store.get_candidate_at_point(frame_x, frame_y, self.last_frame_shape)
        if candidate is None:
            self.status_var.set("Status: no tracked vehicle at clicked point")
            self._set_identity_mode("no tracked vehicle at clicked point")
            return

        detection = self._detection_for_track(candidate.track_id)
        if detection is None or self.last_raw_frame is None:
            self.status_var.set("Status: selected candidate is no longer visible")
            return

        self.identity_manager.select_detection(detection, self.last_raw_frame, persist=False)
        self._refresh_selection_panel()
        self._redraw_current_selection()
        self._set_identity_mode("BBox selected; choose Add Vehicle or select a GID to link")
        self.status_var.set(
            f"Status: selected local track {candidate.track_id}; "
            "choose an action in Vehicle Database"
        )

    def toggle_recording(self) -> None:
        self.recording = not self.recording
        messagebox.showinfo(
            "Recording",
            "Recording scaffold toggled. VideoWriter implementation belongs here.",
        )

    def on_close(self) -> None:
        self.running = False
        self.tracking_server.publish_stop()
        self.tracking_server.stop()
        if self.identity_preview_window is not None:
            self.identity_preview_window.destroy()
            self.identity_preview_window = None
        if self.detector is not None:
            self._close_detector()
            self.detector = None
        self.feature_gallery.close()
        self.identity_store.close()
        self.root.destroy()

    @staticmethod
    def _input_signature(config: InputConfig) -> tuple[object, ...]:
        return (
            config.source_type,
            config.camera_index,
            config.video_path,
            config.video_url,
            config.screen_region,
            config.model_path,
            config.tracker_name,
            config.confidence_threshold,
            config.iou_threshold,
            config.vehicle_classes_only,
        )
