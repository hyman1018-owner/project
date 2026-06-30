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

class IdentityPanelMixin:
    def refresh_identity_db_panel(self, force: bool = True) -> None:
        if not hasattr(self, "identity_tree"):
            return
        now = time()
        if not force and now - self.last_identity_panel_refresh_at < 0.5:
            self._refresh_selection_panel()
            return
        self.last_identity_panel_refresh_at = now
        self.refreshing_identity_panel = True
        selected_tree_ids = set(self.selected_identity_tree_ids)
        summary = self.identity_store.summary(feature_counts=self.feature_gallery.summary_by_vehicle())
        self.identity_summary_var.set(
            "Vehicles: "
            f"{summary.vehicle_count} | Master: {summary.master_feature_count} | "
            f"Pending: {summary.pending_feature_count} | Candidate: {summary.candidate_feature_count}"
        )
        for item in self.identity_tree.get_children():
            self.identity_tree.delete(item)

        selected_gid = self.identity_manager.selected_global_vehicle_id
        for vehicle in summary.vehicles:
            tags: tuple[str, ...] = ()
            if vehicle.vehicle_id in selected_tree_ids:
                tags = ("tree_selected",)
            elif vehicle.vehicle_id == selected_gid:
                tags = ("selected",)
            elif vehicle.master_feature_count <= 0:
                tags = ("no_master",)
            self.identity_tree.insert(
                "",
                "end",
                iid=str(vehicle.vehicle_id),
                values=(
                    vehicle.display_name,
                    vehicle.class_name,
                    vehicle.last_track_id if vehicle.last_track_id is not None else "--",
                    vehicle.master_feature_count,
                    vehicle.pending_feature_count,
                    vehicle.candidate_feature_count,
                    vehicle.last_frame_index,
                    f"{vehicle.confidence:.2f}",
                ),
                tags=tags,
            )
        existing_items = set(self.identity_tree.get_children())
        restore_selection = [str(vehicle_id) for vehicle_id in selected_tree_ids if str(vehicle_id) in existing_items]
        if restore_selection:
            self.identity_tree.selection_set(restore_selection)
        self.refreshing_identity_panel = False
        self._refresh_selection_panel()

    def on_identity_tree_select(self, _event=None) -> str:
        if self.refreshing_identity_panel:
            return "break"
        self.selected_identity_tree_ids = set(self._selected_identity_vehicle_ids())
        if self.selected_identity_tree_ids:
            vehicle_id = sorted(self.selected_identity_tree_ids)[0]
            label = self.identity_store.display_label(vehicle_id)
            self._set_identity_mode(f"Selected GID {label}; choose Link, Find, or Feature action")
        else:
            self._set_identity_mode("Click a bbox or select a database vehicle")
        selected_gid = self.identity_manager.selected_global_vehicle_id
        for item in self.identity_tree.get_children():
            try:
                vehicle_id = int(item)
                master_count = int(self.identity_tree.set(item, "master") or 0)
            except ValueError:
                continue

            tags: tuple[str, ...] = ()
            if vehicle_id in self.selected_identity_tree_ids:
                tags = ("tree_selected",)
            elif vehicle_id == selected_gid:
                tags = ("selected",)
            elif master_count <= 0:
                tags = ("no_master",)
            self.identity_tree.item(item, tags=tags)
        self._refresh_selection_panel()
        return "break"

    def on_identity_tree_motion(self, event) -> None:
        item = self.identity_tree.identify_row(event.y)
        if not item:
            self.hide_identity_preview()
            return
        try:
            vehicle_id = int(item)
        except ValueError:
            self.hide_identity_preview()
            return
        self.show_identity_preview(vehicle_id, event.x_root + 18, event.y_root + 12)

    def show_identity_preview(self, vehicle_id: int, screen_x: int, screen_y: int) -> None:
        if Image is None or ImageTk is None:
            return
        if self.identity_preview_vehicle_id == vehicle_id and self.identity_preview_window is not None:
            self.identity_preview_window.geometry(f"+{screen_x}+{screen_y}")
            return

        crop_jpeg = self.feature_gallery.first_feature_crop_jpeg(vehicle_id)
        if crop_jpeg is None:
            self.hide_identity_preview()
            return

        try:
            image = Image.open(BytesIO(crop_jpeg)).convert("RGB")
        except Exception:
            self.hide_identity_preview()
            return

        image.thumbnail((220, 150))
        photo = ImageTk.PhotoImage(image)
        if self.identity_preview_window is None:
            self.identity_preview_window = tk.Toplevel(self.root)
            self.identity_preview_window.overrideredirect(True)
            self.identity_preview_window.attributes("-topmost", True)
            self.identity_preview_label = ttk.Label(self.identity_preview_window, padding=4)
            self.identity_preview_label.pack()

        assert self.identity_preview_label is not None
        self.identity_preview_photo = photo
        self.identity_preview_vehicle_id = vehicle_id
        self.identity_preview_label.configure(image=photo)
        self.identity_preview_window.geometry(f"+{screen_x}+{screen_y}")
        self.identity_preview_window.deiconify()

    def hide_identity_preview(self, _event=None) -> None:
        self.identity_preview_vehicle_id = None
        if self.identity_preview_window is not None:
            self.identity_preview_window.withdraw()

    def track_selected_identity_from_db(self, _event=None) -> str:
        return self.command_find_gid(actor="Desktop")

    def _run_find_gid_command(self, *, actor: str = "Desktop") -> str:
        if self.refreshing_identity_panel:
            return "break"

        vehicle_ids = self._selected_identity_vehicle_ids()
        if not vehicle_ids:
            self._disable_iphone_motor_tracking("Find GID has no selected identity")
            self._set_identity_mode("select a GID row before Find GID")
            self.status_var.set(f"Status: {actor} Find GID needs a selected GID")
            return "break"
        vehicle_id = vehicle_ids[0]

        if self.last_raw_frame is None:
            self._disable_iphone_motor_tracking("Find GID is waiting for video")
            self._set_identity_mode("Find GID waiting for current frame")
            self.status_var.set(f"Status: no current frame available for {actor} DB identity tracking")
            return "break"

        identity, score = self.identity_manager.select_stored_vehicle(
            vehicle_id,
            self.store.current_detections,
            self.last_raw_frame,
            min_score=self.identity_manager.auto_reid_min_score,
        )
        self.refresh_identity_db_panel()
        if identity is None:
            self._disable_iphone_motor_tracking("Find GID failed")
            self.telemetry_logger.log(
                "find_gid_result",
                actor=actor,
                gid=vehicle_id,
                found=False,
                score=score,
                motor_armed=False,
            )
            self._set_identity_mode(f"GID {vehicle_id} was not found")
            self.status_var.set(f"Status: {actor} vehicle id {vehicle_id} was not found in Identity DB")
            return "break"

        label = self.identity_store.display_label(vehicle_id)
        if identity.last_track_id is None:
            self._disable_iphone_motor_tracking("Find GID target unavailable")
            self.telemetry_logger.log(
                "find_gid_result",
                actor=actor,
                gid=vehicle_id,
                found=True,
                local_track_id=None,
                score=score,
                motor_armed=False,
                reason="target unavailable",
            )
            self._set_identity_mode(f"Find GID searching for GID {label}")
            self.status_var.set(
                f"Status: {actor} no Master feature match for GID {label}; searching "
                f"(score {score:.2f}); motor stopped until target is visible"
            )
        else:
            self.identity_session_links.link(identity.last_track_id, vehicle_id)
            motor_note = self._enable_iphone_motor_tracking("Find GID")
            self.telemetry_logger.log(
                "find_gid_result",
                actor=actor,
                gid=vehicle_id,
                found=True,
                local_track_id=identity.last_track_id,
                score=score,
                motor_armed=True,
            )
            self._set_identity_mode(f"Find GID tracking GID {label}")
            self.status_var.set(
                f"Status: {actor} tracking GID {label} on local track {identity.last_track_id} "
                f"(score {score:.2f}); {motor_note}"
            )
            if self.current_frame_data is not None:
                self._update_images(
                    self._draw_detections(self.last_raw_frame, self.store.current_detections),
                    self.current_frame_data.after_frame,
                )
        return "break"

    def link_selected_bbox(self) -> str:
        vehicle_ids = self._selected_identity_vehicle_ids()
        if not vehicle_ids:
            self.status_var.set("Status: select a GID before Link BBox")
            return "break"
        detection = self._current_visible_detection()
        if detection is None or self.last_raw_frame is None:
            self.status_var.set("Status: click a visible bbox before Link BBox")
            return "break"
        vehicle_id = vehicle_ids[0]
        previous_vehicle_id = self.identity_session_links.vehicle_for_track(detection.track_id)
        if previous_vehicle_id is not None and previous_vehicle_id != vehicle_id:
            previous_label = self.identity_store.display_label(previous_vehicle_id)
            next_label = self.identity_store.display_label(vehicle_id)
            if not messagebox.askyesno(
                "Relink Vehicle",
                f"LID {detection.track_id} is linked to GID {previous_label}. "
                f"Move it to GID {next_label}?",
            ):
                return "break"
            self.identity_store.clear_track_link(previous_vehicle_id, detection.track_id)
        identity = self.identity_manager.link_detection(vehicle_id, detection, self.last_raw_frame)
        if identity is None:
            self.status_var.set(f"Status: vehicle id {vehicle_id} no longer exists")
            return "break"
        self.identity_session_links.link(detection.track_id, vehicle_id)
        label = self.identity_store.display_label(vehicle_id)
        self.refresh_identity_db_panel()
        self._redraw_current_selection()
        self._set_identity_mode(f"Linked LID {detection.track_id} to GID {label}")
        self.status_var.set(f"Status: linked local track {detection.track_id} to GID {label}")
        return "break"

    def add_selected_vehicle(self) -> str:
        detection = self._current_visible_detection()
        if detection is None or self.last_raw_frame is None:
            self.status_var.set("Status: click a visible bbox before Add Vehicle")
            return "break"
        existing_vehicle_id = self.identity_session_links.vehicle_for_track(detection.track_id)
        if existing_vehicle_id is not None:
            label = self.identity_store.display_label(existing_vehicle_id)
            self.selected_identity_tree_ids = {existing_vehicle_id}
            self.refresh_identity_db_panel()
            self._set_identity_mode(f"LID {detection.track_id} already belongs to GID {label}")
            self.status_var.set(
                f"Status: duplicate prevented; LID {detection.track_id} is already GID {label}"
            )
            return "break"
        vehicle_id = self.identity_store.create_vehicle(detection, {"created_manually": True})
        self.identity_manager.link_detection(vehicle_id, detection, self.last_raw_frame)
        self.identity_session_links.link(detection.track_id, vehicle_id)
        self.selected_identity_tree_ids = {vehicle_id}
        self.refresh_identity_db_panel()
        self._redraw_current_selection()
        self._set_identity_mode(f"Added GID {vehicle_id}; add one photo or start Auto Add")
        self.status_var.set(
            f"Status: added local track {detection.track_id} as GID {vehicle_id}"
        )
        return "break"

    def toggle_auto_add_feature(self) -> str:
        if self.auto_feature_sampler.active_vehicle_id is not None:
            vehicle_id = self.auto_feature_sampler.active_vehicle_id
            self.auto_feature_sampler.stop()
            self.auto_feature_status_message = ""
            label = self.identity_store.display_label(vehicle_id)
            self._set_identity_mode(f"Auto Add stopped for GID {label}")
            self.status_var.set(f"Status: stopped automatic feature capture for GID {label}")
            self._refresh_selection_panel()
            return "break"
        return self.start_auto_add_feature()

    def start_auto_add_feature(self) -> str:
        vehicle_ids = self._selected_identity_vehicle_ids()
        vehicle_id = vehicle_ids[0] if vehicle_ids else self.identity_manager.selected_global_vehicle_id
        if vehicle_id is None:
            self._set_identity_mode("select a GID before Auto Add Feature")
            self.status_var.set("Status: select a GID before Auto Add Feature")
            return "break"
        if self.last_raw_frame is None:
            self._set_identity_mode("Auto Add Feature waiting for current frame")
            self.status_var.set("Status: no current frame available for Auto Add Feature")
            return "break"

        detection = self._detection_for_vehicle_id(vehicle_id)
        if detection is None:
            self._set_identity_mode("Auto Add Feature needs visible linked bbox")
            self.status_var.set("Status: Link BBox to a visible vehicle before Auto Add Feature")
            return "break"

        result = self._activate_auto_feature_capture(vehicle_id, detection, self.last_raw_frame)
        self.refresh_identity_db_panel()
        label = self.identity_store.display_label(vehicle_id)
        if result.accepted:
            self._set_identity_mode(f"Auto Add Feature active for GID {label}")
            self.status_var.set(
                f"Status: auto feature capture active for GID {label}; "
                f"added master feature {result.feature_id} (quality {result.quality_score:.2f})"
            )
        else:
            self._set_identity_mode(f"Auto Add Feature waiting for clean GID {label} crop")
            self.status_var.set(
                f"Status: auto feature capture active for GID {label}; first sample rejected: {result.reason}"
            )
        return "break"

    def _activate_auto_feature_capture(self, vehicle_id: int, detection, frame):
        result = self.auto_feature_sampler.start(vehicle_id, detection, frame, self.store)
        label = self.identity_store.display_label(vehicle_id)
        if result.accepted:
            self.auto_feature_status_message = f"GID {label} added {result.feature_id}"
        else:
            self.auto_feature_status_message = f"GID {label} active; waiting for clean crop"
        return result

    def _run_auto_feature_sampling(self, frame) -> None:
        vehicle_id = self.auto_feature_sampler.active_vehicle_id
        if vehicle_id is None:
            return
        detection = self._detection_for_vehicle_id(vehicle_id)
        result = self.auto_feature_sampler.update(detection, frame, self.store)
        if not result.accepted:
            return
        label = self.identity_store.display_label(vehicle_id)
        self.auto_feature_status_message = f"GID {label} added {result.feature_id} q{result.quality_score:.2f}"
        self._set_identity_mode(f"Auto Add Feature added feature {result.feature_id} to GID {label}")

    def _stop_auto_feature_capture_for_scene_change(self) -> None:
        vehicle_id = self.auto_feature_sampler.active_vehicle_id
        if vehicle_id is None:
            return
        label = self.identity_store.display_label(vehicle_id)
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        self._set_identity_mode(f"camera changed; Auto Add Feature stopped for GID {label}")
        self._refresh_selection_panel()

    def add_feature_to_selected_identity(self) -> str:
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        self._refresh_selection_panel()
        vehicle_ids = self._selected_identity_vehicle_ids()
        vehicle_id = vehicle_ids[0] if vehicle_ids else self.identity_manager.selected_global_vehicle_id
        if vehicle_id is None:
            self._set_identity_mode("Manual Add stopped Auto Add Feature; select a GID")
            self.status_var.set("Status: select a GID before Add Feature")
            return "break"
        if self.last_raw_frame is None:
            self._set_identity_mode("Manual Add stopped Auto Add Feature; waiting for current frame")
            self.status_var.set("Status: no current frame available for Add Feature")
            return "break"

        detection = self._detection_for_vehicle_id(vehicle_id)
        if detection is None:
            self._set_identity_mode("Manual Add stopped Auto Add Feature; visible linked bbox required")
            self.status_var.set("Status: Link BBox to a visible vehicle before Add Feature")
            return "break"

        result = self.feature_gallery.add_master_feature(vehicle_id, detection, self.last_raw_frame)
        self.refresh_identity_db_panel()
        label = self.identity_store.display_label(vehicle_id)
        if result.accepted:
            self._set_identity_mode(f"Manual Add added one feature {result.feature_id} to GID {label}")
            self.status_var.set(
                f"Status: added master feature {result.feature_id} to GID {label} "
                f"(quality {result.quality.score:.2f})"
            )
            return "break"
        duplicate = (
            f", duplicate {result.duplicate_score:.3f}"
            if result.duplicate_score is not None
            else ""
        )
        self.status_var.set(
            f"Status: rejected Add Feature for GID {label}: {result.reason}{duplicate}"
        )
        self._set_identity_mode(f"Manual Add did not add a feature to GID {label}")
        return "break"

    def edit_identity_display_name(self, event) -> str:
        if self.identity_tree.identify_column(event.x) != "#1":
            return "break"

        item = self.identity_tree.identify_row(event.y)
        if not item:
            return "break"
        try:
            vehicle_id = int(item)
        except ValueError:
            return "break"

        current_name = self.identity_tree.set(item, "gid")
        new_name = simpledialog.askstring(
            "Edit GID",
            "Vehicle ID label:",
            initialvalue=current_name,
            parent=self.root,
        )
        if new_name is None:
            return "break"

        if self.identity_store.update_display_name(vehicle_id, new_name):
            self.refresh_identity_db_panel()
            label = self.identity_store.display_label(vehicle_id)
            self._set_identity_mode(f"renamed GID {vehicle_id} to {label}")
            self.status_var.set(f"Status: renamed vehicle id {vehicle_id} to {label}")
        return "break"

    def delete_selected_identity(self, _event=None) -> str:
        if not hasattr(self, "identity_tree"):
            return "break"

        vehicle_ids = self._selected_identity_vehicle_ids()
        if not vehicle_ids:
            self._set_identity_mode("select an Identity DB row before deleting")
            self.status_var.set("Status: select an Identity DB row before deleting")
            return "break"

        labels = ", ".join(self.identity_store.display_label(vehicle_id) for vehicle_id in vehicle_ids)
        if not messagebox.askyesno(
            "Delete Vehicle",
            f"Delete GID {labels} and all saved features? This cannot be undone.",
        ):
            return "break"

        deleted_ids: list[int] = []
        for vehicle_id in vehicle_ids:
            if self.identity_store.delete_vehicle(vehicle_id):
                self.feature_gallery.delete_vehicle_features(vehicle_id)
                self.identity_session_links.unlink_vehicle(vehicle_id)
                deleted_ids.append(vehicle_id)

        if self.identity_manager.selected_global_vehicle_id in deleted_ids:
            self.identity_manager.reset()
        if self.auto_feature_sampler.active_vehicle_id in deleted_ids:
            self.auto_feature_sampler.stop()
            self.auto_feature_status_message = ""
        self.selected_identity_tree_ids.difference_update(deleted_ids)

        self.refresh_identity_db_panel()
        if deleted_ids:
            ids = ", ".join(str(vehicle_id) for vehicle_id in deleted_ids)
            self._set_identity_mode(f"deleted GID {ids}")
            self.status_var.set(f"Status: deleted vehicle id {ids}")
        else:
            self._set_identity_mode("selected GID was already deleted")
            self.status_var.set("Status: selected vehicle id was already deleted")
        return "break"

    def _selected_identity_vehicle_ids(self) -> list[int]:
        if not hasattr(self, "identity_tree"):
            return []
        vehicle_ids: list[int] = []
        for item in self.identity_tree.selection():
            try:
                vehicle_ids.append(int(item))
            except ValueError:
                continue
        return vehicle_ids

    def _current_visible_detection(self):
        track_id = self.identity_manager.selected_local_track_id
        if track_id is None:
            return None
        return self._detection_for_track(track_id)

    def _redraw_current_selection(self) -> None:
        if self.last_raw_frame is None or self.current_frame_data is None:
            return
        self._update_images(
            self._draw_detections(self.last_raw_frame, self.store.current_detections),
            self.current_frame_data.after_frame,
        )

    def _refresh_selection_panel(self) -> None:
        if not hasattr(self, "bbox_selection_var") or not hasattr(self, "add_vehicle_button"):
            return

        detection = self._current_visible_detection()
        if detection is None:
            self.bbox_selection_var.set("BBox: none — click a visible detection")
        else:
            self.bbox_selection_var.set(
                f"BBox: LID {detection.track_id} · {detection.class_name} · {detection.confidence:.0%}"
            )

        vehicle_ids = self._selected_identity_vehicle_ids()
        vehicle_id = vehicle_ids[0] if vehicle_ids else None
        if vehicle_id is None:
            self.db_selection_var.set("Database: no GID selected")
        else:
            self.db_selection_var.set(f"Database: GID {self.identity_store.display_label(vehicle_id)}")

        linked_vehicle_id = (
            self.identity_session_links.vehicle_for_track(detection.track_id)
            if detection is not None
            else None
        )
        is_selected_link = linked_vehicle_id is not None and linked_vehicle_id == vehicle_id
        if linked_vehicle_id is not None:
            linked_label = self.identity_store.display_label(linked_vehicle_id)
            if is_selected_link:
                self.link_state_var.set(f"Relation: linked to GID {linked_label}")
            elif vehicle_id is not None:
                self.link_state_var.set(
                    f"Relation: GID {linked_label}; Link will ask before moving"
                )
            else:
                self.link_state_var.set(f"Relation: linked to GID {linked_label}")
        elif detection is not None and vehicle_id is not None:
            self.link_state_var.set("Relation: ready to link")
        else:
            self.link_state_var.set("Relation: select both BBox and GID")

        self._set_button_enabled(
            self.add_vehicle_button,
            detection is not None and linked_vehicle_id is None,
        )
        self._set_button_enabled(
            self.link_bbox_button,
            detection is not None and vehicle_id is not None and not is_selected_link,
        )
        self._set_button_enabled(self.find_gid_button, vehicle_id is not None)
        self._set_button_enabled(self.delete_vehicle_button, vehicle_id is not None)

        feature_detection = self._detection_for_vehicle_id(vehicle_id) if vehicle_id is not None else None
        feature_ready = vehicle_id is not None and feature_detection is not None
        self._set_button_enabled(self.manual_feature_button, feature_ready)
        auto_active = self.auto_feature_sampler.active_vehicle_id is not None
        self.auto_feature_button.configure(text="Stop Auto Add" if auto_active else "Start Auto Add")
        self._set_button_enabled(self.auto_feature_button, auto_active or feature_ready)

    @staticmethod
    def _set_button_enabled(button, enabled: bool) -> None:
        button.state(["!disabled"] if enabled else ["disabled"])
