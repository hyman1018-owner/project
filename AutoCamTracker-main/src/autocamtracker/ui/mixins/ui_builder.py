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

class UIBuilderMixin:
    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("TButton", padding=(5, 2))
        style.configure("TCombobox", padding=1)
        style.configure("Treeview", rowheight=21)
        style.configure("Preview.TLabel", background="#202124", foreground="#f1f3f4")
        style.configure("PreviewTitle.TLabel", font=("TkDefaultFont", 11, "bold"))

        main = ttk.Frame(self.root, padding=6)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        source_controls = ttk.LabelFrame(controls, text="Source", padding=5)
        source_controls.grid(row=0, column=0, sticky="nsew", padx=3, pady=2)
        tracking_controls = ttk.LabelFrame(controls, text="Tracking", padding=5)
        tracking_controls.grid(row=0, column=1, sticky="nsew", padx=3, pady=2)
        playback_controls = ttk.LabelFrame(controls, text="Playback", padding=5)
        playback_controls.grid(row=0, column=2, sticky="nsew", padx=3, pady=2)
        identity_controls = ttk.LabelFrame(main, text="Vehicle Database", padding=7)
        identity_controls.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6, 0), pady=2)
        controls.columnconfigure(0, weight=1, minsize=220)
        controls.columnconfigure(1, weight=1, minsize=250)
        controls.columnconfigure(2, weight=1, minsize=180)

        self.source_var = tk.StringVar(value="iphone")
        self.performance_profile_var = tk.StringVar(value="High FPS")
        self.tracker_var = tk.StringVar(value="bytetrack")
        self.framing_var = tk.StringVar(value="medium")
        self.model_var = tk.StringVar(value=self.config.default_model)
        self.reid_model_var = tk.StringVar(value=self.config.default_reid_model)
        self.auto_reid_threshold_var = tk.StringVar(value=f"{self.identity_manager.auto_reid_min_score:.2f}")
        self.auto_feature_mode_var = tk.StringVar(value=self.auto_feature_sampler.config.mode)
        self.playback_speed_var = tk.StringVar(value="1x")
        self.camera_index_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="No video selected")
        self.video_url_var = tk.StringVar(value="")
        self.video_url_status_var = tk.StringVar(value="No video URL selected")
        self.screen_region_var = tk.StringVar(value="No screen region selected")
        self.identity_summary_var = tk.StringVar(value="Vehicles: 0 | Master: 0 | Pending: 0 | Candidate: 0")
        self.identity_mode_var = tk.StringVar(value="Click a bbox to select a local track")
        self.bbox_selection_var = tk.StringVar(value="BBox: none")
        self.db_selection_var = tk.StringVar(value="Database: no GID selected")
        self.link_state_var = tk.StringVar(value="Relation: not linked")
        self.advanced_identity_visible = tk.BooleanVar(value=False)
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_label_var = tk.StringVar(value="00:00 / 00:00")
        self.track_shot_mode_var = tk.StringVar(value=self.track_shot_controller.mode)
        self.track_shot_state_var = tk.StringVar(value="Shot: AI Tracking · tracking")
        self.in_zone_var = tk.StringVar(value=self.track_shot_controller.in_zone.text())
        self.out_zone_var = tk.StringVar(value=self.track_shot_controller.out_zone.text())

        ttk.Label(source_controls, text="Input").grid(row=0, column=0, sticky="w", padx=4)
        self.source_box = ttk.Combobox(
            source_controls,
            textvariable=self.source_var,
            values=["webcam", "video_file", "video_url", "screen_region", "iphone"],
            width=17,
            state="readonly",
        )
        self.source_box.grid(row=0, column=1, sticky="ew", padx=4)
        self.source_box.bind("<<ComboboxSelected>>", self.on_source_selected)
        self.browse_video_button = ttk.Button(source_controls, text="Browse Video", command=self.choose_video_file)
        self.browse_video_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(5, 0))
        self.screen_region_button = ttk.Button(source_controls, text="Select Screen Region", command=self.select_screen_region)
        self.screen_region_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(5, 0))

        self.url_label = ttk.Label(source_controls, text="URL")
        self.url_label.grid(row=1, column=0, sticky="w", padx=4, pady=(5, 0))
        self.url_entry = ttk.Entry(source_controls, textvariable=self.video_url_var)
        self.url_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(5, 0))
        self.url_entry.bind("<Return>", self.apply_video_url)
        self.url_entry.bind("<FocusOut>", self.apply_video_url)

        self.video_path_label = ttk.Label(source_controls, textvariable=self.video_path_var, wraplength=220)
        self.video_path_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 0))
        self.video_url_status_label = ttk.Label(source_controls, textvariable=self.video_url_status_var, wraplength=220)
        self.video_url_status_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 0))
        self.screen_region_label = ttk.Label(source_controls, textvariable=self.screen_region_var, wraplength=220)
        self.screen_region_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 0))
        self.iphone_connection_var = tk.StringVar(value="iPhone link: off")
        self.iphone_url_var = tk.StringVar(value=self.tracking_server.preferred_url)
        self.iphone_connection_label = ttk.Label(source_controls, textvariable=self.iphone_connection_var, wraplength=145)
        self.iphone_connection_label.grid(row=1, column=0, sticky="w", padx=4, pady=(4, 0))
        self.iphone_url_entry = ttk.Entry(source_controls, textvariable=self.iphone_url_var, state="readonly")
        self.iphone_url_entry.grid(row=2, column=0, sticky="ew", padx=4, pady=(4, 0))
        self.iphone_copy_button = ttk.Button(source_controls, text="Copy", width=7, command=self.copy_iphone_url)
        self.iphone_copy_button.grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
        self.iphone_test_button = ttk.Button(source_controls, text="Test", width=7, command=self.send_iphone_test_pulse)
        self.iphone_test_button.grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        self.iphone_recenter_button = ttk.Button(source_controls, text="Recenter", command=self.send_iphone_recenter)
        self.iphone_recenter_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 0))
        source_controls.columnconfigure(0, weight=1)
        source_controls.columnconfigure(1, weight=1)
        self._update_source_controls()

        ttk.Label(tracking_controls, text="Model").grid(row=0, column=0, sticky="w", padx=4)
        self.model_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.model_var,
            values=[],
            width=17,
            state="readonly",
        )
        self.model_box.grid(row=0, column=1, padx=4, sticky="ew")
        self.model_box.bind("<<ComboboxSelected>>", self.on_tracking_configuration_changed)
        ttk.Button(tracking_controls, text="Refresh", command=self.refresh_model_options).grid(row=0, column=2, sticky="ew", padx=4)

        ttk.Label(tracking_controls, text="Profile").grid(row=1, column=0, sticky="w", padx=4, pady=(6, 0))
        self.performance_profile_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.performance_profile_var,
            values=["High FPS", "Balanced ID"],
            width=13,
            state="readonly",
        )
        self.performance_profile_box.grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        self.performance_profile_box.bind("<<ComboboxSelected>>", self.apply_performance_profile)

        ttk.Label(tracking_controls, text="Tracker").grid(row=2, column=0, sticky="w", padx=4, pady=(6, 0))
        self.tracker_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.tracker_var,
            values=["bytetrack", "botsort", "deepocsort"],
            width=13,
            state="readonly",
        )
        self.tracker_box.grid(row=2, column=1, sticky="ew", padx=4, pady=(6, 0))
        self.tracker_box.bind("<<ComboboxSelected>>", self.on_tracking_configuration_changed)

        ttk.Label(tracking_controls, text="Framing").grid(row=2, column=2, sticky="w", padx=(10, 2), pady=(6, 0))
        self.framing_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.framing_var,
            values=["wide", "medium", "close"],
            width=13,
            state="readonly",
        )
        self.framing_box.grid(row=2, column=3, sticky="ew", padx=4, pady=(6, 0))
        self.framing_box.bind("<<ComboboxSelected>>", lambda _: self.apply_ui_config())
        ttk.Button(tracking_controls, text="Auto Track", command=self.auto_select_one).grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(7, 0))
        ttk.Button(tracking_controls, text="Clear", command=self.clear_selection).grid(row=3, column=2, sticky="ew", padx=4, pady=(7, 0))
        ttk.Button(tracking_controls, text="Reset", command=self.reset_tracking).grid(row=3, column=3, sticky="ew", padx=4, pady=(7, 0))
        ttk.Label(tracking_controls, text="ReID Model").grid(row=4, column=0, sticky="w", padx=4, pady=(7, 0))
        self.reid_model_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.reid_model_var,
            values=[],
            width=18,
            state="readonly",
        )
        self.reid_model_box.grid(row=4, column=1, columnspan=2, sticky="ew", padx=4, pady=(7, 0))
        self.reid_model_box.bind("<<ComboboxSelected>>", lambda _: self.apply_reid_model_config())
        ttk.Button(tracking_controls, text="Refresh ReID", command=self.refresh_reid_model_options).grid(
            row=4, column=3, sticky="ew", padx=4, pady=(7, 0)
        )
        tracking_controls.columnconfigure(1, weight=1)
        tracking_controls.columnconfigure(3, weight=1)

        shot_controls = ttk.LabelFrame(controls, text="Track Shot", padding=5)
        shot_controls.grid(row=1, column=0, columnspan=3, sticky="ew", padx=3, pady=2)
        ttk.Label(shot_controls, text="Mode").grid(row=0, column=0, sticky="w", padx=4)
        self.track_shot_mode_box = ttk.Combobox(
            shot_controls,
            textvariable=self.track_shot_mode_var,
            values=["AI Tracking", "Fixed Cut", "In/Out Auto"],
            state="readonly",
            width=14,
        )
        self.track_shot_mode_box.grid(row=0, column=1, sticky="ew", padx=4)
        self.track_shot_mode_box.bind("<<ComboboxSelected>>", self.apply_track_shot_config)
        ttk.Label(shot_controls, text="In zone").grid(row=0, column=2, sticky="w", padx=(12, 2))
        ttk.Entry(shot_controls, textvariable=self.in_zone_var, width=23).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Label(shot_controls, text="Out zone").grid(row=0, column=4, sticky="w", padx=(12, 2))
        ttk.Entry(shot_controls, textvariable=self.out_zone_var, width=23).grid(row=0, column=5, sticky="ew", padx=4)
        ttk.Button(shot_controls, text="Apply", command=self.apply_track_shot_config).grid(row=0, column=6, padx=4)
        ttk.Button(shot_controls, text="Rearm", command=self.rearm_track_shot).grid(row=0, column=7, padx=4)
        ttk.Label(shot_controls, textvariable=self.track_shot_state_var).grid(row=0, column=8, sticky="w", padx=(10, 4))
        shot_controls.columnconfigure(1, weight=1)
        shot_controls.columnconfigure(3, weight=1)
        shot_controls.columnconfigure(5, weight=1)

        self.start_button = ttk.Button(playback_controls, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.pause_button = ttk.Button(playback_controls, text="Pause", command=self.pause)
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self.stop_button = ttk.Button(playback_controls, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Record", command=self.toggle_recording).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(playback_controls, text="Speed").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Combobox(
            playback_controls,
            textvariable=self.playback_speed_var,
            values=["0.25x", "0.5x", "1x", "1.25x", "1.5x", "3x", "4x", "5x", "6x"],
            width=13,
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        playback_controls.columnconfigure(0, weight=1)
        playback_controls.columnconfigure(1, weight=1)

        header = ttk.Frame(identity_controls)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=self.identity_summary_var).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Refresh", width=8, command=self.refresh_identity_db_panel).grid(row=0, column=1, padx=(6, 0))

        selection_panel = ttk.LabelFrame(identity_controls, text="Current Selection", padding=6)
        selection_panel.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        ttk.Label(selection_panel, textvariable=self.bbox_selection_var).grid(row=0, column=0, sticky="w")
        ttk.Label(selection_panel, textvariable=self.db_selection_var).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(selection_panel, textvariable=self.link_state_var).grid(row=2, column=0, sticky="w", pady=(3, 0))

        actions = ttk.Frame(identity_controls)
        actions.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.add_vehicle_button = ttk.Button(
            actions,
            text="＋ Add Selected Vehicle",
            command=self.add_selected_vehicle,
            style="Accent.TButton",
        )
        self.add_vehicle_button.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.link_bbox_button = ttk.Button(actions, text="Link BBox → GID", command=self.link_selected_bbox)
        self.link_bbox_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.find_gid_button = ttk.Button(actions, text="Find GID", command=self.track_selected_identity_from_db)
        self.find_gid_button.grid(row=2, column=0, sticky="ew", pady=(5, 0), padx=(0, 3))
        self.delete_vehicle_button = ttk.Button(actions, text="Delete Vehicle", command=self.delete_selected_identity)
        self.delete_vehicle_button.grid(row=2, column=1, sticky="ew", pady=(5, 0), padx=(3, 0))

        self.identity_tree = ttk.Treeview(
            identity_controls,
            columns=("gid", "type", "lid", "master", "pending", "candidate", "frame", "conf"),
            show="headings",
            height=12,
        )
        headings = {
            "gid": "GID",
            "type": "Type",
            "lid": "LID",
            "master": "Master",
            "pending": "Pending",
            "candidate": "Candidate",
            "frame": "Frame",
            "conf": "DetConf",
        }
        widths = {"gid": 38, "type": 50, "lid": 38, "master": 48, "pending": 52, "candidate": 58, "frame": 48, "conf": 44}
        for column, label in headings.items():
            self.identity_tree.heading(column, text=label)
            self.identity_tree.column(column, width=widths[column], minwidth=36, anchor="center", stretch=False)
        self.identity_tree.tag_configure("selected", background="#d7ecff")
        self.identity_tree.tag_configure("tree_selected", background="#b9dcff")
        self.identity_tree.tag_configure("no_master", background="#fff4cc")
        self.identity_tree.bind("<<TreeviewSelect>>", self.on_identity_tree_select)
        self.identity_tree.bind("<Double-1>", self.edit_identity_display_name)
        self.identity_tree.bind("<Delete>", self.delete_selected_identity)
        self.identity_tree.bind("<BackSpace>", self.delete_selected_identity)
        self.identity_tree.bind("<Motion>", self.on_identity_tree_motion)
        self.identity_tree.bind("<Leave>", self.hide_identity_preview)
        self.identity_tree.grid(row=3, column=0, sticky="nsew", pady=(7, 0))

        feature_actions = ttk.LabelFrame(identity_controls, text="Features", padding=6)
        feature_actions.grid(row=4, column=0, sticky="ew", pady=(7, 0))
        feature_actions.columnconfigure(0, weight=1)
        feature_actions.columnconfigure(1, weight=1)
        self.manual_feature_button = ttk.Button(
            feature_actions,
            text="Manual Add 1 Photo",
            command=self.add_feature_to_selected_identity,
        )
        self.manual_feature_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.auto_feature_button = ttk.Button(
            feature_actions,
            text="Start Auto Add",
            command=self.toggle_auto_add_feature,
        )
        self.auto_feature_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        ttk.Checkbutton(
            identity_controls,
            text="Advanced ReID settings",
            variable=self.advanced_identity_visible,
            command=self.toggle_identity_advanced,
        ).grid(row=5, column=0, sticky="w", pady=(7, 0))

        self.identity_advanced_frame = ttk.Frame(identity_controls)
        self.identity_advanced_frame.grid(row=6, column=0, sticky="ew", pady=(3, 0))
        self.identity_advanced_frame.columnconfigure(1, weight=1)
        ttk.Label(self.identity_advanced_frame, text="Auto ReID Th").grid(row=0, column=0, sticky="w")
        threshold_entry = ttk.Entry(self.identity_advanced_frame, textvariable=self.auto_reid_threshold_var, width=7)
        threshold_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        threshold_entry.bind("<Return>", self.apply_auto_reid_threshold)
        threshold_entry.bind("<FocusOut>", self.apply_auto_reid_threshold)
        ttk.Label(self.identity_advanced_frame, text="Feature Mode").grid(row=1, column=0, sticky="w", pady=(5, 0))
        feature_mode_box = ttk.Combobox(
            self.identity_advanced_frame,
            textvariable=self.auto_feature_mode_var,
            values=["Balanced", "Diverse", "Strict"],
            width=10,
            state="readonly",
        )
        feature_mode_box.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(5, 0))
        feature_mode_box.bind("<<ComboboxSelected>>", self.apply_auto_feature_mode)
        ttk.Label(identity_controls, textvariable=self.identity_mode_var, wraplength=390).grid(
            row=7,
            column=0,
            sticky="w",
            pady=(7, 0),
        )
        identity_controls.columnconfigure(0, weight=1)
        identity_controls.rowconfigure(3, weight=1)
        self.identity_advanced_frame.grid_remove()

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=400)
        main.rowconfigure(1, weight=1)
        views = ttk.Frame(main)
        views.grid(row=1, column=0, sticky="nsew")
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(1, weight=1)
        views.bind("<Configure>", self.on_views_resize)

        ttk.Label(views, text="Before · Detection", style="PreviewTitle.TLabel").grid(row=0, column=0, pady=(2, 0))
        ttk.Label(views, text="After · Reframe", style="PreviewTitle.TLabel").grid(row=0, column=1, pady=(2, 0))

        self.before_canvas = tk.Canvas(
            views,
            background="#202124",
            borderwidth=0,
            highlightthickness=0,
            width=self.display_width,
            height=self.display_height,
        )
        self.before_canvas.grid(row=1, column=0, padx=(2, 4), pady=4, sticky="nsew")
        self.before_canvas.bind("<Button-1>", self.on_before_click)

        self.after_canvas = tk.Canvas(
            views,
            background="#202124",
            borderwidth=0,
            highlightthickness=0,
            width=self.display_width,
            height=self.display_height,
        )
        self.after_canvas.grid(row=1, column=1, padx=(4, 2), pady=4, sticky="nsew")

        timeline = ttk.Frame(views)
        timeline.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        timeline.columnconfigure(0, weight=1)
        self.timeline_scale = ttk.Scale(
            timeline,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.timeline_var,
            command=self.on_timeline_drag,
        )
        self.timeline_scale.grid(row=0, column=0, sticky="ew")
        self.timeline_scale.bind("<ButtonPress-1>", self.on_timeline_press)
        self.timeline_scale.bind("<ButtonRelease-1>", self.on_timeline_release)
        ttk.Label(timeline, textvariable=self.timeline_label_var, width=16).grid(row=0, column=1, padx=(8, 0))

        self.status_var = tk.StringVar(value="Status: idle")
        self.status_label = ttk.Label(
            main,
            textvariable=self.status_var,
            anchor="w",
            width=1,
        )
        self.status_label.grid(row=2, column=0, sticky="ew")
        self.performance_button = ttk.Button(
            main,
            text="效能評估",
            command=self.open_performance_evaluation_page,
        )
        self.performance_button.grid(row=2, column=1, sticky="e", padx=(8, 0))
        self.refresh_identity_db_panel()
        self._update_transport_actions()

    def apply_ui_config(self) -> None:
        self.input_config = self._ui_input_config()
        self.reframer.set_framing_mode(self.framing_var.get())

    def apply_performance_profile(self, _event=None) -> str:
        profile = self.performance_profile_var.get()
        if profile == "Balanced ID":
            self.model_var.set("yolo26s.pt")
            self.tracker_var.set("botsort")
        else:
            self.performance_profile_var.set("High FPS")
            self.model_var.set("yolo26n.pt")
            self.tracker_var.set("bytetrack")
        self.on_tracking_configuration_changed()
        return "break"

    def apply_track_shot_config(self, _event=None) -> str:
        try:
            in_zone = TrackZone.parse(self.in_zone_var.get())
            out_zone = TrackZone.parse(self.out_zone_var.get())
            self.track_shot_controller.configure_zones(in_zone, out_zone)
            self.track_shot_controller.set_mode(self.track_shot_mode_var.get())
        except ValueError as exc:
            messagebox.showerror("Track Shot", str(exc))
            return "break"
        self.in_zone_var.set(in_zone.text())
        self.out_zone_var.set(out_zone.text())
        self.track_shot_state_var.set(
            f"Shot: {self.track_shot_controller.mode} · {self.track_shot_controller.state}"
        )
        if self.track_shot_controller.mode == "Fixed Cut":
            self.tracking_server.publish_stop()
        return "break"

    def rearm_track_shot(self) -> None:
        self.track_shot_controller.rearm()
        self.track_shot_state_var.set(
            f"Shot: {self.track_shot_controller.mode} · {self.track_shot_controller.state}"
        )
        self.tracking_server.publish_stop()

    def apply_reid_model_config(self) -> None:
        model_path = self.reid_model_options.get(
            self.reid_model_var.get(),
            str(self.config.model_dir / self.reid_model_var.get()),
        )
        self.feature_gallery.set_reid_model(model_path)
        self._set_identity_mode(f"ReID model: {self.reid_model_var.get()}")
        self.status_var.set(f"Status: ReID model set to {self.reid_model_var.get()}")

    def apply_auto_reid_threshold(self, _event=None) -> str:
        raw_value = self.auto_reid_threshold_var.get().strip()
        try:
            threshold = float(raw_value)
        except ValueError:
            threshold = self.identity_manager.auto_reid_min_score

        threshold = max(0.0, min(1.0, threshold))
        self.identity_manager.set_auto_reid_threshold(threshold)
        self.auto_reid_threshold_var.set(f"{threshold:.2f}")
        self._set_identity_mode(f"Find GID threshold: {threshold:.2f}")
        self.status_var.set(f"Status: Auto ReID threshold set to {threshold:.2f}")
        return "break"

    def apply_auto_feature_mode(self, _event=None) -> str:
        mode = self.auto_feature_mode_var.get()
        self.auto_feature_sampler.set_mode(mode)
        self.auto_feature_mode_var.set(self.auto_feature_sampler.config.mode)
        config = self.auto_feature_sampler.config
        self._set_identity_mode(
            f"Auto Feature Mode: {config.mode} "
            f"(quality {config.min_quality_score:.2f}, area {config.min_area_ratio:.3f})"
        )
        self.status_var.set(f"Status: Auto Feature Mode set to {config.mode}")
        return "break"

    def _set_identity_mode(self, message: str) -> None:
        self.identity_mode_var.set(f"Identity Mode: {message}")

    def _ui_input_config(self) -> InputConfig:
        try:
            camera_index = int(self.camera_index_var.get())
        except ValueError:
            camera_index = 0

        high_fps = self.performance_profile_var.get() != "Balanced ID"
        return InputConfig(
            source_type=self.source_var.get(),
            camera_index=camera_index,
            video_path=self.input_config.video_path,
            video_url=self._normalized_video_url(),
            screen_region=self.input_config.screen_region,
            model_path=self.model_options.get(
                self.model_var.get(),
                self.model_var.get() or self.config.default_model,
            ),
            tracker_name=self.tracker_var.get(),
            confidence_threshold=self.input_config.confidence_threshold,
            iou_threshold=self.input_config.iou_threshold,
            vehicle_classes_only=self.input_config.vehicle_classes_only,
            target_source_fps=30.0,
            detector_imgsz=640 if high_fps else None,
            tracker_reid_enabled=not high_fps and self.tracker_var.get() == "botsort",
        )

    def _update_source_controls(self) -> None:
        if not hasattr(self, "browse_video_button"):
            return
        widgets = (
            self.browse_video_button,
            self.screen_region_button,
            self.url_label,
            self.url_entry,
            self.video_path_label,
            self.video_url_status_label,
            self.screen_region_label,
            self.iphone_connection_label,
            self.iphone_url_entry,
            self.iphone_copy_button,
            self.iphone_test_button,
            self.iphone_recenter_button,
        )
        for widget in widgets:
            widget.grid_remove()

        source = self.source_var.get()
        if source == "video_file":
            self.browse_video_button.grid()
            self.video_path_label.grid()
        elif source == "video_url":
            self.url_label.grid()
            self.url_entry.grid()
            self.video_url_status_label.grid()
        elif source == "screen_region":
            self.screen_region_button.grid()
            self.screen_region_label.grid()
        elif source == "iphone":
            self.iphone_connection_label.grid()
            self._refresh_iphone_url()
            self.iphone_url_entry.grid()
            self.iphone_copy_button.grid()
            self.iphone_test_button.grid()
            self.iphone_recenter_button.grid()

    def refresh_model_options(self) -> None:
        model_files = self._discover_model_files()
        options = {self.config.default_model: self.config.default_model}
        for path in model_files:
            options[self._model_label(path)] = str(path)
        self.model_options = options
        if hasattr(self, "model_box"):
            self.model_box.configure(values=list(self.model_options.keys()))
        if self.model_var.get() not in self.model_options:
            self.model_var.set(next(iter(self.model_options)))

    def refresh_reid_model_options(self) -> None:
        asset_names = [
            "yolo26n-reid.onnx",
            "yolo26s-reid.onnx",
            "yolo26m-reid.onnx",
            "yolo26l-reid.onnx",
            "yolo26x-reid.onnx",
        ]
        options = {name: str(self.config.model_dir / name) for name in asset_names}
        if self.config.model_dir.exists():
            for path in sorted(self.config.model_dir.rglob("*-reid.onnx")):
                options[self._model_label(path)] = str(path)
        self.reid_model_options = options
        if hasattr(self, "reid_model_box"):
            self.reid_model_box.configure(values=list(self.reid_model_options.keys()))
        if self.reid_model_var.get() not in self.reid_model_options:
            self.reid_model_var.set(self.config.default_reid_model)
        self.apply_reid_model_config()

    def select_screen_region(self) -> None:
        self.pause()
        self._clear_screen_region_selection()
        screenshot = self._capture_screen_selection_background()
        screen_width = max(1, self.root.winfo_screenwidth())
        screen_height = max(1, self.root.winfo_screenheight())
        selector = tk.Toplevel(self.root)
        selector.withdraw()
        selector.title("Select screen region")
        selector.overrideredirect(True)
        selector.geometry(f"{screen_width}x{screen_height}+0+0")
        selector.attributes("-topmost", True)

        canvas = tk.Canvas(selector, cursor="crosshair", bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        if screenshot is not None:
            selector._screen_selection_image = ImageTk.PhotoImage(screenshot)
            canvas.create_image(0, 0, anchor="nw", image=selector._screen_selection_image)
        canvas.create_text(
            30,
            30,
            anchor="nw",
            text="Drag to select screen region. Press Esc to cancel.",
            fill="white",
            font=("Arial", 24),
        )

        state: dict[str, int | None] = {"start_x": None, "start_y": None, "rect": None}

        def on_press(event) -> None:
            state["start_x"] = event.x_root
            state["start_y"] = event.y_root
            if state["rect"] is not None:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="yellow",
                width=4,
            )

        def on_drag(event) -> None:
            if state["rect"] is None or state["start_x"] is None or state["start_y"] is None:
                return
            local_start_x = state["start_x"] - selector.winfo_rootx()
            local_start_y = state["start_y"] - selector.winfo_rooty()
            canvas.coords(state["rect"], local_start_x, local_start_y, event.x, event.y)

        def on_release(event) -> None:
            if state["start_x"] is None or state["start_y"] is None:
                selector.destroy()
                return
            x1 = int(min(state["start_x"], event.x_root))
            y1 = int(min(state["start_y"], event.y_root))
            x2 = int(max(state["start_x"], event.x_root))
            y2 = int(max(state["start_y"], event.y_root))
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            self.input_config.screen_region = (x1, y1, width, height)
            self.source_var.set("screen_region")
            self._update_source_controls()
            self.screen_region_var.set(f"Screen region: x={x1}, y={y1}, w={width}, h={height}")
            selector.destroy()

        selector.bind("<Escape>", lambda _: selector.destroy())
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        selector.deiconify()
        selector.lift()
        selector.focus_force()

    def _capture_screen_selection_background(self):
        if Image is None or ImageGrab is None:
            return None

        try:
            screenshot = ImageGrab.grab()
        except Exception:
            return None

        screen_width = max(1, self.root.winfo_screenwidth())
        screen_height = max(1, self.root.winfo_screenheight())
        screenshot = screenshot.resize((screen_width, screen_height))
        overlay = Image.new("RGB", screenshot.size, (0, 0, 0))
        return Image.blend(screenshot.convert("RGB"), overlay, 0.22)

    def on_views_resize(self, event) -> None:
        width_limit = max(160, (event.width - 24) // 2)
        height_limit = max(90, event.height - 72)
        self.preview_width_limit = width_limit
        self.preview_height_limit = height_limit
        width, height = self._fit_size_to_source_aspect(width_limit, height_limit)
        if self._set_display_size(width, height) and self.current_frame_data is not None:
            self._update_images(
                self.current_frame_data.before_frame,
                self.current_frame_data.after_frame,
            )

    def toggle_identity_advanced(self) -> None:
        if self.advanced_identity_visible.get():
            self.identity_advanced_frame.grid()
        else:
            self.identity_advanced_frame.grid_remove()

    def _update_transport_actions(self) -> None:
        if not hasattr(self, "start_button"):
            return
        has_source = self.detector is not None
        if self.running:
            self.start_button.configure(text="Running")
            self._set_button_enabled(self.start_button, False)
            self._set_button_enabled(self.pause_button, True)
            self._set_button_enabled(self.stop_button, True)
        elif has_source:
            self.start_button.configure(text="Resume")
            self._set_button_enabled(self.start_button, True)
            self._set_button_enabled(self.pause_button, False)
            self._set_button_enabled(self.stop_button, True)
        else:
            self.start_button.configure(text="Start")
            self._set_button_enabled(self.start_button, True)
            self._set_button_enabled(self.pause_button, False)
            self._set_button_enabled(self.stop_button, False)

        self.source_box.configure(state="readonly")
        self.model_box.configure(state="readonly")
        self.tracker_box.configure(state="readonly")



    def _fit_size_to_source_aspect(self, width_limit: int, height_limit: int) -> tuple[int, int]:
        if self.last_frame_shape is not None:
            frame_h, frame_w = self.last_frame_shape[:2]
        else:
            frame_w, frame_h = self.config.output_width, self.config.output_height
        aspect = max(1, frame_w) / max(1, frame_h)
        width = max(160, int(width_limit))
        height = max(90, int(round(width / aspect)))
        if height > height_limit:
            height = max(90, int(height_limit))
            width = max(160, int(round(height * aspect)))
        return width, height

    @staticmethod
    def _parse_dimension(value: str, fallback: int) -> int:
        try:
            return int(value)
        except ValueError:
            return fallback

    @staticmethod
    def _format_time(seconds: float) -> str:
        total_seconds = int(round(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
