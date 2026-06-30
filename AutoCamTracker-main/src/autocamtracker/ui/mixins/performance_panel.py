from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from autocamtracker.core.performance_evaluation import (
    ConfusionMatrixStats,
    mean_average_precision,
)


class PerformancePanelMixin:
    def open_performance_evaluation_page(self) -> None:
        if getattr(self, "performance_window", None) is not None and self.performance_window.winfo_exists():
            self.performance_window.deiconify()
            self.performance_window.lift()
            self._position_performance_window_bottom_right()
            return

        self._ensure_performance_vars()
        window = tk.Toplevel(self.root)
        self.performance_window = window
        window.title("模型效能評估")
        window.minsize(820, 560)
        window.protocol("WM_DELETE_WINDOW", self.close_performance_evaluation_page)

        outer = ttk.Frame(window, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="模型效能評估", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Close", command=self.close_performance_evaluation_page).grid(row=0, column=1)

        input_panel = ttk.LabelFrame(outer, text="Ground Truth / Confusion Matrix", padding=8)
        input_panel.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        for column in range(8):
            input_panel.columnconfigure(column, weight=1)
        entries = (
            ("TP", self.performance_tp_var),
            ("FP", self.performance_fp_var),
            ("FN", self.performance_fn_var),
            ("TN", self.performance_tn_var),
        )
        for column, (label, variable) in enumerate(entries):
            ttk.Label(input_panel, text=label).grid(row=0, column=column * 2, sticky="e", padx=(0, 4))
            entry = ttk.Entry(input_panel, textvariable=variable, width=8)
            entry.grid(row=0, column=column * 2 + 1, sticky="ew", padx=(0, 10))
            entry.bind("<KeyRelease>", lambda _event: self._refresh_performance_panel())
        ttk.Label(input_panel, text="AP values").grid(row=1, column=0, sticky="e", padx=(0, 4), pady=(8, 0))
        ap_entry = ttk.Entry(input_panel, textvariable=self.performance_ap_values_var)
        ap_entry.grid(row=1, column=1, columnspan=5, sticky="ew", pady=(8, 0), padx=(0, 10))
        ap_entry.bind("<KeyRelease>", lambda _event: self._refresh_performance_panel())
        ttk.Button(input_panel, text="Reset Runtime", command=self.reset_performance_evaluation).grid(
            row=1,
            column=6,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )

        content = ttk.Frame(outer)
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        self.performance_metric_tree = ttk.Treeview(
            content,
            columns=("description", "formula", "direction", "result", "note"),
            show="headings",
            height=9,
        )
        headings = {
            "description": "說明",
            "formula": "計算方式 / 定義",
            "direction": "越高越好？",
            "result": "結果",
            "note": "備註",
        }
        widths = {
            "description": 160,
            "formula": 190,
            "direction": 82,
            "result": 110,
            "note": 150,
        }
        for column, label in headings.items():
            self.performance_metric_tree.heading(column, text=label)
            self.performance_metric_tree.column(column, width=widths[column], minwidth=70, anchor="center")
        self.performance_metric_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        confusion = ttk.LabelFrame(right, text="Confusion Matrix", padding=8)
        confusion.grid(row=0, column=0, sticky="ew")
        self.performance_confusion_labels = {}
        matrix_items = (
            ("TP", "偵測有車 / 實際有車"),
            ("FN", "偵測無車 / 實際有車"),
            ("FP", "偵測有車 / 實際無車"),
            ("TN", "偵測無車 / 實際無車"),
        )
        for index, (key, label) in enumerate(matrix_items):
            frame = ttk.Frame(confusion, padding=5)
            frame.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
            ttk.Label(frame, text=key, font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
            value_label = ttk.Label(frame, text="0")
            value_label.grid(row=0, column=1, sticky="e")
            ttk.Label(frame, text=label, wraplength=130).grid(row=1, column=0, columnspan=2, sticky="w")
            frame.columnconfigure(1, weight=1)
            self.performance_confusion_labels[key] = value_label

        live = ttk.LabelFrame(right, text="Live Runtime Data", padding=8)
        live.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        live.columnconfigure(1, weight=1)
        self.performance_live_labels = {}
        for row, key in enumerate(
            (
                "source",
                "model",
                "tracker",
                "frames",
                "detections",
                "selected",
                "latency",
                "confidence",
            )
        ):
            ttk.Label(live, text=key.title()).grid(row=row, column=0, sticky="w", pady=2)
            value_label = ttk.Label(live, text="--", anchor="e")
            value_label.grid(row=row, column=1, sticky="ew", pady=2)
            self.performance_live_labels[key] = value_label

        self._position_performance_window_bottom_right()
        self._refresh_performance_panel()

    def close_performance_evaluation_page(self) -> None:
        window = getattr(self, "performance_window", None)
        if window is not None and window.winfo_exists():
            window.destroy()
        self.performance_window = None

    def reset_performance_evaluation(self) -> None:
        self.performance_evaluator.reset()
        self._refresh_performance_panel()

    def _ensure_performance_vars(self) -> None:
        if hasattr(self, "performance_tp_var"):
            return
        self.performance_tp_var = tk.StringVar(value="0")
        self.performance_fp_var = tk.StringVar(value="0")
        self.performance_fn_var = tk.StringVar(value="0")
        self.performance_tn_var = tk.StringVar(value="0")
        self.performance_ap_values_var = tk.StringVar(value="")

    def _position_performance_window_bottom_right(self) -> None:
        window = getattr(self, "performance_window", None)
        if window is None or not window.winfo_exists():
            return
        window.update_idletasks()
        width = max(820, window.winfo_width())
        height = max(560, window.winfo_height())
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = max(1, self.root.winfo_width())
        root_height = max(1, self.root.winfo_height())
        x = root_x + max(0, root_width - width - 18)
        y = root_y + max(0, root_height - height - 52)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _refresh_performance_panel(self) -> None:
        window = getattr(self, "performance_window", None)
        if window is None or not window.winfo_exists():
            return
        self._ensure_performance_vars()
        stats = ConfusionMatrixStats(
            true_positive=self._parse_non_negative_int(self.performance_tp_var.get()),
            false_positive=self._parse_non_negative_int(self.performance_fp_var.get()),
            false_negative=self._parse_non_negative_int(self.performance_fn_var.get()),
            true_negative=self._parse_non_negative_int(self.performance_tn_var.get()),
        )
        ap_values = self._parse_ap_values(self.performance_ap_values_var.get())
        snapshot = self.performance_evaluator.snapshot()
        rows = (
            (
                "Precision (精確率)",
                "偵測到的車輛中有多少是真的車輛",
                "TP / (TP + FP)",
                "是",
                self._format_ratio(stats.precision),
                "降低誤偵測",
            ),
            (
                "Recall (召回率)",
                "畫面中真正存在的車輛有多少被偵測到",
                "TP / (TP + FN)",
                "是",
                self._format_ratio(stats.recall),
                "降低漏偵測",
            ),
            (
                "mAP (平均精度均值)",
                "多個 Average Precision 的平均值",
                "mean(AP)",
                "是",
                self._format_ratio(mean_average_precision(ap_values)),
                "輸入 AP values 可計算",
            ),
            (
                "FPS (每秒處理幀數)",
                "每秒可處理的畫面數",
                "Frames Per Second",
                "是",
                self._format_number(snapshot.average_fps, suffix=" FPS"),
                "即時性指標",
            ),
            (
                "Tracking Stability (追蹤穩定性)",
                "目標是否容易遺失或切換",
                "locked frames / sampled frames",
                "是",
                self._format_ratio(snapshot.tracking_stability),
                f"ID switches: {snapshot.id_switches}",
            ),
        )
        tree = self.performance_metric_tree
        tree.delete(*tree.get_children())
        for name, description, formula, direction, result, note in rows:
            tree.insert("", "end", values=(f"{name} - {description}", formula, direction, result, note))

        labels = self.performance_confusion_labels
        labels["TP"].configure(text=str(stats.true_positive))
        labels["FP"].configure(text=str(stats.false_positive))
        labels["FN"].configure(text=str(stats.false_negative))
        labels["TN"].configure(text=str(stats.true_negative))

        live = self.performance_live_labels
        live["source"].configure(text=self.source_var.get())
        live["model"].configure(text=self.model_var.get())
        live["tracker"].configure(text=self.tracker_var.get())
        live["frames"].configure(text=f"{snapshot.frame_count} sampled")
        live["detections"].configure(text=f"{snapshot.detection_count} detections / {snapshot.candidate_count} candidates")
        live["selected"].configure(
            text=(
                f"GID {snapshot.selected_global_vehicle_id} / LID {snapshot.selected_local_track_id}"
                if snapshot.selected_global_vehicle_id is not None or snapshot.selected_local_track_id is not None
                else snapshot.tracking_status
            )
        )
        live["latency"].configure(
            text=(
                f"inf {self._format_number(snapshot.latest_inference_ms, suffix=' ms')} / "
                f"pipe {self._format_number(snapshot.latest_pipeline_ms, suffix=' ms')}"
            )
        )
        live["confidence"].configure(text=self._format_ratio(snapshot.latest_confidence))
        window.after(500, self._refresh_performance_panel)

    @staticmethod
    def _parse_non_negative_int(value: str) -> int:
        try:
            return max(0, int(value.strip() or "0"))
        except ValueError:
            return 0

    @staticmethod
    def _parse_ap_values(value: str) -> list[float]:
        values: list[float] = []
        for item in value.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                raw = float(item)
            except ValueError:
                continue
            values.append(raw / 100.0 if raw > 1.0 else raw)
        return values

    @staticmethod
    def _format_ratio(value: float | None) -> str:
        if value is None:
            return "--"
        return f"{value * 100.0:.1f}%"

    @staticmethod
    def _format_number(value: float | None, suffix: str = "") -> str:
        if value is None:
            return "--"
        return f"{value:.1f}{suffix}"
