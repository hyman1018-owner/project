from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
from time import monotonic, sleep
import unittest


V1_DIR = Path(__file__).resolve().parent
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from autocamtracker.core.desktop_state import IdentitySessionLinks
from autocamtracker.tracking.feature_gallery import DetectionFeatureMatch, FeatureGallery
from autocamtracker.tracking.identity_manager import GlobalIdentityManager
from autocamtracker.core.pipeline_worker import TrackingWorker
from autocamtracker.tracking.vehicle_identity_store import VehicleIdentityStore
from autocamtracker.vision.detector import InputConfig, TrackedDetection, VideoDetector
from autocamtracker.vision.scene_cut import SceneCutDetector


def detection(track_id: int = 12, frame_index: int = 1) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        bbox=(10.0, 20.0, 90.0, 80.0),
        class_id=2,
        class_name="car",
        confidence=0.88,
        center=(50.0, 50.0),
        frame_index=frame_index,
        timestamp=float(frame_index),
        tracker_name="botsort",
    )


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def read_and_track(self):
        sleep(0.02)
        self.calls += 1
        return f"frame-{self.calls}", [self.calls]

    def get_source_fps(self):
        return 15.0

    def reset_tracker_state(self):
        pass


class FakePipeline:
    def process(
        self,
        *,
        frame,
        detections,
        draw_detections,
        reset_tracker_state,
        inference_time_ms,
        source_fps,
        skipped_frames,
        render_preview=True,
        decode_time_ms=0.0,
        receive_latency_ms=None,
    ):
        return {
            "frame": draw_detections(frame, detections),
            "detections": detections,
            "source_fps": source_fps,
            "skipped_frames": skipped_frames,
            "inference_time_ms": inference_time_ms,
            "render_preview": render_preview,
            "decode_time_ms": decode_time_ms,
            "receive_latency_ms": receive_latency_ms,
        }


class TrackingWorkerTests(unittest.TestCase):
    def test_requested_frame_returns_from_background_worker(self) -> None:
        detector = FakeDetector()
        worker = TrackingWorker(
            detector,
            FakePipeline(),
            lambda frame, _detections: frame,
            lambda: 0,
        )
        try:
            self.assertTrue(worker.request_frame())
            self.assertFalse(worker.request_frame())
            deadline = monotonic() + 1.0
            result = None
            while result is None and monotonic() < deadline:
                result = worker.poll()
                sleep(0.005)
            self.assertIsNotNone(result)
            self.assertIsNone(result.error)
            self.assertEqual(result.raw_frame, "frame-1")
            self.assertEqual(result.frame_data["frame"], "frame-1")
            self.assertEqual(result.frame_data["detections"], [1])
            self.assertEqual(result.frame_data["source_fps"], 15.0)
            self.assertGreaterEqual(result.inference_time_ms, 0.0)
        finally:
            worker.close()


class IdentitySessionLinksTests(unittest.TestCase):
    def test_links_can_be_replaced_removed_and_cleared(self) -> None:
        links = IdentitySessionLinks()
        links.link(7, 101)
        self.assertEqual(links.vehicle_for_track(7), 101)
        links.link(7, 202)
        self.assertEqual(links.vehicle_for_track(7), 202)
        links.link(8, 202)
        links.unlink_vehicle(202)
        self.assertIsNone(links.vehicle_for_track(7))
        self.assertIsNone(links.vehicle_for_track(8))
        links.link(9, 303)
        links.clear()
        self.assertIsNone(links.vehicle_for_track(9))


class DetectorRuntimeConfigTests(unittest.TestCase):
    def test_iphone_source_uses_configured_30fps_budget_with_bytetrack(self) -> None:
        detector = VideoDetector(
            InputConfig(source_type="iphone", tracker_name="bytetrack", target_source_fps=30.0),
            frame_provider=lambda: None,
        )

        detector.open_source()

        self.assertEqual(detector.get_source_fps(), 30.0)
        self.assertEqual(detector._tracker_buffer_frames(), 150)
        self.assertIsNotNone(detector._tracker_config_path)
        config_text = detector._tracker_config_path.read_text(encoding="utf-8")
        self.assertIn("tracker_type: bytetrack", config_text)
        self.assertNotIn("with_reid", config_text)

    def test_botsort_reid_can_be_reserved_for_balanced_identity_profile(self) -> None:
        detector = VideoDetector(
            InputConfig(
                source_type="iphone",
                tracker_name="botsort",
                target_source_fps=30.0,
                tracker_reid_enabled=True,
            ),
            frame_provider=lambda: None,
        )

        detector.open_source()

        self.assertIsNotNone(detector._tracker_config_path)
        config_text = detector._tracker_config_path.read_text(encoding="utf-8")
        self.assertIn("tracker_type: botsort", config_text)
        self.assertIn("with_reid: True", config_text)


class VehicleIdentityStoreBatchingTests(unittest.TestCase):
    def test_frame_updates_are_flushed_in_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "identity.sqlite3"
            store = VehicleIdentityStore(db_path, commit_interval_seconds=60.0)
            vehicle_id = store.create_vehicle(detection(frame_index=1))
            store.update_vehicle(vehicle_id, detection(frame_index=2))

            observer = sqlite3.connect(db_path)
            try:
                before_flush = observer.execute(
                    "SELECT last_frame_index FROM vehicles WHERE id = ?", (vehicle_id,)
                ).fetchone()[0]
                self.assertEqual(before_flush, 1)

                # A pending frame update must not hold a SQLite write lock
                # against the independent feature-gallery connection.
                gallery = FeatureGallery(db_path)
                self.assertEqual(gallery.summary_by_vehicle(), {})
                gallery.close()

                store.flush()
                after_flush = observer.execute(
                    "SELECT last_frame_index FROM vehicles WHERE id = ?", (vehicle_id,)
                ).fetchone()[0]
                self.assertEqual(after_flush, 2)
            finally:
                observer.close()
                store.close()

    def test_programmatic_jpg_import_uses_full_image(self) -> None:
        import cv2
        import numpy as np

        class StaticExtractor:
            available = True

            def extract(self, _frame, _bbox):
                return [1.0, 0.0, 0.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "identity.sqlite3"
            jpg_path = Path(temp_dir) / "vehicle.jpg"
            image = np.random.default_rng(4).integers(20, 230, size=(120, 180, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(jpg_path), image))
            store = VehicleIdentityStore(db_path)
            vehicle_id = store.create_vehicle(detection())
            gallery = FeatureGallery(db_path)
            gallery.embedding_extractor = StaticExtractor()

            result = gallery.import_jpg(vehicle_id, jpg_path)

            self.assertTrue(result.accepted)
            self.assertEqual(gallery.summary_by_vehicle()[vehicle_id]["master"], 1)
            gallery.close()
            store.close()


class ReIDRuntimeOptimizationTests(unittest.TestCase):
    def test_embedding_is_reused_for_stable_local_track(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((180, 260, 3), dtype=np.uint8)
        rng = np.random.default_rng(17)
        frame[30:130, 50:170] = rng.integers(35, 225, size=(100, 120, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 30), (170, 130), (255, 255, 255), 2)

        class CountingExtractor:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, _frame, bbox):
                self.calls += 1
                vector = np.asarray([bbox[0] + bbox[2], bbox[1] + bbox[3], 1.0], dtype=np.float32)
                vector /= np.linalg.norm(vector)
                return vector.tolist()

            def extract_batch(self, frame, bboxes):
                return [self.extract(frame, bbox) for bbox in bboxes]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "identity.sqlite3"
            store = VehicleIdentityStore(db_path)
            first = detection(track_id=7, frame_index=1)
            first.bbox = (50.0, 30.0, 170.0, 130.0)
            first.center = (110.0, 80.0)
            vehicle_id = store.create_vehicle(first)
            gallery = FeatureGallery(db_path)
            extractor = CountingExtractor()
            gallery.embedding_extractor = extractor
            self.assertTrue(gallery.add_master_feature(vehicle_id, first, frame).accepted)

            for frame_index in (2, 3, 4):
                candidate = detection(track_id=11, frame_index=frame_index)
                candidate.bbox = (51.0, 30.0, 171.0, 130.0)
                candidate.center = (111.0, 80.0)
                self.assertTrue(gallery.rank_detections_for_vehicle(vehicle_id, [candidate], frame))

            self.assertEqual(extractor.calls, 2)
            gallery.close()
            store.close()

    def test_reid_search_prioritizes_predicted_track_corridor(self) -> None:
        import numpy as np

        class RecordingGallery:
            def __init__(self) -> None:
                self.seen_track_ids: list[int | None] = []

            def has_master_features(self, _vehicle_id: int) -> bool:
                return True

            def rank_detections_for_vehicle(self, _vehicle_id, detections, _frame):
                self.seen_track_ids = [item.track_id for item in detections]
                return []

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        gallery = RecordingGallery()
        manager = GlobalIdentityManager(feature_gallery=gallery)  # type: ignore[arg-type]
        initial = detection(track_id=1, frame_index=1)
        initial.center = (100.0, 180.0)
        initial.bbox = (60.0, 140.0, 140.0, 220.0)
        manager.select_detection(initial, frame, persist=True)
        moved = detection(track_id=1, frame_index=2)
        moved.center = (120.0, 180.0)
        moved.bbox = (80.0, 140.0, 160.0, 220.0)
        manager.update([moved], frame)

        nearby = detection(track_id=2, frame_index=3)
        nearby.center = (145.0, 180.0)
        far = detection(track_id=3, frame_index=3)
        far.center = (560.0, 50.0)
        manager.update([far, nearby], frame)

        self.assertEqual(gallery.seen_track_ids, [2])
        self.assertEqual(manager.selected_global_vehicle_id, 1)
        self.assertEqual(manager.selected_local_track_id, 1)

    def test_find_gid_low_score_preserves_active_local_track(self) -> None:
        import numpy as np

        class LowScoreGallery:
            def rank_detections_for_vehicle(self, _vehicle_id, detections, _frame):
                return [DetectionFeatureMatch(detection=detections[0], score=0.61, matches=[])]

        frame = np.full((180, 260, 3), 90, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "identity.sqlite3"
            store = VehicleIdentityStore(db_path)
            vehicle_id = store.create_vehicle(detection(track_id=7, frame_index=1))
            manager = GlobalIdentityManager(identity_store=store, feature_gallery=LowScoreGallery())  # type: ignore[arg-type]
            manager.link_detection(vehicle_id, detection(track_id=7, frame_index=2), frame)

            identity, score = manager.select_stored_vehicle(
                vehicle_id,
                [detection(track_id=11, frame_index=3)],
                frame,
                min_score=0.72,
            )

            self.assertIsNotNone(identity)
            self.assertEqual(score, 0.61)
            self.assertEqual(manager.status, "tracking")
            self.assertEqual(manager.selected_global_vehicle_id, vehicle_id)
            self.assertEqual(manager.selected_local_track_id, 7)
            store.close()

    def test_auto_reid_can_lock_detection_without_local_track_id(self) -> None:
        import numpy as np

        class StaticGallery:
            def has_master_features(self, _vehicle_id: int) -> bool:
                return True

            def rank_detections_for_vehicle(self, _vehicle_id, detections, _frame):
                from autocamtracker.tracking.feature_gallery import DetectionFeatureMatch

                return [DetectionFeatureMatch(detection=detections[0], score=0.91, matches=[])]

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        manager = GlobalIdentityManager(feature_gallery=StaticGallery())  # type: ignore[arg-type]
        manager.auto_reid_confirm_frames = 1
        initial = detection(track_id=4, frame_index=1)
        manager.select_detection(initial, frame, persist=True)
        manager.update([], frame)

        candidate = detection(track_id=None, frame_index=3)
        candidate.bbox = (12.0, 22.0, 92.0, 82.0)
        candidate.center = (52.0, 52.0)
        targets = manager.update([candidate], frame)

        self.assertEqual(manager.status, "tracking")
        self.assertEqual(manager.selected_global_vehicle_id, 1)
        self.assertIsNone(manager.selected_local_track_id)
        self.assertEqual(targets[0].track_id, -1)

    def test_identity_short_loss_coasts_only_when_safe(self) -> None:
        import numpy as np

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        manager = GlobalIdentityManager(predictive_coast_frames=3)
        initial = detection(track_id=1, frame_index=1)
        initial.center = (220.0, 180.0)
        initial.bbox = (180.0, 140.0, 260.0, 220.0)
        manager.select_detection(initial, frame, persist=False)
        moved = detection(track_id=1, frame_index=2)
        moved.center = (240.0, 180.0)
        moved.bbox = (200.0, 140.0, 280.0, 220.0)
        manager.update([moved], frame)

        targets = manager.update([], frame)

        self.assertEqual(manager.status, "tracking")
        self.assertEqual(targets[0].status, "coasting")
        self.assertEqual(targets[0].lost_frame_count, 1)

    def test_identity_extended_loss_coasts_with_confidence_decay(self) -> None:
        import numpy as np

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        manager = GlobalIdentityManager(coasting_min_confidence=0.24)
        initial = detection(track_id=1, frame_index=1)
        initial.center = (220.0, 180.0)
        initial.bbox = (180.0, 140.0, 260.0, 220.0)
        initial.confidence = 0.90
        manager.select_detection(initial, frame, persist=False)
        moved = detection(track_id=1, frame_index=2)
        moved.center = (230.0, 180.0)
        moved.bbox = (190.0, 140.0, 270.0, 220.0)
        moved.confidence = 0.90
        manager.update([moved], frame)

        target = None
        for _ in range(12):
            target = manager.update([], frame)[0]

        self.assertIsNotNone(target)
        self.assertEqual(manager.status, "tracking")
        self.assertEqual(target.status, "coasting")
        self.assertEqual(target.lost_frame_count, 12)
        self.assertLess(target.confidence, 0.40)
        self.assertGreaterEqual(target.confidence, 0.24)

    def test_identity_coasting_confidence_floor_is_clamped_for_ios_tracking(self) -> None:
        manager = GlobalIdentityManager(coasting_min_confidence=0.05)

        self.assertEqual(manager.coasting_min_confidence, 0.20)

    def test_identity_does_not_coast_at_frame_edge(self) -> None:
        import numpy as np

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        manager = GlobalIdentityManager(predictive_coast_frames=3)
        initial = detection(track_id=1, frame_index=1)
        initial.center = (625.0, 180.0)
        initial.bbox = (585.0, 140.0, 639.0, 220.0)
        manager.select_detection(initial, frame, persist=False)

        targets = manager.update([], frame)

        self.assertEqual(manager.status, "tracking")
        self.assertEqual(targets[0].status, "tracking")
        self.assertEqual(targets[0].lost_frame_count, 1)


class SceneCutDetectorTests(unittest.TestCase):
    def test_scene_cut_requires_consecutive_low_correlation_frames(self) -> None:
        import numpy as np

        detector = SceneCutDetector(threshold=0.99, confirm_frames=2, cooldown_frames=0)
        red = np.zeros((90, 160, 3), dtype=np.uint8)
        green = np.zeros((90, 160, 3), dtype=np.uint8)
        red[:] = (0, 0, 255)
        green[:] = (0, 255, 0)

        self.assertFalse(detector.update(red))
        self.assertFalse(detector.update(green))
        self.assertTrue(detector.update(red))

    def test_scene_cut_cooldown_suppresses_repeated_resets(self) -> None:
        import numpy as np

        detector = SceneCutDetector(threshold=0.99, confirm_frames=1, cooldown_frames=2)
        red = np.zeros((90, 160, 3), dtype=np.uint8)
        green = np.zeros((90, 160, 3), dtype=np.uint8)
        red[:] = (0, 0, 255)
        green[:] = (0, 255, 0)

        self.assertFalse(detector.update(red))
        self.assertTrue(detector.update(green))
        self.assertFalse(detector.update(red))
        self.assertFalse(detector.update(green))


if __name__ == "__main__":
    unittest.main()
