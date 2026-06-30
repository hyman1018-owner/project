from __future__ import annotations

from types import SimpleNamespace
import unittest

from autocamtracker.core.frame_data import FrameData
from autocamtracker.core.performance_evaluation import (
    ConfusionMatrixStats,
    PerformanceEvaluationTracker,
    mean_average_precision,
)


class PerformanceEvaluationTests(unittest.TestCase):
    def test_confusion_matrix_calculates_precision_recall_and_accuracy(self) -> None:
        stats = ConfusionMatrixStats(true_positive=8, false_positive=2, false_negative=4, true_negative=6)

        self.assertAlmostEqual(stats.precision, 0.8)
        self.assertAlmostEqual(stats.recall, 8 / 12)
        self.assertAlmostEqual(stats.accuracy, 0.7)

    def test_confusion_matrix_returns_none_when_denominator_is_missing(self) -> None:
        stats = ConfusionMatrixStats()

        self.assertIsNone(stats.precision)
        self.assertIsNone(stats.recall)
        self.assertIsNone(stats.accuracy)

    def test_mean_average_precision_clamps_samples(self) -> None:
        self.assertAlmostEqual(mean_average_precision([0.8, 1.2, -0.5]), 0.6)
        self.assertIsNone(mean_average_precision([]))

    def test_tracker_records_runtime_snapshot_and_id_switches(self) -> None:
        tracker = PerformanceEvaluationTracker(window_size=4)
        tracker.record_frame(self._frame(24.0, selected_lid=7, confidence=0.9))
        tracker.record_frame(self._frame(30.0, selected_lid=7, confidence=0.8))
        snapshot = tracker.record_frame(self._frame(18.0, selected_lid=9, confidence=0.7, locked=False))

        self.assertEqual(snapshot.frame_count, 3)
        self.assertAlmostEqual(snapshot.average_fps, 24.0)
        self.assertAlmostEqual(snapshot.average_confidence, 0.8)
        self.assertAlmostEqual(snapshot.tracking_stability, 2 / 3)
        self.assertEqual(snapshot.id_switches, 1)
        self.assertEqual(snapshot.detection_count, 2)
        self.assertEqual(snapshot.candidate_count, 1)

    @staticmethod
    def _frame(
        fps: float,
        *,
        selected_lid: int,
        confidence: float,
        locked: bool = True,
    ) -> FrameData:
        target = SimpleNamespace(
            confidence=confidence,
            lost_frame_count=0 if locked else 2,
            status="tracking" if locked else "lost",
        )
        return FrameData(
            raw_frame=None,
            before_frame=None,
            after_frame=None,
            detections=[SimpleNamespace(), SimpleNamespace()],
            candidates=[SimpleNamespace()],
            selected_targets=[target],
            framing_status=SimpleNamespace(crop_window=(0, 0, 1, 1), error_x=0.0, error_y=0.0),
            tracking_status="tracking",
            selected_local_track_id=selected_lid,
            display_fps=fps,
            source_fps=30.0,
            inference_time_ms=12.0,
            pipeline_time_ms=16.0,
            skipped_frames=1,
        )


if __name__ == "__main__":
    unittest.main()
