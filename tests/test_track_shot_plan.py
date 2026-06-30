from types import SimpleNamespace
import unittest

from autocamtracker.core.track_shot_plan import (
    TrackShotController,
    TrackShotDecision,
    TrackZone,
    should_publish_motor_tracking,
)


def frame_data(center=(100.0, 180.0), fresh=True):
    target = SimpleNamespace(
        center=center,
        status="tracking" if fresh else "lost",
        lost_frame_count=0 if fresh else 1,
    )
    return SimpleNamespace(tracking_status="tracking", selected_targets=[target])


class TrackShotControllerTests(unittest.TestCase):
    frame_shape = (360, 640, 3)

    def test_fixed_cut_never_publishes_motor_tracking(self) -> None:
        controller = TrackShotController(mode="Fixed Cut")
        decision = controller.evaluate(frame_data(), self.frame_shape)
        self.assertFalse(decision.publish_tracking)
        self.assertEqual(decision.state, "fixed_cut")

    def test_motor_output_requires_armed_iphone_source(self) -> None:
        decision = TrackShotDecision(True, "tracking", "target locked")

        self.assertTrue(should_publish_motor_tracking("iphone", True, True, decision))
        self.assertFalse(should_publish_motor_tracking("iphone", False, True, decision))
        self.assertFalse(should_publish_motor_tracking("iphone", True, False, decision))
        self.assertFalse(should_publish_motor_tracking("webcam", True, True, decision))
        self.assertFalse(
            should_publish_motor_tracking(
                "iphone",
                True,
                True,
                TrackShotDecision(False, "fixed_cut", "motor blocked"),
            )
        )

    def test_ai_tracking_requires_fresh_visual_target(self) -> None:
        controller = TrackShotController(mode="AI Tracking")
        self.assertTrue(controller.evaluate(frame_data(), self.frame_shape).publish_tracking)
        self.assertFalse(controller.evaluate(frame_data(fresh=False), self.frame_shape).publish_tracking)

    def test_in_out_mode_arms_activates_and_completes(self) -> None:
        controller = TrackShotController(
            mode="In/Out Auto",
            in_zone=TrackZone(0.0, 0.0, 0.25, 1.0),
            out_zone=TrackZone(0.75, 0.0, 1.0, 1.0),
        )
        self.assertFalse(controller.evaluate(frame_data(center=(300.0, 180.0)), self.frame_shape).publish_tracking)
        self.assertTrue(controller.evaluate(frame_data(center=(100.0, 180.0)), self.frame_shape).publish_tracking)
        self.assertTrue(controller.evaluate(frame_data(center=(350.0, 180.0)), self.frame_shape).publish_tracking)
        completed = controller.evaluate(frame_data(center=(550.0, 180.0)), self.frame_shape)
        self.assertFalse(completed.publish_tracking)
        self.assertEqual(completed.state, "complete")

    def test_zone_text_round_trip_and_validation(self) -> None:
        zone = TrackZone.parse("0.10, 0.20, 0.30, 0.90")
        self.assertEqual(TrackZone.parse(zone.text()), zone)
        with self.assertRaises(ValueError):
            TrackZone.parse("0.8,0,0.2,1")


if __name__ == "__main__":
    unittest.main()
