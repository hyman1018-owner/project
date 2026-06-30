import asyncio
import json
import socket
from types import SimpleNamespace
from time import monotonic, sleep
import unittest

from autocamtracker.server.websocket_server import (
    CAMERA_FRAME_ENVELOPE_MAGIC,
    TrackingServerConfig,
    TrackingWebSocketServer,
    frame_tracking_message,
    tracking_message,
)
import autocamtracker.server.websocket_server as websocket_server


class TrackingMessageTests(unittest.TestCase):
    def test_normalizes_pixel_error(self) -> None:
        frame_data = SimpleNamespace(
            selected_targets=[SimpleNamespace(
                confidence=0.91,
                status="tracking",
                lost_frame_count=0,
                center=(480.0, 90.0),
                bbox=(400.0, 50.0, 560.0, 130.0),
            )],
            tracking_status="tracking",
            framing_status=SimpleNamespace(error_x=160.0, error_y=-90.0, framing_mode="medium"),
            selected_global_vehicle_id=12,
            selected_local_track_id=7,
        )

        message = frame_tracking_message(frame_data, (360, 640, 3), sequence=42)

        self.assertTrue(message["target_locked"])
        self.assertEqual(message["target_id"], 12)
        self.assertAlmostEqual(message["error_x"], 0.5)
        self.assertAlmostEqual(message["error_y"], -0.5)
        self.assertEqual(message["sequence"], 42)
        self.assertEqual(message["frame_width"], 640)
        self.assertEqual(message["frame_height"], 360)
        self.assertAlmostEqual(message["target_x"], 0.75)
        self.assertAlmostEqual(message["bbox_width"], 0.25)
        self.assertAlmostEqual(message["zoom_factor"], 1.6)

    def test_lost_target_emits_stop(self) -> None:
        websocket_server._last_locked_zoom_factor = 1.6
        websocket_server._last_unlocked_at = None
        frame_data = SimpleNamespace(
            selected_targets=[],
            tracking_status="lost",
        )

        message = frame_tracking_message(frame_data, (360, 640, 3))

        self.assertFalse(message["target_locked"])
        self.assertEqual(message["error_x"], 0.0)
        self.assertEqual(message["error_y"], 0.0)
        self.assertAlmostEqual(message["zoom_factor"], 1.6)

    def test_stale_selected_bbox_emits_stop(self) -> None:
        websocket_server._last_locked_zoom_factor = 1.6
        websocket_server._last_unlocked_at = None
        frame_data = SimpleNamespace(
            selected_targets=[SimpleNamespace(confidence=0.91, status="lost", lost_frame_count=1)],
            tracking_status="tracking",
        )

        message = frame_tracking_message(frame_data, (360, 640, 3), sequence=43)

        self.assertFalse(message["target_locked"])
        self.assertEqual(message["sequence"], 43)
        self.assertAlmostEqual(message["zoom_factor"], 1.6)

    def test_lost_zoom_ramps_back_to_wide_after_hold(self) -> None:
        original_monotonic = websocket_server.monotonic
        websocket_server._last_locked_zoom_factor = 2.4
        websocket_server._last_unlocked_at = 10.0
        websocket_server.monotonic = lambda: 12.0
        try:
            message = frame_tracking_message(
                SimpleNamespace(selected_targets=[], tracking_status="lost"),
                (360, 640, 3),
            )
        finally:
            websocket_server.monotonic = original_monotonic

        self.assertFalse(message["target_locked"])
        self.assertAlmostEqual(message["zoom_factor"], 1.7)

    def test_wire_values_are_clamped(self) -> None:
        message = tracking_message(
            target_locked=True,
            error_x=8.0,
            error_y=-4.0,
            confidence=3.0,
        )

        self.assertEqual(message["error_x"], 1.0)
        self.assertEqual(message["error_y"], -1.0)
        self.assertEqual(message["confidence"], 1.0)
        self.assertEqual(message["source_version"], "1.75")

    def test_coasted_target_can_emit_predicted_tracking_command(self) -> None:
        frame_data = SimpleNamespace(
            selected_targets=[SimpleNamespace(
                confidence=0.62,
                status="coasting",
                lost_frame_count=2,
                center=(320.0, 180.0),
                bbox=(260.0, 140.0, 380.0, 220.0),
            )],
            tracking_status="tracking",
            framing_status=SimpleNamespace(error_x=0.0, error_y=0.0, framing_mode="medium"),
            selected_global_vehicle_id=12,
            selected_local_track_id=7,
        )

        message = frame_tracking_message(frame_data, (360, 640, 3), sequence=44)

        self.assertTrue(message["target_locked"])
        self.assertTrue(message["predicted_target"])
        self.assertAlmostEqual(message["zoom_factor"], 1.6)

    def test_extended_coasted_target_can_emit_predicted_tracking_command(self) -> None:
        frame_data = SimpleNamespace(
            selected_targets=[SimpleNamespace(
                confidence=0.31,
                status="coasting",
                lost_frame_count=12,
                center=(340.0, 180.0),
                bbox=(280.0, 140.0, 400.0, 220.0),
            )],
            tracking_status="tracking",
            framing_status=SimpleNamespace(error_x=20.0, error_y=0.0, framing_mode="medium"),
            selected_global_vehicle_id=12,
            selected_local_track_id=7,
        )

        message = frame_tracking_message(frame_data, (360, 640, 3), sequence=45)

        self.assertTrue(message["target_locked"])
        self.assertTrue(message["predicted_target"])
        self.assertEqual(message["sequence"], 45)

    def test_motor_status_reports_dockkit_readiness(self) -> None:
        server = TrackingWebSocketServer()

        server._accept_motor_status(
            json.dumps(
                {
                    "type": "motor_status",
                    "docked": True,
                    "manual_ready": True,
                    "system_tracking_enabled": False,
                    "last_error": None,
                    "timestamp_ms": 123,
                    "current_velocity": {"yaw": 0.1, "pitch": -0.2, "roll": 0.0},
                    "last_command": {"sequence": 9, "target_locked": True},
                    "last_stop_reason": None,
                    "camera_zoom_factor": 1.0,
                    "camera_display_zoom_factor": 1.2,
                }
            )
        )

        self.assertTrue(server.motor_ready)
        self.assertEqual(server.motor_status.timestamp_ms, 123)
        self.assertEqual(server.motor_status.current_velocity["yaw"], 0.1)
        self.assertEqual(server.motor_status.last_command["sequence"], 9)
        self.assertEqual(server.motor_status.camera_display_zoom_factor, 1.2)

    def test_control_message_is_routed_to_callback(self) -> None:
        received = []
        server = TrackingWebSocketServer(on_control=received.append)

        server._accept_text_message(
            json.dumps(
                {
                    "type": "control",
                    "action": "find_gid",
                    "gid": 12,
                    "timestamp_ms": 123,
                }
            )
        )

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["action"], "find_gid")
        self.assertEqual(received[0]["gid"], 12)

    def test_new_client_receives_cached_desktop_state(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        server = TrackingWebSocketServer(TrackingServerConfig(host="127.0.0.1", port=port))
        server.publish(
            {
                "type": "desktop_state",
                "version": "1.0",
                "source": "iphone",
                "running": True,
                "tracking": {"target_locked": False},
                "motor": {"armed": False, "ready": False},
                "gids": [{"gid": 12, "trackable": True}],
            }
        )
        server.start()
        deadline = monotonic() + 5.0
        while not server.is_running and monotonic() < deadline:
            sleep(0.01)
        self.assertTrue(server.is_running)
        try:
            initial, state = asyncio.run(self._receive_initial_state(server))
        finally:
            server.stop()

        self.assertEqual(initial["type"], "tracking")
        self.assertEqual(state["type"], "desktop_state")
        self.assertEqual(state["source"], "iphone")
        self.assertEqual(state["gids"][0]["gid"], 12)

    def test_server_round_trip(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        server = TrackingWebSocketServer(TrackingServerConfig(host="127.0.0.1", port=port))
        server.start()
        deadline = monotonic() + 5.0
        while not server.is_running and monotonic() < deadline:
            sleep(0.01)
        self.assertTrue(server.is_running)
        try:
            initial, pulse, camera_frame = asyncio.run(self._receive_server_messages(server))
        finally:
            server.stop()

        self.assertFalse(initial["target_locked"])
        self.assertTrue(pulse["target_locked"])
        self.assertAlmostEqual(pulse["error_x"], 0.12)
        self.assertEqual(camera_frame.shape, (24, 32, 3))

    async def _receive_server_messages(self, server: TrackingWebSocketServer):
        from websockets.asyncio.client import connect
        import cv2
        import numpy as np

        async with connect(f"ws://127.0.0.1:{server.config.port}/ws/tracking") as websocket:
            initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2.0))
            server.publish_test_pulse()
            pulse = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2.0))
            ok, jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
            self.assertTrue(ok)
            await websocket.send(jpeg.tobytes())
            camera_frame = None
            for _ in range(50):
                await asyncio.sleep(0.01)
                camera_frame = server.read_latest_frame()
                if camera_frame is not None:
                    break
            self.assertIsNotNone(camera_frame)
            return initial, pulse, camera_frame

    def test_camera_frame_envelope_records_latency_metadata(self) -> None:
        import cv2
        import numpy as np
        from time import time

        server = TrackingWebSocketServer()
        ok, jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
        self.assertTrue(ok)
        capture_timestamp_ms = int((time() - 0.05) * 1000)
        envelope = (
            CAMERA_FRAME_ENVELOPE_MAGIC
            + capture_timestamp_ms.to_bytes(8, byteorder="big", signed=False)
            + jpeg.tobytes()
        )

        server._accept_camera_frame(envelope)
        frame = server.read_latest_frame()
        timing = server.latest_frame_timing()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (24, 32, 3))
        self.assertEqual(timing["capture_timestamp_ms"], capture_timestamp_ms)
        self.assertGreaterEqual(timing["decode_time_ms"], 0.0)
        self.assertGreater(timing["receive_latency_ms"], 0.0)

    async def _receive_initial_state(self, server: TrackingWebSocketServer):
        from websockets.asyncio.client import connect

        async with connect(f"ws://127.0.0.1:{server.config.port}/ws/tracking") as websocket:
            initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2.0))
            state = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2.0))
            return initial, state


if __name__ == "__main__":
    unittest.main()
