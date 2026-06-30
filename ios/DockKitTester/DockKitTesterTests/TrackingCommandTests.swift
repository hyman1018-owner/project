import Foundation
import XCTest
@testable import DockKitTesterCore

final class TrackingCommandTests: XCTestCase {
    func testDecodesV175SnakeCasePayload() throws {
        let json = #"{"type":"tracking","version":"1.0","source_version":"1.75","sequence":42,"target_locked":true,"target_id":7,"error_x":0.18,"error_y":-0.04,"confidence":0.91,"timestamp_ms":1781770000000,"zoom_factor":2.4,"predicted_target":true}"#

        let command = try JSONDecoder().decode(TrackingCommand.self, from: Data(json.utf8))

        XCTAssertEqual(command.type, "tracking")
        XCTAssertEqual(command.version, "1.0")
        XCTAssertEqual(command.sourceVersion, "1.75")
        XCTAssertEqual(command.sequence, 42)
        XCTAssertTrue(command.targetLocked)
        XCTAssertEqual(command.targetId, 7)
        XCTAssertEqual(command.errorX, 0.18)
        XCTAssertEqual(command.errorY, -0.04)
        XCTAssertEqual(command.confidence, 0.91)
        XCTAssertEqual(command.timestampMs, 1_781_770_000_000)
        XCTAssertEqual(command.zoomFactor, 2.4)
        XCTAssertEqual(command.predictedTarget, true)
    }

    func testSafeDecoderReturnsFailureForMissingFields() {
        let json = Data(#"{"type":"tracking"}"#.utf8)

        let result = JSONDecoder().decodeSafely(TrackingCommand.self, from: json)

        if case .success = result {
            XCTFail("Expected malformed command to fail decoding")
        }
    }

    func testClamp() {
        XCTAssertEqual(clamp(2.0, min: -1.0, max: 1.0), 1.0)
        XCTAssertEqual(clamp(-2.0, min: -1.0, max: 1.0), -1.0)
        XCTAssertEqual(clamp(0.2, min: -1.0, max: 1.0), 0.2)
    }

    func testTrackingRequiresConfidenceAndFiniteErrors() {
        XCTAssertTrue(makeCommand(sequence: 1, confidence: 0.8).isTrackable())
        XCTAssertTrue(makeCommand(sequence: 2, confidence: 0.2).isTrackable())
        XCTAssertFalse(makeCommand(sequence: 2, confidence: 0.19).isTrackable())
        XCTAssertFalse(makeCommand(sequence: 3, confidence: 0.8, errorX: .infinity).isTrackable())
    }

    func testMotorStatusUsesSnakeCaseWireKeys() throws {
        let message = MotorStatusMessage(
            docked: true,
            manualReady: true,
            systemTrackingEnabled: false,
            lastError: nil,
            timestampMs: 123,
            currentVelocity: GimbalVelocity(yaw: 0.1, pitch: -0.2, roll: 0),
            lastCommand: makeCommand(sequence: 9, confidence: 0.8),
            lastStopReason: nil,
            cameraZoomFactor: 1.0,
            cameraDisplayZoomFactor: 1.2
        )

        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(message)) as? [String: Any]

        XCTAssertEqual(object?["type"] as? String, "motor_status")
        XCTAssertEqual(object?["manual_ready"] as? Bool, true)
        XCTAssertEqual(object?["system_tracking_enabled"] as? Bool, false)
        XCTAssertEqual(object?["timestamp_ms"] as? Int, 123)
        XCTAssertNotNil(object?["current_velocity"] as? [String: Any])
        XCTAssertNotNil(object?["last_command"] as? [String: Any])
        XCTAssertEqual(object?["camera_zoom_factor"] as? Double, 1.0)
        XCTAssertEqual(object?["camera_display_zoom_factor"] as? Double, 1.2)
    }

    func testDecodesDesktopStatePayload() throws {
        let json = """
        {
          "type": "desktop_state",
          "version": "1.0",
          "source_version": "1.75",
          "timestamp_ms": 1781770000000,
          "source": "iphone",
          "running": true,
          "tracking": {
            "status": "tracking",
            "target_locked": true,
            "target_id": 12,
            "selected_gid": 12,
            "selected_lid": 7,
            "error_x": 0.2,
            "error_y": -0.1,
            "confidence": 0.92
          },
          "motor": {
            "armed": true,
            "ready": false,
            "client_count": 1,
            "docked": true,
            "manual_ready": true,
            "system_tracking_enabled": true,
            "last_error": null
          },
          "framing": {
            "mode": "medium",
            "crop_window": [10, 20, 320, 180],
            "error_x": 12.5,
            "error_y": -8.0,
            "zoom_factor": 2.4
          },
          "gids": [
            {
              "gid": 12,
              "display_name": "GID 12",
              "class_name": "car",
              "last_track_id": 7,
              "last_frame_index": 40,
              "confidence": 0.88,
              "master_feature_count": 3,
              "pending_feature_count": 0,
              "candidate_feature_count": 1,
              "trackable": true,
              "visible": true,
              "selected": true
            }
          ]
        }
        """

        let state = try JSONDecoder().decode(DesktopState.self, from: Data(json.utf8))

        XCTAssertEqual(state.type, "desktop_state")
        XCTAssertEqual(state.source, "iphone")
        XCTAssertTrue(state.running)
        XCTAssertEqual(state.tracking.selectedGid, 12)
        XCTAssertTrue(state.motor.armed)
        XCTAssertFalse(state.motor.ready)
        XCTAssertEqual(state.framing?.mode, "medium")
        XCTAssertEqual(state.framing?.zoomFactor, 2.4)
        XCTAssertEqual(state.gids.first?.gid, 12)
        XCTAssertTrue(state.gids.first?.trackable == true)
    }

    func testSequenceValidatorRejectsDuplicateAndOutOfOrderCommands() {
        var validator = TrackingCommandSequenceValidator()

        XCTAssertTrue(validator.accept(makeCommand(sequence: 10)))
        XCTAssertFalse(validator.accept(makeCommand(sequence: 10)))
        XCTAssertFalse(validator.accept(makeCommand(sequence: 9)))
        XCTAssertTrue(validator.accept(makeCommand(sequence: 11)))
        validator.reset()
        XCTAssertTrue(validator.accept(makeCommand(sequence: 1)))
    }

    private func makeCommand(
        sequence: Int64,
        confidence: Double = 0.9,
        errorX: Double = 0.1
    ) -> TrackingCommand {
        TrackingCommand(
            type: "tracking",
            sequence: sequence,
            targetLocked: true,
            targetId: 7,
            errorX: errorX,
            errorY: 0,
            confidence: confidence
        )
    }
}
