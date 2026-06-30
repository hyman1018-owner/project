import XCTest
@testable import DockKitTesterCore

final class GimbalVelocityCalculatorTests: XCTestCase {
    func testManualDirectionsMatchAppleAxisMapping() {
        var calculator = GimbalVelocityCalculator()

        XCTAssertEqual(calculator.velocity(for: .panLeft), .init(yaw: -0.2, pitch: 0, roll: 0))
        XCTAssertEqual(calculator.velocity(for: .panRight), .init(yaw: 0.2, pitch: 0, roll: 0))
        XCTAssertEqual(calculator.velocity(for: .tiltUp), .init(yaw: 0, pitch: -0.2, roll: 0))
        XCTAssertEqual(calculator.velocity(for: .tiltDown), .init(yaw: 0, pitch: 0.2, roll: 0))
    }

    func testTrackingAppliesClampAndSmoothing() {
        var calculator = GimbalVelocityCalculator()
        let command = makeCommand(errorX: 4, errorY: 4)

        let velocity = calculator.velocity(for: command)

        XCTAssertEqual(velocity.yaw, 0.105, accuracy: 0.000_001)
        XCTAssertEqual(velocity.pitch, -0.066, accuracy: 0.000_001)
        XCTAssertEqual(velocity.roll, 0)
    }

    func testDeadZoneProducesZero() {
        var calculator = GimbalVelocityCalculator()

        let velocity = calculator.velocity(for: makeCommand(errorX: 0.02, errorY: -0.02))

        XCTAssertEqual(velocity, .zero)
    }

    func testLostTargetStopsAndClearsSmoothingHistory() {
        var calculator = GimbalVelocityCalculator()
        _ = calculator.velocity(for: makeCommand(errorX: 0.5, errorY: 0.2))

        let stopped = calculator.velocity(for: makeCommand(targetLocked: false, errorX: 0.5, errorY: 0.2))

        XCTAssertEqual(stopped, .zero)
        XCTAssertEqual(calculator.previous, .zero)
    }

    func testAxisInversionFlipsTrackingDirections() {
        var calculator = GimbalVelocityCalculator()
        calculator.applyCalibration(
            GimbalCalibrationProfile(yawInverted: true, pitchInverted: true)
        )

        let velocity = calculator.velocity(for: makeCommand(errorX: 0.3, errorY: -0.3))

        XCTAssertLessThan(velocity.yaw, 0)
        XCTAssertLessThan(velocity.pitch, 0)
    }

    func testCalibrationProfileAppliesSafetyLimits() {
        var calculator = GimbalVelocityCalculator()
        calculator.applyCalibration(
            GimbalCalibrationProfile(
                maxYawSpeed: 0.1,
                maxPitchSpeed: 0.08,
                deadZone: 0.12,
                minimumErrorImprovement: 0.04,
                maxNonImprovingUpdates: 5
            )
        )

        let deadZoneVelocity = calculator.velocity(for: makeCommand(errorX: 0.08, errorY: 0.08))
        let trackingVelocity = calculator.velocity(for: makeCommand(errorX: 1.0, errorY: 1.0))

        XCTAssertEqual(deadZoneVelocity, .zero)
        XCTAssertEqual(trackingVelocity.yaw, 0.03, accuracy: 0.000_001)
        XCTAssertEqual(trackingVelocity.pitch, -0.024, accuracy: 0.000_001)
        XCTAssertEqual(calculator.configuration.minimumErrorImprovement, 0.04)
        XCTAssertEqual(calculator.configuration.maxNonImprovingUpdates, 5)
    }

    func testRecoveryCalibrationValuesAreClamped() {
        let fastProfile = GimbalCalibrationProfile(
            lostAutoReturnDelay: 0.1,
            stableLockRequiredFrames: 0
        )
        let slowProfile = GimbalCalibrationProfile(
            lostAutoReturnDelay: 8.0,
            stableLockRequiredFrames: 99
        )

        XCTAssertEqual(fastProfile.clampedLostAutoReturnDelay, 0.5)
        XCTAssertEqual(fastProfile.clampedStableLockRequiredFrames, 1)
        XCTAssertEqual(slowProfile.clampedLostAutoReturnDelay, 5.0)
        XCTAssertEqual(slowProfile.clampedStableLockRequiredFrames, 20)
    }

    func testNonImprovingTrackingTriggersSafetyStop() {
        var calculator = GimbalVelocityCalculator(
            configuration: GimbalControlConfiguration(maxNonImprovingUpdates: 3)
        )

        _ = calculator.velocity(for: makeCommand(errorX: 0.5, errorY: 0))
        _ = calculator.velocity(for: makeCommand(errorX: 0.5, errorY: 0))
        _ = calculator.velocity(for: makeCommand(errorX: 0.5, errorY: 0))
        let stopped = calculator.velocity(for: makeCommand(errorX: 0.5, errorY: 0))

        XCTAssertEqual(stopped, .zero)
        XCTAssertNotNil(calculator.safetyStopReason)
    }

    func testFrameEdgeSafetyStopsOutwardChase() {
        var calculator = GimbalVelocityCalculator()

        let stopped = calculator.velocity(
            for: makeCommand(errorX: 0.8, errorY: 0, targetX: 0.98, targetY: 0.5)
        )

        XCTAssertEqual(stopped, .zero)
        XCTAssertEqual(calculator.safetyStopReason, "target near frame edge; stop before chasing out of view")
    }

    func testFrameEdgeSlowsVelocityNearBoundary() {
        var normal = GimbalVelocityCalculator()
        var nearEdge = GimbalVelocityCalculator()

        let normalVelocity = normal.velocity(for: makeCommand(errorX: 0.5, errorY: 0, targetX: 0.5, targetY: 0.5))
        let edgeVelocity = nearEdge.velocity(for: makeCommand(errorX: 0.5, errorY: 0, targetX: 0.9, targetY: 0.5))

        XCTAssertLessThan(abs(edgeVelocity.yaw), abs(normalVelocity.yaw))
    }

    func testTurnAccelerationUsesMoreResponsiveSmoothingAndFeedForward() {
        var calculator = GimbalVelocityCalculator()

        _ = calculator.velocity(for: makeCommand(errorX: 0.12, errorY: 0))
        let velocity = calculator.velocity(for: makeCommand(errorX: 0.42, errorY: 0))

        XCTAssertGreaterThan(velocity.yaw, 0.18)
    }

    func testPredictedTargetUsesMoreResponsiveSmoothing() {
        var calculator = GimbalVelocityCalculator()

        let velocity = calculator.velocity(for: makeCommand(errorX: 0.4, errorY: 0, predictedTarget: true))

        XCTAssertEqual(velocity.yaw, 0.1925, accuracy: 0.000_001)
    }

    private func makeCommand(
        targetLocked: Bool = true,
        errorX: Double,
        errorY: Double,
        targetX: Double? = nil,
        targetY: Double? = nil,
        predictedTarget: Bool? = nil
    ) -> TrackingCommand {
        TrackingCommand(
            type: "tracking",
            version: "1.3",
            targetLocked: targetLocked,
            targetId: 7,
            errorX: errorX,
            errorY: errorY,
            confidence: 0.91,
            timestampMs: 1_781_770_000_000,
            targetX: targetX,
            targetY: targetY,
            predictedTarget: predictedTarget
        )
    }
}
