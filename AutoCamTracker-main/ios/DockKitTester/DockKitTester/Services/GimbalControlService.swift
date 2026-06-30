import Foundation
import Combine

@MainActor
final class GimbalControlService: ObservableObject {
    @Published private(set) var currentVelocity = GimbalVelocity.zero
    @Published private(set) var calibration = GimbalCalibrationProfile.conservative
    @Published private(set) var lastStopReason: String?
    @Published var lostAutoReturnEnabled = false
    @Published private(set) var homeSet = false
    @Published private(set) var autoReturnPaused = false

    private let dockKitManager: DockKitMotorControlling
    private let logger: AppLogger
    private var calculator = GimbalVelocityCalculator()
    private var commandGeneration = 0
    private var lostStartedAt: Date?
    private var stableLockCount = 0
    private var autoReturnInFlight = false
    private let calibrationKey = "AutoCamTrackerGimbalCalibrationV175"

    init(dockKitManager: DockKitMotorControlling, logger: AppLogger) {
        self.dockKitManager = dockKitManager
        self.logger = logger
        calibration = Self.loadCalibration(key: calibrationKey)
        calculator.applyCalibration(calibration)
    }

    func setYawInverted(_ inverted: Bool) {
        updateCalibration { $0.yawInverted = inverted }
        logger.log(.info, "Tracking yaw direction \(inverted ? "inverted" : "normal").")
    }

    func setPitchInverted(_ inverted: Bool) {
        updateCalibration { $0.pitchInverted = inverted }
        logger.log(.info, "Tracking pitch direction \(inverted ? "inverted" : "normal").")
    }

    func setMaxYawSpeed(_ value: Double) {
        updateCalibration { $0.maxYawSpeed = value }
    }

    func setMaxPitchSpeed(_ value: Double) {
        updateCalibration { $0.maxPitchSpeed = value }
    }

    func setDeadZone(_ value: Double) {
        updateCalibration { $0.deadZone = value }
    }

    func setMinimumErrorImprovement(_ value: Double) {
        updateCalibration { $0.minimumErrorImprovement = value }
    }

    func setMaxNonImprovingUpdates(_ value: Double) {
        updateCalibration { $0.maxNonImprovingUpdates = Int(value.rounded()) }
    }

    func setLostAutoReturnDelay(_ value: Double) {
        updateCalibration { $0.lostAutoReturnDelay = value }
    }

    func setStableLockRequiredFrames(_ value: Double) {
        updateCalibration { $0.stableLockRequiredFrames = Int(value.rounded()) }
    }

    func resetCalibration() {
        calibration = .conservative
        saveCalibration()
        calculator.applyCalibration(calibration)
        logger.log(.info, "Tracking calibration reset to conservative defaults.")
    }

    func setHome() async {
        commandGeneration += 1
        calculator.reset()
        lostStartedAt = nil
        stableLockCount = 0
        autoReturnInFlight = false
        autoReturnPaused = false
        currentVelocity = .zero
        await dockKitManager.stop()
        await dockKitManager.setHome()
        homeSet = true
        lastStopReason = "Home overwritten at current position"
        logger.log(.success, "Set Home / 設定初始位置 overwritten at current position.")
    }

    func execute(_ command: GimbalCommand) async {
        switch command {
        case .stop:
            await emergencyStop(reason: "manual Stop")
        case .recenter:
            commandGeneration += 1
            calculator.reset()
            currentVelocity = .zero
            logger.log(.info, "Recenter requested.")
            await dockKitManager.recenter()
        default:
            commandGeneration += 1
            let generation = commandGeneration
            let velocity = calculator.velocity(for: command)
            currentVelocity = velocity
            await dockKitManager.setAngularVelocity(
                yaw: velocity.yaw,
                pitch: velocity.pitch,
                roll: velocity.roll
            )
            if generation != commandGeneration {
                await dockKitManager.stop()
            }
        }
    }

    func apply(_ trackingCommand: TrackingCommand) async {
        guard trackingCommand.type == "tracking" else {
            logger.log(.error, "Ignored V1.75 message with unsupported type: \(trackingCommand.type).")
            await emergencyStop(reason: "invalid V1.75 message")
            return
        }

        guard trackingCommand.isTrackable() else {
            calculator.reset()
            stableLockCount = 0
            if lostAutoReturnEnabled && !trackingCommand.targetLocked {
                await handleLostAutoReturnTick()
                return
            }
            lostStartedAt = nil
            if currentVelocity != .zero {
                await emergencyStop(reason: "target unavailable or confidence below safety threshold")
            } else {
                lastStopReason = "target unavailable or confidence below safety threshold"
            }
            return
        }

        lostStartedAt = nil
        if autoReturnPaused {
            stableLockCount += 1
            currentVelocity = .zero
            lastStopReason = "waiting for stable Find GID lock after auto return"
            if stableLockCount < calibration.clampedStableLockRequiredFrames {
                await dockKitManager.stop()
                return
            }
            autoReturnPaused = false
            stableLockCount = 0
            logger.log(.success, "Stable Find GID lock restored; motor tracking resumed.")
        }

        commandGeneration += 1
        let generation = commandGeneration
        let velocity = calculator.velocity(for: trackingCommand)
        currentVelocity = velocity
        lastStopReason = nil
        if let reason = calculator.safetyStopReason {
            await emergencyStop(reason: reason)
            return
        }

        logger.log(
            .info,
            String(
                format: "Tracking velocity: seq=%lld target=%d error=(%.3f, %.3f) confidence=%.2f velocity=(yaw %.3f, pitch %.3f, roll %.3f).",
                trackingCommand.sequence ?? -1,
                trackingCommand.targetId ?? -1,
                trackingCommand.errorX,
                trackingCommand.errorY,
                trackingCommand.confidence,
                velocity.yaw,
                velocity.pitch,
                velocity.roll
            )
        )

        await dockKitManager.setAngularVelocity(
            yaw: velocity.yaw,
            pitch: velocity.pitch,
            roll: velocity.roll
        )
        if generation != commandGeneration {
            await dockKitManager.stop()
        }
    }

    func testAngularVelocity() async {
        commandGeneration += 1
        let generation = commandGeneration
        let velocity = GimbalVelocity(yaw: 0.15, pitch: 0, roll: 0)
        currentVelocity = velocity
        logger.log(.info, "Angular velocity test: yaw +0.15 rad/s for 350 ms.")
        await dockKitManager.setAngularVelocity(yaw: velocity.yaw, pitch: 0, roll: 0)
        try? await Task.sleep(for: .milliseconds(350))
        if generation == commandGeneration {
            await emergencyStop(reason: "angular velocity test completed")
        }
    }

    func emergencyStop(reason: String) async {
        commandGeneration += 1
        calculator.reset()
        stableLockCount = 0
        currentVelocity = .zero
        lastStopReason = reason
        logger.log(.warning, "Safety stop: \(reason).")
        await dockKitManager.stop()
    }

    private func handleLostAutoReturnTick() async {
        if lostStartedAt == nil {
            lostStartedAt = Date()
            if currentVelocity != .zero {
                await emergencyStop(reason: "target lost; waiting before auto return")
            } else {
                lastStopReason = "target lost; waiting before auto return"
            }
            return
        }
        guard let lostStartedAt,
              Date().timeIntervalSince(lostStartedAt) >= calibration.clampedLostAutoReturnDelay,
              !autoReturnInFlight,
              !autoReturnPaused else {
            return
        }

        autoReturnInFlight = true
        defer { autoReturnInFlight = false }
        commandGeneration += 1
        calculator.reset()
        currentVelocity = .zero
        stableLockCount = 0
        lastStopReason = "lost auto return: STOP and return Home"
        logger.log(
            .warning,
            String(format: "Lost Auto Return triggered after %.1fs unlocked.", calibration.clampedLostAutoReturnDelay)
        )
        await dockKitManager.stop()
        await dockKitManager.returnHome()
        autoReturnPaused = true
        logger.log(.info, "Motor tracking paused after Home; waiting for stable Find GID lock.")
    }

    private func updateCalibration(_ mutate: (inout GimbalCalibrationProfile) -> Void) {
        mutate(&calibration)
        calibration = calibration
        saveCalibration()
        calculator.applyCalibration(calibration)
    }

    private func saveCalibration() {
        guard let data = try? JSONEncoder().encode(calibration) else { return }
        UserDefaults.standard.set(data, forKey: calibrationKey)
    }

    private static func loadCalibration(key: String) -> GimbalCalibrationProfile {
        guard let data = UserDefaults.standard.data(forKey: key),
              let profile = try? JSONDecoder().decode(GimbalCalibrationProfile.self, from: data) else {
            return .conservative
        }
        return profile
    }
}
