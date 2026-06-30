import Foundation
import Combine
#if canImport(DockKit)
import DockKit
import Spatial
#endif

@MainActor
protocol DockKitMotorControlling: AnyObject {
    func setAngularVelocity(yaw: Double, pitch: Double, roll: Double) async
    func stop() async
    func recenter() async
    func setHome() async
    func returnHome() async
}

@MainActor
final class DockKitManager: ObservableObject, DockKitMotorControlling {
    @Published private(set) var accessoryStatus: AccessoryStatus = .notFound
    @Published private(set) var accessoryName: String?
    @Published private(set) var isSystemTrackingEnabled: Bool?
    @Published private(set) var trackingButtonEnabled: Bool?
    @Published private(set) var isManualModeTransitioning = false
    @Published private(set) var isCapabilityTestRunning = false
    @Published private(set) var lastError: String?
    @Published private(set) var hasHomePosition = false

    private let logger: AppLogger
    private var listeningTask: Task<Void, Never>?

#if !targetEnvironment(simulator)
    private var accessory: DockAccessory?
    private var motorControlMode: MotorControlMode = .unknown
    private var activeOrientationProgress: Progress?
    private var orientationCommandInFlight = false
    private var currentOffset = Vector3D()
    private var homeOffset: Vector3D?
    private var lastVelocityUpdateAt: Date?
#endif

    init(logger: AppLogger) {
        self.logger = logger
    }

    deinit {
        listeningTask?.cancel()
    }

    var isDocked: Bool {
        accessoryStatus == .docked
    }

    var isManualControlReady: Bool {
        isDocked && isSystemTrackingEnabled == false && !isManualModeTransitioning
    }

    func startListening() async {
        guard listeningTask == nil else { return }

#if targetEnvironment(simulator)
        accessoryStatus = .notFound
        logger.log(.warning, "DockKit requires an iPhone and is unavailable in Simulator.")
#else
        accessoryStatus = .connecting
        logger.log(.info, "Starting DockAccessoryManager.accessoryStateChanges listener.")
        listeningTask = Task { @MainActor [weak self] in
            do {
                for await stateChange in try DockAccessoryManager.shared.accessoryStateChanges {
                    guard !Task.isCancelled else { return }
                    self?.handle(stateChange)
                }
            } catch is CancellationError {
                return
            } catch {
                self?.accessoryStatus = .error
                self?.recordError(api: "accessoryStateChanges", error: error)
            }
        }
#endif
    }

    func enableSystemTracking() async {
#if targetEnvironment(simulator)
        logSimulatorFailure(api: "setSystemTrackingEnabled(true)")
#else
        guard accessory != nil else {
            logMissingAccessory(api: "setSystemTrackingEnabled(true)")
            return
        }
        guard !isManualModeTransitioning else {
            logger.log(.warning, "System Tracking change ignored while another mode transition is running.")
            return
        }

        isManualModeTransitioning = true
        defer { isManualModeTransitioning = false }

        do {
            if isSystemTrackingEnabled == false, let accessory {
                try await accessory.setAngularVelocity(Vector3D())
                logger.log(.info, "Manual motor output stopped before restoring System Tracking.")
            }
            try await DockAccessoryManager.shared.setSystemTrackingEnabled(true)
            guard await waitForSystemTracking(expected: true) else {
                throw ManualModeError.trackingStateDidNotChange(expected: true)
            }
            lastError = nil
            logger.log(.success, "setSystemTrackingEnabled(true) succeeded.")
        } catch {
            recordError(api: "setSystemTrackingEnabled(true)", error: error)
        }
#endif
    }

    func disableSystemTracking() async {
        _ = await enterManualMode()
    }

    @discardableResult
    func enterManualMode() async -> Bool {
#if targetEnvironment(simulator)
        logSimulatorFailure(api: "setSystemTrackingEnabled(false)")
        return false
#else
        guard accessory != nil else {
            logMissingAccessory(api: "setSystemTrackingEnabled(false)")
            return false
        }
        if isSystemTrackingEnabled == false {
            logger.log(.info, "Manual Mode is already active; System Tracking is OFF.")
            return true
        }
        guard !isManualModeTransitioning else {
            logger.log(.warning, "Manual Mode request ignored while another mode transition is running.")
            return false
        }

        isManualModeTransitioning = true
        defer { isManualModeTransitioning = false }

        do {
            logger.log(.info, "Entering Manual Mode: requesting System Tracking OFF.")
            try await DockAccessoryManager.shared.setSystemTrackingEnabled(false)
            guard await waitForSystemTracking(expected: false) else {
                throw ManualModeError.trackingStateDidNotChange(expected: false)
            }
            lastError = nil
            logger.log(.success, "Manual Mode ready: System Tracking is confirmed OFF.")
            return true
        } catch {
            recordError(api: "setSystemTrackingEnabled(false)", error: error)
            return false
        }
#endif
    }

    func setAngularVelocity(yaw: Double, pitch: Double, roll: Double) async {
#if targetEnvironment(simulator)
        logSimulatorFailure(api: "setAngularVelocity")
#else
        guard let accessory else {
            logMissingAccessory(api: "setAngularVelocity")
            return
        }
        guard isManualControlReady else {
            let message = "setAngularVelocity blocked: enter Manual Mode and confirm Tracking OFF first."
            lastError = message
            logger.log(.error, message)
            return
        }

        if motorControlMode == .relativeOrientation {
            await applyRelativeOrientationFallback(
                accessory: accessory,
                yaw: yaw,
                pitch: pitch,
                roll: roll
            )
            return
        }

        let velocity = Vector3D(x: pitch, y: yaw, z: roll)
        integrateAngularVelocity(yaw: yaw, pitch: pitch, roll: roll)
        do {
            try await accessory.setAngularVelocity(velocity)
            motorControlMode = .angularVelocity
            lastError = nil
            logger.log(
                .success,
                String(format: "setAngularVelocity(pitch: %.3f, yaw: %.3f, roll: %.3f) succeeded.", pitch, yaw, roll)
            )
        } catch DockKitError.notSupportedByDevice {
            if motorControlMode != .relativeOrientation {
                logger.log(.warning, "Angular velocity is unsupported; switching to relative orientation fallback.")
            }
            motorControlMode = .relativeOrientation
            await applyRelativeOrientationFallback(
                accessory: accessory,
                yaw: yaw,
                pitch: pitch,
                roll: roll
            )
        } catch {
            recordError(
                api: String(format: "setAngularVelocity(pitch: %.3f, yaw: %.3f, roll: %.3f)", pitch, yaw, roll),
                error: error
            )
        }
#endif
    }

    func stop() async {
#if targetEnvironment(simulator)
        logSimulatorFailure(api: "stop / setAngularVelocity(0, 0, 0)")
#else
        guard let accessory else {
            logger.log(.warning, "Stop requested with no DockKit accessory connected.")
            return
        }
        guard isSystemTrackingEnabled == false else {
            logger.log(.info, "STOP skipped: System Tracking owns the motors and no manual velocity is active.")
            return
        }
        if motorControlMode == .relativeOrientation {
            activeOrientationProgress?.cancel()
            activeOrientationProgress = nil
            orientationCommandInFlight = false
            lastVelocityUpdateAt = nil
            lastError = nil
            logger.log(.success, "STOP succeeded: relative orientation command cancelled.")
            return
        }
        integrateAngularVelocity(yaw: 0, pitch: 0, roll: 0)
        lastVelocityUpdateAt = nil
        do {
            try await accessory.setAngularVelocity(Vector3D())
            lastError = nil
            logger.log(.success, "STOP succeeded: all angular velocities are zero.")
        } catch {
            recordError(api: "stop / setAngularVelocity(0, 0, 0)", error: error)
        }
#endif
    }

    func recenter() async {
#if targetEnvironment(simulator)
        logSimulatorFailure(api: "setOrientation(origin)")
#else
        guard let accessory else {
            logMissingAccessory(api: "setOrientation(origin)")
            return
        }
        guard isSystemTrackingEnabled == false else {
            let message = "Recenter blocked: disable system tracking first."
            lastError = message
            logger.log(.error, message)
            return
        }
        do {
            _ = try await accessory.setOrientation(Vector3D(), duration: .seconds(1), relative: false)
            lastError = nil
            logger.log(.success, "setOrientation(origin, 1s, absolute) started.")
        } catch {
            recordError(api: "setOrientation(origin, 1s, absolute)", error: error)
            logger.log(.warning, "Recenter failed; applying STOP fallback.")
            await stop()
        }
#endif
    }

    func setHome() async {
#if targetEnvironment(simulator)
        hasHomePosition = true
        logger.log(.success, "Home position overwritten at simulated current offset.")
#else
        activeOrientationProgress?.cancel()
        activeOrientationProgress = nil
        orientationCommandInFlight = false
        lastVelocityUpdateAt = nil
        homeOffset = currentOffset
        hasHomePosition = true
        logger.log(
            .success,
            String(
                format: "Home overwritten at current relative offset pitch=%.3f yaw=%.3f roll=%.3f.",
                currentOffset.x,
                currentOffset.y,
                currentOffset.z
            )
        )
#endif
    }

    func returnHome() async {
        if !hasHomePosition {
            await setHome()
        }
#if targetEnvironment(simulator)
        logger.log(.info, "Returning gimbal to simulated Home.")
#else
        guard let accessory else {
            logMissingAccessory(api: "return Home")
            return
        }
        guard isSystemTrackingEnabled == false else {
            let message = "Return Home blocked: disable system tracking first."
            lastError = message
            logger.log(.error, message)
            return
        }
        let home = homeOffset ?? Vector3D()
        let delta = Vector3D(
            x: home.x - currentOffset.x,
            y: home.y - currentOffset.y,
            z: home.z - currentOffset.z
        )
        logger.log(
            .info,
            String(format: "Returning to Home via relative delta pitch=%.3f yaw=%.3f roll=%.3f.", delta.x, delta.y, delta.z)
        )
        do {
            activeOrientationProgress?.cancel()
            activeOrientationProgress = try await accessory.setOrientation(delta, duration: .seconds(1), relative: true)
            await waitForProgress(activeOrientationProgress!, timeout: .seconds(2))
            currentOffset = home
            lastVelocityUpdateAt = nil
            lastError = nil
            logger.log(.success, "Return Home completed.")
        } catch {
            recordError(api: "return Home relative orientation", error: error)
            await stop()
        }
#endif
    }

    func runCapabilityDiagnostics() async {
#if targetEnvironment(simulator)
        logSimulatorFailure(api: "DockKit capability diagnostics")
#else
        guard let accessory else {
            logMissingAccessory(api: "DockKit capability diagnostics")
            return
        }
        guard await enterManualMode() else { return }
        guard !isCapabilityTestRunning else {
            logger.log(.warning, "Capability diagnostics are already running.")
            return
        }

        isCapabilityTestRunning = true
        defer { isCapabilityTestRunning = false }
        logger.log(.info, "Capability diagnostics started; keep the gimbal area clear.")

        do {
            let limits = try accessory.limits
            logger.log(.success, "Capability limits succeeded: \(String(reflecting: limits)).")
        } catch {
            recordError(api: "capability limits", error: error)
        }

        do {
            let progress = try await accessory.animate(motion: .yes)
            await waitForProgress(progress, timeout: .seconds(3))
            logger.log(.success, "Capability animate(.yes) succeeded.")
        } catch {
            recordError(api: "capability animate(.yes)", error: error)
        }

        do {
            let progress = try await accessory.setOrientation(
                Vector3D(x: 0, y: 0.08, z: 0),
                duration: .milliseconds(300),
                relative: true
            )
            await waitForProgress(progress, timeout: .seconds(2))
            logger.log(.success, "Capability relative orientation succeeded.")
        } catch {
            recordError(api: "capability relative orientation", error: error)
        }

        do {
            try await accessory.setAngularVelocity(Vector3D(x: 0, y: 0.10, z: 0))
            try await Task.sleep(for: .milliseconds(200))
            try await accessory.setAngularVelocity(Vector3D())
            logger.log(.success, "Capability angular velocity succeeded.")
        } catch {
            recordError(api: "capability angular velocity", error: error)
            try? await accessory.setAngularVelocity(Vector3D())
        }

        logger.log(.info, "Capability diagnostics finished; Manual Mode remains active.")
#endif
    }

#if !targetEnvironment(simulator)
    private func handle(_ stateChange: DockAccessory.StateChange) {
        let previousTracking = isSystemTrackingEnabled
        trackingButtonEnabled = stateChange.trackingButtonEnabled
        let currentTracking = DockAccessoryManager.shared.isSystemTrackingEnabled
        logger.log(
            .info,
            "DockKit state change: state=\(String(describing: stateChange.state)), accessoryPresent=\(stateChange.accessory != nil), trackingButton=\(stateChange.trackingButtonEnabled), systemTracking=\(currentTracking)."
        )
        if stateChange.state == .docked, let newAccessory = stateChange.accessory {
            accessory = newAccessory
            motorControlMode = .unknown
            activeOrientationProgress = nil
            orientationCommandInFlight = false
            currentOffset = Vector3D()
            homeOffset = nil
            lastVelocityUpdateAt = nil
            hasHomePosition = false
            accessoryStatus = .docked
            let model = newAccessory.hardwareModel ?? "DockKit Accessory"
            accessoryName = "\(model) • \(String(describing: newAccessory.identifier))"
            isSystemTrackingEnabled = currentTracking
            lastError = nil
            logger.log(
                .success,
                "Accessory docked: \(accessoryName ?? model); firmware: \(newAccessory.firmwareVersion ?? "unknown")."
            )
            if currentTracking, !isManualModeTransitioning {
                logger.log(.info, "AutoCamTracker disables iPhone System Tracking so only computer tracking is used.")
                Task { @MainActor [weak self] in
                    _ = await self?.enterManualMode()
                }
            }
            if previousTracking == false, currentTracking {
                logger.log(.warning, "Physical tracking button restored System Tracking; turning it OFF again.")
            }
        } else {
            accessory = nil
            motorControlMode = .unknown
            activeOrientationProgress = nil
            orientationCommandInFlight = false
            currentOffset = Vector3D()
            homeOffset = nil
            lastVelocityUpdateAt = nil
            hasHomePosition = false
            accessoryStatus = .notFound
            accessoryName = nil
            isSystemTrackingEnabled = nil
            trackingButtonEnabled = nil
            logger.log(.warning, "DockKit accessory undocked or unavailable; motor commands are disabled.")
        }
    }

    private func waitForSystemTracking(expected: Bool) async -> Bool {
        for _ in 0..<12 {
            let current = DockAccessoryManager.shared.isSystemTrackingEnabled
            isSystemTrackingEnabled = current
            if current == expected {
                return true
            }
            try? await Task.sleep(for: .milliseconds(50))
        }
        isSystemTrackingEnabled = DockAccessoryManager.shared.isSystemTrackingEnabled
        return isSystemTrackingEnabled == expected
    }

    private func waitForProgress(_ progress: Progress, timeout: Duration) async {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while !progress.isFinished && !progress.isCancelled && clock.now < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    private func applyRelativeOrientationFallback(
        accessory: DockAccessory,
        yaw: Double,
        pitch: Double,
        roll: Double
    ) async {
        guard !orientationCommandInFlight else { return }
        orientationCommandInFlight = true
        defer { orientationCommandInFlight = false }

        let yawStep = max(-0.04, min(0.04, yaw * 0.18))
        let pitchStep = max(-0.03, min(0.03, pitch * 0.18))
        let rollStep = max(-0.02, min(0.02, roll * 0.18))
        do {
            let progress = try await accessory.setOrientation(
                Vector3D(x: pitchStep, y: yawStep, z: rollStep),
                duration: .milliseconds(80),
                relative: true
            )
            activeOrientationProgress = progress
            currentOffset = Vector3D(
                x: currentOffset.x + pitchStep,
                y: currentOffset.y + yawStep,
                z: currentOffset.z + rollStep
            )
            lastVelocityUpdateAt = nil
            lastError = nil
            logger.log(
                .success,
                String(format: "Relative fallback(pitch: %.3f, yaw: %.3f, roll: %.3f) started.", pitchStep, yawStep, rollStep)
            )
            await waitForProgress(progress, timeout: .milliseconds(250))
            if activeOrientationProgress === progress {
                activeOrientationProgress = nil
            }
        } catch {
            recordError(api: "relative orientation fallback", error: error)
        }
    }

    private func integrateAngularVelocity(yaw: Double, pitch: Double, roll: Double) {
        let now = Date()
        defer { lastVelocityUpdateAt = now }
        guard let previous = lastVelocityUpdateAt else { return }
        let dt = max(0.0, min(0.25, now.timeIntervalSince(previous)))
        currentOffset = Vector3D(
            x: currentOffset.x + pitch * dt,
            y: currentOffset.y + yaw * dt,
            z: currentOffset.z + roll * dt
        )
    }
#endif

    private func recordError(api: String, error: Error) {
        let detail = "\(api) failed: \(error.localizedDescription) [\(String(reflecting: error))]"
        lastError = detail
        logger.log(.error, detail)
    }

    private func logMissingAccessory(api: String) {
        let detail = "\(api) failed: no docked accessory."
        lastError = detail
        logger.log(.error, detail)
    }

    private func logSimulatorFailure(api: String) {
        let detail = "\(api) unavailable in Simulator; run on a physical iPhone."
        lastError = detail
        logger.log(.error, detail)
    }
}

private enum MotorControlMode {
    case unknown
    case angularVelocity
    case relativeOrientation
}

private enum ManualModeError: LocalizedError {
    case trackingStateDidNotChange(expected: Bool)

    var errorDescription: String? {
        switch self {
        case .trackingStateDidNotChange(let expected):
            "System Tracking did not become \(expected ? "ON" : "OFF") before the verification timeout."
        }
    }
}
