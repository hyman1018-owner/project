import Foundation

struct GimbalVelocity: Codable, Equatable, Sendable {
    var yaw: Double
    var pitch: Double
    var roll: Double

    static let zero = GimbalVelocity(yaw: 0, pitch: 0, roll: 0)
}

struct GimbalControlConfiguration: Equatable, Sendable {
    var manualSpeed = 0.2
    var maxYawSpeed = 0.35
    var maxPitchSpeed = 0.22
    var deadZone = 0.05
    var smoothingOldWeight = 0.7
    var kpYaw = 1.0
    var kpPitch = 1.0
    var yawDirection = 1.0
    var pitchDirection = 1.0
    var minimumErrorImprovement = 0.01
    var maxNonImprovingUpdates = 8
    var edgeStopMargin = 0.04
    var edgeSlowMargin = 0.14
    var feedForwardGain = 0.22
}

struct GimbalCalibrationProfile: Codable, Equatable, Sendable {
    var yawInverted = false
    var pitchInverted = false
    var maxYawSpeed = 0.35
    var maxPitchSpeed = 0.22
    var deadZone = 0.05
    var minimumErrorImprovement = 0.01
    var maxNonImprovingUpdates = 8
    var edgeStopMargin = 0.04
    var edgeSlowMargin = 0.14
    var lostAutoReturnDelay = 1.0
    var stableLockRequiredFrames = 5

    static let conservative = GimbalCalibrationProfile()

    var configuration: GimbalControlConfiguration {
        GimbalControlConfiguration(
            maxYawSpeed: clamped(maxYawSpeed, min: 0.08, max: 0.8),
            maxPitchSpeed: clamped(maxPitchSpeed, min: 0.06, max: 0.5),
            deadZone: clamped(deadZone, min: 0.02, max: 0.2),
            yawDirection: yawInverted ? -1 : 1,
            pitchDirection: pitchInverted ? -1 : 1,
            minimumErrorImprovement: clamped(minimumErrorImprovement, min: 0.0, max: 0.08),
            maxNonImprovingUpdates: max(3, min(30, maxNonImprovingUpdates)),
            edgeStopMargin: clamped(edgeStopMargin, min: 0.02, max: 0.12),
            edgeSlowMargin: clamped(edgeSlowMargin, min: 0.08, max: 0.24)
        )
    }

    var clampedLostAutoReturnDelay: Double {
        clamped(lostAutoReturnDelay, min: 0.5, max: 5.0)
    }

    var clampedStableLockRequiredFrames: Int {
        max(1, min(20, stableLockRequiredFrames))
    }
}

struct GimbalVelocityCalculator: Sendable {
    var configuration: GimbalControlConfiguration
    private(set) var previous = GimbalVelocity.zero
    private(set) var safetyStopReason: String?
    private var previousErrorMagnitude: Double?
    private var previousErrorX: Double?
    private var previousErrorY: Double?
    private var nonImprovingUpdates = 0

    init(configuration: GimbalControlConfiguration = .init()) {
        self.configuration = configuration
    }

    mutating func applyCalibration(_ calibration: GimbalCalibrationProfile) {
        configuration = calibration.configuration
        reset()
    }

    mutating func setTrackingAxisInversion(yawInverted: Bool, pitchInverted: Bool) {
        configuration.yawDirection = yawInverted ? -1 : 1
        configuration.pitchDirection = pitchInverted ? -1 : 1
        reset()
    }

    mutating func velocity(for command: GimbalCommand) -> GimbalVelocity {
        let speed = configuration.manualSpeed
        let output: GimbalVelocity
        switch command {
        case .panLeft:
            output = .init(yaw: -speed, pitch: 0, roll: 0)
        case .panRight:
            output = .init(yaw: speed, pitch: 0, roll: 0)
        case .tiltUp:
            output = .init(yaw: 0, pitch: -speed, roll: 0)
        case .tiltDown:
            output = .init(yaw: 0, pitch: speed, roll: 0)
        case .stop, .recenter:
            output = .zero
        }
        previous = output
        return output
    }

    mutating func velocity(for tracking: TrackingCommand) -> GimbalVelocity {
        safetyStopReason = nil
        guard tracking.isTrackable() else {
            reset()
            return .zero
        }

        let errorX = abs(tracking.errorX) < configuration.deadZone ? 0 : tracking.errorX
        let errorY = abs(tracking.errorY) < configuration.deadZone ? 0 : tracking.errorY
        if errorX == 0, errorY == 0 {
            reset()
            return .zero
        }

        guard shouldTrackNearFrameEdge(tracking: tracking, errorX: errorX, errorY: errorY) else {
            previous = .zero
            previousErrorMagnitude = nil
            nonImprovingUpdates = 0
            safetyStopReason = "target near frame edge; stop before chasing out of view"
            return .zero
        }

        guard shouldContinueTracking(errorX: errorX, errorY: errorY) else {
            previous = .zero
            previousErrorMagnitude = nil
            nonImprovingUpdates = 0
            safetyStopReason = "tracking error did not improve; check yaw/pitch direction"
            return .zero
        }

        let requestedYaw = clamp(
            (errorX * configuration.kpYaw + feedForwardErrorX(current: errorX)) * configuration.yawDirection,
            min: -configuration.maxYawSpeed,
            max: configuration.maxYawSpeed
        )
        let requestedPitch = clamp(
            -(errorY * configuration.kpPitch + feedForwardErrorY(current: errorY)) * configuration.pitchDirection,
            min: -configuration.maxPitchSpeed,
            max: configuration.maxPitchSpeed
        )
        let edgeScale = edgeVelocityScale(tracking: tracking)
        let oldWeight = dynamicSmoothingOldWeight(tracking: tracking, errorX: errorX, errorY: errorY)
        let newWeight = 1 - oldWeight
        let output = GimbalVelocity(
            yaw: (previous.yaw * oldWeight + requestedYaw * newWeight) * edgeScale,
            pitch: (previous.pitch * oldWeight + requestedPitch * newWeight) * edgeScale,
            roll: 0
        )
        previous = output
        previousErrorX = errorX
        previousErrorY = errorY
        return output
    }

    mutating func reset() {
        previous = .zero
        previousErrorMagnitude = nil
        previousErrorX = nil
        previousErrorY = nil
        nonImprovingUpdates = 0
        safetyStopReason = nil
    }

    private func feedForwardErrorX(current errorX: Double) -> Double {
        guard let previousErrorX else { return 0 }
        return clamp(errorX - previousErrorX, min: -0.6, max: 0.6) * configuration.feedForwardGain
    }

    private func feedForwardErrorY(current errorY: Double) -> Double {
        guard let previousErrorY else { return 0 }
        return clamp(errorY - previousErrorY, min: -0.6, max: 0.6) * configuration.feedForwardGain
    }

    private func dynamicSmoothingOldWeight(tracking: TrackingCommand, errorX: Double, errorY: Double) -> Double {
        let deltaX = abs(errorX - (previousErrorX ?? errorX))
        let deltaY = abs(errorY - (previousErrorY ?? errorY))
        let turnOrAcceleration = max(deltaX, deltaY)
        if tracking.predictedTarget == true {
            return min(configuration.smoothingOldWeight, 0.45)
        }
        if turnOrAcceleration >= 0.18 {
            return min(configuration.smoothingOldWeight, 0.45)
        }
        if turnOrAcceleration >= 0.10 {
            return min(configuration.smoothingOldWeight, 0.55)
        }
        return configuration.smoothingOldWeight
    }

    private mutating func shouldContinueTracking(errorX: Double, errorY: Double) -> Bool {
        let magnitude = sqrt(errorX * errorX + errorY * errorY)
        defer { previousErrorMagnitude = magnitude }

        guard let previousErrorMagnitude else {
            nonImprovingUpdates = 0
            return true
        }

        if magnitude < previousErrorMagnitude - configuration.minimumErrorImprovement {
            nonImprovingUpdates = 0
            return true
        }

        nonImprovingUpdates += 1
        return nonImprovingUpdates < configuration.maxNonImprovingUpdates
    }

    private func shouldTrackNearFrameEdge(tracking: TrackingCommand, errorX: Double, errorY: Double) -> Bool {
        let margin = configuration.edgeStopMargin
        if let targetX = tracking.targetX {
            if targetX <= margin, errorX < -configuration.deadZone { return false }
            if targetX >= 1.0 - margin, errorX > configuration.deadZone { return false }
        }
        if let targetY = tracking.targetY {
            if targetY <= margin, errorY < -configuration.deadZone { return false }
            if targetY >= 1.0 - margin, errorY > configuration.deadZone { return false }
        }
        return true
    }

    private func edgeVelocityScale(tracking: TrackingCommand) -> Double {
        let margin = configuration.edgeSlowMargin
        let distances = [
            tracking.targetX.map { min($0, 1.0 - $0) },
            tracking.targetY.map { min($0, 1.0 - $0) }
        ].compactMap { $0 }
        guard let nearest = distances.min(), nearest < margin else { return 1.0 }
        return max(0.35, nearest / margin)
    }
}

private func clamped(_ value: Double, min lowerBound: Double, max upperBound: Double) -> Double {
    max(lowerBound, min(upperBound, value))
}
