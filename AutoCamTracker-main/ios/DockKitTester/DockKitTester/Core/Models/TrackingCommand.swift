import Foundation

struct TrackingCommand: Codable, Equatable, Sendable {
    let type: String
    let version: String?
    let sourceVersion: String?
    let sequence: Int64?
    let targetLocked: Bool
    let targetId: Int?
    let errorX: Double
    let errorY: Double
    let confidence: Double
    let timestampMs: Int64?
    let frameWidth: Int?
    let frameHeight: Int?
    let targetX: Double?
    let targetY: Double?
    let bboxWidth: Double?
    let bboxHeight: Double?
    let zoomFactor: Double?
    let predictedTarget: Bool?

    init(
        type: String,
        version: String? = nil,
        sourceVersion: String? = nil,
        sequence: Int64? = nil,
        targetLocked: Bool,
        targetId: Int? = nil,
        errorX: Double,
        errorY: Double,
        confidence: Double,
        timestampMs: Int64? = nil,
        frameWidth: Int? = nil,
        frameHeight: Int? = nil,
        targetX: Double? = nil,
        targetY: Double? = nil,
        bboxWidth: Double? = nil,
        bboxHeight: Double? = nil,
        zoomFactor: Double? = nil,
        predictedTarget: Bool? = nil
    ) {
        self.type = type
        self.version = version
        self.sourceVersion = sourceVersion
        self.sequence = sequence
        self.targetLocked = targetLocked
        self.targetId = targetId
        self.errorX = errorX
        self.errorY = errorY
        self.confidence = confidence
        self.timestampMs = timestampMs
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.targetX = targetX
        self.targetY = targetY
        self.bboxWidth = bboxWidth
        self.bboxHeight = bboxHeight
        self.zoomFactor = zoomFactor
        self.predictedTarget = predictedTarget
    }

    enum CodingKeys: String, CodingKey {
        case type, version, sequence, confidence
        case sourceVersion = "source_version"
        case targetLocked = "target_locked"
        case targetId = "target_id"
        case errorX = "error_x"
        case errorY = "error_y"
        case timestampMs = "timestamp_ms"
        case frameWidth = "frame_width"
        case frameHeight = "frame_height"
        case targetX = "target_x"
        case targetY = "target_y"
        case bboxWidth = "bbox_width"
        case bboxHeight = "bbox_height"
        case zoomFactor = "zoom_factor"
        case predictedTarget = "predicted_target"
    }

    func isTrackable(minimumConfidence: Double = 0.20) -> Bool {
        targetLocked
            && confidence >= minimumConfidence
            && confidence.isFinite
            && errorX.isFinite
            && errorY.isFinite
    }
}

struct MotorStatusMessage: Codable, Equatable, Sendable {
    let type = "motor_status"
    let docked: Bool
    let manualReady: Bool
    let systemTrackingEnabled: Bool?
    let lastError: String?
    let timestampMs: Int64
    let currentVelocity: GimbalVelocity?
    let lastCommand: TrackingCommand?
    let lastStopReason: String?
    let cameraZoomFactor: Double?
    let cameraDisplayZoomFactor: Double?

    enum CodingKeys: String, CodingKey {
        case type, docked
        case manualReady = "manual_ready"
        case systemTrackingEnabled = "system_tracking_enabled"
        case lastError = "last_error"
        case timestampMs = "timestamp_ms"
        case currentVelocity = "current_velocity"
        case lastCommand = "last_command"
        case lastStopReason = "last_stop_reason"
        case cameraZoomFactor = "camera_zoom_factor"
        case cameraDisplayZoomFactor = "camera_display_zoom_factor"
    }
}

struct ControlMessage: Codable, Equatable, Sendable {
    let type = "control"
    let action: String
    let source: String?
    let gid: Int?
    let framing: String?
    let timestampMs: Int64

    init(action: String, source: String? = nil, gid: Int? = nil, framing: String? = nil) {
        self.action = action
        self.source = source
        self.gid = gid
        self.framing = framing
        self.timestampMs = Int64(Date().timeIntervalSince1970 * 1_000)
    }

    enum CodingKeys: String, CodingKey {
        case type, action, source, gid
        case framing
        case timestampMs = "timestamp_ms"
    }
}

struct DesktopState: Codable, Equatable, Sendable {
    let type: String
    let version: String?
    let sourceVersion: String?
    let timestampMs: Int64?
    let source: String
    let running: Bool
    let tracking: TrackingState
    let motor: MotorState
    let gids: [GIDState]
    let framing: FramingState?

    enum CodingKeys: String, CodingKey {
        case type, version, source, running, tracking, motor, framing, gids
        case sourceVersion = "source_version"
        case timestampMs = "timestamp_ms"
    }

    struct TrackingState: Codable, Equatable, Sendable {
        let status: String
        let targetLocked: Bool
        let targetId: Int?
        let selectedGid: Int?
        let selectedLid: Int?
        let errorX: Double
        let errorY: Double
        let confidence: Double
        let lostFrames: Int?
        let candidateCount: Int?
        let bbox: [Double]?
        let targetCenter: [Double]?

        enum CodingKeys: String, CodingKey {
            case status, confidence, bbox
            case targetLocked = "target_locked"
            case targetId = "target_id"
            case selectedGid = "selected_gid"
            case selectedLid = "selected_lid"
            case errorX = "error_x"
            case errorY = "error_y"
            case lostFrames = "lost_frames"
            case candidateCount = "candidate_count"
            case targetCenter = "target_center"
        }
    }

    struct MotorState: Codable, Equatable, Sendable {
        let armed: Bool
        let ready: Bool
        let clientCount: Int
        let docked: Bool
        let manualReady: Bool
        let systemTrackingEnabled: Bool?
        let lastError: String?
        let currentVelocity: GimbalVelocity?
        let lastCommand: TrackingCommand?
        let lastStopReason: String?
        let cameraZoomFactor: Double?
        let cameraDisplayZoomFactor: Double?

        enum CodingKeys: String, CodingKey {
            case armed, ready, docked
            case clientCount = "client_count"
            case manualReady = "manual_ready"
            case systemTrackingEnabled = "system_tracking_enabled"
            case lastError = "last_error"
            case currentVelocity = "current_velocity"
            case lastCommand = "last_command"
            case lastStopReason = "last_stop_reason"
            case cameraZoomFactor = "camera_zoom_factor"
            case cameraDisplayZoomFactor = "camera_display_zoom_factor"
        }
    }

    struct FramingState: Codable, Equatable, Sendable {
        let mode: String
        let cropWindow: [Double]?
        let errorX: Double
        let errorY: Double
        let zoomFactor: Double?

        enum CodingKeys: String, CodingKey {
            case mode
            case cropWindow = "crop_window"
            case errorX = "error_x"
            case errorY = "error_y"
            case zoomFactor = "zoom_factor"
        }
    }

    struct GIDState: Codable, Equatable, Identifiable, Sendable {
        let gid: Int
        let displayName: String
        let className: String
        let lastTrackId: Int?
        let lastFrameIndex: Int?
        let confidence: Double
        let masterFeatureCount: Int
        let pendingFeatureCount: Int
        let candidateFeatureCount: Int
        let trackable: Bool
        let visible: Bool
        let selected: Bool

        var id: Int { gid }

        enum CodingKeys: String, CodingKey {
            case gid, confidence, trackable, visible, selected
            case displayName = "display_name"
            case className = "class_name"
            case lastTrackId = "last_track_id"
            case lastFrameIndex = "last_frame_index"
            case masterFeatureCount = "master_feature_count"
            case pendingFeatureCount = "pending_feature_count"
            case candidateFeatureCount = "candidate_feature_count"
        }
    }
}

struct TrackingCommandSequenceValidator: Sendable {
    private(set) var lastSequence: Int64?

    mutating func accept(_ command: TrackingCommand) -> Bool {
        guard let sequence = command.sequence else { return true }
        guard lastSequence.map({ sequence > $0 }) ?? true else { return false }
        lastSequence = sequence
        return true
    }

    mutating func reset() {
        lastSequence = nil
    }
}
