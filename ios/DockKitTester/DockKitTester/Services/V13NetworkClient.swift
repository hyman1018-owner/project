import Foundation
import Combine

@MainActor
final class V13NetworkClient: ObservableObject {
    enum ConnectionStatus: String {
        case offline = "Offline"
        case connecting = "Connecting"
        case connected = "Connected"
        case receiving = "Receiving Tracking"
        case timedOut = "Timed Out"
        case failed = "Connection Failed"
    }

    @Published private(set) var status: ConnectionStatus = .offline
    @Published private(set) var lastCommand: TrackingCommand?
    @Published private(set) var desktopState: DesktopState?
    @Published private(set) var cameraFramesSent = 0
    @Published private(set) var cameraFramesDropped = 0
    @Published var serverURL: String {
        didSet { UserDefaults.standard.set(serverURL, forKey: Self.serverURLKey) }
    }

    var onCommand: ((TrackingCommand) async -> Void)?
    var onControl: ((ControlMessage) async -> Void)?
    var onTimeout: (() async -> Void)?

    private let logger: AppLogger
    private var timeoutTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var socketTask: URLSessionWebSocketTask?
    private var latestCameraFrame: Data?
    private var cameraSendInFlight = false
    private var intentionalDisconnect = false
    private var sequenceValidator = TrackingCommandSequenceValidator()
    private let timeout: Duration = .milliseconds(500)
    private static let serverURLKey = "AutoCamTrackerServerURL"

    init(logger: AppLogger) {
        self.logger = logger
        serverURL = UserDefaults.standard.string(forKey: Self.serverURLKey)
            ?? "ws://192.168.1.100:8765/ws/tracking"
    }

    deinit {
        timeoutTask?.cancel()
        receiveTask?.cancel()
        reconnectTask?.cancel()
        socketTask?.cancel(with: .goingAway, reason: nil)
    }

    func connect() async {
        let value = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: value), ["ws", "wss"].contains(url.scheme?.lowercased() ?? "") else {
            status = .failed
            logger.log(.error, "Invalid WebSocket URL: \(value)")
            await onTimeout?()
            return
        }

        closeSocket()
        sequenceValidator.reset()
        intentionalDisconnect = false
        status = .connecting
        logger.log(.info, "Connecting to AutoCamTracker at \(url.absoluteString)")

        let task = URLSession.shared.webSocketTask(with: url)
        socketTask = task
        task.resume()
        receiveTask = Task { @MainActor [weak self, weak task] in
            guard let self, let task else { return }
            await self.receiveLoop(task: task)
        }
        await sendControl(action: "request_state")
    }

    func receive(data: Data) async {
        guard let messageType = messageType(in: data) else {
            logger.log(.info, "Ignored malformed AutoCamTracker message.")
            return
        }

        switch messageType {
        case "tracking":
            markConnectedIfNeeded()
            await receiveTrackingCommand(data)
        case "desktop_state":
            markConnectedIfNeeded()
            receiveDesktopState(data)
        case "control":
            markConnectedIfNeeded()
            await receiveControlMessage(data)
        default:
            logger.log(.info, "Ignored unsupported AutoCamTracker message type: \(messageType).")
        }
    }

    private func receiveTrackingCommand(_ data: Data) async {
        switch JSONDecoder().decodeSafely(TrackingCommand.self, from: data) {
        case .success(let command):
            guard sequenceValidator.accept(command) else {
                logger.log(.error, "V1.75 JSON rejected: duplicate or out-of-order sequence.")
                await triggerTimeout(reason: "stale tracking command")
                return
            }
            lastCommand = command
            logger.log(
                .success,
                String(format: "V1.75 JSON decoded: locked=%@ error=(%.3f, %.3f) confidence=%.2f zoom=%@ predicted=%@.", String(command.targetLocked), command.errorX, command.errorY, command.confidence, command.zoomFactor.map { String(format: "%.2f", $0) } ?? "nil", String(command.predictedTarget ?? false))
            )
            await onCommand?(command)
            if command.targetLocked {
                status = .receiving
                armTimeout()
            } else {
                timeoutTask?.cancel()
                status = .connected
            }
        case .failure(let error):
            logger.log(.error, "V1.75 JSON decode failed: \(error.localizedDescription)")
            await triggerTimeout(reason: "JSON decode failure")
        }
    }

    private func receiveDesktopState(_ data: Data) {
        switch JSONDecoder().decodeSafely(DesktopState.self, from: data) {
        case .success(let state):
            desktopState = state
            if status == .timedOut || status == .connecting {
                status = .connected
            }
            logger.log(
                .info,
                "Desktop state updated: source=\(state.source), motor=\(state.motor.armed ? "armed" : "off"), gids=\(state.gids.count)."
            )
        case .failure(let error):
            logger.log(.error, "Desktop state decode failed: \(error.localizedDescription)")
        }
    }

    private func receiveControlMessage(_ data: Data) async {
        switch JSONDecoder().decodeSafely(ControlMessage.self, from: data) {
        case .success(let message):
            logger.log(.info, "Received desktop control action: \(message.action)")
            await onControl?(message)
        case .failure(let error):
            logger.log(.error, "Desktop control decode failed: \(error.localizedDescription)")
        }
    }

    func sendControl(action: String, source: String? = nil, gid: Int? = nil, framing: String? = nil) async {
        guard let socketTask, status != .offline, status != .failed else { return }
        let message = ControlMessage(action: action, source: source, gid: gid, framing: framing)
        do {
            let data = try JSONEncoder().encode(message)
            guard let text = String(data: data, encoding: .utf8) else { return }
            try await socketTask.send(.string(text))
            logger.log(.info, "Sent iPhone control action: \(action)")
        } catch {
            logger.log(.error, "Control send failed: \(error.localizedDescription)")
        }
    }

    func sendCameraFrame(_ data: Data) async {
        guard status != .offline, status != .failed else { return }
        if cameraSendInFlight {
            if latestCameraFrame != nil {
                cameraFramesDropped += 1
            }
            latestCameraFrame = data
            return
        }

        latestCameraFrame = data
        cameraSendInFlight = true
        defer { cameraSendInFlight = false }

        while let frame = latestCameraFrame {
            latestCameraFrame = nil
            guard let socketTask, status != .offline, status != .failed else { return }
            do {
                try await socketTask.send(.data(frame))
                cameraFramesSent += 1
            } catch {
                latestCameraFrame = nil
                logger.log(.error, "Camera frame send failed: \(error.localizedDescription)")
                return
            }
        }
    }

    func sendMotorStatus(
        docked: Bool,
        manualReady: Bool,
        systemTrackingEnabled: Bool?,
        lastError: String?,
        currentVelocity: GimbalVelocity? = nil,
        lastCommand: TrackingCommand? = nil,
        lastStopReason: String? = nil,
        cameraZoomFactor: Double? = nil,
        cameraDisplayZoomFactor: Double? = nil
    ) async {
        guard let socketTask, status != .offline, status != .failed else { return }
        let message = MotorStatusMessage(
            docked: docked,
            manualReady: manualReady,
            systemTrackingEnabled: systemTrackingEnabled,
            lastError: lastError,
            timestampMs: Int64(Date().timeIntervalSince1970 * 1_000),
            currentVelocity: currentVelocity,
            lastCommand: lastCommand,
            lastStopReason: lastStopReason,
            cameraZoomFactor: cameraZoomFactor,
            cameraDisplayZoomFactor: cameraDisplayZoomFactor
        )
        do {
            let data = try JSONEncoder().encode(message)
            guard let text = String(data: data, encoding: .utf8) else { return }
            try await socketTask.send(.string(text))
        } catch {
            logger.log(.error, "Motor status send failed: \(error.localizedDescription)")
        }
    }

    func sendFakeCommand() async {
        let json = #"{"type":"tracking","version":"1.0","source_version":"1.75","target_locked":true,"target_id":7,"error_x":0.18,"error_y":-0.04,"confidence":0.91,"timestamp_ms":1781770000000,"zoom_factor":2.0}"#
        logger.log(.info, "Injecting a fake V1.75 JSON command.")
        await receive(data: Data(json.utf8))
    }

    func disconnect() async {
        intentionalDisconnect = true
        closeSocket()
        status = .offline
        lastCommand = nil
        desktopState = nil
        sequenceValidator.reset()
        latestCameraFrame = nil
        cameraSendInFlight = false
        cameraFramesSent = 0
        cameraFramesDropped = 0
        logger.log(.warning, "AutoCamTracker client disconnected; requesting safety stop.")
        await onTimeout?()
    }

    private func receiveLoop(task: URLSessionWebSocketTask) async {
        do {
            while !Task.isCancelled, task === socketTask {
                let message = try await task.receive()
                switch message {
                case .data(let data):
                    await receive(data: data)
                case .string(let text):
                    await receive(data: Data(text.utf8))
                @unknown default:
                    logger.log(.warning, "Ignored an unknown WebSocket message type.")
                }
            }
        } catch {
            guard !intentionalDisconnect, task === socketTask else { return }
            socketTask = nil
            status = .failed
            logger.log(.error, "WebSocket receive failed: \(error.localizedDescription)")
            await triggerTimeout(reason: "WebSocket disconnected")
            scheduleReconnect()
        }
    }

    private func closeSocket() {
        timeoutTask?.cancel()
        receiveTask?.cancel()
        reconnectTask?.cancel()
        socketTask?.cancel(with: .goingAway, reason: nil)
        socketTask = nil
        latestCameraFrame = nil
    }

    private func messageType(in data: Data) -> String? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let payload = object as? [String: Any]
        else {
            return nil
        }
        return payload["type"] as? String
    }

    private func markConnectedIfNeeded() {
        if status == .connecting || status == .timedOut {
            status = .connected
            logger.log(.success, "AutoCamTracker WebSocket connected.")
        }
    }

    private func scheduleReconnect() {
        reconnectTask?.cancel()
        reconnectTask = Task { @MainActor [weak self] in
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
            guard let self, !self.intentionalDisconnect else { return }
            self.logger.log(.info, "Retrying AutoCamTracker WebSocket connection.")
            await self.connect()
        }
    }

    private func armTimeout() {
        timeoutTask?.cancel()
        timeoutTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await Task.sleep(for: self.timeout)
                await triggerTimeout(reason: "no V1.75 data for 500 ms")
            } catch {
                return
            }
        }
    }

    private func triggerTimeout(reason: String) async {
        timeoutTask?.cancel()
        status = .timedOut
        logger.log(.warning, "V1.75 safety timeout: \(reason).")
        await onTimeout?()
    }
}
