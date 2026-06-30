@preconcurrency import AVFoundation
import Combine
import CoreImage
import Foundation
import UIKit

enum CameraStreamOrientation: String, CaseIterable, Identifiable {
    case portrait
    case landscapeLeft
    case landscapeRight

    var id: Self { self }
    var label: String {
        switch self {
        case .portrait: "直式"
        case .landscapeLeft, .landscapeRight: "橫式"
        }
    }

    var imageOrientation: CGImagePropertyOrientation {
        switch self {
        case .portrait: .right
        // UIDevice reports the physical rotation, while Core Image describes
        // the transform needed to display the rear-camera sensor upright.
        case .landscapeLeft: .up
        case .landscapeRight: .down
        }
    }
}

enum CameraStreamPreset: String, CaseIterable, Identifiable {
    case lowLatency
    case balanced
    case detail

    var id: Self { self }
    var targetWidth: CGFloat {
        switch self {
        case .lowLatency: 640
        case .balanced: 800
        case .detail: 960
        }
    }
    var minimumFrameInterval: Double {
        switch self {
        case .lowLatency: 1.0 / 30.0
        case .balanced: 1.0 / 24.0
        case .detail: 1.0 / 20.0
        }
    }
    var jpegQuality: CGFloat {
        switch self {
        case .lowLatency: 0.55
        case .balanced: 0.62
        case .detail: 0.70
        }
    }
}

@MainActor
final class CameraSessionService: ObservableObject {
    var session: AVCaptureSession { capture.session }
    var videoDevice: AVCaptureDevice? { capture.camera }
    var onJPEGFrame: (@Sendable (Data) -> Void)? {
        didSet { capture.frameStreamer.onFrame = onJPEGFrame }
    }

    @Published private(set) var isRunning = false
    @Published private(set) var authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
    @Published private(set) var lastError: String?
    @Published private(set) var streamOrientation: CameraStreamOrientation = .portrait
    @Published private(set) var streamPreset: CameraStreamPreset = .lowLatency
    @Published private(set) var zoomFactor: CGFloat = 1
    @Published private(set) var minimumZoomFactor: CGFloat = 1
    @Published private(set) var maximumZoomFactor: CGFloat = 1
    @Published private(set) var displayZoomFactorMultiplier: CGFloat = 1

    var displayZoomFactor: CGFloat { zoomFactor * displayZoomFactorMultiplier }
    var minimumDisplayZoomFactor: CGFloat { minimumZoomFactor * displayZoomFactorMultiplier }
    var maximumDisplayZoomFactor: CGFloat { maximumZoomFactor * displayZoomFactorMultiplier }

    private let logger: AppLogger
    private let sessionQueue = DispatchQueue(label: "com.linen.DockKitTester.camera-session")
    private let capture = CaptureSessionBox()
    private var orientationObserver: AnyCancellable?
    private var lastTrackingZoomUpdate = Date.distantPast

    init(logger: AppLogger) {
        self.logger = logger
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        orientationObserver = NotificationCenter.default.publisher(
            for: UIDevice.orientationDidChangeNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor in self?.updateOrientationFromDevice() }
        }
    }

    func start() async {
        authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)

        if authorizationStatus == .notDetermined {
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
            guard granted else {
                recordError("Camera permission was denied; DockKit camera session cannot start.")
                return
            }
        }

        guard authorizationStatus == .authorized else {
            recordError("Camera permission is not authorized. Enable it in Settings > Privacy & Security > Camera.")
            return
        }

        do {
            try await configureAndStart()
            minimumZoomFactor = capture.minimumZoomFactor
            maximumZoomFactor = capture.maximumZoomFactor
            zoomFactor = capture.zoomFactor
            displayZoomFactorMultiplier = capture.displayZoomFactorMultiplier
            updateOrientationFromDevice()
            isRunning = true
            lastError = nil
            logger.log(.success, "Rear camera capture session started; DockKit discovery is active.")
        } catch {
            recordError("Camera capture session failed: \(error.localizedDescription) [\(String(reflecting: error))]")
        }
    }


    func setZoom(_ requestedFactor: CGFloat) {
        let factor = min(max(requestedFactor, minimumZoomFactor), maximumZoomFactor)
        zoomFactor = factor
        let capture = capture
        sessionQueue.async {
            guard let camera = capture.camera else { return }
            do {
                try camera.lockForConfiguration()
                camera.videoZoomFactor = min(
                    max(factor, camera.minAvailableVideoZoomFactor),
                    camera.maxAvailableVideoZoomFactor
                )
                camera.unlockForConfiguration()
            } catch {
                Task { @MainActor [weak self] in
                    self?.recordError("Camera zoom failed: \(error.localizedDescription)")
                }
            }
        }
    }

    func setDisplayZoom(_ requestedFactor: CGFloat) {
        setZoom(requestedFactor / max(displayZoomFactorMultiplier, 0.01))
    }

    func applyTrackingDisplayZoom(_ requestedFactor: Double?, force: Bool = false) {
        guard let requestedFactor, requestedFactor.isFinite else { return }
        let displayFactor = CGFloat(requestedFactor)
        let clampedDisplayFactor = min(max(displayFactor, minimumDisplayZoomFactor), maximumDisplayZoomFactor)
        guard force || abs(clampedDisplayFactor - displayZoomFactor) >= 0.08 else { return }
        let now = Date()
        guard force || now.timeIntervalSince(lastTrackingZoomUpdate) >= 0.35 else { return }
        lastTrackingZoomUpdate = now
        logger.log(
            .info,
            String(format: "Tracking zoom request: display %.2f -> %.2f.", Double(requestedFactor), Double(clampedDisplayFactor))
        )
        setZoomSmooth(clampedDisplayFactor / max(displayZoomFactorMultiplier, 0.01))
    }

    func resetTrackingDisplayZoom() {
        applyTrackingDisplayZoom(Double(minimumDisplayZoomFactor), force: true)
    }

    func setStreamPreset(_ preset: CameraStreamPreset) {
        streamPreset = preset
        capture.frameStreamer.setPreset(preset)
        logger.log(.info, "Camera stream preset: \(preset.rawValue).")
    }

    func focus(at devicePoint: CGPoint) {
        let capture = capture
        sessionQueue.async {
            guard let camera = capture.camera else { return }
            do {
                try camera.lockForConfiguration()
                if camera.isFocusPointOfInterestSupported {
                    camera.focusPointOfInterest = devicePoint
                    if camera.isFocusModeSupported(.autoFocus) {
                        camera.focusMode = .autoFocus
                    }
                }
                if camera.isExposurePointOfInterestSupported {
                    camera.exposurePointOfInterest = devicePoint
                    if camera.isExposureModeSupported(.continuousAutoExposure) {
                        camera.exposureMode = .continuousAutoExposure
                    }
                }
                camera.isSubjectAreaChangeMonitoringEnabled = true
                camera.unlockForConfiguration()
            } catch {
                Task { @MainActor [weak self] in
                    self?.recordError("Camera focus failed: \(error.localizedDescription)")
                }
            }
        }
    }

    private func updateOrientationFromDevice() {
        let newOrientation: CameraStreamOrientation
        switch UIDevice.current.orientation {
        case .landscapeLeft: newOrientation = .landscapeLeft
        case .landscapeRight: newOrientation = .landscapeRight
        case .portrait, .portraitUpsideDown: newOrientation = .portrait
        default: return
        }
        guard newOrientation != streamOrientation else { return }
        streamOrientation = newOrientation
        capture.frameStreamer.setOrientation(newOrientation)
    }

    func stop() async {
        guard isRunning else { return }
        let capture = capture
        await withCheckedContinuation { continuation in
            sessionQueue.async {
                if capture.session.isRunning {
                    capture.session.stopRunning()
                }
                continuation.resume()
            }
        }
        isRunning = false
        logger.log(.info, "Camera capture session stopped.")
    }

    private func configureAndStart() async throws {
        let capture = capture
        try await withCheckedThrowingContinuation { continuation in
            sessionQueue.async {
                do {
                    if !capture.isConfigured {
                        capture.session.beginConfiguration()
                        capture.session.sessionPreset = .high

                        guard let camera = Self.preferredRearCamera() else {
                            capture.session.commitConfiguration()
                            throw CameraSessionError.rearCameraUnavailable
                        }

                        let input = try AVCaptureDeviceInput(device: camera)
                        guard capture.session.canAddInput(input) else {
                            capture.session.commitConfiguration()
                            throw CameraSessionError.cannotAddInput
                        }
                        capture.session.addInput(input)
                        capture.camera = camera
                        capture.minimumZoomFactor = camera.minAvailableVideoZoomFactor
                        capture.maximumZoomFactor = min(camera.maxAvailableVideoZoomFactor, 10)
                        capture.zoomFactor = camera.videoZoomFactor
                        capture.displayZoomFactorMultiplier = camera.displayVideoZoomFactorMultiplier
                        try Self.configureFrameRate(camera, fps: 30)

                        let output = AVCaptureVideoDataOutput()
                        output.alwaysDiscardsLateVideoFrames = true
                        output.videoSettings = [
                            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                        ]
                        output.setSampleBufferDelegate(capture.frameStreamer, queue: capture.frameQueue)
                        guard capture.session.canAddOutput(output) else {
                            capture.session.commitConfiguration()
                            throw CameraSessionError.cannotAddOutput
                        }
                        capture.session.addOutput(output)
                        capture.session.commitConfiguration()
                        capture.isConfigured = true
                    }

                    if !capture.session.isRunning {
                        capture.session.startRunning()
                    }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func setZoomSmooth(_ requestedFactor: CGFloat) {
        let factor = min(max(requestedFactor, minimumZoomFactor), maximumZoomFactor)
        zoomFactor = factor
        let capture = capture
        sessionQueue.async {
            guard let camera = capture.camera else {
                Task { @MainActor [weak self] in
                    self?.logger.log(.warning, "Tracking zoom skipped: camera is unavailable.")
                }
                return
            }
            do {
                try camera.lockForConfiguration()
                let cameraFactor = min(
                    max(factor, camera.minAvailableVideoZoomFactor),
                    camera.maxAvailableVideoZoomFactor
                )
                camera.ramp(toVideoZoomFactor: cameraFactor, withRate: 2.0)
                camera.unlockForConfiguration()
                Task { @MainActor [weak self] in
                    self?.logger.log(.success, String(format: "Tracking zoom ramp started: %.2f.", Double(cameraFactor)))
                }
            } catch {
                Task { @MainActor [weak self] in
                    self?.recordError("Camera tracking zoom failed: \(error.localizedDescription)")
                }
            }
        }
    }


    private nonisolated static func preferredRearCamera() -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [
                .builtInTripleCamera,
                .builtInDualWideCamera,
                .builtInDualCamera,
                .builtInWideAngleCamera,
            ],
            mediaType: .video,
            position: .back
        )
        let priority: [AVCaptureDevice.DeviceType] = [
            .builtInTripleCamera,
            .builtInDualWideCamera,
            .builtInDualCamera,
            .builtInWideAngleCamera,
        ]
        return priority.lazy.compactMap { type in
            discovery.devices.first(where: { $0.deviceType == type })
        }.first
    }

    private nonisolated static func configureFrameRate(_ camera: AVCaptureDevice, fps: Int32) throws {
        let duration = CMTime(value: 1, timescale: fps)
        let supportsFrameRate = camera.activeFormat.videoSupportedFrameRateRanges.contains { range in
            range.minFrameRate <= Double(fps) && Double(fps) <= range.maxFrameRate
        }
        guard supportsFrameRate else { return }
        try camera.lockForConfiguration()
        camera.activeVideoMinFrameDuration = duration
        camera.activeVideoMaxFrameDuration = duration
        camera.unlockForConfiguration()
    }

    private func recordError(_ message: String) {
        lastError = message
        isRunning = false
        logger.log(.error, message)
    }
}

private final class CaptureSessionBox: @unchecked Sendable {
    let session = AVCaptureSession()
    let frameQueue = DispatchQueue(label: "com.linen.DockKitTester.camera-frames", qos: .userInitiated)
    let frameStreamer = JPEGFrameStreamer()
    var camera: AVCaptureDevice?
    var minimumZoomFactor: CGFloat = 1
    var maximumZoomFactor: CGFloat = 1
    var zoomFactor: CGFloat = 1
    var displayZoomFactorMultiplier: CGFloat = 1
    var isConfigured = false
}

private final class JPEGFrameStreamer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private static let envelopeMagic = Data([0x41, 0x43, 0x54, 0x46, 0x31]) // ACTF1

    var onFrame: (@Sendable (Data) -> Void)?

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private var lastFrameTime = 0.0
    private let orientationLock = NSLock()
    private var orientation: CameraStreamOrientation = .portrait
    private let presetLock = NSLock()
    private var preset: CameraStreamPreset = .lowLatency

    func setOrientation(_ newOrientation: CameraStreamOrientation) {
        orientationLock.lock()
        orientation = newOrientation
        orientationLock.unlock()
    }

    func setPreset(_ newPreset: CameraStreamPreset) {
        presetLock.lock()
        preset = newPreset
        presetLock.unlock()
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard onFrame != nil,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        presetLock.lock()
        let currentPreset = preset
        presetLock.unlock()

        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard timestamp - lastFrameTime >= currentPreset.minimumFrameInterval else { return }
        lastFrameTime = timestamp

        orientationLock.lock()
        let currentOrientation = orientation
        orientationLock.unlock()
        let cameraImage = CIImage(cvPixelBuffer: pixelBuffer)
        let source = cameraImage.oriented(currentOrientation.imageOrientation)
        let targetWidth = currentPreset.targetWidth
        let scale = targetWidth / max(1.0, source.extent.width)
        let resized = source.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cgImage = context.createCGImage(
            resized,
            from: resized.extent,
            format: .RGBA8,
            colorSpace: colorSpace
        ), let jpeg = UIImage(cgImage: cgImage).jpegData(compressionQuality: currentPreset.jpegQuality) else { return }
        onFrame?(Self.envelopedFrame(jpeg, captureTimestampMs: UInt64(Date().timeIntervalSince1970 * 1_000)))
    }

    private static func envelopedFrame(_ jpeg: Data, captureTimestampMs: UInt64) -> Data {
        var payload = Data()
        payload.append(envelopeMagic)
        var timestamp = captureTimestampMs.bigEndian
        withUnsafeBytes(of: &timestamp) { payload.append(contentsOf: $0) }
        payload.append(jpeg)
        return payload
    }
}

private enum CameraSessionError: LocalizedError {
    case rearCameraUnavailable
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .rearCameraUnavailable: "Rear camera is unavailable."
        case .cannotAddInput: "AVCaptureSession rejected the rear camera input."
        case .cannotAddOutput: "AVCaptureSession rejected the video output."
        }
    }
}
