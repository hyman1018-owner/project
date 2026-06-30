import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let videoDevice: AVCaptureDevice?
    let onFocus: (CGPoint) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.videoDevice = videoDevice
        view.onFocus = onFocus
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
        uiView.videoDevice = videoDevice
        uiView.onFocus = onFocus
    }
}

final class PreviewView: UIView {
    var onFocus: ((CGPoint) -> Void)?
    var videoDevice: AVCaptureDevice? {
        didSet { configureRotationCoordinatorIfPossible() }
    }

    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    override init(frame: CGRect) {
        super.init(frame: frame)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(focusTapped(_:))))
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(focusTapped(_:))))
    }

    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        configureRotationCoordinatorIfPossible()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        if let rotationCoordinator {
            applyRotationAngle(rotationCoordinator.videoRotationAngleForHorizonLevelPreview)
        }
    }

    private func configureRotationCoordinatorIfPossible() {
        guard window != nil, let videoDevice else { return }
        if rotationCoordinator?.device === videoDevice { return }

        rotationObservation?.invalidate()
        let coordinator = AVCaptureDevice.RotationCoordinator(
            device: videoDevice,
            previewLayer: previewLayer
        )
        rotationCoordinator = coordinator
        rotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelPreview,
            options: [.initial, .new]
        ) { [weak self] coordinator, _ in
            self?.applyRotationAngle(coordinator.videoRotationAngleForHorizonLevelPreview)
        }
    }

    private func applyRotationAngle(_ angle: CGFloat) {
        guard let connection = previewLayer.connection else { return }
        if connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
        }
    }

    @objc private func focusTapped(_ recognizer: UITapGestureRecognizer) {
        let location = recognizer.location(in: self)
        onFocus?(previewLayer.captureDevicePointConverted(fromLayerPoint: location))
        showFocusReticle(at: location)
    }

    private func showFocusReticle(at point: CGPoint) {
        let reticle = UIView(frame: CGRect(x: 0, y: 0, width: 72, height: 72))
        reticle.center = point
        reticle.layer.borderWidth = 1.5
        reticle.layer.borderColor = UIColor.systemYellow.cgColor
        reticle.layer.cornerRadius = 4
        reticle.alpha = 0
        addSubview(reticle)

        reticle.transform = CGAffineTransform(scaleX: 1.25, y: 1.25)
        UIView.animate(withDuration: 0.18, animations: {
            reticle.alpha = 1
            reticle.transform = .identity
        }) { _ in
            UIView.animate(withDuration: 0.3, delay: 0.65, options: []) {
                reticle.alpha = 0
            } completion: { _ in
                reticle.removeFromSuperview()
            }
        }
    }
}
