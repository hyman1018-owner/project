import SwiftUI

@main
struct DockKitTesterApp: App {
    @StateObject private var logger: AppLogger
    @StateObject private var dockKitManager: DockKitManager
    @StateObject private var cameraSession: CameraSessionService
    @StateObject private var controlService: GimbalControlService
    @StateObject private var networkClient: V13NetworkClient

    init() {
        let logger = AppLogger()
        let manager = DockKitManager(logger: logger)
        _logger = StateObject(wrappedValue: logger)
        _dockKitManager = StateObject(wrappedValue: manager)
        _cameraSession = StateObject(wrappedValue: CameraSessionService(logger: logger))
        _controlService = StateObject(
            wrappedValue: GimbalControlService(dockKitManager: manager, logger: logger)
        )
        _networkClient = StateObject(wrappedValue: V13NetworkClient(logger: logger))
    }

    var body: some Scene {
        WindowGroup {
            ContentView(
                dockKitManager: dockKitManager,
                cameraSession: cameraSession,
                controlService: controlService,
                networkClient: networkClient,
                logger: logger
            )
        }
    }
}
