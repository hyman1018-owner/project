import SwiftUI

struct ContentView: View {
    @ObservedObject var dockKitManager: DockKitManager
    @ObservedObject var cameraSession: CameraSessionService
    @ObservedObject var controlService: GimbalControlService
    @ObservedObject var networkClient: V13NetworkClient
    @ObservedObject var logger: AppLogger
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        TabView {
            cameraTab
                .tabItem { Label("相機", systemImage: "camera.fill") }

            gimbalTab
                .tabItem { Label("雲台", systemImage: "move.3d") }

            connectionTab
                .tabItem { Label("連線", systemImage: "network") }

            logTab
                .tabItem { Label("紀錄", systemImage: "list.bullet.rectangle") }
        }
        .tint(.yellow)
        .task { await prepareServices() }
        .onChange(of: scenePhase) { _, newPhase in
            Task {
                if newPhase == .active {
                    await cameraSession.start()
                } else {
                    await controlService.emergencyStop(reason: "app left foreground")
                    await cameraSession.stop()
                }
            }
        }
        .onChange(of: dockKitManager.isDocked) { _, isDocked in
            Task {
                if !isDocked {
                    await controlService.emergencyStop(reason: "DockKit undocked; motor tracking paused")
                } else {
                    await dockKitManager.startListening()
                }
                await publishMotorStatus()
            }
        }
        .onChange(of: dockKitManager.isManualControlReady) { _, _ in
            Task { await publishMotorStatus() }
        }
        .onChange(of: dockKitManager.lastError) { _, _ in
            Task { await publishMotorStatus() }
        }
        .onChange(of: networkClient.desktopState?.framing?.zoomFactor) { _, newValue in
            cameraSession.applyTrackingDisplayZoom(newValue)
        }
    }

    private var cameraTab: some View {
        NavigationStack {
            CameraControlPage(
                cameraSession: cameraSession,
                dockKitManager: dockKitManager,
                controlService: controlService,
                networkClient: networkClient
            )
            .navigationTitle("AutoCam Camera")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
    }

    private var gimbalTab: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    safetyNotice
                    StatusPanelView(
                        manager: dockKitManager,
                        onTestVelocity: { await controlService.testAngularVelocity() }
                    )
                    ManualControlPadView(
                        isDocked: dockKitManager.isDocked,
                        isManualControlReady: dockKitManager.isManualControlReady,
                        onCommand: { await controlService.execute($0) }
                    )
                    velocityPanel
                    trackingCalibrationPanel
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("雲台控制")
        }
    }

    private var connectionTab: some View {
        NavigationStack {
            ScrollView {
                NetworkTestView(
                    client: networkClient,
                    canInjectCommand: dockKitManager.isManualControlReady
                )
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("電腦連線")
        }
    }

    private var logTab: some View {
        NavigationStack {
            ScrollView {
                LogConsoleView(logger: logger)
                    .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("系統紀錄")
        }
    }

    private var safetyNotice: some View {
        Label(
            "先進入 Manual Mode 並確認 Tracking OFF。方向鍵會持續輸出速度，測完請立即按 STOP。",
            systemImage: "exclamationmark.triangle.fill"
        )
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.orange)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    private var velocityPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("目前命令")
                .font(.headline)
            Text(
                String(
                    format: "yaw %.3f   pitch %.3f   roll %.3f rad/s",
                    controlService.currentVelocity.yaw,
                    controlService.currentVelocity.pitch,
                    controlService.currentVelocity.roll
                )
            )
            .font(.system(.body, design: .monospaced))
            if let reason = controlService.lastStopReason {
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .panelStyle()
    }

    private var trackingCalibrationPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("追蹤校正")
                    .font(.headline)
                Spacer()
                Button("重設") {
                    controlService.resetCalibration()
                }
                .font(.caption.weight(.semibold))
                .buttonStyle(.bordered)
            }
            Toggle(
                "反轉左右追蹤方向",
                isOn: Binding(
                    get: { controlService.calibration.yawInverted },
                    set: { controlService.setYawInverted($0) }
                )
            )
            Toggle(
                "反轉上下追蹤方向",
                isOn: Binding(
                    get: { controlService.calibration.pitchInverted },
                    set: { controlService.setPitchInverted($0) }
                )
            )

            calibrationSlider(
                title: "左右最大速度",
                value: controlService.calibration.maxYawSpeed,
                range: 0.08...0.8,
                format: "%.2f",
                onChange: controlService.setMaxYawSpeed
            )
            calibrationSlider(
                title: "上下最大速度",
                value: controlService.calibration.maxPitchSpeed,
                range: 0.06...0.5,
                format: "%.2f",
                onChange: controlService.setMaxPitchSpeed
            )
            calibrationSlider(
                title: "中心停止區",
                value: controlService.calibration.deadZone,
                range: 0.02...0.2,
                format: "%.2f",
                onChange: controlService.setDeadZone
            )
            calibrationSlider(
                title: "改善門檻",
                value: controlService.calibration.minimumErrorImprovement,
                range: 0.0...0.08,
                format: "%.3f",
                onChange: controlService.setMinimumErrorImprovement
            )
            calibrationSlider(
                title: "無改善停止次數",
                value: Double(controlService.calibration.maxNonImprovingUpdates),
                range: 3...30,
                format: "%.0f",
                onChange: controlService.setMaxNonImprovingUpdates
            )
            calibrationSlider(
                title: "失追回 Home 秒數",
                value: controlService.calibration.lostAutoReturnDelay,
                range: 0.5...5.0,
                format: "%.1f",
                onChange: controlService.setLostAutoReturnDelay
            )
            calibrationSlider(
                title: "重鎖確認幀數",
                value: Double(controlService.calibration.stableLockRequiredFrames),
                range: 1...20,
                format: "%.0f",
                onChange: controlService.setStableLockRequiredFrames
            )

            Text("先用保守速度測試。若 Find GID 後目標越追越遠，先切換左右方向；若網路或辨識不穩，放大中心停止區並降低最大速度。失追回 Home 秒數越短越安全，越長越能容忍短暫遮擋。")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .panelStyle()
    }

    private func calibrationSlider(
        title: String,
        value: Double,
        range: ClosedRange<Double>,
        format: String,
        onChange: @escaping (Double) -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { value },
                    set: onChange
                ),
                in: range
            )
        }
        .font(.subheadline)
    }

    private func prepareServices() async {
        cameraSession.onJPEGFrame = { [weak networkClient, weak dockKitManager, weak controlService, weak cameraSession] data in
            Task { @MainActor in
                guard let networkClient else { return }
                await networkClient.sendCameraFrame(data)
                if networkClient.cameraFramesSent.isMultiple(of: 15), let dockKitManager {
                    await networkClient.sendMotorStatus(
                        docked: dockKitManager.isDocked,
                        manualReady: dockKitManager.isManualControlReady,
                        systemTrackingEnabled: dockKitManager.isSystemTrackingEnabled,
                        lastError: dockKitManager.lastError,
                        currentVelocity: controlService?.currentVelocity,
                        lastCommand: networkClient.lastCommand,
                        lastStopReason: controlService?.lastStopReason,
                        cameraZoomFactor: cameraSession.map { Double($0.zoomFactor) },
                        cameraDisplayZoomFactor: cameraSession.map { Double($0.displayZoomFactor) }
                    )
                }
            }
        }
        networkClient.onCommand = { [weak controlService, weak cameraSession] command in
            cameraSession?.applyTrackingDisplayZoom(command.zoomFactor, force: !command.targetLocked)
            await controlService?.apply(command)
        }
        networkClient.onControl = { [weak controlService] message in
            switch message.action {
            case "recenter":
                await controlService?.execute(.recenter)
            default:
                break
            }
        }
        networkClient.onTimeout = { [weak controlService, weak cameraSession] in
            cameraSession?.resetTrackingDisplayZoom()
            await controlService?.emergencyStop(reason: "V1.75 timeout or disconnect")
        }
        await cameraSession.start()
        await dockKitManager.startListening()
        await networkClient.connect()
        await publishMotorStatus()
    }

    private func publishMotorStatus() async {
        await networkClient.sendMotorStatus(
            docked: dockKitManager.isDocked,
            manualReady: dockKitManager.isManualControlReady,
            systemTrackingEnabled: dockKitManager.isSystemTrackingEnabled,
            lastError: dockKitManager.lastError,
            currentVelocity: controlService.currentVelocity,
            lastCommand: networkClient.lastCommand,
            lastStopReason: controlService.lastStopReason,
            cameraZoomFactor: Double(cameraSession.zoomFactor),
            cameraDisplayZoomFactor: Double(cameraSession.displayZoomFactor)
        )
    }
}

private struct CameraControlPage: View {
    @ObservedObject var cameraSession: CameraSessionService
    @ObservedObject var dockKitManager: DockKitManager
    @ObservedObject var controlService: GimbalControlService
    @ObservedObject var networkClient: V13NetworkClient
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    @State private var selectedRemoteGID: Int?

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                ZStack(alignment: .top) {
                    CameraPreviewView(
                        session: cameraSession.session,
                        videoDevice: cameraSession.videoDevice,
                        onFocus: cameraSession.focus(at:)
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()

                    statusStrip
                        .padding(.horizontal, 12)
                        .padding(.top, 10)

                    if !cameraSession.isRunning {
                        ContentUnavailableView(
                            "相機未啟動",
                            systemImage: "camera.slash",
                            description: Text(cameraSession.lastError ?? "正在等待相機權限")
                        )
                        .foregroundStyle(.white)
                    }
                }
                .frame(height: previewHeight(for: geometry.size))

                ScrollView {
                    controls
                }
                .background(.black)
            }
            .background(.black)
        }
        .background(.black)
        .onChange(of: networkClient.desktopState?.tracking.selectedGid) { _, newValue in
            selectedRemoteGID = newValue
        }
    }

    private var statusStrip: some View {
        HStack(spacing: 8) {
            statusChip(
                cameraSession.isRunning ? "相機" : "相機關閉",
                icon: cameraSession.isRunning ? "camera.fill" : "camera.slash",
                active: cameraSession.isRunning
            )
            statusChip(
                dockKitManager.isDocked ? "雲台" : "未接雲台",
                icon: "move.3d",
                active: dockKitManager.isDocked
            )
            statusChip(
                networkActive ? "電腦" : "未連線",
                icon: "network",
                active: networkActive
            )
            Spacer()
            statusChip(
                cameraSession.streamOrientation.label,
                icon: cameraSession.streamOrientation == .portrait ? "iphone" : "iphone.landscape",
                active: true
            )
        }
    }

    private var controls: some View {
        VStack(spacing: 14) {
            remoteStatusPanel
            remoteCommandRow
            homeReturnControls
            gidControlPanel
            zoomControls
        }
        .padding(.horizontal, 20)
        .padding(.vertical, verticalSizeClass == .compact ? 10 : 18)
        .background(.black)
    }

    private var remoteStatusPanel: some View {
        HStack(spacing: 10) {
            statusMetric(
                title: "Motor",
                value: motorLabel,
                icon: networkClient.desktopState?.motor.armed == true ? "bolt.fill" : "bolt.slash",
                active: networkClient.desktopState?.motor.ready == true
            )
            statusMetric(
                title: "Target",
                value: trackingLabel,
                icon: networkClient.desktopState?.tracking.targetLocked == true ? "scope" : "scope",
                active: networkClient.desktopState?.tracking.targetLocked == true
            )
            statusMetric(
                title: "Error",
                value: errorLabel,
                icon: "point.topleft.down.curvedto.point.bottomright.up",
                active: networkClient.desktopState?.tracking.targetLocked == true
            )
        }
    }

    private var remoteCommandRow: some View {
        HStack(spacing: 10) {
            Button {
                Task { await networkClient.sendControl(action: "auto_track") }
            } label: {
                Label("Auto", systemImage: "scope")
            }
            .buttonStyle(RemoteCommandButtonStyle(tint: .yellow, prominent: true))
            .disabled(!networkActive)

            Button {
                Task { await networkClient.sendControl(action: "stop_motor") }
            } label: {
                Label("STOP", systemImage: "stop.fill")
            }
            .buttonStyle(RemoteCommandButtonStyle(tint: .red, prominent: true))
            .disabled(!networkActive)

            Button {
                guard let selectedGID else { return }
                Task { await networkClient.sendControl(action: "find_gid", gid: selectedGID) }
            } label: {
                Label("Track GID", systemImage: "location.fill")
            }
            .buttonStyle(RemoteCommandButtonStyle(tint: .green, prominent: false))
            .disabled(!networkActive || selectedGID == nil)
        }
    }

    private var homeReturnControls: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button {
                    Task { await controlService.setHome() }
                } label: {
                    Label("Set Home", systemImage: "house.fill")
                }
                .buttonStyle(RemoteCommandButtonStyle(tint: .cyan, prominent: false))
                .disabled(!dockKitManager.isManualControlReady)

                Toggle(
                    "Lost Auto Return",
                    isOn: Binding(
                        get: { controlService.lostAutoReturnEnabled },
                        set: { controlService.lostAutoReturnEnabled = $0 }
                    )
                )
                .toggleStyle(.switch)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack {
                Label(
                    controlService.homeSet || dockKitManager.hasHomePosition ? "Home ready / 再按可覆蓋目前位置" : "設定初始位置",
                    systemImage: controlService.homeSet || dockKitManager.hasHomePosition ? "checkmark.circle.fill" : "house"
                )
                Spacer()
                if controlService.autoReturnPaused {
                    Text("Paused")
                        .fontWeight(.bold)
                        .foregroundStyle(.orange)
                }
            }
            .font(.caption)
            .foregroundStyle(.white.opacity(0.72))
        }
        .padding(.vertical, 2)
    }

    private var gidControlPanel: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Label("GID", systemImage: "number")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.78))
                Spacer()
                Text(selectedGID.map { "Selected \($0)" } ?? "No selection")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.62))
            }

            if sortedGIDs.isEmpty {
                Text("Waiting for GID list")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.55))
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(sortedGIDs) { gid in
                            Button {
                                selectedRemoteGID = gid.gid
                                Task { await networkClient.sendControl(action: "select_gid", gid: gid.gid) }
                            } label: {
                                gidChip(gid)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.vertical, 1)
                }
            }
        }
    }

    private var zoomControls: some View {
        VStack(spacing: 14) {
            HStack(spacing: 12) {
                ForEach(zoomPresets, id: \.self) { factor in
                    Button {
                        cameraSession.setDisplayZoom(factor)
                    } label: {
                        Text(formatZoom(factor))
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(isSelected(factor) ? .black : .yellow)
                            .frame(width: 46, height: 34)
                            .background(
                                isSelected(factor) ? Color.yellow : Color.white.opacity(0.12),
                                in: Capsule()
                            )
                    }
                }
            }

            HStack(spacing: 12) {
                Image(systemName: "minus.magnifyingglass")
                Slider(
                    value: Binding(
                        get: { cameraSession.displayZoomFactor },
                        set: { cameraSession.setDisplayZoom($0) }
                    ),
                    in: cameraSession.minimumDisplayZoomFactor...max(
                        cameraSession.minimumDisplayZoomFactor,
                        cameraSession.maximumDisplayZoomFactor
                    )
                )
                Image(systemName: "plus.magnifyingglass")
            }
            .foregroundStyle(.yellow)

            HStack {
                Label("點按畫面對焦", systemImage: "viewfinder")
                Spacer()
                Text("目前 \(formatZoom(cameraSession.displayZoomFactor))")
                    .fontWeight(.semibold)
                if let zoom = networkClient.lastCommand?.zoomFactor {
                    Text("目標 \(formatZoom(CGFloat(zoom)))")
                }
            }
            .font(.caption)
            .foregroundStyle(.white.opacity(0.75))
        }
    }

    private var zoomPresets: [CGFloat] {
        let candidates: [CGFloat] = [cameraSession.minimumDisplayZoomFactor, 1, 2, 5]
        var result: [CGFloat] = []
        for factor in candidates where factor >= cameraSession.minimumDisplayZoomFactor && factor <= cameraSession.maximumDisplayZoomFactor {
            if !result.contains(where: { abs($0 - factor) < 0.05 }) { result.append(factor) }
        }
        return result
    }

    private var networkActive: Bool {
        networkClient.status == .connected || networkClient.status == .receiving
    }

    private var selectedGID: Int? {
        selectedRemoteGID ?? networkClient.desktopState?.tracking.selectedGid
    }

    private var sortedGIDs: [DesktopState.GIDState] {
        (networkClient.desktopState?.gids ?? []).sorted {
            if $0.selected != $1.selected { return $0.selected }
            if $0.visible != $1.visible { return $0.visible }
            if $0.trackable != $1.trackable { return $0.trackable }
            return $0.gid < $1.gid
        }
    }

    private var motorLabel: String {
        guard let motor = networkClient.desktopState?.motor else { return "Idle" }
        if !motor.armed { return "Off" }
        if motor.ready { return "Ready" }
        if !motor.docked { return "Dock" }
        if !motor.manualReady { return "Manual" }
        return "Wait"
    }

    private var trackingLabel: String {
        guard let tracking = networkClient.desktopState?.tracking else { return "--" }
        if tracking.targetLocked {
            return tracking.targetId.map { "#\($0)" } ?? "Locked"
        }
        return tracking.status.capitalized
    }

    private var errorLabel: String {
        guard let tracking = networkClient.desktopState?.tracking else { return "0.00" }
        return String(format: "%.2f %.2f", tracking.errorX, tracking.errorY)
    }

    private func previewHeight(for size: CGSize) -> CGFloat {
        verticalSizeClass == .compact ? size.height * 0.58 : size.height * 0.54
    }

    private func statusChip(_ text: String, icon: String, active: Bool) -> some View {
        Label(text, systemImage: icon)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .background(active ? Color.black.opacity(0.58) : Color.red.opacity(0.72), in: Capsule())
    }

    private func formatZoom(_ value: CGFloat) -> String {
        value < 1 ? String(format: "%.1f×", value) : String(format: "%.0f×", value)
    }

    private func isSelected(_ factor: CGFloat) -> Bool {
        abs(cameraSession.displayZoomFactor - factor) < 0.08
    }

    private func statusMetric(title: String, value: String, icon: String, active: Bool) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Label(title, systemImage: icon)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.white.opacity(0.58))
            Text(value)
                .font(.caption.monospacedDigit().weight(.bold))
                .foregroundStyle(active ? .yellow : .white)
                .lineLimit(1)
                .minimumScaleFactor(0.72)
        }
        .frame(maxWidth: .infinity, minHeight: 46, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }

    private func gidChip(_ gid: DesktopState.GIDState) -> some View {
        let isSelected = selectedGID == gid.gid || gid.selected
        return VStack(alignment: .leading, spacing: 4) {
            Text(gid.displayName.isEmpty ? "GID \(gid.gid)" : gid.displayName)
                .font(.caption.weight(.bold))
                .lineLimit(1)
                .minimumScaleFactor(0.76)
            HStack(spacing: 5) {
                Image(systemName: gid.trackable ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                Image(systemName: gid.visible ? "eye.fill" : "eye.slash")
                Text("#\(gid.gid)")
                    .font(.caption2.monospacedDigit())
            }
            .font(.caption2)
            .foregroundStyle(isSelected ? .black.opacity(0.70) : .white.opacity(0.62))
        }
        .foregroundStyle(isSelected ? .black : .white)
        .frame(width: 104, height: 54, alignment: .leading)
        .padding(.horizontal, 10)
        .background(
            isSelected ? Color.yellow : Color.white.opacity(gid.trackable ? 0.13 : 0.07),
            in: RoundedRectangle(cornerRadius: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(gid.visible ? Color.green.opacity(0.75) : Color.clear, lineWidth: 1)
        )
    }
}

private extension View {
    func panelStyle() -> some View {
        frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct RemoteCommandButtonStyle: ButtonStyle {
    let tint: Color
    let prominent: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.caption.weight(.bold))
            .lineLimit(1)
            .minimumScaleFactor(0.78)
            .frame(maxWidth: .infinity, minHeight: 42)
            .foregroundStyle(prominent ? .black : tint)
            .background(
                prominent ? tint.opacity(configuration.isPressed ? 0.72 : 1.0) : tint.opacity(configuration.isPressed ? 0.24 : 0.14),
                in: RoundedRectangle(cornerRadius: 8)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(prominent ? Color.clear : tint.opacity(0.45), lineWidth: 1)
            )
            .opacity(configuration.isPressed ? 0.86 : 1)
    }
}
