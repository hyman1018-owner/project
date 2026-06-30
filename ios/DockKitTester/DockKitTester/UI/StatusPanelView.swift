import SwiftUI

struct StatusPanelView: View {
    @ObservedObject var manager: DockKitManager
    let onTestVelocity: () async -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 12, height: 12)
                Text("Accessory: \(manager.accessoryStatus.title)")
                    .font(.headline)
                Spacer()
                Text(trackingTitle)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(trackingColor)
            }

            if let name = manager.accessoryName {
                Text(name)
                    .font(.footnote)
                    .textSelection(.enabled)
            }

            if let trackingButtonEnabled = manager.trackingButtonEnabled {
                Label(
                    "Physical Tracking Button: \(trackingButtonEnabled ? "ON" : "OFF")",
                    systemImage: "dot.radiowaves.left.and.right"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            HStack {
                Button("Enter Manual Mode") {
                    Task { await manager.disableSystemTracking() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!manager.isDocked || manager.isManualControlReady || manager.isManualModeTransitioning)
            }

            Label("iPhone 人像追蹤保持關閉，畫面只交給電腦辨識。", systemImage: "person.crop.circle.badge.xmark")
                .font(.caption)
                .foregroundStyle(.secondary)

            if manager.isManualModeTransitioning {
                ProgressView("Verifying DockKit tracking state…")
                    .font(.caption)
            }

            HStack {
                Button("Test Angular Velocity") {
                    Task { await onTestVelocity() }
                }
                Button("Test Orientation") {
                    Task { await manager.recenter() }
                }
            }
            .buttonStyle(.bordered)
            .disabled(!manager.isManualControlReady)

            Button {
                Task { await manager.runCapabilityDiagnostics() }
            } label: {
                Label(
                    manager.isCapabilityTestRunning ? "Testing Capabilities…" : "Run Capability Diagnostics",
                    systemImage: "stethoscope"
                )
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(!manager.isManualControlReady || manager.isCapabilityTestRunning)

            if let error = manager.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }

    private var statusColor: Color {
        switch manager.accessoryStatus {
        case .notFound: .secondary
        case .connecting: .orange
        case .docked: .green
        case .error: .red
        }
    }

    private var trackingTitle: String {
        if manager.isManualModeTransitioning {
            return "Switching…"
        }
        switch manager.isSystemTrackingEnabled {
        case true: return "System Tracking ON"
        case false: return "Manual Mode"
        case nil: return "Tracking Unknown"
        }
    }

    private var trackingColor: Color {
        if manager.isManualModeTransitioning {
            return .orange
        }
        switch manager.isSystemTrackingEnabled {
        case true: return .blue
        case false: return .green
        case nil: return .secondary
        }
    }
}
