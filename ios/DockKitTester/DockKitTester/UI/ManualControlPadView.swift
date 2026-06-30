import SwiftUI

struct ManualControlPadView: View {
    let isDocked: Bool
    let isManualControlReady: Bool
    let onCommand: (GimbalCommand) async -> Void

    var body: some View {
        VStack(spacing: 12) {
            Text("Manual Control")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            commandButton("Tilt Up", icon: "chevron.up", command: .tiltUp)

            HStack(spacing: 12) {
                commandButton("Pan Left", icon: "chevron.left", command: .panLeft)

                Button {
                    Task { await onCommand(.stop) }
                } label: {
                    Label("STOP", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity, minHeight: 50)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .disabled(!isManualControlReady)

                commandButton("Pan Right", icon: "chevron.right", command: .panRight)
            }

            commandButton("Tilt Down", icon: "chevron.down", command: .tiltDown)

            Button {
                Task { await onCommand(.recenter) }
            } label: {
                Label("Recenter", systemImage: "scope")
                    .frame(maxWidth: .infinity, minHeight: 42)
            }
            .buttonStyle(.bordered)
            .disabled(!isManualControlReady)

            if !isManualControlReady {
                Text(
                    isDocked
                        ? "Tap Enter Manual Mode and wait for Tracking OFF verification."
                        : "Dock the accessory to unlock Manual Mode."
                )
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }

    private func commandButton(_ title: String, icon: String, command: GimbalCommand) -> some View {
        Button {
            Task { await onCommand(command) }
        } label: {
            Label(title, systemImage: icon)
                .frame(maxWidth: .infinity, minHeight: 42)
        }
        .buttonStyle(.bordered)
        .disabled(!isManualControlReady)
    }
}
