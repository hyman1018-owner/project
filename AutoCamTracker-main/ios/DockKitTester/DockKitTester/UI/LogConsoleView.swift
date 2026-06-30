import SwiftUI
import UIKit

struct LogConsoleView: View {
    @ObservedObject var logger: AppLogger

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("API Log (latest 100)")
                    .font(.headline)
                Spacer()
                Button("Copy") {
                    UIPasteboard.general.string = logger.plainText
                }
                Button("Clear") {
                    logger.clear()
                }
            }

            if logger.entries.isEmpty {
                Text("No log entries yet.")
                    .foregroundStyle(.secondary)
            } else {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(logger.entries.reversed()) { entry in
                        Text("[\(entry.time.formatted(date: .omitted, time: .standard))] [\(entry.level.rawValue.uppercased())] \(entry.message)")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(color(for: entry.level))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.88), in: RoundedRectangle(cornerRadius: 14))
        .foregroundStyle(.white)
    }

    private func color(for level: LogLevel) -> Color {
        switch level {
        case .info: .white
        case .success: .green
        case .warning: .yellow
        case .error: .red
        }
    }
}
