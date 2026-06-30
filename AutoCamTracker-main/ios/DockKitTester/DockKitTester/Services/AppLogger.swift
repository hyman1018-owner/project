import Foundation
import Combine

@MainActor
final class AppLogger: ObservableObject {
    @Published private(set) var entries: [AppLogEntry] = []
    private let maximumEntries: Int

    init(maximumEntries: Int = 100) {
        self.maximumEntries = maximumEntries
    }

    func log(_ level: LogLevel, _ message: String) {
        print("[DockKitTester] [\(level.rawValue.uppercased())] \(message)")
        entries.append(AppLogEntry(level: level, message: message))
        if entries.count > maximumEntries {
            entries.removeFirst(entries.count - maximumEntries)
        }
    }

    func clear() {
        entries.removeAll()
    }

    var plainText: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return entries.map {
            "[\(formatter.string(from: $0.time))] [\($0.level.rawValue.uppercased())] \($0.message)"
        }.joined(separator: "\n")
    }
}
