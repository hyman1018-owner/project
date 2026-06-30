import Foundation

struct AppLogEntry: Identifiable, Equatable {
    let id: UUID
    let time: Date
    let level: LogLevel
    let message: String

    init(id: UUID = UUID(), time: Date = Date(), level: LogLevel, message: String) {
        self.id = id
        self.time = time
        self.level = level
        self.message = message
    }
}

enum LogLevel: String, Equatable {
    case info
    case success
    case warning
    case error
}
