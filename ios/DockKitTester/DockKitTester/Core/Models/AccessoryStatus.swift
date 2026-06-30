import Foundation

enum AccessoryStatus: Equatable {
    case notFound
    case connecting
    case docked
    case error

    var title: String {
        switch self {
        case .notFound: "Not Found"
        case .connecting: "Connecting"
        case .docked: "Docked"
        case .error: "Error"
        }
    }
}
