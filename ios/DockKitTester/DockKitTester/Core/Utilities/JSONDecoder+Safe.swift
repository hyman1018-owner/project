import Foundation

extension JSONDecoder {
    func decodeSafely<T: Decodable>(_ type: T.Type, from data: Data) -> Result<T, Error> {
        Result { try decode(type, from: data) }
    }
}
