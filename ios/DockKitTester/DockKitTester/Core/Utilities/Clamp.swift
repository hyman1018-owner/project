func clamp<T: Comparable>(_ value: T, min minimum: T, max maximum: T) -> T {
    Swift.min(Swift.max(value, minimum), maximum)
}
