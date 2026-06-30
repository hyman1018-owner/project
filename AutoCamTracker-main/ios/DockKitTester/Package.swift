// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DockKitTesterCore",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "DockKitTesterCore", targets: ["DockKitTesterCore"])
    ],
    targets: [
        .target(
            name: "DockKitTesterCore",
            path: "DockKitTester/Core"
        ),
        .testTarget(
            name: "DockKitTesterCoreTests",
            dependencies: ["DockKitTesterCore"],
            path: "DockKitTesterTests"
        )
    ]
)
