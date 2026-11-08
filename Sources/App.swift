import SwiftUI

@main
struct EG2TranscribeApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(model: model)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: model.menuBarSymbol)
                if model.isRecording {
                    Text(String(format: "%.0fs", model.recordingSeconds))
                        .font(.caption2.monospacedDigit())
                }
            }
        }
        .menuBarExtraStyle(.window)

        SwiftUI.Window("EG2 Transcribe Studio", id: "studio") {
            ContentView(model: model)
        }
        .defaultSize(width: 720, height: 580)
    }
}
