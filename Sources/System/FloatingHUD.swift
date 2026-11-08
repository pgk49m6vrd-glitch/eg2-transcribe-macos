import AppKit
import SwiftUI

/// Non-activating floating HUD pill displayed during global hotkey dictation so the user gets immediate
/// visual feedback without losing focus in their active application.
@MainActor
final class FloatingHUDController {
    private var panel: NSPanel?
    private var dismissTask: Task<Void, Never>?

    func show(model: AppModel) {
        dismissTask?.cancel()
        if panel == nil {
            createPanel(model: model)
        }
        positionPanel()
        panel?.orderFrontRegardless()
    }

    func hide(after delaySeconds: Double = 0) {
        dismissTask?.cancel()
        if delaySeconds <= 0 {
            panel?.orderOut(nil)
            return
        }
        dismissTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(Int(delaySeconds * 1000)))
            guard !Task.isCancelled else { return }
            self?.panel?.orderOut(nil)
        }
    }

    private func createPanel(model: AppModel) {
        let contentRect = NSRect(x: 0, y: 0, width: 320, height: 56)
        let p = NSPanel(
            contentRect: contentRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.isFloatingPanel = true
        p.level = .statusBar
        p.backgroundColor = .clear
        p.isOpaque = false
        p.hasShadow = true
        p.ignoresMouseEvents = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        p.contentView = NSHostingView(rootView: FloatingHUDView(model: model))
        self.panel = p
    }

    private func positionPanel() {
        guard let panel else { return }
        let screen = NSScreen.main ?? NSScreen.screens.first
        guard let visible = screen?.visibleFrame else { return }
        let width: CGFloat = 320
        let height: CGFloat = 56
        let x = visible.midX - width / 2
        let y = visible.minY + 48
        panel.setFrame(NSRect(x: x, y: y, width: width, height: height), display: true)
    }
}

struct FloatingHUDView: View {
    @Bindable var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            statusIcon
            VStack(alignment: .leading, spacing: 2) {
                Text(headline)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(subline)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 4)
            if model.isRecording {
                AudioMeterBars(level: model.audioLevel)
                    .frame(width: 44, height: 22)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(width: 320, height: 56)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(model.isRecording ? Color.red.opacity(0.6) : Color.primary.opacity(0.15), lineWidth: 1)
        )
    }

    @ViewBuilder
    private var statusIcon: some View {
        if model.isRecording {
            Circle()
                .fill(.red)
                .frame(width: 12, height: 12)
                .shadow(color: .red.opacity(0.8), radius: 6)
        } else if case .transcribing = model.phase {
            ProgressView()
                .controlSize(.small)
        } else {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.system(size: 16))
        }
    }

    private var headline: String {
        if model.isRecording {
            return String(format: "Écoute en cours… %.1f s", model.recordingSeconds)
        }
        if case .transcribing(let w, let total) = model.phase {
            return total > 1 ? "Transcription (\(w)/\(total))…" : "Transcription EmbeddingGemma 2…"
        }
        if let lastAction = model.lastHUDMessage {
            return lastAction
        }
        return "Prêt"
    }

    private var subline: String {
        if model.isRecording {
            return model.hotkeyMode == .holdToTalk
                ? "Relâchez \(model.hotkeyPreset.shortBadge) pour transcrire"
                : "Appuyez sur \(model.hotkeyPreset.shortBadge) pour terminer"
        }
        return model.engineKind.shortTitle
    }
}

/// Animated 5-bar voice level indicator.
struct AudioMeterBars: View {
    let level: Float

    private let multipliers: [Float] = [0.45, 0.75, 1.0, 0.85, 0.55]

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(0..<multipliers.count, id: \.self) { idx in
                let barHeight = max(4, CGFloat(min(1, level * multipliers[idx])) * 20)
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.red.gradient)
                    .frame(width: 4, height: barHeight)
                    .animation(.easeOut(duration: 0.08), value: level)
            }
        }
    }
}
