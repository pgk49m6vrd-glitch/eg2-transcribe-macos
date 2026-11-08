import AppKit
import SwiftUI

/// Rich macOS Menu Bar popover (`MenuBarExtra`) for one-click or push-to-talk dictation, model downloading,
/// engine/language switching, and quick file transcription.
struct MenuBarView: View {
    @Bindable var model: AppModel
    @Environment(\.openWindow) private var openWindow
    @State private var showSettings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            headerSection
            Divider()
            engineAndLanguageSection
            statusOrDownloadSection
            dictationControlsSection
            autoPasteSection
            if !model.transcript.isEmpty {
                Divider()
                latestTranscriptSection
            }
            if !model.history.isEmpty {
                Divider()
                recentHistorySection
            }
            Divider()
            footerSection
        }
        .padding(14)
        .frame(width: 380)
        .task {
            await model.startIfNeeded()
        }
        .onAppear {
            model.refreshAccessibilityStatus()
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(model: model)
        }
    }

    private var headerSection: some View {
        HStack(spacing: 10) {
            Image(systemName: model.menuBarSymbol)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(model.isRecording ? .red : Color.accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text("EG2 Transcribe")
                    .font(.headline)
                Text("EmbeddingGemma 2 · Speech-to-Text Local")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            statusBadge
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch model.phase {
        case .ready:
            Text("Prêt")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.green.opacity(0.18), in: Capsule())
                .foregroundStyle(.green)
        case .recording:
            Text(String(format: "REC %.1fs", model.recordingSeconds))
                .font(.caption2.monospacedDigit().weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.red.opacity(0.2), in: Capsule())
                .foregroundStyle(.red)
        case .transcribing:
            Text("Calcul…")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.orange.opacity(0.18), in: Capsule())
                .foregroundStyle(.orange)
        case .downloading:
            Text("Téléchargement")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.blue.opacity(0.18), in: Capsule())
                .foregroundStyle(.blue)
        default:
            EmptyView()
        }
    }

    private var engineAndLanguageSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Moteur")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Moteur", selection: engineBinding) {
                    ForEach(EngineKind.allCases) { kind in
                        Text(kind.shortTitle).tag(kind)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 260)
                .disabled(model.isBusy || model.isRecording)
            }

            if model.engineKind == .multilingual {
                HStack {
                    Text("Langue")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Picker("Langue", selection: $model.multilingualLanguage) {
                        ForEach(MultilingualLanguage.allCases) { lang in
                            Text(lang.label).tag(lang)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 260)
                    .disabled(model.isBusy || model.isRecording)
                }
            }
        }
    }

    private var engineBinding: Binding<EngineKind> {
        Binding(
            get: { model.engineKind },
            set: { newKind in
                Task { await model.switchEngine(newKind) }
            }
        )
    }

    @ViewBuilder
    private var statusOrDownloadSection: some View {
        switch model.phase {
        case .starting:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Initialisation…").font(.callout).foregroundStyle(.secondary)
            }
        case .noModel(let message):
            VStack(alignment: .leading, spacing: 8) {
                Label(message, systemImage: "arrow.down.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    Button {
                        Task { await model.download() }
                    } label: {
                        Label("Télécharger le modèle", systemImage: "icloud.and.arrow.down")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)

                    Button("Dossier local…") {
                        model.pickModelFolder()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))

        case .downloading(let fraction, let detail):
            VStack(alignment: .leading, spacing: 6) {
                ProgressView(value: fraction)
                Text(detail)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))

        case .loading(let message):
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            .padding(8)

        case .transcribing(let window, let total):
            VStack(alignment: .leading, spacing: 4) {
                if total > 1 {
                    ProgressView(value: Double(window), total: Double(total))
                    Text("Transcription fenêtre \(window) / \(total)…")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Transcription en cours…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(8)

        case .failed(let message):
            VStack(alignment: .leading, spacing: 6) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                HStack {
                    Button("Réessayer") {
                        Task { await model.activateCurrentEngine() }
                    }
                    .controlSize(.small)
                    Button("Télécharger") {
                        Task { await model.download() }
                    }
                    .controlSize(.small)
                    Button("Ignorer") {
                        model.dismissError()
                    }
                    .controlSize(.small)
                }
            }
            .padding(10)
            .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))

        case .ready, .recording:
            EmptyView()
        }
    }

    private var dictationControlsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button {
                    Task { await model.toggleRecording(fromHotkey: false) }
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: model.isRecording ? "stop.circle.fill" : "mic.fill")
                        Text(model.isRecording ? "Arrêter et transcrire" : "Dicter maintenant")
                            .fontWeight(.semibold)
                        if model.isRecording {
                            Spacer()
                            AudioMeterBars(level: model.audioLevel)
                                .frame(width: 36, height: 16)
                        }
                    }
                    .frame(maxWidth: .infinity, minHeight: 32)
                }
                .buttonStyle(.borderedProminent)
                .tint(model.isRecording ? .red : .accentColor)
                .disabled(!model.canTranscribe && !model.isRecording)

                Button {
                    model.pickAudioFileAndTranscribe()
                } label: {
                    Image(systemName: "doc.badge.plus")
                        .help("Transcrire un fichier audio ou vidéo…")
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.bordered)
                .disabled(!model.canTranscribe)
            }

            HStack(spacing: 6) {
                Image(systemName: "keyboard")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(
                    model.hotkeyMode == .holdToTalk
                        ? "Raccourci global : maintenir \(model.hotkeyPreset.label) pour parler"
                        : "Raccourci global : appuyer sur \(model.hotkeyPreset.label)"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var autoPasteSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle("Coller automatiquement dans l'app active (⌘V)", isOn: $model.autoPaste)
                .font(.caption)

            if model.autoPaste && !model.isAccessibilityTrusted {
                HStack(spacing: 8) {
                    Image(systemName: "hand.raised.fill")
                        .foregroundStyle(.orange)
                        .font(.caption)
                    Text("Autorisez l'Accessibilité pour coller directement dans les autres apps.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Autoriser…") {
                        model.requestAccessibility()
                    }
                    .controlSize(.mini)
                    .buttonStyle(.borderedProminent)
                }
                .padding(8)
                .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private var latestTranscriptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Dernière transcription")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if let stats = model.stats {
                    Text(String(format: "%.1fs · ×%.0f", stats.totalSeconds, stats.realtimeFactor))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            ScrollView {
                Text(model.transcript)
                    .font(.callout)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 96)
            .padding(8)
            .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 6))

            HStack(spacing: 6) {
                Button {
                    PasteManager.copyToClipboard(model.transcript)
                } label: {
                    Label("Copier", systemImage: "doc.on.doc")
                }
                .controlSize(.small)

                Button {
                    model.saveTextFile(content: model.transcript, defaultName: "transcription", ext: "txt")
                } label: {
                    Label(".txt", systemImage: "square.and.arrow.down")
                }
                .controlSize(.small)

                if let srt = model.lastSRT, !srt.isEmpty {
                    Button {
                        model.saveTextFile(content: srt, defaultName: "sous-titres", ext: "srt")
                    } label: {
                        Label(".srt", systemImage: "captions.bubble")
                    }
                    .controlSize(.small)
                }
            }
        }
    }

    private var recentHistorySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Historique récent")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Effacer") {
                    model.clearHistory()
                }
                .font(.caption2)
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }

            ForEach(model.history.prefix(3)) { item in
                Button {
                    PasteManager.copyToClipboard(item.text)
                    model.transcript = item.text
                    model.lastSRT = item.srt
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "doc.on.clipboard")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text(item.text)
                            .font(.caption)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Text(item.engineTitle)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.vertical, 3)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var footerSection: some View {
        HStack {
            Button {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "studio")
            } label: {
                Label("Fenêtre Studio", systemImage: "macwindow")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button {
                showSettings = true
            } label: {
                Label("Réglages…", systemImage: "gearshape")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Spacer()

            Button("Quitter") {
                NSApplication.shared.terminate(nil)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }
}
