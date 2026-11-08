import SwiftUI
import UniformTypeIdentifiers

/// Main Studio window supporting drag-and-drop audio/video transcription, live dictation, model downloading,
/// benchmark breakdown, and history.
struct ContentView: View {
    @Bindable var model: AppModel
    @State private var showSettings = false
    @State private var isDropTargeted = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    engineHeaderCard
                    statusCard
                    actionButtons
                    if let stats = model.stats {
                        StatsView(stats: stats, size: model.textSize)
                    }
                    TranscriptView(
                        text: model.transcript,
                        srt: model.lastSRT,
                        file: model.stats?.file,
                        size: model.textSize,
                        onSaveTxt: {
                            model.saveTextFile(content: model.transcript, defaultName: "transcription", ext: "txt")
                        },
                        onSaveSrt: {
                            if let srt = model.lastSRT {
                                model.saveTextFile(content: srt, defaultName: "sous-titres", ext: "srt")
                            }
                        }
                    )
                    if !model.history.isEmpty {
                        historySection
                    }
                }
                .padding(20)
            }
            .navigationTitle("EG2 Transcribe Studio")
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        model.pickModelFolder()
                    } label: {
                        Label("Dossier modèle…", systemImage: "folder")
                    }
                    .disabled(model.isBusy)

                    Button {
                        showSettings = true
                    } label: {
                        Label("Réglages", systemImage: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(model: model)
            }
            .overlay {
                if isDropTargeted {
                    RoundedRectangle(cornerRadius: 14)
                        .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8]))
                        .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
                        .padding(10)
                        .overlay {
                            Label("Déposez le fichier audio/vidéo pour le transcrire", systemImage: "waveform.badge.plus")
                                .font(.title3.weight(.semibold))
                                .padding(16)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                }
            }
            .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
                handleDrop(providers)
            }
        }
        .frame(minWidth: 680, minHeight: 540)
        .task {
            await model.startIfNeeded()
        }
    }

    private var engineHeaderCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Moteur actif", systemImage: "cpu")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Picker("Moteur", selection: engineBinding) {
                    ForEach(EngineKind.allCases) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 420)
                .disabled(model.isBusy || model.isRecording)
            }

            if model.engineKind == .multilingual {
                HStack {
                    Label("Langue", systemImage: "globe")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Picker("Langue", selection: $model.multilingualLanguage) {
                        ForEach(MultilingualLanguage.allCases) { lang in
                            Text(lang.label).tag(lang)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 420)
                }
            }

            HStack(spacing: 14) {
                Label(
                    model.hotkeyMode == .holdToTalk
                        ? "Maintenir \(model.hotkeyPreset.label) pour dicter partout"
                        : "Appuyer sur \(model.hotkeyPreset.label) pour dicter partout",
                    systemImage: "keyboard"
                )
                .font(.caption)
                .foregroundStyle(.secondary)

                Spacer()

                Toggle("Collage auto (⌘V)", isOn: $model.autoPaste)
                    .font(.caption)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    private var engineBinding: Binding<EngineKind> {
        Binding(
            get: { model.engineKind },
            set: { kind in Task { await model.switchEngine(kind) } }
        )
    }

    @ViewBuilder
    private var statusCard: some View {
        switch model.phase {
        case .starting:
            HStack(spacing: 10) {
                ProgressView()
                Text("Démarrage…")
            }
        case .noModel(let message):
            VStack(alignment: .leading, spacing: 10) {
                Label(message, systemImage: "icloud.and.arrow.down")
                    .font(.callout)
                HStack(spacing: 12) {
                    Button {
                        Task { await model.download() }
                    } label: {
                        Label("Télécharger maintenant", systemImage: "arrow.down.circle.fill")
                            .frame(minHeight: 34)
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Choisir un dossier local…") {
                        model.pickModelFolder()
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(14)
            .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))

        case .downloading(let fraction, let detail):
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(value: fraction)
                Text(detail)
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))

        case .loading(let message):
            HStack(spacing: 10) {
                ProgressView()
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        case .transcribing(let window, let total):
            VStack(alignment: .leading, spacing: 6) {
                if total > 1 {
                    ProgressView(value: Double(window), total: Double(total))
                    Text("Transcription fenêtre \(window) sur \(total)…")
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else {
                    HStack(spacing: 10) {
                        ProgressView()
                        Text("Transcription en cours…")
                    }
                }
            }
        case .ready, .recording:
            if model.engineKind == .coreAI, let form = model.form {
                Label("\(form.title) · overlap \(model.overlap ? "actif" : "inactif")", systemImage: "checkmark.circle.fill")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else if model.engineKind == .multilingual {
                Label(
                    "Lecteur Multilingue v4 prêt (\(model.multilingualLanguage.label), \(model.useBeamSearch ? "N-Gram Beam Search" : "Greedy"))",
                    systemImage: "checkmark.circle.fill"
                )
                .font(.callout)
                .foregroundStyle(.secondary)
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                HStack {
                    Button("OK") { model.dismissError() }
                    Button("Télécharger le modèle") { Task { await model.download() } }
                    Button("Choisir le dossier…") { model.pickModelFolder() }
                }
            }
            .padding(12)
            .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var actionButtons: some View {
        HStack(spacing: 12) {
            Button {
                model.pickAudioFileAndTranscribe()
            } label: {
                Label("Choisir un fichier audio/vidéo…", systemImage: "waveform")
                    .frame(maxWidth: .infinity, minHeight: 42)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.canTranscribe)

            Button {
                Task { await model.toggleRecording(fromHotkey: false) }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: model.isRecording ? "stop.circle.fill" : "mic.fill")
                    Text(
                        model.isRecording
                            ? String(format: "Arrêter (%.1f s)", model.recordingSeconds)
                            : "Dicter au microphone"
                    )
                    if model.isRecording {
                        AudioMeterBars(level: model.audioLevel)
                            .frame(width: 40, height: 18)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 42)
            }
            .buttonStyle(.bordered)
            .tint(model.isRecording ? .red : .accentColor)
            .disabled(!model.canTranscribe && !model.isRecording)
        }
    }

    private var historySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Historique des transcriptions")
                    .font(.headline)
                Spacer()
                Button("Tout effacer") {
                    model.clearHistory()
                }
                .font(.caption)
            }

            ForEach(model.history.prefix(10)) { item in
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text(item.sourceName)
                                .font(.caption.weight(.semibold))
                            Text(item.engineTitle)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.quaternary, in: Capsule())
                            Text(String(format: "%.1fs audio en %.2fs", item.audioSeconds, item.totalSeconds))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Text(item.text)
                            .font(.callout)
                            .lineLimit(2)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    Button {
                        PasteManager.copyToClipboard(item.text)
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .help("Copier dans le presse-papiers")
                }
                .padding(10)
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard model.canTranscribe, let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            let url: URL?
            if let data = item as? Data {
                url = URL(dataRepresentation: data, relativeTo: nil)
            } else if let directURL = item as? URL {
                url = directURL
            } else {
                url = nil
            }
            if let url {
                Task { @MainActor in
                    await model.transcribe(url, fromDictation: false)
                }
            }
        }
        return true
    }
}
