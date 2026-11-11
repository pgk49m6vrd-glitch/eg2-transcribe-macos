import AppKit
import Foundation
import Observation
import UniformTypeIdentifiers

/// Which speech-to-text engine is active.
enum EngineKind: String, CaseIterable, Identifiable, Sendable {
    case frenchSteno = "french_steno"
    case coreAI = "coreai_en"
    case multilingual = "multilingual_cjk"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .frenchSteno:
            "Français — Sténo v1 (EmbeddingGemma 2 CTC + 4-Gram LM + ITN)"
        case .coreAI:
            "Anglais Ultra-Rapide (Core AI Reader v2 · 200k mots)"
        case .multilingual:
            "Multilingual zh / en / ja / ko (Core ML + N-Gram v4)"
        }
    }

    var shortTitle: String {
        switch self {
        case .frenchSteno: "Français"
        case .coreAI: "Anglais"
        case .multilingual: "zh/en/ja/ko"
        }
    }

    var downloadSizeDescription: String {
        switch self {
        case .frenchSteno: "~245 Mo (Sténo v1 Large Core ML + 4-Gramme Français lm_fr.bin)"
        case .coreAI: "~933 Mo (Hugging Face: mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI)"
        case .multilingual: "~318 Mo (GitHub Release: TimHu950112/eg2-transcribe-zh-en-ja-ko v4.0)"
        }
    }
}

/// Language selection when the Multilingual (zh/en/ja/ko) engine is active.
enum MultilingualLanguage: String, CaseIterable, Identifiable, Sendable {
    case auto = "auto"
    case zhHant = "zh_hant"
    case zhHans = "zh_hans"
    case en = "en"
    case ja = "ja"
    case ko = "ko"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto: "Auto (zh / en / ja / ko)"
        case .zhHant: "中文 (繁體)"
        case .zhHans: "中文 (简体)"
        case .en: "English"
        case .ja: "日本語"
        case .ko: "한국어"
        }
    }

    var modelLangCode: String {
        switch self {
        case .auto: "auto"
        case .zhHant, .zhHans: "zh"
        case .en: "en"
        case .ja: "ja"
        case .ko: "ko"
        }
    }

    var isSimplifiedChinese: Bool {
        self == .zhHans
    }
}

/// One entry in the recent transcriptions history.
struct TranscriptHistoryItem: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    let date: Date
    let sourceName: String
    let engineTitle: String
    let audioSeconds: Double
    let totalSeconds: Double
    let text: String
    let srt: String?
}

/// Performance breakdown for the most recent transcription.
struct RunStats: Sendable, Equatable {
    let file: String
    let audioSeconds: Double
    let totalSeconds: Double
    let decodeSeconds: Double
    let logMelSeconds: Double
    let windowSeconds: Double
    let stageSeconds: [String]
    let stageSums: [Double]
    let hostSeconds: Double
    let windows: Int
    let form: String
    let overlap: Bool

    var realtimeFactor: Double { audioSeconds / max(totalSeconds, 1e-6) }
}

/// Central observable state for both the macOS menu bar extra and the Studio window.
@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case starting
        case noModel(String)
        case downloading(Double, String)
        case loading(String)
        case ready
        case recording
        case transcribing(window: Int, of: Int)
        case failed(String)
    }

    var phase: Phase = .starting
    var engineKind: EngineKind = {
        let raw = UserDefaults.standard.string(forKey: "engineKind") ?? EngineKind.frenchSteno.rawValue
        return EngineKind(rawValue: raw) ?? .frenchSteno
    }() {
        didSet {
            UserDefaults.standard.set(engineKind.rawValue, forKey: "engineKind")
        }
    }

    var multilingualLanguage: MultilingualLanguage = {
        let raw = UserDefaults.standard.string(forKey: "multilingualLanguage") ?? MultilingualLanguage.auto.rawValue
        return MultilingualLanguage(rawValue: raw) ?? .auto
    }() {
        didSet {
            UserDefaults.standard.set(multilingualLanguage.rawValue, forKey: "multilingualLanguage")
        }
    }

    var useBeamSearch: Bool = {
        if UserDefaults.standard.object(forKey: "useBeamSearch") == nil { return true }
        return UserDefaults.standard.bool(forKey: "useBeamSearch")
    }() {
        didSet {
            UserDefaults.standard.set(useBeamSearch, forKey: "useBeamSearch")
        }
    }

    var useFrenchITN: Bool = {
        if UserDefaults.standard.object(forKey: "useFrenchITN") == nil { return true }
        return UserDefaults.standard.bool(forKey: "useFrenchITN")
    }() {
        didSet {
            UserDefaults.standard.set(useFrenchITN, forKey: "useFrenchITN")
        }
    }

    var autoPaste: Bool = {
        if UserDefaults.standard.object(forKey: "autoPaste") == nil { return true }
        return UserDefaults.standard.bool(forKey: "autoPaste")
    }() {
        didSet {
            UserDefaults.standard.set(autoPaste, forKey: "autoPaste")
        }
    }

    var hotkeyEnabled: Bool = {
        if UserDefaults.standard.object(forKey: "hotkeyEnabled") == nil { return true }
        return UserDefaults.standard.bool(forKey: "hotkeyEnabled")
    }() {
        didSet {
            UserDefaults.standard.set(hotkeyEnabled, forKey: "hotkeyEnabled")
            hotkeyManager.isEnabled = hotkeyEnabled
        }
    }

    var hotkeyPreset: HotkeyPreset = {
        let raw = UserDefaults.standard.string(forKey: "hotkeyPreset") ?? HotkeyPreset.rightOption.rawValue
        return HotkeyPreset(rawValue: raw) ?? .rightOption
    }() {
        didSet {
            UserDefaults.standard.set(hotkeyPreset.rawValue, forKey: "hotkeyPreset")
            hotkeyManager.preset = hotkeyPreset
        }
    }

    var hotkeyMode: HotkeyTriggerMode = {
        let raw = UserDefaults.standard.string(forKey: "hotkeyMode") ?? HotkeyTriggerMode.holdToTalk.rawValue
        return HotkeyTriggerMode(rawValue: raw) ?? .holdToTalk
    }() {
        didSet {
            UserDefaults.standard.set(hotkeyMode.rawValue, forKey: "hotkeyMode")
            hotkeyManager.mode = hotkeyMode
        }
    }

    var hideDockIcon: Bool = UserDefaults.standard.bool(forKey: "hideDockIcon") {
        didSet {
            UserDefaults.standard.set(hideDockIcon, forKey: "hideDockIcon")
            applyDockIconPolicy()
        }
    }

    private(set) var folder: ModelFolder?
    var forms: [PipelineForm] = []
    var formID = ""
    var overlap = true

    var transcript = ""
    var lastSRT: String?
    var stats: RunStats?
    var history: [TranscriptHistoryItem] = []

    var audioLevel: Float = 0
    var recordingSeconds: TimeInterval = 0
    var lastHUDMessage: String?
    var isAccessibilityTrusted: Bool = PasteManager.isAccessibilityTrusted

    var textSize = UserDefaults.standard.double(forKey: "textSize") {
        didSet { UserDefaults.standard.set(textSize, forKey: "textSize") }
    }

    private var frenchTranscriber: FrenchTranscriber?
    private var coreAITranscriber: Transcriber?
    private var multilingualTranscriber: MultilingualTranscriber?
    private let recorder = AudioRecorder()
    private let hotkeyManager = HotkeyManager()
    private let hud = FloatingHUDController()
    private var started = false

    private static let frenchFolderKey = "frenchModelFolderPath"
    private static let folderKey = "modelFolderPath"
    private static let multilingualFolderKey = "multilingualModelFolderPath"
    private static let historyKey = "transcriptHistory"

    init() {
        loadHistory()
        recorder.onMeterUpdate = { [weak self] level, elapsed in
            guard let self else { return }
            self.audioLevel = level
            self.recordingSeconds = elapsed
        }
        setupHotkeys()
    }

    var canTranscribe: Bool {
        if case .ready = phase {
            switch engineKind {
            case .frenchSteno: return frenchTranscriber != nil
            case .coreAI: return coreAITranscriber != nil
            case .multilingual: return multilingualTranscriber != nil
            }
        }
        return false
    }

    var isBusy: Bool {
        switch phase {
        case .loading, .downloading, .transcribing: true
        default: false
        }
    }

    var isRecording: Bool { phase == .recording }

    var form: PipelineForm? { forms.first { $0.id == formID } }

    var menuBarSymbol: String {
        switch phase {
        case .recording: "record.circle.fill"
        case .transcribing, .loading, .downloading: "waveform.badge.ellipsis"
        case .noModel, .failed: "waveform.badge.exclamationmark"
        default: "waveform"
        }
    }

    // MARK: - Startup & Engine switching

    func startIfNeeded() async {
        guard !started else { return }
        started = true
        applyDockIconPolicy()
        refreshAccessibilityStatus()
        await activateCurrentEngine()
    }

    func switchEngine(_ kind: EngineKind) async {
        guard kind != engineKind || !canTranscribe else { return }
        engineKind = kind
        await activateCurrentEngine()
    }

    func activateCurrentEngine() async {
        let defaults = UserDefaults.standard
        switch engineKind {
        case .frenchSteno:
            let preferred = defaults.string(forKey: Self.frenchFolderKey)
                .map { URL(fileURLWithPath: $0, isDirectory: true) }
            await openFrench(FrenchStore.locate(preferred: preferred))
        case .coreAI:
            let preferred = (defaults.string(forKey: "models") ?? defaults.string(forKey: Self.folderKey))
                .map { URL(fileURLWithPath: $0, isDirectory: true) }
            await openCoreAI(ModelStore.locate(preferred: preferred))
        case .multilingual:
            let preferred = defaults.string(forKey: Self.multilingualFolderKey)
                .map { URL(fileURLWithPath: $0, isDirectory: true) }
            await openMultilingual(MultilingualStore.locate(preferred: preferred))
        }
    }

    func applyDockIconPolicy() {
        NSApp?.setActivationPolicy(hideDockIcon ? .accessory : .regular)
    }

    func refreshAccessibilityStatus() {
        isAccessibilityTrusted = PasteManager.isAccessibilityTrusted
    }

    func requestAccessibility() {
        _ = PasteManager.requestAccessibilityPermission()
        PasteManager.openAccessibilitySettings()
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1))
            self.refreshAccessibilityStatus()
        }
    }

    // MARK: - French Sténo v1 (EmbeddingGemma 2 CTC)

    func openFrench(_ root: URL?) async {
        guard let root else {
            phase = .noModel(
                "Modèle Français Sténo v1 non téléchargé (\(EngineKind.frenchSteno.downloadSizeDescription)). Cliquez sur Télécharger pour l'installer automatiquement."
            )
            return
        }
        frenchTranscriber = nil
        phase = .loading("Chargement du lecteur Français Sténo v1…")
        do {
            let loaded = try await FrenchTranscriber.load(root: root) { message in
                Task { @MainActor in self.phase = .loading(message) }
            }
            frenchTranscriber = loaded
            UserDefaults.standard.set(root.path, forKey: Self.frenchFolderKey)
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: - Core AI (English Reader v2)

    func openCoreAI(_ root: URL?) async {
        guard let root else {
            phase = .noModel(
                "Modèle Core AI Anglais v2 non téléchargé (\(EngineKind.coreAI.downloadSizeDescription)). Cliquez sur Télécharger pour l'installer automatiquement."
            )
            return
        }
        do {
            let folder = try ModelFolder(root: root)
            self.folder = folder
            forms = folder.availableForms
            guard let form = folder.defaultForm else {
                phase = .failed("Aucun pipeline de pipeline.json n'est compatible avec les bundles dans \(root.lastPathComponent).")
                return
            }
            formID = form.id
            overlap = folder.defaultOverlap
            UserDefaults.standard.set(root.path, forKey: Self.folderKey)
            await loadCoreAI()
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func loadCoreAI() async {
        guard let folder, let form else { return }
        coreAITranscriber = nil
        phase = .loading("Chargement de \(form.title)…")
        do {
            coreAITranscriber = try await Transcriber.load(folder: folder, form: form) { message in
                Task { @MainActor in self.phase = .loading(message) }
            }
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func selectForm(_ id: String) async {
        guard id != formID, forms.contains(where: { $0.id == id }) else { return }
        formID = id
        await loadCoreAI()
    }

    // MARK: - Multilingual (zh / en / ja / ko v4)

    func openMultilingual(_ root: URL?) async {
        guard let root else {
            phase = .noModel(
                "Modèle Multilingue zh/en/ja/ko v4 non téléchargé (\(EngineKind.multilingual.downloadSizeDescription)). Cliquez sur Télécharger pour l'installer automatiquement."
            )
            return
        }
        multilingualTranscriber = nil
        phase = .loading("Chargement du lecteur multilingue zh/en/ja/ko…")
        do {
            let loaded = try await MultilingualTranscriber.load(root: root) { message in
                Task { @MainActor in self.phase = .loading(message) }
            }
            multilingualTranscriber = loaded
            UserDefaults.standard.set(root.path, forKey: Self.multilingualFolderKey)
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: - Downloading models

    func download() async {
        switch engineKind {
        case .frenchSteno:
            phase = .downloading(0, "Téléchargement du modèle Français Sténo v1 (EmbeddingGemma 2 CTC)…")
            do {
                let root = try await FrenchStore.download { fraction, status in
                    Task { @MainActor in
                        self.phase = .downloading(fraction, status)
                    }
                }
                await openFrench(root)
            } catch {
                phase = .failed(error.localizedDescription)
            }
        case .coreAI:
            phase = .downloading(0, "Téléchargement de EmbeddingGemma-2-Transcribe-CoreAI…")
            do {
                let root = try await ModelStore.download { fraction in
                    Task { @MainActor in
                        self.phase = .downloading(fraction, "Téléchargement Core AI Anglais v2 (\(Int(fraction * 100))%)…")
                    }
                }
                await openCoreAI(root)
            } catch {
                phase = .failed(error.localizedDescription)
            }
        case .multilingual:
            phase = .downloading(0, "Téléchargement du modèle multilingue zh/en/ja/ko…")
            do {
                let root = try await MultilingualStore.download { fraction, status in
                    Task { @MainActor in
                        self.phase = .downloading(fraction, status)
                    }
                }
                await openMultilingual(root)
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    // MARK: - Transcribing & Dictation

    @discardableResult
    func transcribe(_ url: URL, fromDictation: Bool = false) async -> String? {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        phase = .transcribing(window: 0, of: 0)
        transcript = ""
        lastSRT = nil

        do {
            switch engineKind {
            case .frenchSteno:
                guard let transcriber = frenchTranscriber else {
                    phase = .noModel("Téléchargez d'abord le modèle Français Sténo v1.")
                    return nil
                }
                let result = try await transcriber.transcribe(
                    url,
                    useBeamSearch: useBeamSearch,
                    useITN: useFrenchITN
                ) { done, total in
                    Task { @MainActor in self.phase = .transcribing(window: done, of: total) }
                }
                transcript = result.text
                lastSRT = result.srt
                let decodeLabel = result.usedBeamSearch ? "4-Gram Beam" : "Greedy"
                let itnSuffix = result.usedITN ? " + ITN" : ""
                stats = RunStats(
                    file: url.lastPathComponent,
                    audioSeconds: result.audioSeconds,
                    totalSeconds: result.totalSeconds,
                    decodeSeconds: result.decodeSeconds,
                    logMelSeconds: 0,
                    windowSeconds: result.modelSeconds + result.postSeconds,
                    stageSeconds: ["coreml (EG2+CTC)", "\(decodeLabel)\(itnSuffix)"],
                    stageSums: [result.modelSeconds, result.postSeconds],
                    hostSeconds: 0,
                    windows: result.windows,
                    form: "Sténo Français v1 Large (\(decodeLabel)\(itnSuffix))",
                    overlap: false
                )
                appendHistory(
                    sourceName: fromDictation ? "Dictée vocale (FR)" : url.lastPathComponent,
                    engineTitle: "Français Sténo v1 (\(decodeLabel))",
                    audioSeconds: result.audioSeconds,
                    totalSeconds: result.totalSeconds,
                    text: result.text,
                    srt: result.srt
                )
                phase = .ready

            case .coreAI:
                guard let transcriber = coreAITranscriber else {
                    phase = .noModel("Téléchargez d'abord le modèle Core AI.")
                    return nil
                }
                let result = try await transcriber.transcribe(url, overlap: overlap) { done, total in
                    Task { @MainActor in self.phase = .transcribing(window: done, of: total) }
                }
                transcript = result.text
                let names = transcriber.stages.map(\.spec.name)
                let sums = result.run.stageMilliseconds.map { $0.filter(\.isFinite).reduce(0, +) / 1000 }
                stats = RunStats(
                    file: url.lastPathComponent,
                    audioSeconds: result.audioSeconds,
                    totalSeconds: result.totalSeconds,
                    decodeSeconds: result.decodeSeconds,
                    logMelSeconds: result.logMelSeconds,
                    windowSeconds: result.run.wallSeconds,
                    stageSeconds: names,
                    stageSums: sums,
                    hostSeconds: result.run.host.total,
                    windows: result.run.windows,
                    form: transcriber.form.title,
                    overlap: result.run.overlap
                )
                appendHistory(
                    sourceName: fromDictation ? "Dictée vocale (EN)" : url.lastPathComponent,
                    engineTitle: "Core AI EN v2",
                    audioSeconds: result.audioSeconds,
                    totalSeconds: result.totalSeconds,
                    text: result.text,
                    srt: nil
                )
                phase = .ready

            case .multilingual:
                guard let transcriber = multilingualTranscriber else {
                    phase = .noModel("Téléchargez d'abord le modèle multilingue.")
                    return nil
                }
                let result = try await transcriber.transcribe(
                    url,
                    language: multilingualLanguage.modelLangCode,
                    simplifiedChinese: multilingualLanguage.isSimplifiedChinese,
                    useBeamSearch: useBeamSearch
                ) { done, total in
                    Task { @MainActor in self.phase = .transcribing(window: done, of: total) }
                }
                transcript = result.text
                lastSRT = result.srt
                stats = RunStats(
                    file: url.lastPathComponent,
                    audioSeconds: result.audioSeconds,
                    totalSeconds: result.totalSeconds,
                    decodeSeconds: result.decodeSeconds,
                    logMelSeconds: 0,
                    windowSeconds: result.modelSeconds + result.beamSeconds,
                    stageSeconds: ["coreml", "beam (\(result.language))"],
                    stageSums: [result.modelSeconds, result.beamSeconds],
                    hostSeconds: 0,
                    windows: result.windows,
                    form: "Core ML + N-Gram (\(result.language.uppercased()))",
                    overlap: true
                )
                appendHistory(
                    sourceName: fromDictation ? "Dictée vocale (\(result.language))" : url.lastPathComponent,
                    engineTitle: "Multilingue (\(result.language.uppercased()))",
                    audioSeconds: result.audioSeconds,
                    totalSeconds: result.totalSeconds,
                    text: result.text,
                    srt: result.srt
                )
                phase = .ready
            }

            if fromDictation && !transcript.isEmpty {
                let pasted = await PasteManager.copyAndPaste(transcript, autoPaste: autoPaste)
                lastHUDMessage = pasted ? "Collé dans l'app active ✓" : "Copié dans le presse-papiers ✓"
                hud.show(model: self)
                hud.hide(after: 1.4)
            } else {
                hud.hide()
            }
            return transcript
        } catch {
            hud.hide()
            phase = .failed(error.localizedDescription)
            return nil
        }
    }

    // MARK: - Recording & Global Hotkey

    private func setupHotkeys() {
        hotkeyManager.preset = hotkeyPreset
        hotkeyManager.mode = hotkeyMode
        hotkeyManager.isEnabled = hotkeyEnabled

        hotkeyManager.onDictationStart = { [weak self] in
            guard let self else { return }
            Task { await self.startDictationFromHotkey() }
        }
        hotkeyManager.onDictationStop = { [weak self] in
            guard let self else { return }
            Task { await self.stopDictationFromHotkey() }
        }
        hotkeyManager.onDictationToggle = { [weak self] in
            guard let self else { return }
            Task { await self.toggleRecording(fromHotkey: true) }
        }
        hotkeyManager.start()
    }

    func startDictationFromHotkey() async {
        guard !isRecording, canTranscribe else { return }
        await toggleRecording(fromHotkey: true)
    }

    func stopDictationFromHotkey() async {
        guard isRecording else { return }
        await toggleRecording(fromHotkey: true)
    }

    func toggleRecording(fromHotkey: Bool = false) async {
        if isRecording {
            phase = .ready
            audioLevel = 0
            if let url = recorder.stop() {
                await transcribe(url, fromDictation: true)
            } else {
                hud.hide()
            }
            return
        }
        guard canTranscribe else { return }
        guard await recorder.permission() else {
            phase = .failed("Accès au microphone refusé : autorisez-le dans Réglages Système > Confidentialité et sécurité > Microphone.")
            return
        }
        do {
            recordingSeconds = 0
            audioLevel = 0
            try recorder.start()
            phase = .recording
            if fromHotkey {
                hud.show(model: self)
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func dismissError() {
        if frenchTranscriber != nil || coreAITranscriber != nil || multilingualTranscriber != nil {
            phase = .ready
        }
    }

    // MARK: - File Dialogs & Export

    func pickAudioFileAndTranscribe() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .movie]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.prompt = "Transcrire"
        if panel.runModal() == .OK, let url = panel.url {
            Task { await transcribe(url, fromDictation: false) }
        }
    }

    func pickModelFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choisir le dossier du modèle"
        if panel.runModal() == .OK, let url = panel.url {
            Task {
                switch engineKind {
                case .frenchSteno: await openFrench(url)
                case .coreAI: await openCoreAI(url)
                case .multilingual: await openMultilingual(url)
                }
            }
        }
    }

    func saveTextFile(content: String, defaultName: String, ext: String) {
        guard !content.isEmpty else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(defaultName).\(ext)"
        if let utType = UTType(filenameExtension: ext) {
            panel.allowedContentTypes = [utType]
        }
        if panel.runModal() == .OK, let url = panel.url {
            try? content.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    // MARK: - History Persistence

    private func appendHistory(
        sourceName: String,
        engineTitle: String,
        audioSeconds: Double,
        totalSeconds: Double,
        text: String,
        srt: String?
    ) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let item = TranscriptHistoryItem(
            id: UUID(),
            date: Date(),
            sourceName: sourceName,
            engineTitle: engineTitle,
            audioSeconds: audioSeconds,
            totalSeconds: totalSeconds,
            text: trimmed,
            srt: srt
        )
        history.insert(item, at: 0)
        if history.count > 30 {
            history.removeLast(history.count - 30)
        }
        saveHistory()
    }

    func clearHistory() {
        history.removeAll()
        saveHistory()
    }

    private func loadHistory() {
        guard let data = UserDefaults.standard.data(forKey: Self.historyKey),
              let decoded = try? JSONDecoder().decode([TranscriptHistoryItem].self, from: data)
        else { return }
        history = decoded
    }

    private func saveHistory() {
        if let data = try? JSONEncoder().encode(history) {
            UserDefaults.standard.set(data, forKey: Self.historyKey)
        }
    }
}
