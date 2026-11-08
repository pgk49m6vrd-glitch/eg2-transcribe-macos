import CoreML
import Foundation

/// Runs the multilingual (zh / en / ja / ko) EmbeddingGemma 2 + CTC + N-Gram reader (`JoinTranscribe.mlpackage`)
/// via Core ML, with 30 s overlapping windows, language auto-detection, and prefix beam search.
final class MultilingualTranscriber: @unchecked Sendable {
    static let slotSamples = 320   // 20 ms at 16 kHz
    static let edgeSlots = 50      // 1.0 s dropped at inner window edges
    static let tailSlots = 25      // 0.5 s tail allowance

    struct Meta: Decodable, Sendable {
        let version: Int
        let sample_rate: Int
        let window_samples: Int
        let slot_ms: Int
        let topk: Int
        let langs: [String]
        let units: [[String]]
        let lm: [String: [Float]]
    }

    struct Result: Sendable {
        let language: String
        let audioSeconds: Double
        let decodeSeconds: Double
        let modelSeconds: Double
        let beamSeconds: Double
        let windows: Int
        let lines: [TranscriptLine]
        let text: String
        let srt: String

        var totalSeconds: Double { decodeSeconds + modelSeconds + beamSeconds }
        var realtimeFactor: Double { audioSeconds / max(totalSeconds, 1e-6) }
    }

    let root: URL
    let meta: Meta
    let decoder: TranscribeDecoder
    let hans: SimplifiedChineseConverter
    private let model: MLModel
    private let lms: [String: NGramLM]
    private let lock = NSLock()

    var windowSamples: Int { meta.window_samples }
    var topK: Int { meta.topk }
    var slotsPerWindow: Int { windowSamples / Self.slotSamples }
    var strideSamples: Int { windowSamples - 2 * Self.edgeSlots * Self.slotSamples }

    private init(
        root: URL,
        meta: Meta,
        model: MLModel,
        decoder: TranscribeDecoder,
        hans: SimplifiedChineseConverter,
        lms: [String: NGramLM]
    ) {
        self.root = root
        self.meta = meta
        self.model = model
        self.decoder = decoder
        self.hans = hans
        self.lms = lms
    }

    static func load(root: URL, status: @Sendable (String) -> Void = { _ in }) async throws -> MultilingualTranscriber {
        status("Lecture de transcribe_meta.json…")
        let metaURL = root.appending(path: "transcribe_meta.json")
        let metaData = try Data(contentsOf: metaURL)
        let meta = try JSONDecoder().decode(Meta.self, from: metaData)

        let hansURL = root.appending(path: "zh_hans.json")
        let hans = try SimplifiedChineseConverter(url: hansURL)
        let decoder = TranscribeDecoder(units: meta.units, topK: meta.topk)

        let compiledCacheURL = root.appending(path: "JoinTranscribe.mlmodelc", directoryHint: .isDirectory)
        let packageURL = root.appending(path: "JoinTranscribe.mlpackage", directoryHint: .isDirectory)

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledCacheURL.path) {
            modelURL = compiledCacheURL
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            status("Compilation Core ML de JoinTranscribe.mlpackage (1re ouverture)…")
            let tempCompiled = try await MLModel.compileModel(at: packageURL)
            try? FileManager.default.removeItem(at: compiledCacheURL)
            do {
                try FileManager.default.moveItem(at: tempCompiled, to: compiledCacheURL)
                modelURL = compiledCacheURL
            } catch {
                modelURL = tempCompiled
            }
        } else {
            throw TranscribeError.model("JoinTranscribe.mlpackage introuvable dans \(root.path)")
        }

        status("Chargement du modèle Core ML multilingue (GPU)…")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        let mlModel = try await MLModel.load(contentsOf: modelURL, configuration: config)

        status("Chargement des modèles de langue N-gram (zh / en / ja / ko)…")
        var loadedLMs: [String: NGramLM] = [:]
        for lang in meta.langs {
            let lmURL = root.appending(path: "lm_\(lang).bin")
            if let lm = NGramLM(file: lmURL) {
                loadedLMs[lang] = lm
            }
        }

        return MultilingualTranscriber(
            root: root,
            meta: meta,
            model: mlModel,
            decoder: decoder,
            hans: hans,
            lms: loadedLMs
        )
    }

    /// Transcribes an audio file using the multilingual reader.
    /// - Parameters:
    ///   - url: Audio file URL
    ///   - language: `"auto"`, `"zh"`, `"en"`, `"ja"`, or `"ko"`
    ///   - simplifiedChinese: When `true`, converts Traditional Chinese output to Simplified Chinese
    ///   - useBeamSearch: When `true`, runs CTC prefix beam search with the language's N-gram LM
    func transcribe(
        _ url: URL,
        language: String = "auto",
        simplifiedChinese: Bool = false,
        useBeamSearch: Bool = true,
        progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }
    ) async throws -> Result {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let decoded = try AudioDecoder.decode(url)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let decodeSeconds = Double(t1 - t0) / 1e9

        let forcedLang = meta.langs.contains(language) ? language : nil
        let langCode: Int32 = forcedLang.flatMap { Int32((meta.langs.firstIndex(of: $0) ?? 0) + 1) } ?? 0

        let samples = decoded.samples
        let totalWindows = max(1, (max(0, samples.count - windowSamples)) / strideSamples + (samples.count > windowSamples ? 2 : 1))

        var allIDs: [Int32] = []
        var allLogP: [Float] = []
        var pos = 0
        var w = 0

        let tModelStart = DispatchTime.now().uptimeNanoseconds
        while true {
            let valid = min(windowSamples, max(0, samples.count - pos))
            if valid == 0 && w > 0 { break }

            let (windowIDs, windowLogP) = try predictWindow(samples: samples, offset: pos, validCount: valid, langCode: langCode)
            let usable = min(slotsPerWindow, (valid + Self.slotSamples - 1) / Self.slotSamples + Self.tailSlots)
            let isLast = (pos + valid >= samples.count) || valid < windowSamples
            let a = (w == 0) ? 0 : Self.edgeSlots
            let b = isLast ? usable : (slotsPerWindow - Self.edgeSlots)

            if a < b {
                let sliceRange = (a * topK)..<(b * topK)
                allIDs.append(contentsOf: windowIDs[sliceRange])
                allLogP.append(contentsOf: windowLogP[sliceRange])
            }

            w += 1
            progress(w, max(w, totalWindows))
            if isLast { break }
            pos += strideSamples
        }
        let modelSeconds = Double(DispatchTime.now().uptimeNanoseconds - tModelStart) / 1e9

        let tBeamStart = DispatchTime.now().uptimeNanoseconds
        let greedySymbols = decoder.greedy(ids: allIDs)
        let chosenLang = forcedLang ?? TranscribeDecoder.guessLanguage(greedySymbols, among: Set(meta.langs))

        let finalSymbols: [TranscribeDecoder.Emitted]
        if useBeamSearch, !allIDs.isEmpty, let lm = lms[chosenLang] {
            let weights = meta.lm[chosenLang] ?? [0.3, 1.0]
            let alpha = weights.count > 0 ? weights[0] : 0.3
            let beta = weights.count > 1 ? weights[1] : 1.0
            finalSymbols = decoder.beam(ids: allIDs, logp: allLogP, lm: lm, alpha: alpha, beta: beta)
        } else {
            finalSymbols = greedySymbols
        }

        let lines = TranscriptText.lines(finalSymbols, simplified: simplifiedChinese, hans: hans)
        let beamSeconds = Double(DispatchTime.now().uptimeNanoseconds - tBeamStart) / 1e9

        let joiner = (chosenLang == "en" || chosenLang == "ko") ? " " : ""
        let fullText = lines.map(\.text).joined(separator: joiner.isEmpty ? "\n" : " ")
        let srt = TranscriptText.formatSRT(lines)

        return Result(
            language: chosenLang,
            audioSeconds: decoded.seconds,
            decodeSeconds: decodeSeconds,
            modelSeconds: modelSeconds,
            beamSeconds: beamSeconds,
            windows: w,
            lines: lines,
            text: fullText,
            srt: srt
        )
    }

    private func predictWindow(
        samples: [Float],
        offset: Int,
        validCount: Int,
        langCode: Int32
    ) throws -> (ids: [Int32], logp: [Float]) {
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: windowSamples)], dataType: .float32)
        let audioPtr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
        audioPtr.initialize(repeating: 0, count: windowSamples)
        if validCount > 0 {
            samples.withUnsafeBufferPointer { src in
                audioPtr.update(from: src.baseAddress! + offset, count: validCount)
            }
        }

        let langArray = try MLMultiArray(shape: [1], dataType: .int32)
        langArray.dataPointer.assumingMemoryBound(to: Int32.self)[0] = langCode

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArrayValue: audioArray),
            "lang": MLFeatureValue(multiArrayValue: langArray),
        ])

        lock.lock()
        let out: MLFeatureProvider
        do {
            out = try model.prediction(from: provider)
        } catch {
            lock.unlock()
            throw error
        }
        lock.unlock()

        guard let idsMulti = out.featureValue(for: "ids")?.multiArrayValue,
              let logpMulti = out.featureValue(for: "logp")?.multiArrayValue
        else {
            throw TranscribeError.contract("JoinTranscribe n'a pas retourné les tenseurs ids / logp")
        }

        let ids = Self.extractInt32s(idsMulti, slots: slotsPerWindow, topK: topK)
        let logp = Self.extractFloats(logpMulti, slots: slotsPerWindow, topK: topK)
        return (ids, logp)
    }

    private static func extractInt32s(_ array: MLMultiArray, slots: Int, topK: Int) -> [Int32] {
        var out = [Int32](repeating: 0, count: slots * topK)
        let strides = array.strides.map(\.intValue)
        let ptr = array.dataPointer.assumingMemoryBound(to: Int32.self)
        if strides.count == 3 {
            let s1 = strides[1], s2 = strides[2]
            for t in 0..<slots {
                let rowOffset = t * s1
                for k in 0..<topK {
                    out[t * topK + k] = ptr[rowOffset + k * s2]
                }
            }
        } else {
            for i in 0..<(slots * topK) { out[i] = ptr[i] }
        }
        return out
    }

    private static func extractFloats(_ array: MLMultiArray, slots: Int, topK: Int) -> [Float] {
        var out = [Float](repeating: 0, count: slots * topK)
        let strides = array.strides.map(\.intValue)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        if strides.count == 3 {
            let s1 = strides[1], s2 = strides[2]
            for t in 0..<slots {
                let rowOffset = t * s1
                for k in 0..<topK {
                    out[t * topK + k] = ptr[rowOffset + k * s2]
                }
            }
        } else {
            for i in 0..<(slots * topK) { out[i] = ptr[i] }
        }
        return out
    }
}
