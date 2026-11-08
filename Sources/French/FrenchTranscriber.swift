import CoreML
import Foundation

/// Runs the French Sténo EmbeddingGemma 2 + CTC Reader (`StenoFr.mlpackage`) via Core ML.
/// Outputs punctuated, capitalized French directly from 30,002 Gemma word pieces in a single pass.
final class FrenchTranscriber: @unchecked Sendable {
    struct Meta: Decodable, Sendable {
        let version: Int
        let model_id: String?
        let sample_rate: Int
        let window_samples: Int
        let encoder_frames: Int
        let frame_ms: Int
        let topk: Int
        let blank_id: Int
        let unk_id: Int
        let symbols: [String]
    }

    struct WordTiming: Sendable, Equatable {
        let word: String
        let start: Double
        let end: Double
        let confidence: Double
    }

    struct Result: Sendable {
        let audioSeconds: Double
        let decodeSeconds: Double
        let modelSeconds: Double
        let postSeconds: Double
        let windows: Int
        let words: [WordTiming]
        let text: String
        let srt: String

        var totalSeconds: Double { decodeSeconds + modelSeconds + postSeconds }
        var realtimeFactor: Double { audioSeconds / max(totalSeconds, 1e-6) }
    }

    let root: URL
    let meta: Meta
    private let model: MLModel
    private let lock = NSLock()

    var windowSamples: Int { meta.window_samples }
    var encoderFrames: Int { meta.encoder_frames }
    var topK: Int { meta.topk }
    var frameSeconds: Double { Double(meta.frame_ms) / 1000.0 }

    private init(root: URL, meta: Meta, model: MLModel) {
        self.root = root
        self.meta = meta
        self.model = model
    }

    static func load(root: URL, status: @Sendable (String) -> Void = { _ in }) async throws -> FrenchTranscriber {
        status("Lecture du vocabulaire français (steno_fr_meta.json · 30 002 pièces)…")
        let metaURL = root.appending(path: "steno_fr_meta.json")
        let metaData = try Data(contentsOf: metaURL)
        let meta = try JSONDecoder().decode(Meta.self, from: metaData)

        let compiledCacheURL = root.appending(path: "StenoFr.mlmodelc", directoryHint: .isDirectory)
        let packageURL = root.appending(path: "StenoFr.mlpackage", directoryHint: .isDirectory)

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledCacheURL.path) {
            modelURL = compiledCacheURL
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            status("Compilation Core ML de StenoFr.mlpackage (1re ouverture)…")
            let tempCompiled = try await MLModel.compileModel(at: packageURL)
            try? FileManager.default.removeItem(at: compiledCacheURL)
            do {
                try FileManager.default.moveItem(at: tempCompiled, to: compiledCacheURL)
                modelURL = compiledCacheURL
            } catch {
                modelURL = tempCompiled
            }
        } else {
            throw TranscribeError.model("StenoFr.mlpackage introuvable dans \(root.path)")
        }

        status("Chargement du modèle Français Sténo v1 (EmbeddingGemma 2 CTC · GPU)…")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        let mlModel = try await MLModel.load(contentsOf: modelURL, configuration: config)

        return FrenchTranscriber(root: root, meta: meta, model: mlModel)
    }

    /// Transcribes an audio or video file into punctuated French text and SRT subtitles.
    func transcribe(
        _ url: URL,
        progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }
    ) async throws -> Result {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let decoded = try AudioDecoder.decode(url)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let decodeSeconds = Double(t1 - t0) / 1e9

        let samples = decoded.samples
        let blocks = Self.splitPoints(samples: samples, sampleRate: meta.sample_rate)
        let totalWindows = max(1, blocks.count)

        var allWords: [WordTiming] = []
        let tModelStart = DispatchTime.now().uptimeNanoseconds
        var postNanoseconds: UInt64 = 0

        for (index, block) in blocks.enumerated() {
            let validSamples = block.end - block.start
            if validSamples <= 0 { continue }

            let validFrames = Self.computeEncoderFrames(
                sampleCount: validSamples,
                maxFrames: encoderFrames
            )
            let (topIDs, topProbs) = try predictBlock(
                samples: samples,
                start: block.start,
                count: validSamples,
                validFrames: validFrames
            )

            let tPost0 = DispatchTime.now().uptimeNanoseconds
            let offsetSeconds = Double(block.start) / Double(meta.sample_rate)
            let blockWords = greedyWords(
                ids: topIDs,
                probs: topProbs,
                offsetSeconds: offsetSeconds
            )
            allWords.append(contentsOf: blockWords)
            postNanoseconds += DispatchTime.now().uptimeNanoseconds - tPost0

            progress(index + 1, totalWindows)
        }

        let totalLoopSeconds = Double(DispatchTime.now().uptimeNanoseconds - tModelStart) / 1e9
        let postSeconds = Double(postNanoseconds) / 1e9
        let modelSeconds = max(0, totalLoopSeconds - postSeconds)

        let fullText = allWords.map(\.word).joined(separator: " ")
        let srt = Self.formatSRT(words: allWords)

        return Result(
            audioSeconds: decoded.seconds,
            decodeSeconds: decodeSeconds,
            modelSeconds: modelSeconds,
            postSeconds: postSeconds,
            windows: blocks.count,
            words: allWords,
            text: fullText,
            srt: srt
        )
    }

    // MARK: - Core ML Prediction

    private func predictBlock(
        samples: [Float],
        start: Int,
        count: Int,
        validFrames: Int
    ) throws -> (ids: [Int], probs: [Double]) {
        let clampedCount = min(windowSamples, max(0, count))
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: windowSamples)], dataType: .float32)
        let audioPtr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
        audioPtr.initialize(repeating: 0, count: windowSamples)
        if clampedCount > 0 {
            samples.withUnsafeBufferPointer { src in
                audioPtr.update(from: src.baseAddress! + start, count: clampedCount)
            }
        }

        let framesArray = try MLMultiArray(shape: [1], dataType: .int32)
        framesArray.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(validFrames)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "n_frames": MLFeatureValue(multiArray: framesArray),
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
            throw TranscribeError.contract("StenoFr n'a pas retourné les tenseurs ids / logp")
        }

        let n = min(validFrames, encoderFrames)
        var topIDs = [Int](repeating: 0, count: n)
        var topProbs = [Double](repeating: 0, count: n)

        let idStrides = idsMulti.strides.map(\.intValue)
        let lpStrides = logpMulti.strides.map(\.intValue)
        let idPtr = idsMulti.dataPointer.assumingMemoryBound(to: Int32.self)
        let lpPtr = logpMulti.dataPointer.assumingMemoryBound(to: Float.self)

        let idFrameStride = idStrides.count >= 3 ? idStrides[1] : topK
        let lpFrameStride = lpStrides.count >= 3 ? lpStrides[1] : topK

        for f in 0..<n {
            topIDs[f] = Int(idPtr[f * idFrameStride])
            topProbs[f] = exp(Double(lpPtr[f * lpFrameStride]))
        }

        return (topIDs, topProbs)
    }

    // MARK: - Audio Segmentation at Quiet Points (steno_asr/audio.py)

    static func computeEncoderFrames(sampleCount: Int, maxFrames: Int = 750) -> Int {
        let melFrames = max(1, (sampleCount - 321 + 160) / 160 + 1)
        let encFrames = ((melFrames + 1) / 2 + 1) / 2
        return min(maxFrames, max(1, encFrames))
    }

    /// Splits audio longer than 30 s into blocks of at most 30 s, cutting at the quietest 20 ms
    /// within the last 10 s of each block (exact port of `steno_asr.audio.split_points`).
    static func splitPoints(
        samples: [Float],
        sampleRate: Int = 16000,
        maxSeconds: Double = 30.0,
        searchSeconds: Double = 10.0
    ) -> [(start: Int, end: Int)] {
        let n = samples.count
        guard n > 0 else { return [] }
        let hop = sampleRate / 50  // 320 samples (20 ms)
        let nFrames = n / hop
        var rms = [Float](repeating: 0, count: max(1, nFrames))
        if nFrames > 0 {
            samples.withUnsafeBufferPointer { buf in
                for f in 0..<nFrames {
                    var sumSq: Float = 0
                    let base = f * hop
                    for i in 0..<hop {
                        let v = buf[base + i]
                        sumSq += v * v
                    }
                    rms[f] = sqrt(sumSq / Float(hop))
                }
            }
        }

        let maxLen = Int(maxSeconds * Double(sampleRate))
        let searchLen = Int(searchSeconds * Double(sampleRate))
        var out: [(start: Int, end: Int)] = []
        var start = 0

        while n - start > maxLen {
            let lo = max(0, (start + maxLen - searchLen) / hop)
            let hi = min(rms.count, (start + maxLen) / hop)
            let cut: Int
            if hi > lo {
                var minIdx = lo
                var minVal = rms[lo]
                for idx in (lo + 1)..<hi {
                    if rms[idx] < minVal {
                        minVal = rms[idx]
                        minIdx = idx
                    }
                }
                cut = max(minIdx * hop, start + sampleRate)
            } else {
                cut = start + maxLen
            }
            out.append((start: start, end: cut))
            start = cut
        }
        out.append((start: start, end: n))
        return out
    }

    // MARK: - Greedy CTC Word-Piece Decoding & SRT Formatting (steno_asr/decode.py)

    private func greedyWords(
        ids: [Int],
        probs: [Double],
        offsetSeconds: Double
    ) -> [WordTiming] {
        let blank = meta.blank_id
        let unk = meta.unk_id
        let symbols = meta.symbols
        let frameSec = frameSeconds

        var words: [WordTiming] = []
        var curWord: String?
        var curStart: Int = 0
        var curEnd: Int = 0
        var curPeaks: [Double] = []
        var prev = blank

        func flush() {
            guard let w = curWord, !w.isEmpty, !curPeaks.isEmpty else { return }
            let logSum = curPeaks.reduce(0.0) { $0 + log(max($1, 1e-6)) }
            let conf = exp(logSum / Double(curPeaks.count))
            let startSec = (offsetSeconds + Double(curStart) * frameSec * 1000.0).rounded() / 1000.0
            let endSec = (offsetSeconds + Double(curEnd + 1) * frameSec * 1000.0).rounded() / 1000.0
            words.append(WordTiming(word: w, start: startSec, end: endSec, confidence: (conf * 1000.0).rounded() / 1000.0))
        }

        for f in 0..<min(ids.count, probs.count) {
            let i = ids[f]
            let p = probs[f]
            if i == prev && i != blank {
                if curWord != nil && i != unk && !curPeaks.isEmpty {
                    curPeaks[curPeaks.count - 1] = max(curPeaks[curPeaks.count - 1], p)
                    curEnd = f
                }
                continue
            }
            prev = i
            if i == blank || i == unk || i < 0 || i >= symbols.count {
                continue
            }
            let piece = symbols[i]
            if piece.hasPrefix("<0x") {
                continue
            }
            if piece.hasPrefix("▁") || curWord == nil {
                flush()
                let stripped = String(piece.drop { $0 == "▁" })
                curWord = stripped
                curStart = f
                curEnd = f
                curPeaks = [p]
            } else {
                curWord = (curWord ?? "") + piece
                curEnd = f
                curPeaks.append(p)
            }
        }
        flush()
        return words
    }

    static func formatSRT(
        words: [WordTiming],
        maxChars: Int = 42,
        maxLines: Int = 2,
        maxSeconds: Double = 6.0
    ) -> String {
        struct Cue {
            let start: Double
            let end: Double
            let text: String
        }

        var cues: [Cue] = []
        var lines: [String] = []
        var line = ""
        var cueStart: Double?
        var cueEnd: Double = 0.0

        func closeCue() {
            if !line.isEmpty {
                lines.append(line)
            }
            if !lines.isEmpty, let s = cueStart {
                cues.append(Cue(start: s, end: cueEnd, text: lines.joined(separator: "\n")))
            }
            lines.removeAll()
            line = ""
            cueStart = nil
        }

        for w in words {
            if let s = cueStart, w.end - s > maxSeconds {
                closeCue()
            }
            let candidate = line.isEmpty ? w.word : "\(line) \(w.word)"
            if candidate.count > maxChars && !line.isEmpty {
                lines.append(line)
                line = ""
                if lines.count == maxLines {
                    closeCue()
                }
            }
            let updatedLine = line.isEmpty ? w.word : "\(line) \(w.word)"
            if cueStart == nil {
                cueStart = w.start
            }
            line = updatedLine
            cueEnd = w.end
            if let lastChar = w.word.last, ".?!".contains(lastChar), (!lines.isEmpty || line.count > maxChars / 2) {
                closeCue()
            }
        }
        closeCue()

        var out = ""
        for (idx, cue) in cues.enumerated() {
            out += "\(idx + 1)\n\(formatTimestamp(cue.start)) --> \(formatTimestamp(cue.end))\n\(cue.text)\n\n"
        }
        return out
    }

    private static func formatTimestamp(_ seconds: Double) -> String {
        let totalMs = max(0, Int((seconds * 1000.0).rounded()))
        let h = totalMs / 3_600_000
        let rem1 = totalMs % 3_600_000
        let m = rem1 / 60_000
        let rem2 = rem1 % 60_000
        let s = rem2 / 1_000
        let ms = rem2 % 1_000
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }
}
