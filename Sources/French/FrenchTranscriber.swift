import CoreML
import Foundation

/// Runs the French Sténo EmbeddingGemma 2 + CTC Reader (`StenoFr.mlpackage`) via Core ML,
/// paired with a memory-mapped 4-gram French language model (`lm_fr.bin`) and deterministic French ITN.
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
        let usedBeamSearch: Bool
        let usedITN: Bool
        let words: [WordTiming]
        let text: String
        let srt: String

        var totalSeconds: Double { decodeSeconds + modelSeconds + postSeconds }
        var realtimeFactor: Double { audioSeconds / max(totalSeconds, 1e-6) }
    }

    let root: URL
    let meta: Meta
    private let model: MLModel
    private let lm: NGramLM?
    private let lock = NSLock()

    private static let ln10: Float = 2.302585093
    private static let negInf: Float = -1e30
    private static let lmTrimSet = CharacterSet(charactersIn: ".,;:!?'\"()[]{}«»—–-… \t\r\n")

    var windowSamples: Int { meta.window_samples }
    var encoderFrames: Int { meta.encoder_frames }
    var topK: Int { meta.topk }
    var frameSeconds: Double { Double(meta.frame_ms) / 1000.0 }
    var hasLanguageModel: Bool { lm != nil }

    private init(root: URL, meta: Meta, model: MLModel, lm: NGramLM?) {
        self.root = root
        self.meta = meta
        self.model = model
        self.lm = lm
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

        status("Chargement du modèle de langue 4-Gram Français (lm_fr_v2.bin · mmap)…")
        let lmV2URL = root.appending(path: "lm_fr_v2.bin")
        let lmV1URL = root.appending(path: "lm_fr.bin")
        let loadedLM = NGramLM(file: lmV2URL) ?? NGramLM(file: lmV1URL)

        return FrenchTranscriber(root: root, meta: meta, model: mlModel, lm: loadedLM)
    }

    /// Transcribes an audio or video file into punctuated French text and SRT subtitles.
    func transcribe(
        _ url: URL,
        useBeamSearch: Bool = true,
        useITN: Bool = true,
        progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }
    ) async throws -> Result {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let decoded = try AudioDecoder.decode(url)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let decodeSeconds = Double(t1 - t0) / 1e9

        let samples = decoded.samples
        let blocks = Self.splitPoints(samples: samples, sampleRate: meta.sample_rate)
        let totalWindows = max(1, blocks.count)
        let activeLM = useBeamSearch ? lm : nil

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
            let (blockIDs, blockLogP) = try predictBlock(
                samples: samples,
                start: block.start,
                count: validSamples,
                validFrames: validFrames
            )

            let tPost0 = DispatchTime.now().uptimeNanoseconds
            let offsetSeconds = Double(block.start) / Double(meta.sample_rate)
            let blockWords: [WordTiming]
            if let activeLM {
                blockWords = beamWords(
                    ids: blockIDs,
                    logp: blockLogP,
                    validFrames: validFrames,
                    offsetSeconds: offsetSeconds,
                    lm: activeLM
                )
            } else {
                blockWords = greedyWords(
                    ids: blockIDs,
                    logp: blockLogP,
                    validFrames: validFrames,
                    offsetSeconds: offsetSeconds
                )
            }
            allWords.append(contentsOf: blockWords)
            postNanoseconds += DispatchTime.now().uptimeNanoseconds - tPost0

            progress(index + 1, totalWindows)
        }

        let tITN0 = DispatchTime.now().uptimeNanoseconds
        if useITN {
            allWords = FrenchITN.normalizeWords(allWords)
        }
        postNanoseconds += DispatchTime.now().uptimeNanoseconds - tITN0

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
            usedBeamSearch: activeLM != nil,
            usedITN: useITN,
            words: allWords,
            text: fullText,
            srt: srt
        )
    }

    // MARK: - Core ML Prediction (Full Top-K Extraction)

    private func predictBlock(
        samples: [Float],
        start: Int,
        count: Int,
        validFrames: Int
    ) throws -> (ids: [Int32], logp: [Float]) {
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
        let kCount = topK
        var outIDs = [Int32](repeating: 0, count: n * kCount)
        var outLogP = [Float](repeating: Self.negInf, count: n * kCount)

        let idStrides = idsMulti.strides.map(\.intValue)
        let lpStrides = logpMulti.strides.map(\.intValue)
        let idPtr = idsMulti.dataPointer.assumingMemoryBound(to: Int32.self)
        let lpPtr = logpMulti.dataPointer.assumingMemoryBound(to: Float.self)

        let idS1 = idStrides.count >= 3 ? idStrides[1] : kCount
        let idS2 = idStrides.count >= 3 ? idStrides[2] : 1
        let lpS1 = lpStrides.count >= 3 ? lpStrides[1] : kCount
        let lpS2 = lpStrides.count >= 3 ? lpStrides[2] : 1

        for f in 0..<n {
            let idRow = f * idS1
            let lpRow = f * lpS1
            let dstBase = f * kCount
            for k in 0..<kCount {
                outIDs[dstBase + k] = idPtr[idRow + k * idS2]
                outLogP[dstBase + k] = lpPtr[lpRow + k * lpS2]
            }
        }

        return (outIDs, outLogP)
    }

    // MARK: - Word-Piece CTC Prefix Beam Search + 4-Gram LM

    @inline(__always)
    private static func logSumExp(_ a: Float, _ b: Float) -> Float {
        let (hi, lo) = a > b ? (a, b) : (b, a)
        return lo <= negInf ? hi : hi + log1p(exp(lo - hi))
    }

    static func cleanWordForLM(_ raw: String) -> String {
        let normalized = raw
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .trimmingCharacters(in: lmTrimSet)
            .lowercased()
        guard normalized.unicodeScalars.contains(where: { CharacterSet.letters.contains($0) }) else {
            return ""
        }
        return normalized
    }

    private static let cliticExpansions: [String: String] = [
        "j": "je", "l": "le", "d": "de", "n": "ne", "s": "se",
        "c": "ce", "m": "me", "t": "te", "qu": "que",
        "jusqu": "jusque", "lorsqu": "lorsque", "puisqu": "puisque"
    ]

    /// Scores a completed word against the 4-gram LM, handling direct vocabulary hits,
    /// hyphenated compounds ("soixante-trois", "peut-être"), and elided clitic contractions ("j'habite", "l'application").
    private static func scoreCompletedWord(
        rawWord: String,
        history: [String],
        lm: NGramLM,
        alpha: Float,
        beta: Float,
        maxHist: Int
    ) -> (scoreDelta: Float, updatedHistory: [String]) {
        let canonicalRaw = FrenchITN.canonicalizeGluedNumberWord(rawWord)
        let wasGluedWithoutHyphen = (canonicalRaw != rawWord)
        let clean = cleanWordForLM(canonicalRaw)
        guard !clean.isEmpty else {
            return (0.0, history)
        }

        let directLP = lm.logp(history: history, symbol: clean)
        var bestLP = directLP
        var nextHistory = history
        var appendedHistory = false

        if clean.contains("'"), !lm.containsUnigram(clean) {
            let parts = clean.split(separator: "'", maxSplits: 1, omittingEmptySubsequences: true).map(String.init)
            if parts.count == 2 {
                let clitic = cliticExpansions[parts[0]] ?? parts[0]
                let stem = parts[1]
                let lp1 = lm.logp(history: history, symbol: clitic)
                var h2 = history
                h2.append(clitic)
                if h2.count > maxHist {
                    h2.removeFirst(h2.count - maxHist)
                }
                let lp2 = lm.logp(history: h2, symbol: stem)
                let avgLP = 0.5 * (lp1 + lp2)
                if avgLP > bestLP {
                    bestLP = avgLP
                    h2.append(stem)
                    if h2.count > maxHist {
                        h2.removeFirst(h2.count - maxHist)
                    }
                    nextHistory = h2
                    appendedHistory = true
                }
            }
        }

        if clean.contains("-") {
            let subParts = clean.split(separator: "-", omittingEmptySubsequences: true).map(String.init)
            if subParts.count >= 2 {
                var tempHist = history
                var sumLP: Float = 0
                for part in subParts {
                    sumLP += lm.logp(history: tempHist, symbol: part)
                    tempHist.append(part)
                    if tempHist.count > maxHist {
                        tempHist.removeFirst(tempHist.count - maxHist)
                    }
                }
                let avgLP = sumLP / Float(subParts.count)
                if avgLP > bestLP {
                    bestLP = avgLP
                }
            }
        }

        if !appendedHistory {
            nextHistory.append(clean)
            if nextHistory.count > maxHist {
                nextHistory.removeFirst(nextHistory.count - maxHist)
            }
        }

        // Slight penalty if the acoustic path dropped the hyphen in a compound number ("soixantetrois")
        // so that the explicit hyphen path ("soixante-trois") wins when both are in the beam.
        let gluePenalty: Float = wasGluedWithoutHyphen ? -0.6 : 0.0
        let delta = alpha * bestLP * ln10 + beta + gluePenalty
        return (delta, nextHistory)
    }

    private struct BeamHypothesis {
        let pieceKey: UInt64
        let lastUnit: Int
        var pBlank: Float
        var pNonBlank: Float
        let completedWords: [WordTiming]
        let curWord: String?
        let curStart: Int
        var curEnd: Int
        var curPeaks: [Double]
        let history: [String]
        let committedLMScore: Float

        var totalScore: Float {
            FrenchTranscriber.logSumExp(pBlank, pNonBlank) + committedLMScore
        }

        func finalScore(lm: NGramLM, alpha: Float, beta: Float, maxHist: Int) -> Float {
            guard let w = curWord, !w.isEmpty else {
                return totalScore
            }
            let (delta, _) = FrenchTranscriber.scoreCompletedWord(
                rawWord: w,
                history: history,
                lm: lm,
                alpha: alpha,
                beta: beta,
                maxHist: maxHist
            )
            return totalScore + delta
        }

        func flushAll(offsetSeconds: Double, frameSeconds: Double) -> [WordTiming] {
            var result = completedWords
            if let w = curWord, !w.isEmpty, !curPeaks.isEmpty {
                let canonicalWord = FrenchITN.canonicalizeGluedNumberWord(w)
                let logSum = curPeaks.reduce(0.0) { $0 + log(max($1, 1e-6)) }
                let conf = exp(logSum / Double(curPeaks.count))
                let startSec = (offsetSeconds + Double(curStart) * frameSeconds * 1000.0).rounded() / 1000.0
                let endSec = (offsetSeconds + Double(curEnd + 1) * frameSeconds * 1000.0).rounded() / 1000.0
                result.append(
                    WordTiming(
                        word: canonicalWord,
                        start: startSec,
                        end: endSec,
                        confidence: (conf * 1000.0).rounded() / 1000.0
                    )
                )
            }
            return result
        }
    }

    private func beamWords(
        ids: [Int32],
        logp: [Float],
        validFrames: Int,
        offsetSeconds: Double,
        lm: NGramLM,
        alpha: Float = 0.4,
        beta: Float = 1.4,
        beamWidth: Int = 16,
        prune: Float = -10.0
    ) -> [WordTiming] {
        let blank = meta.blank_id
        let unk = meta.unk_id
        let symbols = meta.symbols
        let frameSec = frameSeconds
        let kCount = topK
        let maxHist = max(1, lm.order - 1)

        let rootKey: UInt64 = 0xCBF29CE484222325
        let root = BeamHypothesis(
            pieceKey: rootKey,
            lastUnit: blank,
            pBlank: 0.0,
            pNonBlank: Self.negInf,
            completedWords: [],
            curWord: nil,
            curStart: 0,
            curEnd: 0,
            curPeaks: [],
            history: [],
            committedLMScore: 0.0
        )

        var beams: [UInt64: BeamHypothesis] = [rootKey: root]

        func extendHypothesis(
            _ state: BeamHypothesis,
            unit u: Int,
            frame f: Int,
            peakProb: Double
        ) -> BeamHypothesis? {
            guard u >= 0 && u < symbols.count else { return nil }
            let piece = symbols[u]
            if piece.hasPrefix("<0x") { return nil }

            let childKey = (state.pieceKey ^ UInt64(bitPattern: Int64(u))) &* 0x100000001B3

            if piece.hasPrefix("▁") || state.curWord == nil {
                var newCompleted = state.completedWords
                var newHistory = state.history
                var newCommittedLM = state.committedLMScore

                if let prevWord = state.curWord, !prevWord.isEmpty, !state.curPeaks.isEmpty {
                    let canonicalPrev = FrenchITN.canonicalizeGluedNumberWord(prevWord)
                    let logSum = state.curPeaks.reduce(0.0) { $0 + log(max($1, 1e-6)) }
                    let conf = exp(logSum / Double(state.curPeaks.count))
                    let startSec = (offsetSeconds + Double(state.curStart) * frameSec * 1000.0).rounded() / 1000.0
                    let endSec = (offsetSeconds + Double(state.curEnd + 1) * frameSec * 1000.0).rounded() / 1000.0
                    newCompleted.append(
                        WordTiming(
                            word: canonicalPrev,
                            start: startSec,
                            end: endSec,
                            confidence: (conf * 1000.0).rounded() / 1000.0
                        )
                    )
                    let (delta, updatedHist) = Self.scoreCompletedWord(
                        rawWord: prevWord,
                        history: newHistory,
                        lm: lm,
                        alpha: alpha,
                        beta: beta,
                        maxHist: maxHist
                    )
                    newCommittedLM += delta
                    newHistory = updatedHist
                }

                let stripped = String(piece.drop { $0 == "▁" })

                return BeamHypothesis(
                    pieceKey: childKey,
                    lastUnit: u,
                    pBlank: Self.negInf,
                    pNonBlank: Self.negInf,
                    completedWords: newCompleted,
                    curWord: stripped,
                    curStart: f,
                    curEnd: f,
                    curPeaks: [peakProb],
                    history: newHistory,
                    committedLMScore: newCommittedLM
                )
            } else {
                let updatedWord = (state.curWord ?? "") + piece
                var newPeaks = state.curPeaks
                newPeaks.append(peakProb)

                return BeamHypothesis(
                    pieceKey: childKey,
                    lastUnit: u,
                    pBlank: Self.negInf,
                    pNonBlank: Self.negInf,
                    completedWords: state.completedWords,
                    curWord: updatedWord,
                    curStart: state.curStart,
                    curEnd: f,
                    curPeaks: newPeaks,
                    history: state.history,
                    committedLMScore: state.committedLMScore
                )
            }
        }

        for f in 0..<validFrames {
            let rowBase = f * kCount
            let bestLP = logp[rowBase]
            var next: [UInt64: BeamHypothesis] = [:]
            next.reserveCapacity(beamWidth * 2)

            for (key, state) in beams {
                let totalProb = Self.logSumExp(state.pBlank, state.pNonBlank)

                for k in 0..<kCount {
                    let lp = logp[rowBase + k]
                    if lp < bestLP + prune { continue }
                    let u = Int(ids[rowBase + k])

                    if u == blank || u == unk {
                        if var existing = next[key] {
                            existing.pBlank = Self.logSumExp(existing.pBlank, totalProb + lp)
                            next[key] = existing
                        } else {
                            var copy = state
                            copy.pBlank = totalProb + lp
                            copy.pNonBlank = Self.negInf
                            next[key] = copy
                        }
                        continue
                    }

                    let prob = exp(Double(lp))

                    if u == state.lastUnit {
                        // 1. Repeat without blank: collapse into same hypothesis
                        if state.pNonBlank > Self.negInf {
                            if var existing = next[key] {
                                existing.pNonBlank = Self.logSumExp(existing.pNonBlank, state.pNonBlank + lp)
                                existing.curEnd = max(existing.curEnd, f)
                                if !existing.curPeaks.isEmpty {
                                    existing.curPeaks[existing.curPeaks.count - 1] = max(
                                        existing.curPeaks[existing.curPeaks.count - 1],
                                        prob
                                    )
                                }
                                next[key] = existing
                            } else {
                                var copy = state
                                copy.pBlank = Self.negInf
                                copy.pNonBlank = state.pNonBlank + lp
                                copy.curEnd = f
                                if !copy.curPeaks.isEmpty {
                                    copy.curPeaks[copy.curPeaks.count - 1] = max(
                                        copy.curPeaks[copy.curPeaks.count - 1],
                                        prob
                                    )
                                }
                                next[key] = copy
                            }
                        }

                        // 2. Repeat after a blank: emit a new copy of piece u
                        if state.pBlank > -1e20 {
                            let childKey = (state.pieceKey ^ UInt64(bitPattern: Int64(u))) &* 0x100000001B3
                            if var existingChild = next[childKey] {
                                existingChild.pNonBlank = Self.logSumExp(existingChild.pNonBlank, state.pBlank + lp)
                                next[childKey] = existingChild
                            } else if var created = extendHypothesis(state, unit: u, frame: f, peakProb: prob) {
                                created.pNonBlank = state.pBlank + lp
                                next[childKey] = created
                            }
                        }
                    } else {
                        // Emit new piece u != state.lastUnit
                        let childKey = (state.pieceKey ^ UInt64(bitPattern: Int64(u))) &* 0x100000001B3
                        if var existingChild = next[childKey] {
                            existingChild.pNonBlank = Self.logSumExp(existingChild.pNonBlank, totalProb + lp)
                            next[childKey] = existingChild
                        } else if var created = extendHypothesis(state, unit: u, frame: f, peakProb: prob) {
                            created.pNonBlank = totalProb + lp
                            next[childKey] = created
                        }
                    }
                }
            }

            if next.count <= beamWidth {
                beams = next
            } else {
                let topStates = next.values.sorted { $0.totalScore > $1.totalScore }.prefix(beamWidth)
                beams.removeAll(keepingCapacity: true)
                for s in topStates {
                    beams[s.pieceKey] = s
                }
            }
        }

        guard let best = beams.values.max(by: {
            $0.finalScore(lm: lm, alpha: alpha, beta: beta, maxHist: maxHist)
                < $1.finalScore(lm: lm, alpha: alpha, beta: beta, maxHist: maxHist)
        }) else {
            return []
        }
        return best.flushAll(offsetSeconds: offsetSeconds, frameSeconds: frameSec)
    }

    // MARK: - Greedy CTC Word-Piece Decoding & SRT Formatting (steno_asr/decode.py)

    private func greedyWords(
        ids: [Int32],
        logp: [Float],
        validFrames: Int,
        offsetSeconds: Double
    ) -> [WordTiming] {
        let blank = meta.blank_id
        let unk = meta.unk_id
        let symbols = meta.symbols
        let frameSec = frameSeconds
        let kCount = topK

        var words: [WordTiming] = []
        var curWord: String?
        var curStart: Int = 0
        var curEnd: Int = 0
        var curPeaks: [Double] = []
        var prev = blank

        func flush() {
            guard let w = curWord, !w.isEmpty, !curPeaks.isEmpty else { return }
            let canonicalWord = FrenchITN.canonicalizeGluedNumberWord(w)
            let logSum = curPeaks.reduce(0.0) { $0 + log(max($1, 1e-6)) }
            let conf = exp(logSum / Double(curPeaks.count))
            let startSec = (offsetSeconds + Double(curStart) * frameSec * 1000.0).rounded() / 1000.0
            let endSec = (offsetSeconds + Double(curEnd + 1) * frameSec * 1000.0).rounded() / 1000.0
            words.append(WordTiming(word: canonicalWord, start: startSec, end: endSec, confidence: (conf * 1000.0).rounded() / 1000.0))
        }

        for f in 0..<validFrames {
            let i = Int(ids[f * kCount])
            let p = exp(Double(logp[f * kCount]))
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
