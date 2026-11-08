import Foundation

/// One timestamped sentence emitted by the multilingual transcriber.
struct TranscriptLine: Sendable, Equatable, Identifiable {
    var id: String { "\(startMs)-\(endMs)-\(text)" }
    let startMs: Int
    let endMs: Int
    let text: String
}

/// Converts Traditional Chinese (Taiwan phrasing) to Simplified Chinese using `zh_hans.json`.
struct SimplifiedChineseConverter: Sendable {
    let chars: [Character: String]
    let phrases: [(key: String, value: String)]

    private struct RawJSON: Decodable {
        let chars: [String: String]
        let phrases: [String: String]
    }

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode(RawJSON.self, from: data)
        var charMap: [Character: String] = [:]
        for (k, v) in raw.chars {
            if let ch = k.first, k.count == 1 {
                charMap[ch] = v
            }
        }
        self.chars = charMap
        self.phrases = raw.phrases
            .map { (key: $0.key, value: $0.value) }
            .sorted { $0.key.count > $1.key.count }
    }

    func convert(_ input: String) -> String {
        var text = input
        for (src, dst) in phrases where text.contains(src) {
            text = text.replacingOccurrences(of: src, with: dst)
        }
        var out = String()
        out.reserveCapacity(text.count)
        for ch in text {
            if let mapped = chars[ch] {
                out.append(mapped)
            } else {
                out.append(ch)
            }
        }
        return out
    }
}

/// Text normalization, punctuation rendering, and sentence splitting for the multilingual reader (zh/en/ja/ko).
enum TranscriptText {
    static let comma = "<,>"
    static let period = "<.>"
    static let question = "<?>"
    static let exclaim = "<!>"
    static let enumeration = "<、>"
    static let space = "▁"
    static let blank = "<b>"

    static let punctMarks: Set<String> = [comma, period, question, exclaim, enumeration]
    static let sentenceEnds: Set<String> = [period, question, exclaim]

    private static let fullWidthPunct: [String: String] = [
        comma: "，",
        period: "。",
        question: "？",
        exclaim: "！",
        enumeration: "、",
    ]

    private static let halfWidthPunct: [String: String] = [
        comma: ",",
        period: ".",
        question: "?",
        exclaim: "!",
        enumeration: ",",
    ]

    @inline(__always)
    static func isMark(_ symbol: String) -> Bool {
        punctMarks.contains(symbol)
    }

    @inline(__always)
    static func isCJK(_ scalar: UnicodeScalar) -> Bool {
        let v = scalar.value
        return (0x4E00...0x9FFF).contains(v)
            || (0x3400...0x4DBF).contains(v)
            || (0x20000...0x2FFFF).contains(v)
            || (0xF900...0xFAFF).contains(v)
            || (0x3040...0x30FF).contains(v)
            || v == 0x30FC
            || (0x31F0...0x31FF).contains(v)
    }

    @inline(__always)
    static func isHangul(_ scalar: UnicodeScalar) -> Bool {
        (0xAC00...0xD7A3).contains(scalar.value)
    }

    @inline(__always)
    private static func isAlnum(_ scalar: UnicodeScalar) -> Bool {
        CharacterSet.alphanumerics.contains(scalar)
    }

    /// Converts a sequence of emitted symbols into display text, setting full-width punctuation after CJK clauses
    /// and inserting spaces between CJK and Latin segments.
    static func render(_ symbols: [String], simplified: Bool = false, hans: SimplifiedChineseConverter? = nil) -> String {
        var merged: [String] = []
        merged.reserveCapacity(symbols.count)
        for s in symbols {
            if isMark(s), let last = merged.last, isMark(last) {
                merged[merged.count - 1] = s
            } else if s == space, merged.last == space {
                continue
            } else {
                merged.append(s)
            }
        }

        var out: [String] = []
        var clauseCJK = false
        let cjkPunctScalars = Set("，。？！、".unicodeScalars)

        for (i, s) in merged.enumerated() {
            if s == space {
                if let last = out.last, last != " " {
                    out.append(" ")
                }
            } else if isMark(s) {
                while out.last == " " {
                    out.removeLast()
                }
                let mapped = (clauseCJK ? fullWidthPunct[s] : halfWidthPunct[s]) ?? ""
                out.append(mapped)
                if !clauseCJK && i + 1 < merged.count {
                    out.append(" ")
                }
                clauseCJK = false
            } else if let firstScalar = s.unicodeScalars.first {
                let cjk = isCJK(firstScalar)
                if let prevStr = out.last, let prevScalar = prevStr.unicodeScalars.last {
                    if prevStr != " "
                        && !cjkPunctScalars.contains(prevScalar)
                        && (isCJK(prevScalar) != cjk)
                        && (isAlnum(prevScalar) || isCJK(prevScalar))
                        && (isAlnum(firstScalar) || cjk)
                        && !isHangul(firstScalar)
                    {
                        out.append(" ")
                    }
                }
                clauseCJK = clauseCJK || cjk
                out.append(s)
            }
        }

        let text = out.joined().trimmingCharacters(in: .whitespacesAndNewlines)
        if simplified, let hans {
            return hans.convert(text)
        }
        return text
    }

    /// Splits emitted timed symbols into sentences (after 。？！, at pauses > 1.2 s, or at commas on long lines).
    static func lines(
        _ symbols: [TranscribeDecoder.Emitted],
        simplified: Bool = false,
        hans: SimplifiedChineseConverter? = nil
    ) -> [TranscriptLine] {
        var result: [TranscriptLine] = []
        var current: [String] = []
        var firstSlot = -1
        var lastSlot = -1

        func flush() {
            let text = render(current, simplified: simplified, hans: hans)
            if !text.isEmpty {
                let startMs = max(0, firstSlot * 20 - 200)
                let endMs = lastSlot * 20 + 300
                result.append(TranscriptLine(startMs: startMs, endMs: endMs, text: text))
            }
            current.removeAll(keepingCapacity: true)
            firstSlot = -1
        }

        for item in symbols {
            let s = item.symbol
            let slot = item.slot
            if !current.isEmpty && (slot - lastSlot > 60) && !isMark(s) {
                flush()
            }
            if current.isEmpty {
                if s == space || isMark(s) {
                    continue
                }
                firstSlot = slot
            }
            current.append(s)
            lastSlot = slot
            if sentenceEnds.contains(s) || (s == comma && current.count > 90) {
                flush()
            }
        }
        if !current.isEmpty {
            flush()
        }
        return result
    }

    /// Formats timestamped lines as SubRip (.srt) subtitles.
    static func formatSRT(_ lines: [TranscriptLine]) -> String {
        lines.enumerated().map { idx, line in
            "\(idx + 1)\n\(srtTimestamp(line.startMs)) --> \(srtTimestamp(line.endMs))\n\(line.text)\n"
        }.joined(separator: "\n")
    }

    private static func srtTimestamp(_ msTotal: Int) -> String {
        let clamped = max(0, msTotal)
        let h = clamped / 3_600_000
        let rem1 = clamped % 3_600_000
        let m = rem1 / 60_000
        let rem2 = rem1 % 60_000
        let s = rem2 / 1_000
        let ms = rem2 % 1_000
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }
}
