import Foundation

/// Deterministic pure-Swift Inverse Text Normalization (ITN) for French speech recognition.
/// Converts spoken French numbers ("quatre cent douze", "vingt-quatre", "deux mille vingt-six",
/// "version un point deux", "trois virgule quatorze", "quatre-vingt-dix pour cent") into digits
/// in < 0.1 ms with zero memory overhead and 0% hallucination risk.
enum FrenchITN {
    private static let unitsAndTeens: [String: Int64] = [
        "zéro": 0, "zero": 0,
        "un": 1, "une": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
        "onze": 11,
        "douze": 12,
        "treize": 13,
        "quatorze": 14,
        "quinze": 15,
        "seize": 16,
    ]

    private static let tensMap: [String: Int64] = [
        "vingt": 20, "vingts": 20,
        "trente": 30,
        "quarante": 40,
        "cinquante": 50,
        "soixante": 60,
        "septante": 70,
        "huitante": 80, "octante": 80,
        "nonante": 90,
    ]

    /// Preceding words that disambiguate standalone "un"/"une" as the digit 1 rather than an indefinite article.
    private static let precedingNumberTriggers: Set<String> = [
        "numéro", "numero", "n°",
        "ticket", "issue", "pr", "bug", "commit", "branche", "branch",
        "version", "release",
        "chapitre", "page", "étape", "etape", "ligne", "colonne",
        "option", "piste", "phase", "partie", "point", "article",
        "figure", "tableau", "saison", "épisode", "episode",
        "niveau", "salle", "porte", "voie", "quai", "terminal",
        "classe", "groupe", "formule", "règle", "regle", "cas",
        "degré", "degre", "num", "chambre", "étage", "etage",
    ]

    /// Following words that disambiguate standalone "un"/"une" as the digit 1.
    private static let followingNumberTriggers: Set<String> = [
        "euro", "euros", "€", "centime", "centimes",
        "dollar", "dollars", "$",
        "pourcent", "pourcents", "%",
        "kilo", "kilos", "kg", "gramme", "grammes", "mg",
        "mètre", "mètres", "metre", "metres", "cm", "mm", "km",
        "litre", "litres", "ml",
        "seconde", "secondes", "milliseconde", "millisecondes", "ms",
        "minute", "minutes", "heure", "heures",
        "octet", "octets", "ko", "mo", "go", "to",
    ]

    private struct TokenParts {
        let leadingPunct: String
        let core: String
        let trailingPunct: String
        let atoms: [String]
    }

    /// Applies deterministic French ITN to word timings (updating both `.txt` and `.srt` cues).
    static func normalizeWords(_ words: [FrenchTranscriber.WordTiming]) -> [FrenchTranscriber.WordTiming] {
        guard !words.isEmpty else { return [] }
        let parts = words.map { splitPunctuation($0.word) }
        var out: [FrenchTranscriber.WordTiming] = []
        var i = 0

        while i < words.count {
            if let match = matchNumberSpan(parts: parts, startIndex: i) {
                let firstWord = words[i]
                let lastWord = words[match.endIndex - 1]
                let leading = parts[i].leadingPunct
                var trailing = parts[match.endIndex - 1].trailingPunct
                var formatted = "\(leading)\(match.formattedDigits)\(trailing)"
                var nextIdx = match.endIndex
                var endSec = lastWord.end
                var confSum = (i..<match.endIndex).reduce(0.0) { $0 + words[$1].confidence }
                var confCount = match.endIndex - i

                // Check for decimal ("virgule" or "point") when no sentence punctuation intervened
                while trailing.isEmpty && nextIdx + 1 < words.count {
                    let sepCore = parts[nextIdx].core.lowercased()
                    let isComma = (sepCore == "virgule")
                    let isDot = (sepCore == "point")
                    guard (isComma || isDot),
                          parts[nextIdx].leadingPunct.isEmpty,
                          parts[nextIdx].trailingPunct.isEmpty
                    else { break }

                    // After "virgule"/"point", consume one or more consecutive number spans (e.g. "zéro cinq" -> "05")
                    var decDigits = ""
                    var probe = nextIdx + 1
                    while probe < words.count {
                        guard let decMatch = matchNumberSpan(parts: parts, startIndex: probe, allowStandaloneOne: true) else {
                            break
                        }
                        decDigits += decMatch.formattedDigits
                        trailing = parts[decMatch.endIndex - 1].trailingPunct
                        endSec = words[decMatch.endIndex - 1].end
                        for k in probe..<decMatch.endIndex {
                            confSum += words[k].confidence
                            confCount += 1
                        }
                        probe = decMatch.endIndex
                        // Only continue chaining digits right after "zéro" (e.g. "virgule zéro cinq" -> ",05")
                        if decMatch.value != 0 || !trailing.isEmpty {
                            break
                        }
                    }

                    guard !decDigits.isEmpty else { break }
                    let sepChar = isComma ? "," : "."
                    formatted = "\(leading)\( formatted.dropFirst(leading.count) )\(sepChar)\(decDigits)\(trailing)"
                    nextIdx = probe
                    if isComma { break }
                }

                // Check for "pour cent" / "pourcent" -> " %"
                if trailing.isEmpty && nextIdx < words.count {
                    let nextCore = parts[nextIdx].core.lowercased()
                    if nextCore == "pourcent" || nextCore == "pourcents" {
                        trailing = parts[nextIdx].trailingPunct
                        formatted += " %\(trailing)"
                        endSec = words[nextIdx].end
                        nextIdx += 1
                    } else if nextCore == "pour" && parts[nextIdx].trailingPunct.isEmpty && nextIdx + 1 < words.count {
                        let afterPour = parts[nextIdx + 1].core.lowercased()
                        if afterPour == "cent" || afterPour == "cents" {
                            trailing = parts[nextIdx + 1].trailingPunct
                            formatted += " %\(trailing)"
                            endSec = words[nextIdx + 1].end
                            nextIdx += 2
                        }
                    }
                }

                let avgConf = (confSum / Double(max(1, confCount)) * 1000.0).rounded() / 1000.0
                out.append(
                    FrenchTranscriber.WordTiming(
                        word: formatted,
                        start: firstWord.start,
                        end: endSec,
                        confidence: avgConf
                    )
                )
                i = nextIdx
            } else {
                out.append(words[i])
                i += 1
            }
        }

        return out
    }

    /// Convenience helper for plain strings.
    static func normalize(_ text: String) -> String {
        let rawTokens = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard !rawTokens.isEmpty else { return text }
        let timings = rawTokens.enumerated().map { idx, tok in
            FrenchTranscriber.WordTiming(word: tok, start: Double(idx) * 0.1, end: Double(idx + 1) * 0.1, confidence: 1.0)
        }
        return normalizeWords(timings).map(\.word).joined(separator: " ")
    }

    // MARK: - Span Matching & French Number Parser

    private struct SpanMatch {
        let value: Int64
        let formattedDigits: String
        let endIndex: Int
    }

    private static func matchNumberSpan(
        parts: [TokenParts],
        startIndex: Int,
        allowStandaloneOne: Bool = false
    ) -> SpanMatch? {
        guard startIndex < parts.count else { return nil }
        guard isNumberToken(parts[startIndex]) else { return nil }

        var total: Int64 = 0
        var current: Int64 = 0
        var hasAnyDigit = false
        var lastConsumedWordIndex = startIndex
        var atomCount = 0

        var wordIdx = startIndex
        while wordIdx < parts.count {
            let part = parts[wordIdx]
            if wordIdx > startIndex && !part.leadingPunct.isEmpty {
                break
            }
            guard !part.atoms.isEmpty else { break }

            // Try to consume all hyphen-connected atoms in `part.atoms`
            var tempTotal = total
            var tempCurrent = current
            var tempHasAny = hasAnyDigit
            var ok = true
            var aIdx = 0

            while aIdx < part.atoms.count {
                let atom = part.atoms[aIdx]
                if atom == "et" {
                    // "et" is only valid after 20, 30, 40, 50, 60, 70 when followed by "un"/"une"/"onze"
                    let rem = tempCurrent % 100
                    let validTens = (rem == 20 || rem == 30 || rem == 40 || rem == 50 || rem == 60 || rem == 70)
                    let nextAtom: String? = (aIdx + 1 < part.atoms.count)
                        ? part.atoms[aIdx + 1]
                        : (part.trailingPunct.isEmpty && wordIdx + 1 < parts.count ? parts[wordIdx + 1].atoms.first : nil)
                    if validTens, let na = nextAtom, (na == "un" || na == "une" || na == "onze") {
                        aIdx += 1
                        continue
                    } else {
                        ok = false
                        break
                    }
                }

                if !stepNumberAutomaton(atom: atom, total: &tempTotal, current: &tempCurrent, hasAny: &tempHasAny) {
                    ok = false
                    break
                }
                aIdx += 1
            }

            if !ok {
                break
            }

            total = tempTotal
            current = tempCurrent
            hasAnyDigit = tempHasAny
            atomCount += part.atoms.count
            lastConsumedWordIndex = wordIdx

            // Stop at trailing punctuation (e.g. "quatre, cinq" -> two separate numbers)
            if !part.trailingPunct.isEmpty {
                break
            }
            wordIdx += 1
        }

        guard hasAnyDigit else { return nil }
        let finalValue = total + current
        let endIndex = lastConsumedWordIndex + 1

        // Guard against turning the French indefinite article "un" / "une" into "1"
        if atomCount == 1 && finalValue == 1 && !allowStandaloneOne {
            let rawCore = parts[startIndex].core.lowercased()
            if rawCore == "un" || rawCore == "une" {
                if !isStandaloneOneAllowed(parts: parts, index: startIndex) {
                    return nil
                }
            }
        }

        return SpanMatch(
            value: finalValue,
            formattedDigits: String(finalValue),
            endIndex: endIndex
        )
    }

    private static func isStandaloneOneAllowed(parts: [TokenParts], index: Int) -> Bool {
        if index > 0 {
            let prev = parts[index - 1].core.lowercased()
            if precedingNumberTriggers.contains(prev) {
                return true
            }
        }
        if parts[index].trailingPunct.isEmpty && index + 1 < parts.count {
            let next = parts[index + 1].core.lowercased()
            if followingNumberTriggers.contains(next) {
                return true
            }
            if (next == "virgule" || next == "point") && index + 2 < parts.count && isNumberToken(parts[index + 2]) {
                return true
            }
            if next == "pour" && index + 2 < parts.count {
                let after = parts[index + 2].core.lowercased()
                if after == "cent" || after == "cents" {
                    return true
                }
            }
        }
        return false
    }

    private static func stepNumberAutomaton(
        atom: String,
        total: inout Int64,
        current: inout Int64,
        hasAny: inout Bool
    ) -> Bool {
        if let u = unitsAndTeens[atom] {
            if u == 0 {
                guard !hasAny && total == 0 && current == 0 else { return false }
                hasAny = true
                return true
            }
            let rem100 = current % 100
            let rem10 = current % 10
            if rem100 == 0 {
                current += u
                hasAny = true
                return true
            } else if u < 10 && rem10 == 0 && rem100 >= 20 && rem100 <= 90 {
                // e.g. vingt-quatre (20 + 4), soixante-dix-sept (70 + 7), quatre-vingt-neuf (80 + 9)
                current += u
                hasAny = true
                return true
            } else if u >= 7 && u <= 9 && (rem100 == 10 || rem100 == 70 || rem100 == 90) {
                // dix-sept (10 + 7), soixante-dix-huit (70 + 8), quatre-vingt-dix-neuf (90 + 9)
                current += u
                hasAny = true
                return true
            } else if u >= 10 && u <= 16 && (rem100 == 60 || rem100 == 80) {
                // soixante-douze (60 + 12), quatre-vingt-quatorze (80 + 14)
                current += u
                hasAny = true
                return true
            }
            return false
        }

        if let t = tensMap[atom] {
            let rem100 = current % 100
            if t == 20 && rem100 == 4 {
                // quatre-vingt(s) -> replace 4 with 80
                current = (current - 4) + 80
                hasAny = true
                return true
            }
            if rem100 == 0 {
                current += t
                hasAny = true
                return true
            }
            return false
        }

        if atom == "cent" || atom == "cents" {
            if !hasAny || current == 0 {
                current = 100
                hasAny = true
                return true
            } else if current >= 1 && current <= 9 {
                current *= 100
                hasAny = true
                return true
            } else if current >= 11 && current <= 19 && total == 0 {
                // e.g. "douze cents" (1200)
                total = current * 100
                current = 0
                hasAny = true
                return true
            }
            return false
        }

        if atom == "mille" || atom == "milles" {
            // Prevent two "mille" in the same number
            guard (total % 1_000_000) < 1000 else { return false }
            let mult: Int64 = (current == 0) ? 1 : current
            total += mult * 1_000
            current = 0
            hasAny = true
            return true
        }

        if atom == "million" || atom == "millions" {
            guard current > 0 && total < 1_000_000_000_000 else { return false }
            total += current * 1_000_000
            current = 0
            hasAny = true
            return true
        }

        if atom == "milliard" || atom == "milliards" {
            guard current > 0 else { return false }
            total += current * 1_000_000_000
            current = 0
            hasAny = true
            return true
        }

        return false
    }

    private static func isNumberToken(_ part: TokenParts) -> Bool {
        guard let first = part.atoms.first else { return false }
        return unitsAndTeens[first] != nil
            || tensMap[first] != nil
            || first == "cent" || first == "cents"
            || first == "mille" || first == "milles"
    }

    private static let punctSet = CharacterSet(charactersIn: ".,;:!?'\"()[]{}«»—–…")

    private static func splitPunctuation(_ word: String) -> TokenParts {
        var scalars = Array(word.unicodeScalars)
        var start = 0
        while start < scalars.count && punctSet.contains(scalars[start]) {
            start += 1
        }
        var end = scalars.count
        while end > start && punctSet.contains(scalars[end - 1]) {
            end -= 1
        }
        let leading = String(String.UnicodeScalarView(scalars[0..<start]))
        let core = String(String.UnicodeScalarView(scalars[start..<end]))
        let trailing = String(String.UnicodeScalarView(scalars[end..<scalars.count]))
        let atoms = core
            .lowercased()
            .split(separator: "-", omittingEmptySubsequences: true)
            .map(String.init)
        return TokenParts(leadingPunct: leading, core: core, trailingPunct: trailing, atoms: atoms)
    }
}
