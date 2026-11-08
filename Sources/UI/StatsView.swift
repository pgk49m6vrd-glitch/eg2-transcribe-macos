import SwiftUI

/// Displays the benchmark numbers of the last transcription: audio length, wall clock, and realtime factor (`×RT`).
struct StatsView: View {
    let stats: RunStats
    let size: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(headline)
                .font(size > 0 ? .system(size: size * 0.8, weight: .semibold) : .title3.weight(.semibold))
                .monospacedDigit()
            Text(line)
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var headline: String {
        "\(Self.clock(stats.audioSeconds)) audio · \(Self.duration(stats.totalSeconds)) · "
            + String(format: "×%.0f temps réel", stats.realtimeFactor)
    }

    private var line: String {
        var parts = [
            "décodage " + Self.duration(stats.decodeSeconds)
        ]
        if stats.logMelSeconds > 0 {
            parts.append("log-mel " + Self.duration(stats.logMelSeconds))
        }
        parts += zip(stats.stageSeconds, stats.stageSums).map { "\($0) " + Self.duration($1) }
        if stats.hostSeconds > 0 {
            parts.append("hôte " + Self.duration(stats.hostSeconds))
        }
        parts.append("\(stats.windows) fenêtre\(stats.windows == 1 ? "" : "s")")
        parts.append(stats.form)
        return parts.joined(separator: " · ")
    }

    static func duration(_ seconds: Double) -> String {
        seconds < 1 ? String(format: "%.0f ms", seconds * 1000) : String(format: "%.2f s", seconds)
    }

    static func clock(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s >= 3600
            ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
            : String(format: "%d:%02d", s / 60, s % 60)
    }
}
