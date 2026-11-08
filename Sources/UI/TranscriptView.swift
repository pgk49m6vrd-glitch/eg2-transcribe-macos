import AppKit
import SwiftUI

/// Displays the selectable transcript with Copy, Paste, and Export (.txt / .srt) actions.
struct TranscriptView: View {
    let text: String
    let srt: String?
    let file: String?
    let size: Double
    let onSaveTxt: () -> Void
    let onSaveSrt: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Transcription")
                    .font(.headline)
                if let file {
                    Text(file)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                Button {
                    PasteManager.copyToClipboard(text)
                } label: {
                    Label("Copier", systemImage: "doc.on.doc")
                }
                .disabled(text.isEmpty)

                Button(action: onSaveTxt) {
                    Label("Exporter .txt", systemImage: "square.and.arrow.down")
                }
                .disabled(text.isEmpty)

                if let srt, !srt.isEmpty {
                    Button(action: onSaveSrt) {
                        Label("Exporter .srt", systemImage: "captions.bubble")
                    }
                }
            }

            Text(
                text.isEmpty
                    ? "Glissez-déposez un fichier audio/vidéo ici, cliquez sur « Choisir un audio », ou maintenez le raccourci global pour dicter."
                    : text
            )
            .font(size > 0 ? .system(size: size) : .body)
            .foregroundStyle(text.isEmpty ? .secondary : .primary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
            .padding(14)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}
