import Foundation

/// Locates and downloads the community multilingual (zh / en / ja / ko) CTC + N-Gram reader v4.0
/// from `TimHu950112/eg2-transcribe-zh-en-ja-ko`.
enum MultilingualStore {
    static let releaseBaseURL = URL(string: "https://github.com/TimHu950112/eg2-transcribe-zh-en-ja-ko/releases/download/v4.0")!

    static var localFolder: URL {
        URL.documentsDirectory.appending(path: "models/eg2-multilingual-v4", directoryHint: .isDirectory)
    }

    static var downloadedFolder: URL {
        URL.cachesDirectory.appending(path: "huggingface/eg2-multilingual-v4", directoryHint: .isDirectory)
    }

    struct AssetSpec: Sendable {
        let filename: String
        let expectedBytes: Int64
    }

    static let requiredAssets: [AssetSpec] = [
        AssetSpec(filename: "transcribe_meta.json", expectedBytes: 58_527),
        AssetSpec(filename: "zh_hans.json", expectedBytes: 37_769),
        AssetSpec(filename: "lm_zh.bin", expectedBytes: 57_546_092),
        AssetSpec(filename: "lm_en.bin", expectedBytes: 46_759_172),
        AssetSpec(filename: "lm_ja.bin", expectedBytes: 43_670_120),
        AssetSpec(filename: "lm_ko.bin", expectedBytes: 47_445_272),
    ]

    static let mlpackageZip = AssetSpec(filename: "JoinTranscribe.mlpackage.zip", expectedBytes: 123_787_759)

    /// Returns the first directory that holds a complete multilingual model installation.
    static func locate(preferred: URL? = nil) -> URL? {
        let candidates = [preferred, localFolder, downloadedFolder].compactMap { $0 }
        return candidates.first { isComplete($0) }
    }

    static func isComplete(_ root: URL) -> Bool {
        let fm = FileManager.default
        let hasModel = fm.fileExists(atPath: root.appending(path: "JoinTranscribe.mlmodelc").path)
            || fm.fileExists(atPath: root.appending(path: "JoinTranscribe.mlpackage/Manifest.json").path)
        guard hasModel else { return false }
        for asset in requiredAssets {
            let path = root.appending(path: asset.filename).path
            guard fm.fileExists(atPath: path) else { return false }
        }
        return true
    }

    /// Downloads any missing files into `downloadedFolder` and unpacks `JoinTranscribe.mlpackage.zip`.
    static func download(progress: @escaping @Sendable (Double, String) -> Void) async throws -> URL {
        let fm = FileManager.default
        let root = downloadedFolder
        try fm.createDirectory(at: root, withIntermediateDirectories: true)

        let needModel = !fm.fileExists(atPath: root.appending(path: "JoinTranscribe.mlmodelc").path)
            && !fm.fileExists(atPath: root.appending(path: "JoinTranscribe.mlpackage/Manifest.json").path)

        var queue: [AssetSpec] = []
        for asset in requiredAssets {
            let target = root.appending(path: asset.filename)
            let size = (try? fm.attributesOfItem(atPath: target.path)[.size] as? Int64) ?? 0
            if size < asset.expectedBytes / 2 {
                queue.append(asset)
            }
        }
        if needModel {
            queue.append(mlpackageZip)
        }

        let totalBytes = max(1, queue.reduce(Int64(0)) { $0 + $1.expectedBytes })
        var completedBytes: Int64 = 0

        for asset in queue {
            let url = releaseBaseURL.appending(path: asset.filename)
            let destination = root.appending(path: asset.filename)
            let tempDestination = root.appending(path: asset.filename + ".part")
            try? fm.removeItem(at: tempDestination)

            progress(Double(completedBytes) / Double(totalBytes), "Téléchargement de \(asset.filename)…")
            try await downloadFile(
                from: url,
                to: tempDestination,
                expectedBytes: asset.expectedBytes
            ) { fileBytes in
                let fraction = Double(completedBytes + fileBytes) / Double(totalBytes)
                progress(min(0.98, fraction), "Téléchargement de \(asset.filename)…")
            }

            try? fm.removeItem(at: destination)
            try fm.moveItem(at: tempDestination, to: destination)
            completedBytes += asset.expectedBytes

            if asset.filename == mlpackageZip.filename {
                progress(0.99, "Extraction de JoinTranscribe.mlpackage…")
                try unpackZip(destination, into: root)
                try? fm.removeItem(at: destination)
            }
        }

        progress(1.0, "Modèle multilingue prêt")
        return root
    }

    private static func downloadFile(
        from url: URL,
        to destination: URL,
        expectedBytes: Int64,
        onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw TranscribeError.download("HTTP \(http.statusCode) pour \(url.lastPathComponent)")
        }
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }

        var buffer = Data()
        let chunkSize = 256 * 1024
        buffer.reserveCapacity(chunkSize)
        var received: Int64 = 0
        var lastReport: Int64 = 0

        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= chunkSize {
                try handle.write(contentsOf: buffer)
                received += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if received - lastReport >= 512 * 1024 {
                    lastReport = received
                    onBytes(min(received, expectedBytes))
                }
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            received += Int64(buffer.count)
        }
        onBytes(min(received, expectedBytes))
    }

    private static func unpackZip(_ zipURL: URL, into directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", zipURL.path, directory.path]
        let errPipe = Pipe()
        process.standardError = errPipe
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let errText = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw TranscribeError.download("Impossible d'extraire \(zipURL.lastPathComponent): \(errText)")
        }
    }
}
