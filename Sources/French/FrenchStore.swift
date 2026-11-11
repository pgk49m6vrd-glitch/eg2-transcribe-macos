import Foundation

/// Locates and downloads the French EmbeddingGemma 2 + CTC Reader (`StenoFr.mlpackage` + `steno_fr_meta.json`)
/// and its memory-mapped 4-gram French language model (`lm_fr.bin`).
enum FrenchStore {
    static let releaseBaseURL = URL(string: "https://github.com/pgk49m6vrd-glitch/eg2-transcribe-macos/releases/latest/download")!

    static var localFolder: URL {
        URL.documentsDirectory.appending(path: "models/eg2-steno-fr-v1", directoryHint: .isDirectory)
    }

    static var downloadedFolder: URL {
        URL.cachesDirectory.appending(path: "huggingface/eg2-steno-fr-v1", directoryHint: .isDirectory)
    }

    struct AssetSpec: Sendable {
        let filename: String
        let expectedBytes: Int64
        let minValidBytes: Int64
    }

    static let metaAsset = AssetSpec(
        filename: "steno_fr_meta.json",
        expectedBytes: 302_116,
        minValidBytes: 100_000
    )

    static let lmAsset = AssetSpec(
        filename: "lm_fr.bin",
        expectedBytes: 48_000_000,
        minValidBytes: 5_000_000
    )

    static let mlpackageZip = AssetSpec(
        filename: "StenoFr.mlpackage.zip",
        expectedBytes: 205_984_851,
        minValidBytes: 100_000_000
    )

    /// Returns the first directory that holds a complete French Sténo + 4-Gram LM installation.
    static func locate(preferred: URL? = nil) -> URL? {
        let candidates = [preferred, localFolder, downloadedFolder].compactMap { $0 }
        return candidates.first { isComplete($0) }
    }

    static func isComplete(_ root: URL) -> Bool {
        let fm = FileManager.default
        let hasModel = fm.fileExists(atPath: root.appending(path: "StenoFr.mlmodelc").path)
            || fm.fileExists(atPath: root.appending(path: "StenoFr.mlpackage/Manifest.json").path)
        guard hasModel else { return false }

        for asset in [metaAsset, lmAsset] {
            let path = root.appending(path: asset.filename).path
            guard fm.fileExists(atPath: path) else { return false }
            let size = (try? fm.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
            if size < asset.minValidBytes { return false }
        }
        return true
    }

    /// Downloads any missing files into `downloadedFolder` and unpacks `StenoFr.mlpackage.zip`.
    static func download(progress: @escaping @Sendable (Double, String) -> Void) async throws -> URL {
        let fm = FileManager.default
        let root = downloadedFolder
        try fm.createDirectory(at: root, withIntermediateDirectories: true)

        let needModel = !fm.fileExists(atPath: root.appending(path: "StenoFr.mlmodelc").path)
            && !fm.fileExists(atPath: root.appending(path: "StenoFr.mlpackage/Manifest.json").path)

        var queue: [AssetSpec] = []
        for asset in [metaAsset, lmAsset] {
            let target = root.appending(path: asset.filename)
            let size = (try? fm.attributesOfItem(atPath: target.path)[.size] as? Int64) ?? 0
            if size < asset.minValidBytes {
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

            let baseBytes = completedBytes
            let assetName = asset.filename
            progress(Double(baseBytes) / Double(totalBytes), "Téléchargement de \(assetName)…")
            try await downloadFile(
                from: url,
                to: tempDestination,
                expectedBytes: asset.expectedBytes
            ) { fileBytes in
                let fraction = Double(baseBytes + fileBytes) / Double(totalBytes)
                progress(min(0.98, fraction), "Téléchargement Français Sténo + 4-Gram (\(Int(min(0.98, fraction) * 100))%)…")
            }

            try? fm.removeItem(at: destination)
            try fm.moveItem(at: tempDestination, to: destination)
            completedBytes += asset.expectedBytes

            if asset.filename == mlpackageZip.filename {
                progress(0.99, "Extraction de StenoFr.mlpackage…")
                try unpackZip(destination, into: root)
                try? fm.removeItem(at: destination)
            }
        }

        progress(1.0, "Modèle Français Sténo + 4-Gram prêt")
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
