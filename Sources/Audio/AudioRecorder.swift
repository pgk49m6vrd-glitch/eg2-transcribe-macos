import AVFoundation
import Foundation

/// Records the microphone to a 16 kHz mono 16-bit WAV file in the temporary folder and exposes live audio levels.
@MainActor
final class AudioRecorder {
    private var recorder: AVAudioRecorder?
    private(set) var url: URL?
    private var meterTimer: Timer?
    private var startTime: Date?

    var isRecording: Bool { recorder?.isRecording ?? false }
    var onMeterUpdate: ((Float, TimeInterval) -> Void)?

    /// Asks for the microphone once (the system shows its dialog the first time).
    func permission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    func start() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "recording-\(Int(Date().timeIntervalSince1970 * 1000)).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: AudioDecoder.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder.isMeteringEnabled = true
        guard recorder.record() else { throw TranscribeError.audio("the microphone did not start") }
        self.recorder = recorder
        self.url = url
        self.startTime = Date()
        startMetering()
    }

    /// Stops and returns the recorded file.
    func stop() -> URL? {
        stopMetering()
        recorder?.stop()
        recorder = nil
        startTime = nil
        return url
    }

    private func startMetering() {
        stopMetering()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let rec = self.recorder, rec.isRecording else { return }
                rec.updateMeters()
                let power = rec.averagePower(forChannel: 0) // -160 ... 0 dB
                let normalized = max(0, min(1, (power + 50) / 50))
                let elapsed = self.startTime.map { Date().timeIntervalSince($0) } ?? 0
                self.onMeterUpdate?(normalized, elapsed)
            }
        }
    }

    private func stopMetering() {
        meterTimer?.invalidate()
        meterTimer = nil
    }
}
