import AppKit
import ApplicationServices
import Foundation

/// Handles clipboard copy, Accessibility permission verification, and automatic `⌘V` pasting into the active app.
@MainActor
enum PasteManager {
    /// Whether the current process has Accessibility permission to synthesize keyboard events (`CGEvent.post`).
    static var isAccessibilityTrusted: Bool {
        AXIsProcessTrusted()
    }

    /// Prompts macOS to show the Accessibility trust dialog for this app.
    @discardableResult
    static func requestAccessibilityPermission() -> Bool {
        let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    /// Opens System Settings directly on the Accessibility privacy pane.
    static func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    /// Copies `text` to the system clipboard.
    static func copyToClipboard(_ text: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
    }

    /// Copies `text` to the clipboard and, if `autoPaste` is enabled and Accessibility is trusted, synthesizes `⌘V`
    /// into the frontmost application. Returns `true` if `⌘V` was posted.
    @discardableResult
    static func copyAndPaste(_ text: String, autoPaste: Bool) async -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        copyToClipboard(trimmed)
        guard autoPaste, isAccessibilityTrusted else { return false }

        // Give the clipboard and modifier keys a brief moment to settle before posting Cmd+V.
        try? await Task.sleep(for: .milliseconds(70))
        postCommandV()
        return true
    }

    private static func postCommandV() {
        guard let source = CGEventSource(stateID: .combinedSessionState) else { return }
        let vKeyCode: CGKeyCode = 0x09 // kVK_ANSI_V
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false)
        else { return }

        keyDown.flags = .maskCommand
        keyUp.flags = .maskCommand
        keyDown.post(tap: .cghidEventTap)
        keyUp.post(tap: .cghidEventTap)
    }
}
