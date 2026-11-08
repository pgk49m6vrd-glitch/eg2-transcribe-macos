import AppKit
import Foundation

/// Global shortcut presets for push-to-talk or toggle dictation anywhere in macOS.
enum HotkeyPreset: String, CaseIterable, Identifiable, Sendable {
    case rightOption = "right_option"
    case fnKey = "fn_key"
    case optionSpace = "option_space"
    case ctrlOptionSpace = "ctrl_option_space"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .rightOption: "⌥ Option Droit"
        case .fnKey: "🌐 Fn / Globe"
        case .optionSpace: "⌥ + Espace"
        case .ctrlOptionSpace: "⌃ + ⌥ + Espace"
        }
    }

    var shortBadge: String {
        switch self {
        case .rightOption: "⌥ Droit"
        case .fnKey: "Fn"
        case .optionSpace: "⌥Espace"
        case .ctrlOptionSpace: "⌃⌥Espace"
        }
    }
}

/// How the global dictation shortcut behaves.
enum HotkeyTriggerMode: String, CaseIterable, Identifiable, Sendable {
    case holdToTalk = "hold"
    case toggle = "toggle"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .holdToTalk: "Maintenir pour parler (Push-to-Talk)"
        case .toggle: "Appuyer pour démarrer / arrêter"
        }
    }
}

/// Monitors global and local keyboard events for push-to-talk or toggle dictation.
@MainActor
final class HotkeyManager {
    var preset: HotkeyPreset = .rightOption
    var mode: HotkeyTriggerMode = .holdToTalk
    var isEnabled: Bool = true

    var onDictationStart: (() -> Void)?
    var onDictationStop: (() -> Void)?
    var onDictationToggle: (() -> Void)?

    private var globalFlagsMonitor: Any?
    private var localFlagsMonitor: Any?
    private var globalKeyDownMonitor: Any?
    private var globalKeyUpMonitor: Any?
    private var localKeyDownMonitor: Any?
    private var localKeyUpMonitor: Any?

    private var isHotkeyPressed = false

    func start() {
        stop()

        globalFlagsMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            Task { @MainActor [weak self] in
                self?.handleFlagsChanged(event)
            }
        }
        localFlagsMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.handleFlagsChanged(event)
            return event
        }

        globalKeyDownMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            Task { @MainActor [weak self] in
                self?.handleKeyDown(event)
            }
        }
        globalKeyUpMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyUp) { [weak self] event in
            Task { @MainActor [weak self] in
                self?.handleKeyUp(event)
            }
        }

        localKeyDownMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if self?.handleKeyDown(event) == true {
                return nil
            }
            return event
        }
        localKeyUpMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyUp) { [weak self] event in
            if self?.handleKeyUp(event) == true {
                return nil
            }
            return event
        }
    }

    func stop() {
        for monitor in [
            globalFlagsMonitor, localFlagsMonitor,
            globalKeyDownMonitor, globalKeyUpMonitor,
            localKeyDownMonitor, localKeyUpMonitor,
        ].compactMap({ $0 }) {
            NSEvent.removeMonitor(monitor)
        }
        globalFlagsMonitor = nil
        localFlagsMonitor = nil
        globalKeyDownMonitor = nil
        globalKeyUpMonitor = nil
        localKeyDownMonitor = nil
        localKeyUpMonitor = nil
        isHotkeyPressed = false
    }

    private func handleFlagsChanged(_ event: NSEvent) {
        guard isEnabled else { return }
        switch preset {
        case .rightOption:
            // Virtual key 61 (0x3D) is Right Option on macOS keyboards.
            guard event.keyCode == 61 else { return }
            let optionDown = event.modifierFlags.contains(.option)
            transitionKeyState(down: optionDown)
        case .fnKey:
            // Virtual key 63 (0x3F) is Fn / Globe.
            guard event.keyCode == 63 else { return }
            let fnDown = event.modifierFlags.contains(.function)
            transitionKeyState(down: fnDown)
        case .optionSpace, .ctrlOptionSpace:
            // If user releases modifier while holding Space in hold-to-talk mode, end recording cleanly.
            if isHotkeyPressed && mode == .holdToTalk {
                let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                let stillHeld = (preset == .optionSpace)
                    ? flags.contains(.option)
                    : (flags.contains(.option) && flags.contains(.control))
                if !stillHeld {
                    transitionKeyState(down: false)
                }
            }
        }
    }

    @discardableResult
    private func handleKeyDown(_ event: NSEvent) -> Bool {
        guard isEnabled else { return false }
        guard matchesKeyCombo(event) else { return false }
        if event.isARepeat { return true }
        transitionKeyState(down: true)
        return true
    }

    @discardableResult
    private func handleKeyUp(_ event: NSEvent) -> Bool {
        guard isEnabled else { return false }
        guard event.keyCode == 49 else { return false } // Space
        guard preset == .optionSpace || preset == .ctrlOptionSpace else { return false }
        if isHotkeyPressed {
            transitionKeyState(down: false)
            return true
        }
        return false
    }

    private func matchesKeyCombo(_ event: NSEvent) -> Bool {
        guard event.keyCode == 49 else { return false } // Space bar
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        switch preset {
        case .optionSpace:
            return flags.contains(.option) && !flags.contains(.command) && !flags.contains(.control)
        case .ctrlOptionSpace:
            return flags.contains(.option) && flags.contains(.control) && !flags.contains(.command)
        case .rightOption, .fnKey:
            return false
        }
    }

    private func transitionKeyState(down: Bool) {
        switch mode {
        case .holdToTalk:
            if down && !isHotkeyPressed {
                isHotkeyPressed = true
                onDictationStart?()
            } else if !down && isHotkeyPressed {
                isHotkeyPressed = false
                onDictationStop?()
            }
        case .toggle:
            if down && !isHotkeyPressed {
                isHotkeyPressed = true
                onDictationToggle?()
            } else if !down {
                isHotkeyPressed = false
            }
        }
    }
}
