import SwiftUI

/// Comprehensive settings for engine selection, global hotkey dictation, auto-paste, Core AI compute forms, and Dock visibility.
struct SettingsView: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Moteur & Langues") {
                    Picker("Moteur de transcription", selection: engineSelection) {
                        ForEach(EngineKind.allCases) { kind in
                            Text(kind.title).tag(kind)
                        }
                    }

                    if model.engineKind == .multilingual {
                        Picker("Langue cible", selection: $model.multilingualLanguage) {
                            ForEach(MultilingualLanguage.allCases) { lang in
                                Text(lang.label).tag(lang)
                            }
                        }
                        Toggle("Recherche en faisceau N-Gram (Beam Search)", isOn: $model.useBeamSearch)
                        Text("Le N-Gram réduit d'un tiers les erreurs d'homophones en chinois, japonais, coréen et anglais.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    } else {
                        if !model.forms.isEmpty {
                            Picker("Pipeline Core AI", selection: formSelection) {
                                ForEach(model.forms) { form in
                                    Text(form.title).tag(form.id)
                                }
                            }
                        }
                        Toggle("Chevauchement GPU ∥ Neural Engine (Overlap)", isOn: $model.overlap)
                    }
                }

                Section("Dictée Globale & Collage Automatique") {
                    Toggle("Activer le raccourci clavier global", isOn: $model.hotkeyEnabled)
                    Picker("Raccourci", selection: $model.hotkeyPreset) {
                        ForEach(HotkeyPreset.allCases) { preset in
                            Text(preset.label).tag(preset)
                        }
                    }
                    Picker("Comportement", selection: $model.hotkeyMode) {
                        ForEach(HotkeyTriggerMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    Toggle("Coller automatiquement dans l'application active (⌘V)", isOn: $model.autoPaste)

                    HStack {
                        Label(
                            model.isAccessibilityTrusted
                                ? "Permission Accessibilité accordée"
                                : "Permission Accessibilité requise pour le collage ⌘V",
                            systemImage: model.isAccessibilityTrusted ? "checkmark.shield.fill" : "exclamationmark.shield.fill"
                        )
                        .foregroundStyle(model.isAccessibilityTrusted ? .green : .orange)
                        Spacer()
                        Button("Ouvrir Accessibilité…") {
                            model.requestAccessibility()
                        }
                    }
                }

                Section("Apparence & Barre de menus") {
                    Toggle("Masquer l'icône du Dock (mode barre de menus uniquement)", isOn: $model.hideDockIcon)
                    Picker("Taille du texte transcrit", selection: $model.textSize) {
                        Text("Standard").tag(0.0)
                        Text("Grand").tag(22.0)
                        Text("Très grand").tag(32.0)
                    }
                }

                if model.engineKind == .coreAI, let folder = model.folder {
                    Section("Dossier du modèle Core AI") {
                        LabeledContent(
                            "Fenêtre",
                            value: String(format: "%.0f s (%d tokens de contexte gauche)", folder.reader.windowSeconds, folder.reader.leftOverlapTokens)
                        )
                        LabeledContent("Reader", value: folder.reader.version)
                        Text(folder.root.path)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Réglages EG2 Transcribe")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Fermer") { dismiss() }
                }
            }
        }
        .frame(minWidth: 540, minHeight: 500)
    }

    private var engineSelection: Binding<EngineKind> {
        Binding(
            get: { model.engineKind },
            set: { kind in Task { await model.switchEngine(kind) } }
        )
    }

    private var formSelection: Binding<String> {
        Binding(
            get: { model.formID },
            set: { id in Task { await model.selectForm(id) } }
        )
    }
}
