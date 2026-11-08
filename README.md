# EG2 Transcribe pour macOS

Application macOS native prête à l'emploi (`.app` / `.dmg`, barre de menus + fenêtre Studio + dictée globale Push-to-Talk) basée sur l'encodeur audio de **Google EmbeddingGemma 2** et les décodeurs **CTC / lecteurs vectoriels ultra-légers** de la communauté.

---

## Fonctionnalités

1. **Trois moteurs locaux EmbeddingGemma 2 + CTC au choix (téléchargés à la demande au 1er lancement)** :
   - 🇫🇷 **Français — Sténo v1 Large (`guigssss/steno-fr-v1-large` / `GuigsEvt/steno-asr`)** *(Moteur par défaut)* :
     - Encodeur audio EmbeddingGemma 2 (couches 1–9 figées, couches 10–12 affinées pour le français, mélange appris des 13 états cachés) + lecteur CTC 10 blocs sur **30 002 pièces de mots Gemma** exporté en Core ML 8-bit (`StenoFr.mlpackage`, ~365 Mo).
     - Produit directement du **français ponctué avec majuscules** et sous-titres `.srt` en un seul passage (WER **8,8 %** sur Multilingual LibriSpeech FR et **9,7 %** sur FLEURS FR), sans hallucinations de type Whisper.
   - 🇬🇧 **Core AI Anglais v2 (`mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI`)** :
     - Encodeur audio EmbeddingGemma 2 figé (GPU, fp16) + Reader v2 (19,6 M paramètres, Neural Engine / GPU, table de 200 000 mots anglais dans l'espace 768-d).
     - WER LibriSpeech `test-clean` **4,21 %**, vitesse jusqu'à **×548 temps réel** sur Apple Silicon (M4 Max) grâce au chevauchement GPU ∥ Neural Engine.
   - 🌐 **Multilingue zh / en / ja / ko v4 (`TimHu950112/eg2-transcribe-zh-en-ja-ko`)** :
     - Modèle Core ML 8-bit (`JoinTranscribe.mlpackage`, ~318 Mo) + recherche en faisceau CTC avec modèles de langue **N-Gram** memory-mappés (`lm_zh.bin`, `lm_en.bin`, `lm_ja.bin`, `lm_ko.bin`).
     - Détection automatique de la langue ou sélection explicite (`中文 繁體`, `中文 简体`, `English`, `日本語`, `한국어`), ponctuation automatique et découpage en sous-titres `.srt`.
2. **Application de Barre de Menus (`MenuBarExtra`) + Fenêtre Studio** :
   - Accès instantané depuis la barre de menus macOS (avec option pour masquer l'icône du Dock).
   - Glisser-déposer de fichiers audio/vidéo (`.wav`, `.mp3`, `.m4a`, `.flac`, `.caf`, `.aiff`) et export en `.txt` ou `.srt`.
3. **Dictée Globale Push-to-Talk & Collage Automatique (`⌘V`)** :
   - Maintenez **`⌥ Option Droit`** (ou `🌐 Fn`, `⌥ + Espace`, `⌃ + ⌥ + Espace`) depuis n'importe quelle application pour parler : une pilule flottante (HUD) non-intrusive affiche le niveau sonore en direct.
   - Relâchez la touche : la transcription s'exécute localement et se colle automatiquement dans l'application active !

---

## Installation (`.dmg` précompilé par GitHub Actions)

1. Téléchargez **`EG2Transcribe-macOS-arm64.dmg`** depuis l'onglet [**Releases**](../../releases) ou les artefacts GitHub Actions.
2. Ouvrez le `.dmg` et glissez **`EG2Transcribe.app`** dans **`Applications`**.
3. Comme l'application est signée de manière ad-hoc (sans abonnement Apple Developer payant) :
   - Faites **Clic droit > Ouvrir** sur `EG2Transcribe.app` dans `/Applications` (ou double-cliquez sur `Ouvrir sans Gatekeeper.command` inclus dans le `.dmg`),
   - Ou lancez dans le Terminal :
     ```bash
     xattr -cr /Applications/EG2Transcribe.app
     open /Applications/EG2Transcribe.app
     ```
4. Pour activer le collage automatique `⌘V` dans les autres applications, cliquez sur **« Autoriser… »** dans le panneau de la barre de menus pour accorder la permission **Accessibilité** (`Réglages Système > Confidentialité et sécurité > Accessibilité`).

---

## Compilation locale & Export Core ML Français (macOS 27+ / Xcode 27+ / Apple Silicon)

```bash
brew install xcodegen
xcodegen generate
xcodebuild -project EG2Transcribe.xcodeproj -scheme EG2TranscribeMac -configuration Release build
./scripts/package_dmg.sh build/Build/Products/Release/EG2Transcribe.app dist

# Optionnel : régénérer StenoFr.mlpackage depuis Hugging Face (guigssss/steno-fr-v1-large)
python3 scripts/export_steno_fr_coreml.py --model large --out dist/steno_fr
```

---

## Crédits & Licences

- Code de l'application : **Apache-2.0** ([`LICENSE`](LICENSE)).
- Modèle de base : **Google EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache-2.0, soumis à la [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)).
- Pipeline Français Sténo v1 : basé sur [`GuigsEvt/steno-asr`](https://github.com/GuigsEvt/steno-asr) et [`guigssss/steno-fr-v1-large`](https://huggingface.co/guigssss/steno-fr-v1-large) par Guillaume Evrat (Apache-2.0).
- Pipeline Core AI & Reader Anglais v2 : basé sur [`john-rocky/eg2-transcribe`](https://github.com/john-rocky/eg2-transcribe) et [`mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI`](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI) (Apache-2.0).
- Pipeline Multilingue zh/en/ja/ko v4 : basé sur [`TimHu950112/eg2-transcribe-zh-en-ja-ko`](https://github.com/TimHu950112/eg2-transcribe-zh-en-ja-ko) (code & lecteur Apache-2.0, modèles N-gram CC BY-SA 4.0).
