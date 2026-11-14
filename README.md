# EG2 Transcribe for macOS

Native macOS application ready to use (`.app` / `.dmg`, menu bar + Studio window + global Push-to-Talk dictation) based on the audio encoder from **Google EmbeddingGemma 2** and community **CTC decoders / ultra-light vector readers**.

---

## Features

1. **Three local EmbeddingGemma 2 + CTC engines to choose from (downloaded on first launch as needed)**:
   - 🇫🇷 **French — Sténo v1 Large (`guigssss/steno-fr-v1-large` / `GuigsEvt/steno-asr`) + 4-Gram Beam Search + ITN** *(Default engine)*:
     - EmbeddingGemma 2 audio encoder (layers 1–9 frozen, layers 10–12 fine-tuned for French, learned mix of 13 hidden states) + 10-block CTC reader over **30,002 Gemma word pieces** exported in 8-bit Core ML (`StenoFr.mlpackage`, ~196 MB).
     - **French CTC Prefix Beam Search + 4-Gram Language Model (`lm_fr.bin`, ~48 MB memory-mapped via `mmap`, 0 MB RAM)**: uses the Top-8 candidates per 40 ms frame with interpolated Kneser-Ney smoothing to disambiguate grammatical homophones (`a/à`, `c'est/s'est`, `ou/où`, `on/ont`, `-é/-er/-ez`) in under 2 ms without altering Franglais or technical terms.
     - **Deterministic French ITN in pure Swift (`FrenchITN`)**: instantly converts spoken numbers to digits (< 0.1 ms, 0 MB RAM, 0% hallucination) (`"ticket quatre cent douze"` → `"ticket 412"`, `"version un point deux"` → `"version 1.2"`, `"quatre-vingt-dix pour cent"` → `"90 %"`) while protecting the indefinite article `un/une`.
     - Produces directly **punctuated French with capitalization** and `.srt` subtitles in one pass, with no Whisper-style hallucinations.
   - 🇬🇧 **Core AI English v2 (`mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI`)**:
     - Frozen EmbeddingGemma 2 audio encoder (GPU, fp16) + Reader v2 (19.6 M parameters, Neural Engine / GPU, 200,000-word English vocabulary table in 768-d space).
     - LibriSpeech `test-clean` WER **4.21%**, reaching up to **×548 real-time** on Apple Silicon (M4 Max) thanks to GPU ∥ Neural Engine overlap.
   - 🌐 **Multilingual zh / en / ja / ko v4 (`TimHu950112/eg2-transcribe-zh-en-ja-ko`)**:
     - 8-bit Core ML model (`JoinTranscribe.mlpackage`, ~318 MB) + CTC beam search with **N-Gram** language models memory-mapped (`lm_zh.bin`, `lm_en.bin`, `lm_ja.bin`, `lm_ko.bin`).
     - Automatic language detection or explicit selection (`中文 繁體`, `中文 简体`, `English`, `日本語`, `한국어`), automatic punctuation, and `.srt` subtitle segmentation.
2. **Menu Bar app (`MenuBarExtra`) + Studio window**:
   - Instant access from the macOS menu bar (with an option to hide the Dock icon).
   - Drag-and-drop audio/video files (`.wav`, `.mp3`, `.m4a`, `.flac`, `.caf`, `.aiff`) and export to `.txt` or `.srt`.
3. **Global Push-to-Talk Dictation & Auto Paste (`⌘V`)**:
   - Hold **`⌥ Right Option`** (or `🌐 Fn`, `⌥ + Space`, `⌃ + ⌥ + Space`) from any application to speak: a subtle floating pill (HUD) shows the live audio level.
   - Release the key: transcription runs locally and is pasted automatically into the active application!

---

## Installation (`.dmg` built by GitHub Actions)

1. Download **`EG2Transcribe-macOS-arm64.dmg`** from the [**Releases**](../../releases) tab or GitHub Actions artifacts.
2. Open the `.dmg` and drag **`EG2Transcribe.app`** into **Applications**.
3. Since the app is signed ad hoc (without a paid Apple Developer subscription):
   - Use **Right Click > Open** on `EG2Transcribe.app` in `/Applications` (or double-click `Open without Gatekeeper.command` included in the `.dmg`),
   - Or run in Terminal:
     ```bash
     xattr -cr /Applications/EG2Transcribe.app
     open /Applications/EG2Transcribe.app
     ```
4. To enable automatic paste `⌘V` in other apps, click **"Allow…"** in the menu bar panel to grant **Accessibility** permission (`System Settings > Privacy & Security > Accessibility`).

---

## Local Build & French Core ML Export (macOS 27+ / Xcode 27+ / Apple Silicon)

```bash
brew install xcodegen
xcodegen generate
xcodebuild -project EG2Transcribe.xcodeproj -scheme EG2TranscribeMac -configuration Release build
./scripts/package_dmg.sh build/Build/Products/Release/EG2Transcribe.app dist

# Optional: regenerate StenoFr.mlpackage from Hugging Face (guigssss/steno-fr-v1-large)
python3 scripts/export_steno_fr_coreml.py --model large --out dist/steno_fr
```

---

## Credits & Licenses

- Application code: **Apache-2.0** ([`LICENSE`](LICENSE)).
- Base model: **Google EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache-2.0, subject to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)).
- French Sténo v1 pipeline: based on [`GuigsEvt/steno-asr`](https://github.com/GuigsEvt/steno-asr) and [`guigssss/steno-fr-v1-large`](https://huggingface.co/guigssss/steno-fr-v1-large) by Guillaume Evrat (Apache-2.0).
- English Core AI & Reader v2 pipeline: based on [`john-rocky/eg2-transcribe`](https://github.com/john-rocky/eg2-transcribe) and [`mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI`](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI) (Apache-2.0).
- Multilingual zh/en/ja/ko v4 pipeline: based on [`TimHu950112/eg2-transcribe-zh-en-ja-ko`](https://github.com/TimHu950112/eg2-transcribe-zh-en-ja-ko) (code & reader Apache-2.0, N-gram models CC BY-SA 4.0).
