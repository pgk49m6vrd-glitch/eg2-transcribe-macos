#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${1:?Usage: package_dmg.sh <path/to/EG2Transcribe.app> <output_dir>}"
OUT_DIR="${2:-dist}"
APP_NAME="EG2Transcribe"
DMG_NAME="EG2Transcribe-macOS-arm64.dmg"
ZIP_NAME="EG2Transcribe-macOS-arm64.zip"

mkdir -p "$OUT_DIR"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "==> Copie de ${APP_PATH} vers le dossier de staging…"
ditto "$APP_PATH" "$STAGING/${APP_NAME}.app"

echo "==> Signature ad-hoc de ${APP_NAME}.app…"
codesign --force --deep --sign - "$STAGING/${APP_NAME}.app"

ln -s /Applications "$STAGING/Applications"

cat > "$STAGING/Ouvrir sans Gatekeeper.command" <<'EOF'
#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "/Applications/EG2Transcribe.app" ]; then
  TARGET="/Applications/EG2Transcribe.app"
elif [ -d "$DIR/EG2Transcribe.app" ]; then
  echo "Copie de EG2Transcribe.app dans /Applications..."
  ditto "$DIR/EG2Transcribe.app" "/Applications/EG2Transcribe.app"
  TARGET="/Applications/EG2Transcribe.app"
else
  echo "EG2Transcribe.app introuvable."
  exit 1
fi
echo "Suppression de l'attribut de quarantaine Gatekeeper sur $TARGET..."
xattr -cr "$TARGET" || true
echo "Lancement de EG2Transcribe..."
open "$TARGET"
EOF
chmod +x "$STAGING/Ouvrir sans Gatekeeper.command"

cat > "$STAGING/LISEZMOI - Premiere Ouverture.txt" <<'EOF'
EG2 Transcribe pour macOS (Apple Silicon, macOS 27+)
====================================================

1. Glissez « EG2Transcribe.app » dans le dossier « Applications ».
2. Comme l'application est distribuée sans certificat Apple Developer payant :
   - Faites Clic droit > Ouvrir sur « EG2Transcribe.app » dans /Applications,
   - OU double-cliquez sur « Ouvrir sans Gatekeeper.command »,
   - OU exécutez dans le Terminal :
     xattr -cr /Applications/EG2Transcribe.app && open /Applications/EG2Transcribe.app

3. Au premier lancement :
   - Cliquez sur l'icône d'onde dans la barre de menus (ou dans la fenêtre Studio).
   - Choisissez votre moteur :
     • Core AI Anglais v2 (ultra-rapide, 200 000 mots, GPU + Neural Engine)
     • Multilingue zh / en / ja / ko v4 (Core ML + N-Gram Beam Search)
   - Cliquez sur « Télécharger le modèle ».
   - Maintenez « ⌥ Option Droit » (ou votre raccourci choisi) pour dicter dans n'importe quelle application !
EOF

echo "==> Création de l'archive ZIP (${ZIP_NAME})…"
ditto -c -k --sequesterRsrc --keepParent "$STAGING/${APP_NAME}.app" "$OUT_DIR/$ZIP_NAME"

echo "==> Création de l'image disque DMG (${DMG_NAME})…"
rm -f "$OUT_DIR/$DMG_NAME"
hdiutil create \
  -volname "EG2 Transcribe" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$OUT_DIR/$DMG_NAME"

shasum -a 256 "$OUT_DIR/$DMG_NAME" "$OUT_DIR/$ZIP_NAME" > "$OUT_DIR/SHA256SUMS.txt"
ls -lh "$OUT_DIR"
