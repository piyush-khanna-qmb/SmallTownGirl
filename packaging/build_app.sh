#!/bin/bash
# Assemble SmallTownGirl.app and a distributable .dmg.
# Usage: packaging/build_app.sh [version]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VERSION="${1:-0.2.0}"

BUILD="$ROOT/build"
APP="$BUILD/SmallTownGirl.app"
rm -rf "$BUILD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

sed "s/__VERSION__/$VERSION/g" "$HERE/Info.plist" > "$APP/Contents/Info.plist"
cp "$HERE/launcher.sh" "$APP/Contents/MacOS/SmallTownGirl"
chmod +x "$APP/Contents/MacOS/SmallTownGirl"
cp "$ROOT/smalltowngirl.py" "$ROOT/menubar.py" "$ROOT/requirements.txt" "$APP/Contents/Resources/"

plutil -lint "$APP/Contents/Info.plist" >/dev/null
echo "Built app: $APP"

# .dmg with an /Applications drop target
DMG="$BUILD/SmallTownGirl-$VERSION.dmg"
STAGE="$BUILD/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "SmallTownGirl" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
echo "Built dmg: $DMG"
shasum -a 256 "$DMG"
