#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
npm --prefix "$ROOT/desktop" ci
npm --prefix "$ROOT/desktop" run dist
mkdir -p "$HOME/Applications"
rm -rf "$HOME/Applications/Speaker Captions.app"
cp -R "$ROOT/desktop/dist/mac-arm64/Speaker Captions.app" "$HOME/Applications/"
echo "Installed: $HOME/Applications/Speaker Captions.app"
