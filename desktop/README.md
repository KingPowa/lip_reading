# Speaker Captions desktop shell

Electron wrapper around the Python live server (`python -m src.app.live`). It starts the backend on a free localhost port, shows a loading screen while models warm up, then opens the page in a window.

- Dev: `npm install && npm start` (uses `../.venv/bin/python`; run `uv sync` in the repo root first).
- Build: `npm run dist` produces `dist/mac-arm64/Speaker Captions.app` (unsigned). The app records the project root in `backend.json`, so it needs this repo and its `.venv` to stay in place.
- Install: `../scripts/install-desktop-app.sh` builds and copies the app to `~/Applications`.
- Language preference: `language` (e.g. `"it,en"`) in `prefs.json` in the app's userData folder (`~/Library/Application Support/speaker-captions`); default is the system language if supported, else `en`.
- Logs: `backend.log` in the same folder.
- Env: `SPEAKER_CAPTIONS_ROOT` overrides the project root; `SPEAKER_CAPTIONS_SMOKE=1` with `SPEAKER_CAPTIONS_SMOKE_OUT=file.json` runs a headless check and quits.
- Tests: `npm test`.
- First launch asks for camera and microphone access. If you denied it, enable it in System Settings > Privacy & Security.
