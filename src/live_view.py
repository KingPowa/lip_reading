"""The dependency-free browser interface for live camera captioning."""

import json
from importlib.resources import files

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live speaker captions</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { background: #111827; color: #f9fafb; margin: 0; }
main { display: grid; gap: 1rem; grid-template-columns: minmax(0, 1fr) minmax(18rem, 26rem); margin: auto; max-width: 90rem; padding: 1rem; }
header, #controls { grid-column: 1 / -1; }
h1 { font-size: 1.3rem; margin: 0; } p { color: #cbd5e1; margin: .4rem 0; }
button, select { font: inherit; padding: .55rem .75rem; } button { cursor: pointer; }
#controls { align-items: center; display: flex; flex-wrap: wrap; gap: .6rem; }
#status { color: #93c5fd; min-height: 1.5rem; }
#stage { min-width: 0; } .video-shell { background: #030712; position: relative; }
video { display: block; max-height: 72vh; max-width: 100%; width: 100%; }
#faces { height: 100%; inset: 0; pointer-events: none; position: absolute; width: 100%; }
.speaker { cursor: help; pointer-events: auto; }
.speaker:focus { outline: 3px solid #facc15; } #caption { background: #030712; font-size: 1.05rem; min-height: 2rem; padding: .65rem; }
aside { background: #1f2937; border-radius: .5rem; min-height: 12rem; overflow: hidden; }
aside.collapsed { display: none; } #transcript-toggle { float: right; margin: .5rem; }
#transcript { clear: both; max-height: 70vh; overflow: auto; padding: .5rem; }
.event { background: none; border: 0; border-radius: .3rem; color: inherit; display: block; padding: .6rem; text-align: left; width: 100%; }
.event.active { background: #155e75; } .event time { color: #93c5fd; display: block; font-size: .8rem; }
@media (max-width: 52rem) { main { grid-template-columns: 1fr; } aside { grid-row: 4; } }
</style>
<main>
  <header><h1>Live speaker captions</h1><p>Your camera remains live. Local captions arrive after analysis. Highlights follow the last attributed speaker's current position; a silhouette appears when its person mask is unambiguous.</p></header>
  <section id="controls" aria-label="Camera controls">
    <label>Camera <select id="camera-select"></select></label>
    <label>Language <select id="language"><option value="">Auto detect</option><option value="en">English</option><option value="it">Italiano</option><option value="es">Español</option><option value="fr">Français</option><option value="de">Deutsch</option></select></label>
    <button id="start">Start camera</button><button id="stop" disabled>Stop</button><button id="retry" hidden>Retry recognition</button><button id="transcript-toggle" aria-controls="transcript-panel" aria-expanded="true">Collapse transcript</button>
    <span id="status" role="status">Choose Start camera to grant camera and microphone access.</span>
  </section>
  <section id="stage">
    <h2>Live camera</h2><div class="video-shell"><video id="camera" autoplay muted playsinline></video><svg id="faces" preserveAspectRatio="none" aria-label="Current speaker highlights"></svg></div>
    <div id="caption" aria-live="polite">Waiting for a recognized clip.</div>
  </section>
  <aside id="transcript-panel"><h2>Transcript</h2><div id="transcript" aria-label="All recognized speech"></div></aside>
</main>
<script>const configuration = __CONFIGURATION__;</script>
<script>__CLIENT_SCRIPT__
</script></html>"""


def build_page(chunk_milliseconds: int, language: str | None = None) -> str:
    """Return the standalone camera interface with its server configuration."""
    return PAGE_TEMPLATE.replace(
        "__CLIENT_SCRIPT__",
        files("src").joinpath("live_client.js").read_text(encoding="utf-8"),
    ).replace(
        "__CONFIGURATION__",
        json.dumps({"chunkMilliseconds": chunk_milliseconds, "language": language}),
    )
