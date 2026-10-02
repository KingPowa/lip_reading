"""The dependency-free browser interface for live camera captioning."""

import json


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
#faces { inset: 0; pointer-events: none; position: absolute; }
.speaker { background: color-mix(in srgb, var(--speaker-color) 22%, transparent); border: 4px solid var(--speaker-color); box-sizing: border-box; cursor: help; pointer-events: auto; position: absolute; }
.speaker:focus { outline: 3px solid #facc15; } #caption { background: #030712; font-size: 1.05rem; min-height: 2rem; padding: .65rem; }
aside { background: #1f2937; border-radius: .5rem; min-height: 12rem; overflow: hidden; }
aside.collapsed { display: none; } #transcript-toggle { float: right; margin: .5rem; }
#transcript { clear: both; max-height: 70vh; overflow: auto; padding: .5rem; }
.event { background: none; border: 0; border-radius: .3rem; color: inherit; display: block; padding: .6rem; text-align: left; width: 100%; }
.event.active { background: #155e75; } .event time { color: #93c5fd; display: block; font-size: .8rem; }
@media (max-width: 52rem) { main { grid-template-columns: 1fr; } aside { grid-row: 4; } }
</style>
<main>
  <header><h1>Live speaker captions</h1><p>Your camera remains live. Captions and speaker highlights appear as each short segment finishes local analysis; recorded clips are never played back.</p></header>
  <section id="controls" aria-label="Camera controls">
    <label>Camera <select id="camera-select"></select></label>
    <label>Language <select id="language"><option value="">Auto detect</option><option value="en">English</option><option value="it">Italiano</option><option value="es">Español</option><option value="fr">Français</option><option value="de">Deutsch</option></select></label>
    <button id="start">Start camera</button><button id="stop" disabled>Stop</button>
    <span id="status" role="status">Choose Start camera to grant camera and microphone access.</span>
  </section>
  <section id="stage">
    <h2>Live camera</h2><div class="video-shell"><video id="camera" autoplay muted playsinline></video><div id="faces" aria-live="polite"></div></div>
    <div id="caption" aria-live="polite">Waiting for a recognized clip.</div>
  </section>
  <aside id="transcript-panel"><button id="transcript-toggle" aria-controls="transcript" aria-expanded="true">Collapse transcript</button><h2>Transcript</h2><div id="transcript" aria-label="All recognized speech"></div></aside>
</main>
<script>const configuration = __CONFIGURATION__;</script>
<script>
const camera = document.querySelector("#camera");
const faces = document.querySelector("#faces");
const caption = document.querySelector("#caption");
const status = document.querySelector("#status");
const cameraSelect = document.querySelector("#camera-select");
const languageSelect = document.querySelector("#language");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const transcript = document.querySelector("#transcript");
const transcriptPanel = document.querySelector("#transcript-panel");
const transcriptToggle = document.querySelector("#transcript-toggle");
let stream; let recorder; let captureTimer; let capturing = false; let captureGeneration = 0; let chunkNumber = 0;
const activeEvents = [];
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const browserLanguages = { en: "en-US", it: "it-IT", es: "es-ES", fr: "fr-FR", de: "de-DE" };
let speechRecognition; let liveText = "";

if (configuration.language) {
  if (!browserLanguages[configuration.language]) languageSelect.add(new Option(configuration.language, configuration.language));
  languageSelect.value = configuration.language;
} else {
  const browserLanguage = navigator.language.split("-")[0].toLowerCase();
  if (browserLanguages[browserLanguage]) languageSelect.value = browserLanguage;
}

function setStatus(message) { status.textContent = message; }
function formatTime(seconds) { const minutes = Math.floor(seconds / 60); return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`; }
function boxAt(track, timestamp) {
  if (!track || timestamp < track.start || timestamp > track.end + .04) return null;
  let before = track.boxes[0]; let after = track.boxes.at(-1);
  for (const box of track.boxes) { if (box.timestamp <= timestamp) before = box; if (box.timestamp >= timestamp) { after = box; break; } }
  const span = after.timestamp - before.timestamp; const fraction = span ? (timestamp - before.timestamp) / span : 0;
  const interpolate = (first, second) => Object.fromEntries(["x", "y", "width", "height"].map((key) => [key, first[key] + (second[key] - first[key]) * fraction])); return { ...interpolate(before, after), bodyBox: interpolate(before.bodyBox, after.bodyBox) };
}
function render() {
  const now = performance.now();
  while (activeEvents[0]?.expiresAt <= now) activeEvents.shift();
  faces.replaceChildren();
  for (const item of activeEvents) {
    const { event, box } = item; if (!box) continue;
    const face = document.createElement("button"); face.className = "speaker"; face.title = `${event.speaker}: ${event.text}`;
    face.setAttribute("aria-label", face.title); face.style.setProperty("--speaker-color", event.color); Object.assign(face.style, { left: `${100 * box.bodyBox.x / item.width}%`, top: `${100 * box.bodyBox.y / item.height}%`, width: `${100 * box.bodyBox.width / item.width}%`, height: `${100 * box.bodyBox.height / item.height}%` }); faces.append(face);
  }
  const captions = activeEvents.map(({ event }) => event);
  if (captions.length) caption.replaceChildren(...captions.flatMap((event, index) => { const line = document.createElement("span"); line.textContent = `${event.speaker}: ${event.text}`; line.style.color = event.color; return index ? [document.createTextNode("  "), line] : [line]; }));
  else caption.textContent = liveText || "Listening…";
  requestAnimationFrame(render);
}
function addTranscript(events) {
  events.forEach((event) => {
    const button = document.createElement("button"); button.className = "event"; button.innerHTML = `<time>${formatTime(event.start)} — ${event.speaker}</time>`; button.style.color = event.color; button.append(document.createTextNode(event.text));
    transcript.append(button);
  });
}
async function recognize(blob, number) {
  setStatus(`Recognizing clip ${number}…`);
  const response = await fetch("/recognize", { method: "POST", body: blob, headers: { "X-Whisper-Language": languageSelect.value } });
  const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "Recognition failed");
  if (number !== chunkNumber) return;
  const tracks = new Map(payload.tracks.map((track) => [track.track_id, track]));
  const expiresAt = performance.now() + configuration.chunkMilliseconds;
  payload.events.forEach((event) => activeEvents.push({ event, box: boxAt(tracks.get(event.faceTrack), event.end), width: payload.width, height: payload.height, expiresAt }));
  addTranscript(payload.events); setStatus(`Clip ${number} captioned. Camera remains live.`);
}
function queueClip(blob) { const number = ++chunkNumber; recognize(blob, number).catch((error) => setStatus(`Clip ${number}: ${error.message}`)); }
function startLiveTranscription() {
  if (!SpeechRecognition) return false;
  speechRecognition = new SpeechRecognition(); speechRecognition.continuous = true; speechRecognition.interimResults = true;
  if (languageSelect.value) speechRecognition.lang = browserLanguages[languageSelect.value] || languageSelect.value;
  speechRecognition.onresult = (event) => { liveText = Array.from(event.results).slice(event.resultIndex).map((result) => result[0].transcript).join(""); };
  speechRecognition.onend = () => { if (capturing) speechRecognition.start(); };
  try { speechRecognition.start(); return true; } catch { return false; }
}
function recorderOptions() {
  const mimeTypes = ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
  const mimeType = mimeTypes.find((candidate) => MediaRecorder.isTypeSupported(candidate));
  return mimeType ? { mimeType } : undefined;
}
function startChunk(generation = captureGeneration) {
  if (!capturing || !stream || generation !== captureGeneration) return;
  const pieces = []; const options = recorderOptions(); const currentRecorder = options ? new MediaRecorder(stream, options) : new MediaRecorder(stream); recorder = currentRecorder;
  currentRecorder.ondataavailable = (event) => { if (event.data.size) pieces.push(event.data); };
  currentRecorder.onstop = () => { if (pieces.length) queueClip(new Blob(pieces, { type: currentRecorder.mimeType })); if (capturing && generation === captureGeneration) startChunk(generation); };
  currentRecorder.start(); captureTimer = setTimeout(() => currentRecorder.stop(), configuration.chunkMilliseconds);
}
async function refreshCameras() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
  cameraSelect.replaceChildren(...devices.map((device, index) => new Option(device.label || `Camera ${index + 1}`, device.deviceId)));
}
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error("This browser does not support camera recording.");
  stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: cameraSelect.value ? { deviceId: { exact: cameraSelect.value } } : true });
  camera.srcObject = stream; await refreshCameras(); capturing = true; captureGeneration += 1; startButton.disabled = true; stopButton.disabled = false; languageSelect.disabled = true; const instantTranscript = startLiveTranscription(); startChunk(); setStatus(instantTranscript ? "Listening and recognizing camera segments…" : "Recognizing camera segments; use Chromium for immediate browser transcription.");
}
function stopCamera() { capturing = false; captureGeneration += 1; clearTimeout(captureTimer); if (recorder?.state === "recording") recorder.stop(); speechRecognition?.abort(); stream?.getTracks().forEach((track) => track.stop()); stream = undefined; startButton.disabled = false; stopButton.disabled = true; languageSelect.disabled = false; setStatus("Camera stopped. Remaining clips will finish recognition."); }
startButton.onclick = () => startCamera().catch((error) => setStatus(error.message)); stopButton.onclick = stopCamera;
cameraSelect.onchange = () => { if (capturing) { stopCamera(); startCamera().catch((error) => setStatus(error.message)); } };
transcriptToggle.onclick = () => { const collapsed = transcriptPanel.classList.toggle("collapsed"); transcriptToggle.setAttribute("aria-expanded", String(!collapsed)); transcriptToggle.textContent = collapsed ? "Show transcript" : "Collapse transcript"; };
requestAnimationFrame(render);
</script></html>"""


def build_page(chunk_milliseconds: int, language: str | None = None) -> str:
    """Return the standalone camera interface with its server configuration."""
    return PAGE_TEMPLATE.replace(
        "__CONFIGURATION__",
        json.dumps({"chunkMilliseconds": chunk_milliseconds, "language": language}),
    )
