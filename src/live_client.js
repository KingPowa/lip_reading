const camera = document.querySelector("#camera");
const faces = document.querySelector("#faces");
const caption = document.querySelector("#caption");
const status = document.querySelector("#status");
const cameraSelect = document.querySelector("#camera-select");
const languageSelect = document.querySelector("#language");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const retryButton = document.querySelector("#retry");
const transcript = document.querySelector("#transcript");
const transcriptPanel = document.querySelector("#transcript-panel");
const transcriptToggle = document.querySelector("#transcript-toggle");
const languageLocales = {
  en: "English",
  it: "Italiano",
  es: "Español",
  fr: "Français",
  de: "Deutsch",
};
const RECORDING_WINDOW_MILLISECONDS = 30000;
const RECORDING_OVERLAP_MILLISECONDS = 4000;
const TRACKING_INTERVAL_MILLISECONDS = 250;
const TRACKING_EXPIRY_MILLISECONDS = 1500;
const RETRY_MILLISECONDS = 1000;
const MAXIMUM_BUFFER_BYTES = 160 * 1024 * 1024;
const MAXIMUM_UPLOAD_BYTES = 100 * 1024 * 1024;
const MAXIMUM_BACKLOG_SECONDS = 60;
const JPEG_QUALITY = 0.8;
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const activeEvents = [];
const pendingClips = new Map();
const recorders = new Set();
const renderedOutlines = new Map();
const frameCanvas = document.createElement("canvas");
let stream;
let capturing = false;
let captureGeneration = 0;
let sessionId = "";
let sessionStartedAt = 0;
let chunkNumber = 0;
let windowNumber = 0;
let uploading = false;
let failedClip;
let trackingTimer;
let currentGeometry;
let geometryReceivedAt = 0;
let provisionalText = "";
let committedUntil = 0;
let visualRequests = 0;
let sessionEnding = false;
let renderedCaption = "";

if (configuration.language) {
  if (!languageLocales[configuration.language])
    languageSelect.add(
      new Option(configuration.language, configuration.language),
    );
  languageSelect.value = configuration.language;
} else {
  const browserLanguage = navigator.language.split("-")[0].toLowerCase();
  if (languageLocales[browserLanguage]) languageSelect.value = browserLanguage;
}

function setStatus(message) {
  status.textContent = message;
}
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}
function sessionSeconds() {
  return (performance.now() - sessionStartedAt) / 1000;
}

function render() {
  const now = performance.now();
  const visibleEvents = activeEvents
    .filter((item) => item.startsAt <= now && now < item.endsAt)
    .map((item) => item.event);
  for (let index = activeEvents.length - 1; index >= 0; index -= 1) {
    if (activeEvents[index].endsAt <= now) activeEvents.splice(index, 1);
  }
  const geometryFresh =
    currentGeometry && now - geometryReceivedAt < TRACKING_EXPIRY_MILLISECONDS;
  const highlighted = new Set();
  for (const event of visibleEvents) {
    if (!geometryFresh || !event.faceTrack || highlighted.has(event.faceTrack))
      continue;
    const person = currentGeometry.faces.find(
      (face) => face.trackId === event.faceTrack,
    );
    if (!person) continue;
    highlighted.add(event.faceTrack);
    let outline = renderedOutlines.get(event.faceTrack);
    if (!outline) {
      outline = document.createElementNS(SVG_NAMESPACE, "polygon");
      outline.classList.add("speaker");
      outline.setAttribute("fill-opacity", ".2");
      outline.setAttribute("stroke-width", "3");
      outline.setAttribute("tabindex", "0");
      outline.setAttribute("role", "img");
      outline.append(document.createElementNS(SVG_NAMESPACE, "title"));
      renderedOutlines.set(event.faceTrack, outline);
      faces.append(outline);
    }
    outline.setAttribute("stroke", event.color);
    outline.setAttribute("fill", person.contour.length ? event.color : "none");
    outline.setAttribute("aria-label", `${event.speaker}: ${event.text}`);
    const { x, y, width, height } = person.box;
    const points = person.contour.length
      ? person.contour
      : [
          [x, y],
          [x + width, y],
          [x + width, y + height],
          [x, y + height],
        ];
    outline.setAttribute(
      "points",
      points.map((point) => point.join(",")).join(" "),
    );
    outline.firstChild.textContent = `${event.speaker}: ${event.text}`;
  }
  for (const [trackId, outline] of renderedOutlines) {
    if (highlighted.has(trackId)) continue;
    outline.remove();
    renderedOutlines.delete(trackId);
  }
  const captionText =
    visibleEvents.map((event) => `${event.speaker}: ${event.text}`).join(" ") ||
    provisionalText ||
    String(capturing);
  if (captionText === renderedCaption) {
    requestAnimationFrame(render);
    return;
  }
  renderedCaption = captionText;
  if (visibleEvents.length) {
    caption.replaceChildren(
      ...visibleEvents.map((event) => {
        const line = document.createElement("span");
        line.textContent = `${event.speaker}: ${event.text} `;
        line.style.color = event.color;
        return line;
      }),
    );
    requestAnimationFrame(render);
    return;
  }
  caption.textContent = capturing ? "Listening…" : "Camera stopped.";
  if (provisionalText) caption.textContent = `Provisional: ${provisionalText}`;
  requestAnimationFrame(render);
}

function addTranscript(events) {
  for (const event of events) {
    const entry = document.createElement("div");
    entry.className = "event";
    entry.style.color = event.color;
    const time = document.createElement("time");
    time.textContent = `${formatTime(event.start)} — ${event.speaker}`;
    entry.append(time, document.createTextNode(event.text));
    transcript.append(entry);
  }
}

function applyResult(payload, item) {
  if (item.sessionId !== sessionId || payload.sessionId !== sessionId) return;
  provisionalText = payload.provisionalText || "";
  committedUntil = payload.committedUntil ?? committedUntil;
  const firstTime = payload.events[0]?.start || 0;
  const displayStart = performance.now();
  const lastTime = Math.max(
    firstTime,
    ...payload.events.map((event) => event.end),
  );
  const playbackScale = Math.min(
    1,
    configuration.chunkMilliseconds / ((lastTime - firstTime) * 1000 || 1),
  );
  // ponytail: the transcript keeps every event; preview captions yield to the newest update instead of accumulating delay.
  if (payload.events.length) activeEvents.length = 0;
  for (const event of payload.events) {
    activeEvents.push({
      event,
      startsAt: displayStart + (event.start - firstTime) * 1000 * playbackScale,
      endsAt: displayStart + (event.end - firstTime) * 1000 * playbackScale,
    });
  }
  addTranscript(payload.events);
  const delay = Math.max(0, sessionSeconds() - payload.captureEnd);
  setStatus(
    `Captioned through ${formatTime(payload.captureEnd)}. Processing ${(payload.timings?.total || 0).toFixed(1)}s; backlog ${delay.toFixed(1)}s.`,
  );
}

async function recognize(item) {
  const response = await fetch("/recognize", {
    method: "POST",
    body: item.blob,
    headers: {
      "X-Whisper-Language": item.language,
      "X-Capture-Session": item.sessionId,
      "X-Capture-Offset": String(item.offset),
      "X-Capture-Job": String(item.number),
      "X-Capture-Final": String(item.final),
    },
  });
  if (response.status === 429 || response.status >= 500) return false;
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Recognition failed");
  applyResult(payload, item);
  return true;
}

async function drainClips() {
  if (uploading || failedClip || !pendingClips.size) return;
  uploading = true;
  const [key, item] = pendingClips.entries().next().value;
  pendingClips.delete(key);
  try {
    const completed = await recognize(item);
    if (!completed) {
      // ponytail: one retained retry at a time; a worker pool only helps with independent model capacity.
      const latest = pendingClips.get(key) || item;
      const retained = [...pendingClips.entries()].filter(
        ([candidate]) => candidate !== key,
      );
      pendingClips.clear();
      pendingClips.set(key, latest);
      retained.forEach(([candidate, clip]) =>
        pendingClips.set(candidate, clip),
      );
      setStatus("Recognizer busy. All captured speech is buffered for retry.");
      uploading = false;
      setTimeout(drainClips, RETRY_MILLISECONDS);
      return;
    }
  } catch (error) {
    failedClip = item;
    stopCamera();
    retryButton.hidden = false;
    setStatus(
      `Recording retained: ${error.message}. Choose Retry recognition.`,
    );
  }
  uploading = false;
  finishCapture();
  if (pendingClips.size && !failedClip) drainClips();
}

function queueClip(blob, recording, final) {
  if (!blob.size || recording.generation !== captureGeneration) return;
  const key = `${recording.sessionId}:${recording.id}`;
  const item = {
    blob,
    offset: recording.offset,
    sessionId: recording.sessionId,
    number: ++chunkNumber,
    final,
    language: languageSelect.value,
  };
  pendingClips.set(key, item);
  const bufferedBytes = [...pendingClips.values()].reduce(
    (total, clip) => total + clip.blob.size,
    0,
  );
  if (
    blob.size > MAXIMUM_UPLOAD_BYTES ||
    bufferedBytes > MAXIMUM_BUFFER_BYTES ||
    sessionSeconds() - committedUntil > MAXIMUM_BACKLOG_SECONDS
  ) {
    stopCamera();
    setStatus(
      "Capture paused: recognition is behind. Buffered speech is finishing; restart after it drains.",
    );
  }
  drainClips();
}

function recorderOptions() {
  const mimeTypes = ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
  const mimeType = mimeTypes.find((candidate) =>
    MediaRecorder.isTypeSupported(candidate),
  );
  return mimeType ? { mimeType } : undefined;
}

function startWindow(generation = captureGeneration) {
  if (!capturing || generation !== captureGeneration) return;
  const recording = {
    id: ++windowNumber,
    offset: sessionSeconds(),
    sessionId,
    generation,
  };
  const pieces = [];
  const currentRecorder = new MediaRecorder(stream, recorderOptions());
  recorders.add(currentRecorder);
  currentRecorder.ondataavailable = (event) => {
    if (!event.data.size) return;
    pieces.push(event.data);
    if (currentRecorder.state !== "inactive")
      queueClip(
        new Blob(pieces, { type: currentRecorder.mimeType }),
        recording,
        false,
      );
  };
  const rotateTimer = setTimeout(
    () => startWindow(generation),
    RECORDING_WINDOW_MILLISECONDS - RECORDING_OVERLAP_MILLISECONDS,
  );
  const stopTimer = setTimeout(() => {
    if (currentRecorder.state !== "inactive") currentRecorder.stop();
  }, RECORDING_WINDOW_MILLISECONDS);
  currentRecorder.onstop = () => {
    clearTimeout(rotateTimer);
    clearTimeout(stopTimer);
    recorders.delete(currentRecorder);
    queueClip(
      new Blob(pieces, { type: currentRecorder.mimeType }),
      recording,
      !capturing,
    );
    pieces.length = 0;
    finishCapture();
  };
  currentRecorder.start(configuration.chunkMilliseconds);
}

async function trackCurrentFrame(generation) {
  if (!capturing || generation !== captureGeneration) return;
  visualRequests += 1;
  try {
    if (camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      frameCanvas.width = camera.videoWidth;
      frameCanvas.height = camera.videoHeight;
      frameCanvas.getContext("2d").drawImage(camera, 0, 0);
      const timestamp = sessionSeconds();
      const frame = await new Promise((resolve) =>
        frameCanvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
      );
      const response = await fetch("/track", {
        method: "POST",
        body: frame,
        headers: {
          "X-Capture-Session": sessionId,
          "X-Capture-Offset": String(timestamp),
        },
      });
      const payload = await response.json();
      if (!response.ok)
        throw new Error(payload.error || "Visual tracking unavailable");
      if (generation !== captureGeneration || !capturing) return;
      currentGeometry = payload;
      geometryReceivedAt = performance.now();
      faces.setAttribute("viewBox", `0 0 ${payload.width} ${payload.height}`);
    }
  } catch (error) {
    currentGeometry = undefined;
    setStatus(
      `Captions continue; visual tracking unavailable: ${error.message}`,
    );
  } finally {
    visualRequests -= 1;
    finishCapture();
  }
  if (capturing && generation === captureGeneration)
    trackingTimer = setTimeout(
      () => trackCurrentFrame(generation),
      TRACKING_INTERVAL_MILLISECONDS,
    );
}

async function refreshCameras() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
    (device) => device.kind === "videoinput",
  );
  cameraSelect.replaceChildren(
    ...devices.map(
      (device, index) =>
        new Option(device.label || `Camera ${index + 1}`, device.deviceId),
    ),
  );
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
    throw new Error("This browser does not support camera recording.");
  startButton.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: {
        width: { ideal: 640, max: 1280 },
        height: { ideal: 360, max: 720 },
        ...(cameraSelect.value
          ? { deviceId: { exact: cameraSelect.value } }
          : {}),
      },
    });
    camera.srcObject = stream;
    await refreshCameras();
    sessionId = crypto.randomUUID();
    sessionStartedAt = performance.now();
    captureGeneration += 1;
    capturing = true;
    activeEvents.length = 0;
    renderedOutlines.clear();
    faces.replaceChildren();
    renderedCaption = "";
    transcript.replaceChildren();
    currentGeometry = undefined;
    provisionalText = "";
    committedUntil = 0;
    sessionEnding = false;
    stopButton.disabled = false;
    languageSelect.disabled = true;
    cameraSelect.disabled = true;
    startWindow();
    trackCurrentFrame(captureGeneration);
    setStatus(
      "Recording continuous speech; local captions are provisional until committed.",
    );
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    stream = undefined;
    startButton.disabled = false;
    throw error;
  }
}

function stopCamera() {
  capturing = false;
  clearTimeout(trackingTimer);
  for (const currentRecorder of recorders)
    if (currentRecorder.state !== "inactive") currentRecorder.stop();
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  currentGeometry = undefined;
  stopButton.disabled = true;
  startButton.disabled =
    uploading ||
    pendingClips.size > 0 ||
    recorders.size > 0 ||
    Boolean(failedClip);
  languageSelect.disabled = startButton.disabled;
  cameraSelect.disabled = startButton.disabled;
  setStatus("Camera stopped. Buffered speech will finish recognition.");
  finishCapture();
}

async function finishCapture() {
  const busy =
    capturing ||
    uploading ||
    pendingClips.size ||
    failedClip ||
    recorders.size ||
    visualRequests;
  if (busy || sessionEnding || !sessionId) return;
  sessionEnding = true;
  startButton.disabled = true;
  try {
    const response = await fetch("/end", {
      method: "POST",
      body: "",
      headers: { "X-Capture-Session": sessionId },
    });
    if (!response.ok) throw new Error("Could not release the capture session");
    startButton.disabled = false;
    languageSelect.disabled = false;
    cameraSelect.disabled = false;
  } catch (error) {
    sessionEnding = false;
    setStatus(error.message);
    setTimeout(finishCapture, RETRY_MILLISECONDS);
  }
}

startButton.onclick = () =>
  startCamera().catch((error) => setStatus(error.message));
stopButton.onclick = stopCamera;
retryButton.onclick = () => {
  if (!failedClip) return;
  const retained = [
    [`${failedClip.sessionId}:retry`, failedClip],
    ...pendingClips.entries(),
  ];
  pendingClips.clear();
  retained.forEach(([key, item]) => pendingClips.set(key, item));
  failedClip = undefined;
  retryButton.hidden = true;
  drainClips();
};
transcriptToggle.onclick = () => {
  const collapsed = transcriptPanel.classList.toggle("collapsed");
  transcriptToggle.setAttribute("aria-expanded", String(!collapsed));
  transcriptToggle.textContent = collapsed
    ? "Show transcript"
    : "Collapse transcript";
};
requestAnimationFrame(render);
