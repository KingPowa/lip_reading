const cameraSelect = document.querySelector("#camera-select");
const languageSelect = document.querySelector("#language");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const retryButton = document.querySelector("#retry");
const transcriptToggle = document.querySelector("#transcript-toggle");
const transcriptPanel = document.querySelector("#transcript-panel");
const frameCanvas = document.createElement("canvas");
let sessionStartedAt = 0;
const sessionSeconds = () => (performance.now() - sessionStartedAt) / 1000;
const renderer = new LiveRenderer(configuration, sessionSeconds);
let sessionId = "";
let generation = 0;
let capturing = false;
let stopping = false;
let stream;
let audioContext;
let microphoneSource;
let microphoneNode;
let silentOutput;
let flushCompleted;
let audioBuffer;
let audioUploading = false;
let activityUploading = false;
let visualUploading = false;
let pendingActivity;
let failedAudio;
let retainedAudio;
let nextJob = 0;
let audioTimer;
let trackingTimer;
let closingSession = false;
let finalAudioSent = false;

if (configuration.language) {
  if (
    ![...languageSelect.options].some(
      (option) => option.value === configuration.language,
    )
  )
    languageSelect.add(
      new Option(configuration.language, configuration.language),
    );
  languageSelect.value = configuration.language;
} else {
  const locale = navigator.language.split("-")[0].toLowerCase();
  if ([...languageSelect.options].some((option) => option.value === locale))
    languageSelect.value = locale;
}

async function post(
  path,
  body,
  offset,
  extraHeaders = {},
  requestSession = sessionId,
) {
  const response = await fetch(path, {
    method: "POST",
    body,
    signal: AbortSignal.timeout(
      path === "/start"
        ? configuration.modelLoadingTimeoutMilliseconds
        : configuration.requestTimeoutMilliseconds,
    ),
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Capture-Session": requestSession,
      "X-Capture-Offset": String(offset),
      ...extraHeaders,
    },
  });
  if (response.status === 429 || response.status >= 500) return undefined;
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Live request failed");
  return payload;
}

function pcmBytes(samples) {
  const bytes = new ArrayBuffer(samples.length * 2);
  const view = new DataView(bytes);
  for (let index = 0; index < samples.length; index += 1)
    view.setInt16(index * 2, samples[index], true);
  return bytes;
}

function receiveAudio(message, captureGeneration, origin) {
  if (message.stopped) {
    flushCompleted?.();
    return;
  }
  if (captureGeneration !== generation || (!capturing && !stopping)) return;
  const offset = message.timestamp + origin;
  try {
    audioBuffer.append(message.samples, offset);
  } catch (error) {
    renderer.setStatus(error.message);
    stopCamera();
    return;
  }
  pendingActivity = {
    samples: message.samples,
    end: offset + message.samples.length / configuration.sampleRate,
    sessionId,
    generation: captureGeneration,
  };
  drainActivity();
  if (!audioUploading && !failedAudio) scheduleAudio(0);
  if (
    audioBuffer.backlogSeconds >=
    configuration.maximumBufferSeconds - configuration.packetSeconds
  ) {
    stopCamera();
    renderer.setStatus(
      "Capture paused: the audio decoder cannot keep up. Retained speech is finishing.",
    );
  }
}

async function drainActivity() {
  if (activityUploading || !pendingActivity) return;
  activityUploading = true;
  const item = pendingActivity;
  pendingActivity = undefined;
  try {
    const payload = await post(
      "/activity",
      pcmBytes(item.samples),
      item.end,
      {},
      item.sessionId,
    );
    if (
      payload &&
      item.generation === generation &&
      payload.sessionId === sessionId
    ) {
      renderer.updateSpeaker(payload.speaker);
      renderer.record("activity", Math.max(0, sessionSeconds() - item.end));
    }
  } catch (error) {
    renderer.setStatus(
      `Audio activity unavailable: ${error.message}. Captions continue.`,
    );
  } finally {
    activityUploading = false;
    if (pendingActivity) drainActivity();
    finishCapture();
  }
}

function scheduleAudio(
  delay = configuration.transcriptionIntervalMilliseconds,
) {
  clearTimeout(audioTimer);
  audioTimer = setTimeout(drainAudio, delay);
}

async function drainAudio() {
  if (audioUploading || failedAudio || !audioBuffer || finalAudioSent) return;
  const snapshot = retainedAudio || audioBuffer.snapshot(stopping);
  if (!snapshot) {
    if (capturing) scheduleAudio();
    finishCapture();
    return;
  }
  const minimumNewAudio =
    configuration.transcriptionIntervalMilliseconds / 1000;
  const insufficientNewAudio =
    !stopping && snapshot.end - audioBuffer.lastSubmittedEnd < minimumNewAudio;
  if (!retainedAudio && insufficientNewAudio) {
    scheduleAudio();
    return;
  }
  const item = retainedAudio || {
    ...snapshot,
    job: ++nextJob,
    sessionId,
    generation,
    language: languageSelect.value,
  };
  retainedAudio = item;
  audioUploading = true;
  try {
    const payload = await post(
      "/audio",
      pcmBytes(item.samples),
      item.offset,
      {
        "X-Whisper-Language": item.language,
        "X-Capture-Job": String(item.job),
        "X-Capture-Final": String(item.final),
      },
      item.sessionId,
    );
    if (!payload) {
      renderer.setStatus(
        "Audio decoder busy. Speech is retained; camera tracking continues.",
      );
      scheduleAudio(configuration.retryMilliseconds);
      return;
    }
    if (item.generation !== generation || payload.sessionId !== sessionId)
      return;
    audioBuffer.acknowledge(payload.acknowledgedUntil);
    audioBuffer.lastSubmittedEnd = item.end;
    renderer.updateCaptions(payload);
    retainedAudio = undefined;
    finalAudioSent = item.final;
    const pendingSeconds = audioBuffer.latestEnd - item.end;
    const catchUp = stopping || pendingSeconds >= minimumNewAudio;
    scheduleAudio(
      catchUp ? 0 : configuration.transcriptionIntervalMilliseconds,
    );
  } catch (error) {
    failedAudio = item;
    retryButton.hidden = false;
    stopCamera();
    renderer.setStatus(
      `Speech retained: ${error.message}. Choose Retry recognition.`,
    );
  } finally {
    audioUploading = false;
    finishCapture();
  }
}

async function trackCurrentFrame(captureGeneration) {
  if (!capturing || captureGeneration !== generation) return;
  visualUploading = true;
  const requestStarted = performance.now();
  try {
    const camera = renderer.camera;
    if (camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      const scale = Math.min(
        1,
        configuration.maximumFrameWidth / camera.videoWidth,
        configuration.maximumFrameHeight / camera.videoHeight,
      );
      frameCanvas.width = Math.round(camera.videoWidth * scale);
      frameCanvas.height = Math.round(camera.videoHeight * scale);
      frameCanvas
        .getContext("2d")
        .drawImage(camera, 0, 0, frameCanvas.width, frameCanvas.height);
      const timestamp = sessionSeconds();
      const frame = await new Promise((resolve) =>
        frameCanvas.toBlob(resolve, "image/jpeg", configuration.jpegQuality),
      );
      if (!frame) throw new Error("Could not encode the current frame");
      const payload = await post("/track", frame, timestamp);
      if (
        payload &&
        capturing &&
        captureGeneration === generation &&
        payload.sessionId === sessionId
      )
        renderer.updateGeometry(payload);
    }
  } catch (error) {
    renderer.setStatus(
      `Camera detection unavailable: ${error.message}. Microphone captions continue.`,
    );
  } finally {
    visualUploading = false;
    finishCapture();
  }
  if (capturing && captureGeneration === generation) {
    const elapsed = performance.now() - requestStarted;
    trackingTimer = setTimeout(
      () => trackCurrentFrame(captureGeneration),
      Math.max(0, configuration.trackingIntervalMilliseconds - elapsed),
    );
  }
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
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode)
    throw new Error(
      "This browser requires secure microphone access and AudioWorklet support.",
    );
  startButton.disabled = true;
  try {
    sessionId = crypto.randomUUID();
    renderer.setStatus("Preparing the selected speech language…");
    const ready = await post("/start", "", 0, {
      "X-Whisper-Language": languageSelect.value,
    });
    if (!ready) throw new Error("Speech model is busy. Try starting again.");
    audioContext = new AudioContext({ latencyHint: "interactive" });
    await audioContext.resume();
    await audioContext.audioWorklet.addModule("/audio-worklet.js");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: {
        width: { ideal: configuration.maximumFrameWidth },
        height: { ideal: configuration.maximumFrameHeight },
        ...(cameraSelect.value
          ? { deviceId: { exact: cameraSelect.value } }
          : {}),
      },
    });
    renderer.camera.srcObject = stream;
    await refreshCameras();
    sessionStartedAt = performance.now();
    generation += 1;
    capturing = true;
    stopping = false;
    closingSession = false;
    finalAudioSent = false;
    audioBuffer = new AudioBufferQueue(configuration);
    pendingActivity = undefined;
    retainedAudio = undefined;
    failedAudio = undefined;
    renderer.reset();
    renderer.capturing = true;
    microphoneSource = audioContext.createMediaStreamSource(stream);
    microphoneNode = new AudioWorkletNode(audioContext, "microphone-pcm", {
      processorOptions: configuration,
    });
    silentOutput = audioContext.createGain();
    silentOutput.gain.value = 0;
    const captureGeneration = generation;
    let audioClockOrigin;
    microphoneNode.port.onmessage = (event) => {
      if (event.data.samples && audioClockOrigin === undefined)
        audioClockOrigin =
          Math.max(
            0,
            sessionSeconds() -
              event.data.samples.length / configuration.sampleRate,
          ) - event.data.timestamp;
      receiveAudio(event.data, captureGeneration, audioClockOrigin);
    };
    microphoneSource
      .connect(microphoneNode)
      .connect(silentOutput)
      .connect(audioContext.destination);
    stopButton.disabled = false;
    languageSelect.disabled = true;
    cameraSelect.disabled = true;
    scheduleAudio();
    trackCurrentFrame(captureGeneration);
    renderer.setStatus(
      "Microphone captions and camera speaker detection are running.",
    );
  } catch (error) {
    capturing = false;
    renderer.capturing = false;
    stream?.getTracks().forEach((track) => track.stop());
    stream = undefined;
    await audioContext?.close();
    audioContext = undefined;
    try {
      await post("/end", "", 0);
    } catch {}
    startButton.disabled = false;
    throw error;
  }
}

async function stopCamera() {
  if (!capturing) return;
  capturing = false;
  stopping = true;
  renderer.capturing = false;
  stopButton.disabled = true;
  clearTimeout(trackingTimer);
  if (microphoneNode) {
    await new Promise((resolve) => {
      flushCompleted = resolve;
      microphoneNode.port.postMessage("flush");
    });
    flushCompleted = undefined;
    microphoneSource.disconnect();
    microphoneNode.disconnect();
    silentOutput.disconnect();
    microphoneNode = undefined;
  }
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  await audioContext?.close();
  audioContext = undefined;
  if (!audioBuffer.initialized) finalAudioSent = true;
  scheduleAudio(0);
  finishCapture();
}

async function finishCapture() {
  const busy =
    capturing ||
    microphoneNode ||
    audioContext ||
    audioUploading ||
    activityUploading ||
    visualUploading ||
    pendingActivity ||
    failedAudio;
  if (busy || !finalAudioSent || closingSession || !sessionId) return;
  closingSession = true;
  try {
    const payload = await post("/end", "", 0);
    if (!payload) {
      closingSession = false;
      setTimeout(finishCapture, configuration.retryMilliseconds);
      return;
    }
    startButton.disabled = false;
    languageSelect.disabled = false;
    cameraSelect.disabled = false;
    stopping = false;
    renderer.setStatus("Capture stopped. All retained speech has finished.");
  } catch (error) {
    closingSession = false;
    renderer.setStatus(error.message);
    setTimeout(finishCapture, configuration.retryMilliseconds);
  }
}

startButton.onclick = () =>
  startCamera().catch((error) => renderer.setStatus(error.message));
stopButton.onclick = () =>
  stopCamera().catch((error) => renderer.setStatus(error.message));
retryButton.onclick = () => {
  if (!failedAudio) return;
  retainedAudio = failedAudio;
  failedAudio = undefined;
  retryButton.hidden = true;
  scheduleAudio(0);
};
transcriptToggle.onclick = () => {
  const collapsed = transcriptPanel.classList.toggle("collapsed");
  transcriptToggle.setAttribute("aria-expanded", String(!collapsed));
  transcriptToggle.textContent = collapsed
    ? "Show transcript"
    : "Collapse transcript";
};
renderer.render();
