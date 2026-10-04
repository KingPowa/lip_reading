class LiveRenderer {
  constructor(settings, sessionSeconds) {
    this.settings = settings;
    this.sessionSeconds = sessionSeconds;
    this.camera = document.querySelector("#camera");
    this.faces = document.querySelector("#faces");
    this.caption = document.querySelector("#caption");
    this.status = document.querySelector("#status");
    this.speakerState = document.querySelector("#speaker-state");
    this.transcript = document.querySelector("#transcript");
    this.outlines = new Map();
    this.geometry = undefined;
    this.speaker = undefined;
    this.captionUntil = 0;
    this.captionText = "";
    this.capturing = false;
    this.provisionalEntry = undefined;
    this.metrics = {
      activity: [],
      vision: [],
      captions: [],
      transcription: [],
    };
  }

  reset() {
    this.faces.replaceChildren();
    this.transcript.replaceChildren();
    this.outlines.clear();
    this.geometry = undefined;
    this.speaker = undefined;
    this.captionUntil = 0;
    this.captionText = "";
    this.provisionalEntry = undefined;
    this.metrics = {
      activity: [],
      vision: [],
      captions: [],
      transcription: [],
    };
    window.liveMetrics = this.metrics;
  }

  record(channel, seconds) {
    this.metrics[channel].push(seconds);
    if (this.metrics[channel].length > this.settings.maximumMetricItems)
      this.metrics[channel].shift();
  }

  setStatus(message) {
    this.status.textContent = message;
  }

  updateSpeaker(estimate) {
    if (this.speaker && estimate.revision < this.speaker.revision) return;
    this.speaker = estimate;
  }

  updateGeometry(payload) {
    if (this.geometry && payload.timestamp < this.geometry.timestamp) return;
    this.geometry = payload;
    this.faces.setAttribute(
      "viewBox",
      `0 0 ${payload.width} ${payload.height}`,
    );
    this.updateSpeaker(payload.speaker);
    this.record(
      "vision",
      Math.max(0, this.sessionSeconds() - payload.timestamp),
    );
  }

  updateCaptions(payload) {
    this.provisionalEntry?.remove();
    this.provisionalEntry = undefined;
    for (const event of payload.events) {
      const entry = document.createElement("div");
      entry.className = "event";
      entry.style.color = event.color;
      const time = document.createElement("time");
      const minutes = Math.floor(event.start / 60);
      time.textContent = `${minutes}:${String(Math.floor(event.start % 60)).padStart(2, "0")} — ${event.speaker}`;
      entry.append(time, document.createTextNode(event.text));
      this.transcript.append(entry);
      this.record("captions", Math.max(0, this.sessionSeconds() - event.end));
    }
    const text = payload.events.map((event) => event.text).join(" ");
    const provisional = payload.provisionalText || "";
    if (provisional) {
      this.provisionalEntry = document.createElement("div");
      this.provisionalEntry.className = "event";
      const label = document.createElement("time");
      label.textContent = "Live speech — provisional";
      this.provisionalEntry.append(label, document.createTextNode(provisional));
      this.transcript.append(this.provisionalEntry);
    }
    if (text || provisional) {
      this.captionText = [text, provisional]
        .filter(Boolean)
        .join(" ")
        .split(/\s+/)
        .slice(-this.settings.maximumPreviewWords)
        .join(" ");
      this.captionUntil =
        performance.now() + this.settings.captionDisplayMilliseconds;
    }
    const delay = Math.max(0, this.sessionSeconds() - payload.captureEnd);
    this.record("transcription", payload.timings.transcription);
    const targetMissed = delay > this.settings.latencyTargetSeconds;
    this.setStatus(
      `Captions ${delay.toFixed(2)}s behind${targetMissed ? " — above the one second target; check the device load." : "."}`,
    );
  }

  render() {
    const now = this.sessionSeconds();
    const freshSpeaker =
      this.capturing &&
      this.speaker &&
      now - this.speaker.timestamp < this.settings.geometryExpirySeconds;
    const freshGeometry =
      this.geometry &&
      now - this.geometry.timestamp < this.settings.geometryExpirySeconds;
    const estimate = freshSpeaker ? this.speaker : undefined;
    const identity =
      estimate?.state === "visible" && freshGeometry
        ? estimate.faceTrack
        : undefined;
    const face = this.geometry?.faces.find(
      (candidate) => candidate.trackId === identity,
    );
    for (const [trackId, outline] of this.outlines) {
      if (face && trackId === identity) continue;
      outline.remove();
      this.outlines.delete(trackId);
    }
    if (face) this.drawFace(face, estimate);
    const labels = {
      silent: "No speech detected.",
      off_screen: "Voice outside the camera.",
      uncertain: "Voice detected; location uncertain.",
    };
    this.speakerState.textContent =
      estimate?.state === "visible"
        ? `${estimate.speaker} is speaking.`
        : labels[estimate?.state] ||
          "Waiting for current audio and camera evidence.";
    let captionText = this.capturing ? "Listening…" : "Camera stopped.";
    if (performance.now() < this.captionUntil) captionText = this.captionText;
    this.caption.textContent = captionText;
    requestAnimationFrame(() => this.render());
  }

  drawFace(face, estimate) {
    const namespace = "http://www.w3.org/2000/svg";
    let outline = this.outlines.get(face.trackId);
    if (!outline) {
      outline = document.createElementNS(namespace, "polygon");
      outline.classList.add("speaker");
      outline.setAttribute("fill", "none");
      outline.setAttribute("stroke-width", "3");
      outline.setAttribute("tabindex", "0");
      outline.setAttribute("role", "img");
      outline.append(document.createElementNS(namespace, "title"));
      this.faces.append(outline);
      this.outlines.set(face.trackId, outline);
    }
    const { x, y, width, height } = face.box;
    const points = [
      [x, y],
      [x + width, y],
      [x + width, y + height],
      [x, y + height],
    ];
    outline.setAttribute(
      "points",
      points.map((point) => point.join(",")).join(" "),
    );
    outline.setAttribute("stroke", estimate.color);
    outline.setAttribute("aria-label", `${estimate.speaker} is speaking`);
    outline.firstChild.textContent = `${estimate.speaker} is speaking`;
  }
}
