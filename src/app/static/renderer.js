class LiveRenderer {
  constructor(settings, sessionSeconds) {
    this.settings = settings;
    this.sessionSeconds = sessionSeconds;
    this.camera = document.querySelector("#camera");
    this.faces = document.querySelector("#faces");
    this.caption = document.querySelector("#caption");
    this.captionSpeaker = document.querySelector("#caption-speaker");
    this.captionFinal = document.querySelector("#caption-final");
    this.captionProvisional = document.querySelector("#caption-provisional");
    this.status = document.querySelector("#status");
    this.speakerState = document.querySelector("#speaker-state");
    this.pill = document.querySelector("#pill");
    this.pillDot = document.querySelector("#pill-dot");
    this.offCamera = document.querySelector("#offcamera");
    this.placeholder = document.querySelector("#placeholder");
    this.loadingOverlay = document.querySelector("#loading");
    this.toast = document.querySelector("#toast");
    this.toastText = document.querySelector("#toast-text");
    this.toastClose = document.querySelector("#toast-close");
    this.transcript = document.querySelector("#transcript");
    this.empty = document.querySelector("#empty");
    this.jump = document.querySelector("#jump");
    this.phase = "idle";
    this.stickToBottom = true;
    this.captionSpeakerName = "";
    this.captionColor = "";
    this.finalText = "";
    this.provisionalText = "";
    this.transcript.onscroll = () => {
      const distance =
        this.transcript.scrollHeight -
        this.transcript.scrollTop -
        this.transcript.clientHeight;
      this.stickToBottom = !(distance > 48);
      if (this.stickToBottom) this.jump.hidden = true;
    };
    this.jump.onclick = () => this.scrollToLatest();
    this.toastClose.onclick = () => {
      this.toast.hidden = true;
    };
    this.jump.hidden = true;
    this.toast.hidden = true;
    this.outlines = new Map();
    this.geometry = undefined;
    this.speaker = undefined;
    this.captionUntil = 0;
    this.capturing = false;
    this.multilingual = false;
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
    this.finalText = "";
    this.provisionalText = "";
    this.captionSpeakerName = "";
    this.provisionalEntry = undefined;
    this.lastCard = undefined;
    this.stickToBottom = true;
    this.jump.hidden = true;
    this.updateEmpty();
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

  setStatus(message, error = false) {
    this.status.textContent = message;
    if (error) this.showError(message);
  }

  showError(message) {
    this.toastText.textContent = message;
    this.toast.hidden = false;
  }

  setPhase(phase) {
    this.phase = phase;
    this.loadingOverlay.hidden = phase !== "loading";
  }

  updateEmpty() {
    this.empty.hidden = this.transcript.children.length > 0;
  }

  scrollToLatest() {
    this.transcript.scrollTop = this.transcript.scrollHeight;
    this.stickToBottom = true;
    this.jump.hidden = true;
  }

  afterTranscriptChange() {
    this.updateEmpty();
    if (this.stickToBottom) this.scrollToLatest();
    else this.jump.hidden = false;
  }

  initials(name) {
    const person = /^Person (\d+)$/.exec(name);
    if (person) return `P${person[1]}`;
    return (
      String(name)
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((word) => word[0].toUpperCase())
        .join("") || "?"
    );
  }

  makeCard({ name, color, time, language, offCamera, text, provisional }) {
    const entry = document.createElement("div");
    entry.className = provisional ? "event provisional" : "event";
    entry.style.borderLeftColor = color || "";
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.style.background = color || "";
    avatar.textContent = provisional ? "…" : this.initials(name);
    avatar.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    body.className = "body";
    const meta = document.createElement("div");
    meta.className = "meta";
    const nameElement = document.createElement("span");
    nameElement.className = "name";
    nameElement.textContent = name;
    meta.append(nameElement);
    if (time) {
      const stamp = document.createElement("time");
      stamp.textContent = time;
      meta.append(stamp);
    }
    if (this.multilingual && language) {
      const badge = document.createElement("span");
      badge.className = "badge language";
      badge.textContent = language.toUpperCase();
      meta.append(badge);
    }
    if (offCamera) {
      const badge = document.createElement("span");
      badge.className = "badge off-camera";
      badge.textContent = "off camera";
      meta.append(badge);
    }
    const content = document.createElement("div");
    content.className = "text";
    content.textContent = text;
    body.append(meta, content);
    entry.append(avatar, body);
    return entry;
  }

  setText(element, text) {
    if (element.textContent !== text) element.textContent = text;
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
      const minutes = Math.floor(event.start / 60);
      const offCamera = this.isNamedOffScreen(event);
      const last = this.lastCard;
      if (
        last &&
        last.name === event.speaker &&
        last.language === event.language &&
        last.offCamera === offCamera &&
        event.start - last.end < 4 &&
        event.end - last.start < 30 // keep paragraphs short and timestamped
      ) {
        last.text.textContent += ` ${event.text}`;
        last.end = event.end;
      } else {
        const entry = this.makeCard({
          name: event.speaker,
          color: event.color,
          time: `${minutes}:${String(Math.floor(event.start % 60)).padStart(2, "0")}`,
          language: event.language,
          offCamera,
          text: event.text,
        });
        this.transcript.append(entry);
        this.lastCard = {
          name: event.speaker,
          language: event.language,
          offCamera,
          start: event.start,
          end: event.end,
          text: entry.children[1].children[1],
        };
      }
      this.record("captions", Math.max(0, this.sessionSeconds() - event.end));
    }
    const text = payload.events.map((event) => event.text).join(" ");
    const provisional = payload.provisionalText || "";
    if (provisional) {
      this.provisionalEntry = this.makeCard({
        name: "Live speech",
        time: "provisional",
        language: payload.language,
        text: provisional,
        provisional: true,
      });
      this.transcript.append(this.provisionalEntry);
    }
    this.afterTranscriptChange();
    if (text || provisional) {
      const last = payload.events.at(-1);
      if (last) {
        this.captionSpeakerName = last.speaker;
        this.captionColor = last.color;
      } else if (this.speaker?.speaker) {
        this.captionSpeakerName = this.speaker.speaker;
        this.captionColor = this.speaker.color;
      }
      const words = [text, provisional]
        .filter(Boolean)
        .join(" ")
        .split(/\s+/)
        .slice(-this.settings.maximumPreviewWords);
      const provisionalCount = Math.min(
        provisional.split(/\s+/).filter(Boolean).length,
        words.length,
      );
      this.finalText = words.slice(0, words.length - provisionalCount).join(" ");
      this.provisionalText = words.slice(words.length - provisionalCount).join(" ");
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
    const named = this.isNamedOffScreen(estimate);
    this.renderStatus(estimate, named);
    const showing = this.capturing && performance.now() < this.captionUntil;
    const caption = this.caption;
    caption.hidden = !showing;
    if (showing) {
      this.setText(this.captionSpeaker, this.captionSpeakerName);
      this.captionSpeaker.style.color = this.captionColor || "";
      this.setText(this.captionFinal, this.finalText);
      this.setText(
        this.captionProvisional,
        this.provisionalText ? ` ${this.provisionalText}` : "",
      );
    }
    this.placeholder.hidden = this.capturing || this.phase !== "idle";
    requestAnimationFrame(() => this.render());
  }

  renderStatus(estimate, named) {
    let tone = "ready";
    let text = "Ready";
    let color = "";
    let chip = "";
    if (this.phase === "loading") {
      tone = "loading";
      text = "Loading models…";
    } else if (this.phase === "stopping" && !this.capturing) {
      tone = "loading";
      text = "Finishing captions…";
    } else if (this.capturing) {
      tone = "listening";
      text = "Listening";
      if (estimate?.state === "visible") {
        tone = "speaking";
        text = `${estimate.speaker} speaking`;
        color = estimate.color;
      } else if (named) {
        tone = "speaking";
        text = `${estimate.speaker} · off camera`;
        color = estimate.color;
        chip = `\u{1F399} ${estimate.speaker} · off camera`;
      } else if (estimate?.state === "off_screen") {
        tone = "offscreen";
        text = "Someone off camera";
        chip = "\u{1F399} Voice off camera";
      } else if (estimate?.state === "uncertain") {
        tone = "uncertain";
        text = "Not sure who is speaking";
      }
    }
    this.setText(this.speakerState, text);
    this.pill.setAttribute("data-tone", tone);
    this.pillDot.style.color = color;
    this.pillDot.style.background = color;
    this.offCamera.hidden = !chip;
    if (chip) {
      this.setText(this.offCamera, chip);
      this.offCamera.style.color = color;
    }
  }

  isNamedOffScreen(item) {
    return item?.state === "off_screen" && /^Person \d+$/.test(item.speaker);
  }

  drawFace(face, estimate) {
    const namespace = "http://www.w3.org/2000/svg";
    let group = this.outlines.get(face.trackId);
    if (!group) {
      group = document.createElementNS(namespace, "g");
      group.classList.add("speaker");
      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "img");
      const rectangle = document.createElementNS(namespace, "rect");
      rectangle.setAttribute("class", "outline");
      rectangle.setAttribute("rx", "10");
      rectangle.setAttribute("vector-effect", "non-scaling-stroke");
      const chip = document.createElementNS(namespace, "rect");
      chip.setAttribute("rx", "7");
      chip.setAttribute("height", "18");
      const label = document.createElementNS(namespace, "text");
      group.append(rectangle, chip, label);
      this.faces.append(group);
      this.outlines.set(face.trackId, group);
    }
    const [rectangle, chip, label] = group.children;
    const { x, y, width, height } = face.box;
    const name = estimate.speaker;
    const chipWidth = name.length * 6.6 + 14;
    rectangle.setAttribute("x", String(x));
    rectangle.setAttribute("y", String(y));
    rectangle.setAttribute("width", String(width));
    rectangle.setAttribute("height", String(height));
    rectangle.setAttribute("stroke", estimate.color);
    rectangle.style.color = estimate.color;
    const chipY = y >= 24 ? y - 23 : y + 5;
    chip.setAttribute("x", String(x));
    chip.setAttribute("y", String(chipY));
    chip.setAttribute("width", String(chipWidth));
    chip.setAttribute("fill", estimate.color);
    label.setAttribute("x", String(x + 7));
    label.setAttribute("y", String(chipY + 13));
    label.textContent = name;
    group.setAttribute("aria-label", `${name} is speaking`);
  }
}
