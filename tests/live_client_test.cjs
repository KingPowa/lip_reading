const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function element() {
  const attributes = new Map();
  return {
    value: "it",
    options: [{ value: "it" }],
    disabled: false,
    children: [],
    style: {},
    classList: { toggle() {}, add() {} },
    setAttribute(name, value) {
      attributes.set(name, value);
    },
    getAttribute(name) {
      return attributes.get(name);
    },
    get firstChild() {
      return this.children[0];
    },
    remove() {
      if (this.parent)
        this.parent.children = this.parent.children.filter(
          (child) => child !== this,
        );
    },
    add() {},
    replaceChildren(...children) {
      this.children = children;
    },
    append(...children) {
      children.forEach((child) => {
        child.parent = this;
      });
      this.children.push(...children);
    },
  };
}

const settings = {
  transcriptionIntervalMilliseconds: 350,
  language: "it",
  maximumSessionLanguages: 3,
  sampleRate: 16000,
  contextSeconds: 0,
  maximumWindowSeconds: 1,
  maximumBufferSeconds: 12,
  packetSeconds: 0.1,
  geometryExpirySeconds: 0.7,
  retryMilliseconds: 200,
  requestTimeoutMilliseconds: 10000,
  latencyTargetSeconds: 1,
  maximumPreviewWords: 14,
  maximumMetricItems: 2000,
  captionDisplayMilliseconds: 2500,
};

function load(context, name) {
  vm.runInContext(
    fs.readFileSync(`src/app/static/${name}.js`, "utf8"),
    context,
  );
}

async function check() {
  const elements = new Map();
  const requests = [];
  const timers = [];
  let releaseFirst;
  let now = 1000;
  const context = vm.createContext({
    configuration: settings,
    document: {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, element());
        return elements.get(selector);
      },
      createElement: element,
      createElementNS: element,
      createTextNode(text) {
        return { text };
      },
    },
    navigator: { language: "it-IT" },
    performance: { now: () => now },
    requestAnimationFrame() {},
    setTimeout(callback) {
      timers.push(callback);
    },
    clearTimeout() {},
    ArrayBuffer,
    DataView,
    Int16Array,
    AbortSignal,
    window: {},
    fetch: async (path, options) => {
      requests.push({ path, ...options });
      if (requests.length === 1)
        await new Promise((resolve) => {
          releaseFirst = resolve;
        });
      return {
        ok: true,
        status: 200,
        json: async () => ({
          sessionId: "test-session",
          captureEnd: 0.4,
          acknowledgedUntil: requests.length === 1 ? 0.4 : 0.8,
          events: [],
          provisionalText: "hello",
          timings: { transcription: 0.1 },
        }),
      };
    },
  });
  load(context, "audio-buffer");
  load(context, "renderer");
  load(context, "client");
  vm.runInContext(
    "sessionId = 'test-session'; sessionStartedAt = 0; capturing = true; audioBuffer = new AudioBufferQueue(configuration); audioBuffer.append(new Int16Array(6400).fill(1), 0); drainAudio();",
    context,
  );
  await Promise.resolve();
  vm.runInContext(
    "audioBuffer.append(new Int16Array(6400).fill(2), 0.4); drainAudio();",
    context,
  );
  assert.equal(requests.length, 1, "ASR uploads must be serialized");
  assert.equal(requests[0].path, "/audio");
  assert.equal(requests[0].headers["X-Whisper-Language"], "it");
  releaseFirst();
  await new Promise((resolve) => setImmediate(resolve));
  vm.runInContext("drainAudio();", context);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    requests[1].body.byteLength,
    12800,
    "pending snapshots retain all uncommitted audio",
  );
  const bytes = new DataView(requests[1].body);
  assert.equal(bytes.getInt16(0, true), 2);
  assert.equal(vm.runInContext("audioBuffer.chunks.length", context), 0);

  vm.runInContext(
    `renderer.capturing = true; renderer.updateGeometry({ timestamp: 0.9, width: 640, height: 360, faces: [{trackId:'face-1',box:{x:10,y:10,width:20,height:20}}], speaker: {timestamp:0.9,revision:2,state:'visible',faceTrack:'face-1',speaker:'Person 1',color:'#fff'}}); renderer.render();`,
    context,
  );
  const outline = elements.get("#faces").children[0];
  assert.ok(outline, "speaker outline must appear before any transcript event");
  vm.runInContext(
    "renderer.geometry.faces[0].box.x = 25; renderer.captionUntil = 0; renderer.render();",
    context,
  );
  assert.equal(
    elements.get("#faces").children[0],
    outline,
    "caption expiry must not expire a speaking outline",
  );
  const [outlineRect, nameChip, nameLabel] = outline.children;
  assert.equal(outlineRect.getAttribute("x"), "25");
  assert.equal(nameLabel.textContent, "Person 1", "face carries a name chip");
  assert.equal(nameChip.getAttribute("fill"), "#fff");
  assert.equal(elements.get("#speaker-state").textContent, "Person 1 speaking");
  vm.runInContext(
    "renderer.updateSpeaker({timestamp:0.9,revision:1,state:'silent'}); renderer.render();",
    context,
  );
  assert.equal(
    elements.get("#faces").children.length,
    1,
    "late activity replies must not overwrite newer fusion evidence",
  );
  now = 2000;
  vm.runInContext("renderer.render();", context);
  assert.equal(
    elements.get("#faces").children.length,
    0,
    "freshness uses capture age, not response arrival",
  );

  vm.runInContext(
    "renderer.updateSpeaker({timestamp:2.0,revision:9,state:'off_screen',faceTrack:null,speaker:'Person 2',color:'#abc'}); renderer.render();",
    context,
  );
  assert.equal(
    elements.get("#speaker-state").textContent,
    "Person 2 · off camera",
  );
  assert.equal(elements.get("#pill-dot").style.background, "#abc");
  assert.equal(elements.get("#offcamera").hidden, false, "named chip shows");
  assert.match(elements.get("#offcamera").textContent, /Person 2 · off camera/);
  assert.equal(elements.get("#offcamera").style.color, "#abc");
  assert.equal(elements.get("#faces").children.length, 0, "no box off camera");
  vm.runInContext(
    "renderer.updateSpeaker({timestamp:2.0,revision:10,state:'off_screen',faceTrack:null,speaker:'Off-screen speaker',color:'#888'}); renderer.render();",
    context,
  );
  assert.equal(
    elements.get("#speaker-state").textContent,
    "Someone off camera",
  );
  assert.match(elements.get("#offcamera").textContent, /Voice off camera/);
  assert.equal(elements.get("#offcamera").style.color, "", "unknown voice is neutral");
  vm.runInContext(
    "renderer.updateSpeaker({timestamp:2.0,revision:11,state:'uncertain',faceTrack:null,speaker:'Unknown',color:'#888'}); renderer.render();",
    context,
  );
  assert.equal(
    elements.get("#speaker-state").textContent,
    "Not sure who is speaking",
  );
  assert.equal(elements.get("#offcamera").hidden, true);

  const tags = () =>
    elements.get("#transcript").children.map((entry) => {
      const [name, ...rest] = entry.children[1].children[0].children;
      return [name.textContent, ...rest.map((item) => item.textContent)].join("|");
    });
  const badges = (entry) =>
    entry.children[1].children[0].children.filter((item) =>
      (item.className || "").startsWith("badge"),
    );
  const event = { start: 1, end: 2, speaker: "Person 1", text: "ciao", color: "#fff" };
  const reply = { captureEnd: 2, timings: { transcription: 0.1 }, provisionalText: "hi" };
  vm.runInContext("renderer.reset()", context);
  context.reply = { ...reply, language: "en", events: [{ ...event, language: "it" }] };
  vm.runInContext("renderer.updateCaptions(reply)", context);
  assert.ok(
    elements
      .get("#transcript")
      .children.every((entry) => badges(entry).length === 0),
    "single-language sessions show no language badge",
  );
  context.reply = { ...reply, language: "en", events: [{ ...event, language: "it", start: 10, end: 11 }] };
  vm.runInContext("renderer.multilingual = true; renderer.updateCaptions(reply)", context);
  const cards = elements.get("#transcript").children;
  assert.equal(cards.length, 3, "provisional card is replaced, not duplicated");
  assert.equal(tags()[1], "Person 1|0:10|IT");
  assert.equal(cards[1].children[0].textContent, "P1", "avatar initials");
  assert.equal(cards[1].style.borderLeftColor, "#fff");
  assert.equal(tags().at(-1), "Live speech|provisional|EN");
  assert.equal(elements.get("#empty").hidden, true, "empty state disappears");
  vm.runInContext("renderer.render()", context);
  assert.equal(elements.get("#caption").hidden, false);
  assert.equal(
    elements.get("#caption-speaker").textContent,
    "Person 1",
    "caption bar names the speaker",
  );
  assert.equal(elements.get("#caption-provisional").textContent, " hi");

  const textOf = (card) => card.children[1].children[1].textContent;
  const send = (events) => {
    context.merge = { ...reply, provisionalText: "", events };
    vm.runInContext("renderer.updateCaptions(merge)", context);
  };
  vm.runInContext("renderer.reset()", context);
  send([{ ...event, start: 1, end: 2 }]);
  send([{ ...event, start: 3, end: 4, text: "mondo" }]);
  assert.equal(elements.get("#transcript").children.length, 1, "same speaker merges");
  assert.equal(textOf(elements.get("#transcript").children[0]), "ciao mondo");
  send([{ ...event, start: 4, end: 5, speaker: "Person 2", text: "x" }]);
  send([{ ...event, start: 5, end: 6, speaker: "Person 2", text: "y" }]);
  send([{ ...event, start: 10, end: 11, speaker: "Person 2", text: "z" }]);
  assert.equal(elements.get("#transcript").children.length, 3, "new speaker and long gap start cards");
  send([{ ...event, start: 13, end: 41, speaker: "Person 2", text: "w" }]);
  assert.equal(elements.get("#transcript").children.length, 4, "paragraphs stop at 30 seconds");
  vm.runInContext("renderer.reset()", context);
  send([{ ...event, language: "it" }]);
  context.reply = { ...reply, language: "en", events: [{ ...event, language: "it", start: 10, end: 11 }] };
  vm.runInContext("renderer.updateCaptions(reply)", context);

  context.offReply = {
    ...reply,
    language: "en",
    events: [{ ...event, language: "it", speaker: "Person 2", state: "off_screen" }],
  };
  vm.runInContext("renderer.updateCaptions(offReply)", context);
  const offCameraCard = elements.get("#transcript").children.at(-2);
  assert.ok(
    badges(offCameraCard).some((item) => item.textContent === "off camera"),
    "named off-screen events carry an off camera badge",
  );

  const transcript = elements.get("#transcript");
  const jump = elements.get("#jump");
  assert.equal(jump.hidden, true, "jump button hidden while following");
  transcript.scrollHeight = 1000;
  transcript.clientHeight = 300;
  transcript.scrollTop = 0;
  transcript.onscroll();
  vm.runInContext("renderer.updateCaptions(reply)", context);
  assert.equal(jump.hidden, false, "scrolled-up readers get a jump button");
  assert.equal(transcript.scrollTop, 0, "scrolled-up readers are not moved");
  jump.onclick();
  assert.equal(transcript.scrollTop, 1000);
  assert.equal(jump.hidden, true);

  vm.runInContext("renderer.showError(\"boom\")", context);
  assert.equal(elements.get("#toast").hidden, false);
  assert.equal(elements.get("#toast-text").textContent, "boom");
  elements.get("#toast-close").onclick();
  assert.equal(elements.get("#toast").hidden, true, "toast is dismissible");

  let attempts = 0;
  context.fetch = async (path, options) => {
    attempts += 1;
    requests.push({ path, ...options });
    return {
      ok: attempts > 1,
      status: attempts === 1 ? 429 : 200,
      json: async () => ({
        sessionId: "test-session",
        captureEnd: 0.8,
        acknowledgedUntil: 1.2,
        events: [],
        timings: { transcription: 0.1 },
      }),
    };
  };
  vm.runInContext(
    "audioBuffer = new AudioBufferQueue(configuration); audioBuffer.append(new Int16Array(6400).fill(7), 0.8); drainAudio();",
    context,
  );
  await new Promise((resolve) => setImmediate(resolve));
  const retained = requests.at(-1);
  vm.runInContext(
    "audioBuffer.append(new Int16Array(6400).fill(3), 1.2); drainAudio();",
    context,
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    requests.at(-1).headers["X-Capture-Job"],
    retained.headers["X-Capture-Job"],
  );
  assert.deepEqual(
    new Uint8Array(requests.at(-1).body),
    new Uint8Array(retained.body),
    "retry keeps an identical idempotent job",
  );

  vm.runInContext(
    "const queue = new AudioBufferQueue(configuration); queue.append(new Int16Array(16000).fill(4),0); queue.append(new Int16Array(16000).fill(5),1); queue.acknowledge(1.8); const tail = queue.snapshot(true);",
    context,
  );
  assert.equal(vm.runInContext("tail.offset", context), 1.8);
  assert.equal(vm.runInContext("tail.samples.length", context), 3200);
  assert.equal(vm.runInContext("tail.final", context), true);
  assert.throws(
    () => vm.runInContext("queue.append(new Int16Array(1600),3)", context),
    /discontinuity/,
  );

  for (const rate of [44100, 48000]) {
    let Processor;
    const messages = [];
    const worklet = vm.createContext({
      sampleRate: rate,
      currentTime: 0,
      Int16Array,
      AudioWorkletProcessor: class {
        constructor() {
          this.port = {
            postMessage(message) {
              messages.push(message);
            },
          };
        }
      },
      registerProcessor(_name, processor) {
        Processor = processor;
      },
    });
    load(worklet, "audio-worklet");
    const processor = new Processor({ processorOptions: settings });
    for (let frame = 0; frame < rate; frame += 128) {
      worklet.currentTime = frame / rate;
      processor.process([
        [new Float32Array(Math.min(128, rate - frame)).fill(0.5)],
      ]);
    }
    processor.port.onmessage({ data: "flush" });
    const packets = messages.filter((message) => message.samples);
    const sampleCount = packets.reduce(
      (sum, message) => sum + message.samples.length,
      0,
    );
    assert.ok(
      Math.abs(sampleCount - 16000) <= 1,
      `${rate} resampling must retain phase`,
    );
    assert.ok(
      packets.every((packet) =>
        packet.samples.every((sample) => sample === 16384),
      ),
    );
    assert.equal(
      messages.at(-1).stopped,
      true,
      "Stop flushes the last partial PCM packet",
    );
  }
  const picker = vm.createContext({
    configuration: { ...settings, language: undefined },
    document: {
      querySelector(selector) {
        const item = element();
        if (selector === "#language")
          item.options = ["en", "it", "es", "fr"].map((value) => ({ value }));
        return item;
      },
      createElement: element,
      createElementNS: element,
    },
    navigator: { language: "it-IT" },
    performance: { now: () => 0 },
    requestAnimationFrame() {},
    setTimeout() {},
    AbortSignal,
    window: {},
  });
  load(picker, "audio-buffer");
  load(picker, "renderer");
  load(picker, "client");
  assert.equal(vm.runInContext("selectedLanguages().join()", picker), "it");
  assert.equal(vm.runInContext("chips.length", picker), 4, "one chip per language");
  assert.equal(
    vm.runInContext("chips.map((chip) => chip.getAttribute('aria-pressed')).join()", picker),
    "false,true,false,false",
    "the locale language chip is pressed",
  );
  vm.runInContext("chips[0].onclick(); chips[2].onclick();", picker);
  assert.equal(vm.runInContext("selectedLanguages().join()", picker), "en,it,es");
  vm.runInContext("chips[3].onclick();", picker);
  assert.equal(
    vm.runInContext("selectedLanguages().join()", picker),
    "en,it,es",
    "chips cap at three languages",
  );
  assert.match(vm.runInContext("languageHint.textContent", picker), /between 1 and 3/);
  assert.equal(vm.runInContext("chips[3].getAttribute('aria-pressed')", picker), "false");
  vm.runInContext("chips[0].onclick(); chips[2].onclick(); chips[1].onclick();", picker);
  assert.equal(vm.runInContext("selectedLanguages().join()", picker), "it", "last chip cannot be deselected");
  vm.runInContext("setPhase('running')", picker);
  assert.equal(vm.runInContext("chips.every((chip) => chip.disabled)", picker), true);
  vm.runInContext("chips[0].onclick();", picker);
  assert.equal(vm.runInContext("selectedLanguages().join()", picker), "it", "chips are locked during capture");
  assert.equal(vm.runInContext("toggleButton.textContent", picker), "Stop");
  vm.runInContext("setPhase('idle')", picker);
  assert.equal(vm.runInContext("toggleButton.disabled", picker), false);
  vm.runInContext("selectLanguages(['it'])", picker);
  vm.runInContext("selectLanguages(['it', 'en', 'es'])", picker);
  vm.runInContext("languageSelect.options[3].selected = true; languageSelect.onchange()", picker);
  assert.equal(
    vm.runInContext("selectedLanguages().join()", picker),
    "it",
    "a fourth language is reverted to the previous choice",
  );
  vm.runInContext("languageSelect.options[0].selected = true; languageSelect.onchange()", picker);
  assert.equal(vm.runInContext("selectedLanguages().join()", picker), "en,it");
  vm.runInContext("selectLanguages([]); languageSelect.onchange()", picker);
  assert.equal(
    vm.runInContext("selectedLanguages().join()", picker),
    "en,it",
    "at least one language stays selected",
  );
  console.log(
    "live client: independent outlines, PCM retention, retry, freshness, stop flush, and resampling passed",
  );
}

check().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
