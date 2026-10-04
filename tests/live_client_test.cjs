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
  assert.match(outline.getAttribute("points"), /^25,/);
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
  console.log(
    "live client: independent outlines, PCM retention, retry, freshness, stop flush, and resampling passed",
  );
}

check().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
