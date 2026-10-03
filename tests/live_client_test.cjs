const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function element() {
  const attributes = new Map();
  return {
    value: "it",
    disabled: false,
    children: [],
    style: { setProperty() {} },
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

async function check() {
  const elements = new Map();
  const requests = [];
  const timers = [];
  let releaseFirst;
  const context = vm.createContext({
    configuration: { chunkMilliseconds: 2000, language: "it" },
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
    performance: { now: () => 10000 },
    requestAnimationFrame() {},
    setTimeout(callback) {
      timers.push(callback);
    },
    clearTimeout() {},
    Blob,
    Map,
    Set,
    crypto: { randomUUID: () => "test-session" },
    fetch: async (_path, options) => {
      requests.push(options);
      if (requests.length === 1)
        await new Promise((resolve) => {
          releaseFirst = resolve;
        });
      return {
        ok: true,
        json: async () => ({
          sessionId: "test-session",
          events: [
            {
              start: requests.length,
              end: requests.length + 0.5,
              speaker: "Person 1",
              text: `speech ${requests.length}`,
              color: "#fff",
              faceTrack: "face-1",
            },
          ],
          provisionalText: "",
          timings: { total: 3 },
          captureEnd: 4,
        }),
      };
    },
  });
  vm.runInContext(fs.readFileSync("src/live_client.js", "utf8"), context);
  vm.runInContext(
    "sessionId = 'test-session'; sessionStartedAt = 0; capturing = true;",
    context,
  );
  vm.runInContext(
    "queueClip(new Blob(['first']), { id: 1, offset: 0, sessionId, generation: captureGeneration }, false);",
    context,
  );
  await Promise.resolve();
  vm.runInContext(
    "queueClip(new Blob(['first', 'second']), { id: 1, offset: 0, sessionId, generation: captureGeneration }, false);",
    context,
  );
  vm.runInContext(
    "queueClip(new Blob(['first', 'second', 'third']), { id: 1, offset: 0, sessionId, generation: captureGeneration }, true);",
    context,
  );
  assert.equal(requests.length, 1, "uploads must be serialized");
  releaseFirst();
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    requests.length,
    2,
    "pending cumulative snapshots should coalesce",
  );
  assert.equal(
    await requests[1].body.text(),
    "firstsecondthird",
    "speech captured under load must survive",
  );
  assert.equal(
    elements.get("#transcript").children.length,
    2,
    "valid older results must be committed",
  );
  assert.equal(vm.runInContext("activeEvents.length", context), 1);
  assert.equal(requests[1].headers["X-Capture-Final"], "true");
  vm.runInContext(
    "currentGeometry = { faces: [{trackId: 'face-1', box: {x: 10, y: 10, width: 20, height: 20}, contour: []}] }; geometryReceivedAt = performance.now(); render();",
    context,
  );
  const outline = elements.get("#faces").children[0];
  vm.runInContext("currentGeometry.faces[0].box.x = 25; render();", context);
  assert.equal(
    elements.get("#faces").children[0],
    outline,
    "tracking updates must preserve the focused overlay",
  );
  assert.match(
    outline.getAttribute("points"),
    /^25,/,
    "outline must follow current geometry",
  );
  vm.runInContext(
    "geometryReceivedAt -= TRACKING_EXPIRY_MILLISECONDS + 1; render();",
    context,
  );
  assert.equal(
    elements.get("#faces").children.length,
    0,
    "stale geometry must disappear",
  );
  assert.equal(
    vm.runInContext(
      "activeEvents.every(event => event.endsAt <= performance.now() + configuration.chunkMilliseconds)",
      context,
    ),
    true,
    "preview captions must not queue beyond the update cadence",
  );
  vm.runInContext(
    "sessionId = 'new-session'; applyResult({sessionId: 'test-session', events: [{text: 'stale'}]}, {sessionId: 'test-session'});",
    context,
  );
  assert.equal(
    elements.get("#transcript").children.length,
    2,
    "previous-session replies must not leak",
  );
  vm.runInContext("sessionId = 'test-session';", context);
  let attempts = 0;
  context.fetch = async (_path, options) => {
    requests.push(options);
    attempts += 1;
    return {
      status: attempts === 1 ? 429 : 200,
      ok: attempts > 1,
      json: async () => ({
        sessionId: "test-session",
        events: [],
        captureEnd: 10,
      }),
    };
  };
  vm.runInContext(
    "queueClip(new Blob(['retained']), {id: 2, offset: 4, sessionId, generation: captureGeneration}, false);",
    context,
  );
  vm.runInContext(
    "queueClip(new Blob(['retained', ' speech']), {id: 2, offset: 4, sessionId, generation: captureGeneration}, false);",
    context,
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    vm.runInContext("pendingClips.size", context),
    1,
    "busy response must retain audio",
  );
  timers.at(-1)();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    await requests.at(-1).body.text(),
    "retained speech",
    "retry must include newer buffered speech",
  );
  assert.equal(vm.runInContext("pendingClips.size", context), 0);
  const transcriptEntries = elements.get("#transcript").children.length;
  vm.runInContext(
    "applyResult({sessionId, captureEnd: 20, events: [{start: 1, end: 11, text: 'first long phrase'}, {start: 11, end: 20, text: 'second long phrase'}]}, {sessionId});",
    context,
  );
  assert.equal(
    vm.runInContext(
      "activeEvents[0].endsAt <= activeEvents[1].startsAt",
      context,
    ),
    true,
    "compressed captions must retain spoken order",
  );
  assert.equal(
    vm.runInContext(
      "activeEvents.at(-1).endsAt <= performance.now() + configuration.chunkMilliseconds",
      context,
    ),
    true,
    "a long recognition response must fit within one update cadence",
  );
  vm.runInContext(
    "applyResult({sessionId, captureEnd: 21, events: [{start: 20, end: 21, text: 'latest phrase'}]}, {sessionId});",
    context,
  );
  assert.equal(vm.runInContext("activeEvents.length", context), 1);
  assert.equal(
    elements.get("#transcript").children.length,
    transcriptEntries + 3,
    "refreshing the preview must preserve every transcript event",
  );
  console.log(
    "live client: serialized capture, coalescing, result lifecycle, caption scheduling, session isolation passed",
  );
}

check().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
