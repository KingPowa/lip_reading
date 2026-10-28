// Feed recorded /audio responses through a worktree's LiveRenderer and count
// the transcript cards a user would see. Usage: node render_sim.cjs ROOT responses.json
const fs = require("node:fs");
const vm = require("node:vm");

const [root, responsesPath] = process.argv.slice(2);
const responses = JSON.parse(fs.readFileSync(responsesPath, "utf8"));

function element(tag = "div") {
  const attributes = new Map();
  const target = {
    tagName: tag,
    value: "it",
    options: [],
    children: [],
    style: {},
    dataset: {},
    textContent: "",
    className: "",
    hidden: false,
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    setAttribute(name, value) { attributes.set(name, value); },
    getAttribute(name) { return attributes.get(name); },
    removeAttribute(name) { attributes.delete(name); },
    get firstChild() { return this.children[0]; },
    get lastChild() { return this.children.at(-1); },
    get lastElementChild() { return this.children.at(-1); },
    get childElementCount() { return this.children.length; },
    remove() {
      if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    },
    replaceChildren(...children) { this.children = children; children.forEach((c) => (c.parent = this)); },
    append(...children) {
      children.forEach((c) => { if (typeof c === "object") c.parent = this; });
      this.children.push(...children.filter((c) => typeof c === "object"));
    },
    appendChild(child) { this.append(child); return child; },
    insertBefore(child, ref) {
      child.parent = this;
      const i = this.children.indexOf(ref);
      if (i < 0) this.children.push(child); else this.children.splice(i, 0, child);
      return child;
    },
    querySelector(selector) { return find(this, selector); },
    querySelectorAll(selector) { return findAll(this, selector); },
    addEventListener() {},
    scrollTo() {},
    scrollIntoView() {},
    focus() {},
    animate() { return { finished: Promise.resolve() }; },
    getBoundingClientRect() { return { x: 0, y: 0, width: 0, height: 0 }; },
  };
  return new Proxy(target, {
    get(object, key) {
      if (key in object) return object[key];
      if (typeof key === "symbol") return undefined;
      return undefined;
    },
  });
}

function matches(node, selector) {
  if (selector.startsWith(".")) return String(node.className).split(/\s+/).includes(selector.slice(1));
  return node.tagName === selector;
}
function findAll(node, selector, out = []) {
  for (const child of node.children || []) {
    if (matches(child, selector)) out.push(child);
    findAll(child, selector, out);
  }
  return out;
}
function find(node, selector) { return findAll(node, selector)[0] || null; }

let now = 0;
const elements = new Map();
const context = vm.createContext({
  configuration: {
    transcriptionIntervalMilliseconds: 100, language: "it,en", maximumSessionLanguages: 3,
    sampleRate: 16000, packetSeconds: 0.1, geometryExpirySeconds: 0.7, retryMilliseconds: 200,
    requestTimeoutMilliseconds: 10000, latencyTargetSeconds: 1, maximumPreviewWords: 14,
    maximumMetricItems: 2000, captionDisplayMilliseconds: 2500, maximumBufferSeconds: 12,
  },
  document: {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, element());
      return elements.get(selector);
    },
    querySelectorAll() { return []; },
    createElement: element,
    createElementNS: (_, tag) => element(tag),
    createTextNode(text) { return { text }; },
    addEventListener() {},
  },
  navigator: { language: "it-IT" },
  performance: { now: () => now * 1000 },
  requestAnimationFrame() {},
  setTimeout() { return 0; },
  clearTimeout() {},
  setInterval() { return 0; },
  clearInterval() {},
  window: {},
  console,
});
vm.runInContext(fs.readFileSync(`${root}/src/app/static/renderer.js`, "utf8"), context);
vm.runInContext("var renderer = new LiveRenderer(configuration, () => globalThis.__now); renderer.multilingual = true; renderer.capturing = true;", context);
for (const response of responses) {
  now = response.receivedAt;
  context.__now = now;
  context.__payload = response;
  vm.runInContext("renderer.updateCaptions(__payload)", context);
}
const transcript = elements.get("#transcript");
const cards = transcript.children.filter((card) => !String(card.className).includes("provisional"));
const texts = cards.map((card) => findAll(card, ".text").map((n) => n.textContent).join(" "));
const names = cards.map((card) => findAll(card, ".name").map((n) => n.textContent).join(" "));
const words = texts.map((text) => text.split(/\s+/).filter(Boolean).length);
const speakers = {};
names.forEach((name) => { speakers[name] = (speakers[name] || 0) + 1; });
process.stdout.write(JSON.stringify({
  cards: cards.length,
  wordsPerCard: cards.length ? +(words.reduce((a, b) => a + b, 0) / cards.length).toFixed(2) : 0,
  speakers,
  sample: texts.slice(0, 5).map((text, i) => `${names[i]}: ${text}`),
}));
