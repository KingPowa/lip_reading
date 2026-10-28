"use strict";
const { app, BrowserWindow, Menu, dialog, ipcMain, session, shell, systemPreferences } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const net = require("net");
const http = require("http");
const path = require("path");
const perms = require("./permissions");

const SMOKE = process.env.SPEAKER_CAPTIONS_SMOKE === "1";
const MAX_LOG_LINES = 200;

let win = null;
let backend = null;
let port = 0;
let root = null;
let logLines = [];
let logStream = null;
let pollTimer = null;
let runId = 0;
let quitting = false;

if (!app.requestSingleInstanceLock()) {
  app.quit();
  process.exit(0);
}
app.on("second-instance", () => {
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});

function prefsPath() {
  return path.join(app.getPath("userData"), "prefs.json");
}

function loadLanguage() {
  try {
    const p = JSON.parse(fs.readFileSync(prefsPath(), "utf8"));
    const langs = String(p.language || "").split(",").filter((l) => perms.SUPPORTED_LANGUAGES.includes(l));
    if (langs.length) return langs.slice(0, 3).join(",");
  } catch {}
  return perms.defaultLanguage(app.getLocale());
}

function findRoot() {
  if (process.env.SPEAKER_CAPTIONS_ROOT) return process.env.SPEAKER_CAPTIONS_ROOT;
  if (app.isPackaged) {
    try {
      return JSON.parse(fs.readFileSync(path.join(process.resourcesPath, "backend.json"), "utf8")).root;
    } catch {}
  }
  return path.resolve(__dirname, "..");
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port: p } = srv.address();
      srv.close(() => resolve(p));
    });
  });
}

function send(channel, payload) {
  if (win && !win.isDestroyed() && win.webContents.getURL().startsWith("file:")) {
    win.webContents.send(channel, payload);
  }
}

function pushLog(text) {
  for (const line of String(text).split(/\r?\n/)) {
    if (!line) continue;
    logLines.push(line);
    if (logLines.length > MAX_LOG_LINES) logLines.shift();
  }
  if (logStream) logStream.write(text);
  send("log", logLines.slice(-60));
}

function showError(title, message) {
  stopPolling();
  const info = { title, message, lines: logLines.slice(-60) };
  const show = () => send("error", info);
  if (win && win.webContents.getURL().startsWith("file:")) show();
  else loadLoading().then(() => setTimeout(show, 100));
}

function loadLoading() {
  return win.loadFile(path.join(__dirname, "loading.html"));
}

function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

function killBackend() {
  const child = backend;
  backend = null;
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      try { child.kill("SIGKILL"); } catch {}
    }, 3000);
    child.once("exit", () => { clearTimeout(timer); resolve(); });
    try { child.kill("SIGTERM"); } catch { clearTimeout(timer); resolve(); }
  });
}

function killBackendSync() {
  // Last resort on exit: make sure nothing is left behind.
  if (backend && backend.exitCode === null) {
    try { backend.kill("SIGKILL"); } catch {}
  }
}

function startBackend() {
  const python = path.join(root, ".venv", "bin", "python");
  if (!fs.existsSync(python)) {
    showError("Python environment not found", `Run \`uv sync\` in ${root}`);
    return false;
  }
  const language = loadLanguage();
  const myRun = ++runId;
  const child = spawn(
    python,
    ["-m", "src.app.live", "--host", "127.0.0.1", "--port", String(port), "--language", language],
    {
      cwd: root,
      env: { ...process.env, MPLCONFIGDIR: path.join(app.getPath("userData"), "mpl"), PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  backend = child;
  pushLog(`[shell] starting backend on port ${port}, language ${language}\n`);
  child.stdout.on("data", (d) => pushLog(d.toString()));
  child.stderr.on("data", (d) => pushLog(d.toString()));
  child.on("error", (err) => {
    if (myRun === runId) showError("Could not start the backend", String(err.message || err));
  });
  child.on("exit", (code, signal) => {
    if (myRun !== runId || quitting || backend !== child) return;
    backend = null;
    showError("The backend stopped", `Exit code ${code ?? signal}. See the log below (full log: ${path.join(app.getPath("userData"), "backend.log")}).`);
  });
  return true;
}

function probe() {
  return new Promise((resolve) => {
    const req = http.get({ host: "127.0.0.1", port, path: "/", timeout: 1000 }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });
}

function pollUntilReady(myRun) {
  const tick = async () => {
    if (myRun !== runId || quitting) return;
    if (await probe()) {
      pollTimer = null;
      onBackendReady();
      return;
    }
    pollTimer = setTimeout(tick, 500);
  };
  tick();
}

async function ensureMediaAccess() {
  if (process.platform !== "darwin" || SMOKE) return true;
  const cam = await systemPreferences.askForMediaAccess("camera");
  const mic = await systemPreferences.askForMediaAccess("microphone");
  return cam && mic;
}

async function onBackendReady() {
  const ok = await ensureMediaAccess();
  if (!ok) {
    showError(
      "Camera or microphone access denied",
      "Open System Settings > Privacy & Security > Camera / Microphone, enable Speaker Captions, then press Retry.",
    );
    return;
  }
  if (!win || win.isDestroyed()) return;
  await win.loadURL(`${perms.backendOrigin(port)}/`);
  if (SMOKE) runSmoke();
}

async function runSmoke() {
  let result;
  try {
    const page = await win.webContents.executeJavaScript(
      `({ mediaDevices: navigator.mediaDevices !== undefined, audioWorklet: typeof AudioWorkletNode === "function", title: document.title, secure: window.isSecureContext })`,
    );
    result = { ok: page.mediaDevices && page.audioWorklet, ...page, port };
  } catch (err) {
    result = { ok: false, error: String(err) };
  }
  if (process.env.SPEAKER_CAPTIONS_SMOKE_OUT) {
    fs.writeFileSync(process.env.SPEAKER_CAPTIONS_SMOKE_OUT, JSON.stringify(result));
  }
  app.quit();
}

function launch() {
  stopPolling();
  logLines = [];
  send("log", []);
  if (!startBackend()) return;
  pollUntilReady(runId);
}

async function retry() {
  stopPolling();
  runId++;
  await killBackend();
  if (win && !win.isDestroyed()) await loadLoading();
  launch();
}

function setupSession() {
  const ses = session.defaultSession;
  ses.setPermissionRequestHandler((wc, permission, callback, details) => {
    callback(perms.isPermissionAllowed(permission, details.requestingUrl || wc.getURL(), port));
  });
  ses.setPermissionCheckHandler((wc, permission, requestingOrigin) => {
    return perms.isPermissionAllowed(permission, requestingOrigin, port);
  });
}

function buildMenu() {
  const template = [
    ...(process.platform === "darwin" ? [{ role: "appMenu" }] : []),
    { role: "editMenu" },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        ...(app.isPackaged ? [] : [{ role: "toggleDevTools" }]),
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    { role: "windowMenu" },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: "#0f1115",
    title: "Speaker Captions",
    show: !SMOKE,
    ...(process.platform === "darwin" ? { titleBarStyle: "hiddenInset" } : {}),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (!perms.isNavigationAllowed(url, port)) {
      event.preventDefault();
      if (/^https?:/.test(url)) shell.openExternal(url);
    }
  });
  win.on("closed", () => { win = null; });
}

ipcMain.on("retry", () => { retry(); });

app.whenReady().then(async () => {
  root = findRoot();
  port = await freePort();
  setupSession();
  buildMenu();
  fs.mkdirSync(app.getPath("userData"), { recursive: true });
  logStream = fs.createWriteStream(path.join(app.getPath("userData"), "backend.log"), { flags: "w" });
  createWindow();
  await loadLoading();
  launch();
});

app.on("window-all-closed", () => app.quit());

app.on("before-quit", (event) => {
  if (quitting) return;
  quitting = true;
  stopPolling();
  if (backend) {
    event.preventDefault();
    killBackend().finally(() => app.quit());
  }
});
app.on("will-quit", killBackendSync);
