const { contextBridge, ipcRenderer } = require("electron");

// The hidden-inset title bar puts the window buttons over the page's top-left corner.
if (process.platform === "darwin") {
  window.addEventListener("DOMContentLoaded", () => {
    document.documentElement.dataset.shell = "mac";
  });
}

contextBridge.exposeInMainWorld("shell", {
  retry: () => ipcRenderer.send("retry"),
  onLog: (cb) => ipcRenderer.on("log", (_e, lines) => cb(lines)),
  onError: (cb) => ipcRenderer.on("error", (_e, info) => cb(info)),
  ready: () => ipcRenderer.send("loading-ready"),
});
