// Writes backend.json with the absolute project root so the packaged app can find the Python backend.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..", "..");
fs.writeFileSync(path.join(__dirname, "..", "backend.json"), JSON.stringify({ root }, null, 2) + "\n");
console.log("backend.json ->", root);
