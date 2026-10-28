"use strict";

const SUPPORTED_LANGUAGES = ["en", "it", "es", "fr", "de"];

function backendOrigin(port) {
  return `http://127.0.0.1:${port}`;
}

function originOf(url) {
  try {
    return new URL(url).origin;
  } catch {
    return null;
  }
}

// Grant only "media" to the backend origin; everything else is denied.
function isPermissionAllowed(permission, requestingUrl, port) {
  return permission === "media" && originOf(requestingUrl) === backendOrigin(port);
}

function isNavigationAllowed(url, port) {
  if (url.startsWith("data:") || url.startsWith("file:")) return true;
  return originOf(url) === backendOrigin(port);
}

function defaultLanguage(locale) {
  const lang = String(locale || "").toLowerCase().split(/[-_]/)[0];
  return SUPPORTED_LANGUAGES.includes(lang) ? lang : "en";
}

module.exports = { SUPPORTED_LANGUAGES, backendOrigin, isPermissionAllowed, isNavigationAllowed, defaultLanguage };
