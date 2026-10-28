const test = require("node:test");
const assert = require("node:assert");
const p = require("../permissions");

test("media granted only for backend origin", () => {
  assert.equal(p.isPermissionAllowed("media", "http://127.0.0.1:5000/", 5000), true);
  assert.equal(p.isPermissionAllowed("media", "http://127.0.0.1:5000/x?y=1", 5000), true);
  assert.equal(p.isPermissionAllowed("media", "http://127.0.0.1:5001/", 5000), false);
  assert.equal(p.isPermissionAllowed("media", "https://127.0.0.1:5000/", 5000), false);
  assert.equal(p.isPermissionAllowed("media", "http://evil.com/", 5000), false);
  assert.equal(p.isPermissionAllowed("media", "http://127.0.0.1:5000.evil.com/", 5000), false);
  assert.equal(p.isPermissionAllowed("media", "", 5000), false);
});

test("other permissions denied", () => {
  for (const perm of ["geolocation", "notifications", "clipboard-read", "mediaKeySystem", "openExternal"]) {
    assert.equal(p.isPermissionAllowed(perm, "http://127.0.0.1:5000/", 5000), false);
  }
});

test("navigation allowed only to backend origin or local pages", () => {
  assert.equal(p.isNavigationAllowed("http://127.0.0.1:5000/a", 5000), true);
  assert.equal(p.isNavigationAllowed("https://example.com", 5000), false);
});

test("default language", () => {
  assert.equal(p.defaultLanguage("it-IT"), "it");
  assert.equal(p.defaultLanguage("pt-BR"), "en");
  assert.equal(p.defaultLanguage(undefined), "en");
});
