# Live identity re-entry — ECC TDD evidence, 2026-10-04

## Intent and workflow

Journeys came from the user's request, without a separate plan file: retain a person's anonymous label when they leave and return, distinguish them from other previously observed people, and preserve off-screen speech detection.

Applied ECC's `skills/tdd-workflow/SKILL.md` from the installed ECC 2.2.3 plugin. The project uses Python `unittest`, as documented in README. No commits or pushes were made: the user's AGENTS.md prohibits them unless explicitly requested, overriding the skill's checkpoint-commit guidance.

## Change

Live face crops now produce normalized 128-dimensional features using the local [OpenCV SFace model](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Eye, nose, and mouth landmarks align each crop before feature inference. The first startup obtains the model and checks the pinned SHA-256; subsequent detectors reuse one encoder per model directory, with serialized native inference. Custom `--model-directory` applies to the identity model too.

The capture's tracker owns an independent gallery of up to 128 identities, each with five recent feature examples. Geometry still expires after two seconds, while features survive until session removal or capacity eviction. Matching uses cosine similarity >= 0.45 and a >= 0.08 margin over the next identity. These are conservative application thresholds, not an accuracy calibration. Feature matches are reserved across the whole frame before box continuity, preventing a face without landmarks from taking a matched identity. A person receives at most one box per frame. Weak or ambiguous matches create a new identity; missing features allow only short geometry continuity.

Mouth history is removed when a person is absent from the current frame. Restoring an identity cannot restore stale speaking evidence. Galleries never enter public JSON payloads and are not persisted between captures.

## RED and GREEN evidence

Command: `MPLCONFIGDIR=/private/tmp/lip-reading-matplotlib .venv/bin/python -m unittest tests.test_live_identity -v`.

Initial RED: all ten tests executed and errored because the previous observation contract did not accept identity features. After adding feature transport alone, six behavioral tests failed: re-entry changed `face-1` to `face-2`, the returning speaker changed `Person 1` to `Person 2`, replacements inherited old IDs, and swapped positions swapped identities. Four existing-compatible behaviors passed.

GREEN: adding the session gallery made those same ten tests pass, alongside the existing nine tracking tests. Seven encoder tests then passed, covering alignment, normalized output, missing landmarks, corrupt weights, atomic verified downloads, network failures, and local model reuse.

Integration RED: detector tests failed because current frames returned no features, and the service ignored the configured model directory. GREEN: detector feature extraction and model-directory wiring passed both tests. The service payload test also passed after its assertion was corrected to the existing schema: face payloads carry `trackId`; display colors belong to speaker payloads.

A further RED test demonstrated stale mouth history after a brief absence: mouth motion remained `0.07` on the first return observation. Clearing absent mouth histories made the same test pass with motion `0.0`, the original identity, and an `off_screen` speaker estimate until fresh motion is available.

## Guarantees

| Guarantee | Evidence | Type | Result |
| --- | --- | --- | --- |
| Return beyond geometry expiry, at another position, restores the original ID | `tests/test_live_identity.py` | Unit | PASS |
| A different face in the same place does not inherit the previous ID | `tests/test_live_identity.py` | Unit | PASS |
| Two people returning in reverse order or crossing retain distinct IDs | `tests/test_live_identity.py` | Unit | PASS |
| Ambiguous matches are not assigned either known ID | `tests/test_live_identity.py` | Unit | PASS |
| Identity assignment is one-to-one; feature matches precede geometry fallback | `tests/test_live_identity.py` | Unit | PASS |
| Missing/invalid features do not corrupt identity memory | `tests/test_live_identity.py` | Unit | PASS |
| Capacity eviction, copied feature arrays, out-of-order rejection, and session isolation behave correctly | `tests/test_live_identity.py` | Unit | PASS |
| Off-screen speech survives absence; restored IDs require fresh mouth motion | `tests/test_live_identity.py`, `tests/test_live.py` | Unit/integration | PASS |
| Current-frame extraction sends features to tracking without exposing them in JSON | `tests/test_live_tracking.py`, `tests/test_live.py` | Integration | PASS |
| Encoder alignment and model provisioning reject invalid inputs and weights | `tests/test_face_identity.py` | Unit | PASS |
| Four actual faces retain IDs after absence and mirroring; two actual faces retain IDs after swapping positions | `output/identity-check/reentry.py` | Real-model replay | PASS |

## Final validation

```bash
MPLCONFIGDIR=/private/tmp/lip-reading-matplotlib \
PYTHONPATH=/private/tmp/lip-reading-check-tools \
COVERAGE_FILE=/private/tmp/lip-reading-identity.coverage \
.venv/bin/python -m coverage run --branch \
  --source=src.core.tracking.identity,src.core.tracking.realtime,src.core.vision.face_identity,src.core.vision.live,src.app.service \
  -m unittest

PYTHONPATH=/private/tmp/lip-reading-check-tools \
COVERAGE_FILE=/private/tmp/lip-reading-identity.coverage \
.venv/bin/python -m coverage report -m
```

Actual result: **101 tests passed**, with no skips. Coverage of the five affected behavioral modules, including branches: **95% overall**; identity gallery 98%, real-time tracker 97%, encoder 91%, live detector 90%, service 96%. Ruff and coverage were installed into `/private/tmp/lip-reading-check-tools`; project dependencies and lockfile were unchanged.

Also passed: `node tests/live_client_test.cjs`, `.venv/bin/python -m compileall -q src tests`, Ruff lint and formatting checks for all changed Python files, and `git diff --check`. This Python project has no npm formatting script.

## Real-model checks and limits

Ran `MPLCONFIGDIR=/private/tmp/lip-reading-matplotlib PYTHONPATH=. .venv/bin/python output/identity-check/reentry.py` with real MediaPipe and SFace on frames from `samples/giuseppeconte.mp4`. MediaPipe required macOS graphics access outside the sandbox. A blank frame simulated absence. Four faces returned after fourteen seconds with all original IDs; horizontal mirroring preserved their ID set. Two distinct cropped faces, enlarged and composited into one frame, returned after nineteen seconds at swapped positions with their original respective IDs. Speech during the blank frame remained `off_screen`.

Raw results and reproducible scripts are in the ignored `output/identity-check/` directory. Replay face processing took approximately 38 ms for two faces and 73 ms for four faces after the first inference; the first four-face frame took 114 ms on this Apple M3. These few measurements are not a general latency benchmark.

The broader ten-frame diagnostic across thirty seconds retained several feature-backed IDs across multi-second gaps. Small faces without mesh landmarks still received new IDs after geometry expiry; one face also fragmented when its appearance changed. The real-image re-entry checks reuse source images and do not establish accuracy for arbitrary lighting, pose changes, lookalikes, occlusion, or every webcam. No new interactive-browser replay was performed; browser control tests passed and the public payload schema remained compatible.

Uncovered paths include native model-load errors, some malformed-image branches, and incompatible feature dimensions. No identity gallery is retained across session boundaries. At capacity, the oldest absent identity can be forgotten. This report preserves the RED/GREEN evidence without Git checkpoint commits.
