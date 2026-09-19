# 01 — What the sample showed

## What I checked

- The current plan, pipeline, generated `speech.json`, `faces.json`, `events.json`, WebVTT file, and an extracted annotated frame.
- The sample is Italian parliamentary speech. The pipeline used Whisper `base`, which is the smallest multilingual Whisper model, and its saved transcript visibly contains repeated and misrecognised words.
- The renderer wrapped captions by character count. Characters do not have equal pixel widths, so a line could run past the right edge even when it was below the configured character limit.
- When the voice-clustering stage found one speaker, Light-ASD evaluated only the first ten seconds of every face track and copied that score across the full track duration. That can associate a speaker with a face that was only a good match at the beginning of the video.

## Decisions

1. Use Whisper `small` by default. It is still local and multilingual, but has materially more capacity for Italian than `base`. The command-line option remains available for users who need a faster or larger model.
2. Remove the ten-second Light-ASD shortcut. Association evidence must come from the timestamp being attributed, not from a score copied from an earlier shot.
3. Wrap burned-in captions by their measured OpenCV pixel width. This preserves every word while keeping rendered text inside the frame.
4. Keep every fifth decoded frame for visual analysis. Holding a 568-second 640×360 video at 25 fps required roughly 10 GB before model inference and caused the verification run to be terminated. The retained frames preserve their original timestamps, so the common timeline remains correct.
5. Draw only currently active face tracks. The old renderer kept drawing every track that was active somewhere in the video in a neutral colour. If a face track was fragmented or revisited, that produced a second box over the actual speaker. The plan explicitly permits an absent border for non-speaking faces, so removing those neutral boxes is both simpler and clearer.
6. Use the existing OpenCV cascade fallback when MediaPipe cannot make a stable track. On the supplied video, MediaPipe produced occasional detections but zero usable tracks, while the cascade detected faces on every inspected frame. Sparse detections are not useful association evidence and must not silently disable face association.
7. Render captions with Pillow and the system Verdana font instead of OpenCV's ASCII-only Hershey font. The verified output replaced Italian characters such as `é` with `?`; a Unicode font is required to render the transcript faithfully.
8. Stream the original video during rendering rather than reusing the 5 fps analysis frames. Reusing them at the source frame rate shortened the output to one fifth of its intended duration. The renderer now reads only compact presentation timestamps into memory and writes every source frame.

## Known trade-off

Full-timeline Light-ASD scoring costs more CPU time than the removed shortcut. It is necessary evidence for reliable speaker-to-face association; the existing eight-track cap remains to bound that work.

Visual analysis uses 5 fps rather than every source frame. This is the current MVP ceiling; raise it only after streaming face crops replaces the in-memory frame list.

## Verification on `samples/giuseppeconte.mp4`

- Whisper `small` with `--language it` produced 97 turns and 1,170 timestamped words. Its opening caption is `Signor Presidente, signora Presidente, onorevoli colleghi.`, replacing the earlier repeated and misspelled opening.
- The cascade fallback created 121 stable face tracks after MediaPipe could not create any stable tracks for this video.
- Light-ASD created 49,938 timestamped activity scores for the eight largest candidate tracks. The final events contain 79 visible associations and 18 explicitly uncertain events; uncertainty is retained instead of guessing a face.
- The final annotated video is 568.32 seconds, 640×360 at 25 fps, with its audio stream preserved. Visual checks at 6.5 and 30 seconds showed one active face box and captions inside the video frame; the 6.5-second caption visibly renders `realtà` without replacement characters.
