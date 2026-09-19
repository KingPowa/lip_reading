"""Combine voice turns, words, and face activity into caption events."""

import re
from collections import defaultdict
from dataclasses import replace

from .config import ACTIVE_SPEAKER_THRESHOLD, ASSIGNMENT_MARGIN
from .contracts import CaptionEvent, FaceActivity, FaceTrack, SpeakerTurn, Word


OFF_SCREEN_SPEAKER = "Off-screen speaker"
UNCERTAIN_SPEAKER = "Uncertain speaker"
DEFAULT_SPEECH_TEXT = "[speech detected]"


def associate_speakers(
    turns: list[SpeakerTurn],
    words: list[Word],
    tracks: list[FaceTrack],
    activities: list[FaceActivity],
) -> list[CaptionEvent]:
    """Attribute transcribed turns to visually active tracked faces.

    Args:
        turns: Voice-clustered speech turns on the video timeline.
        words: Whisper words on the same timeline.
        tracks: Visible face tracks.
        activities: Per-face active-speaker probabilities.

    Returns:
        Caption events that retain visible, uncertain, off-screen, and overlap state.
    """
    voice_labels: dict[str, str] = {}
    track_labels: dict[str, str] = {}
    track_ids = {track.track_id for track in tracks}
    events: list[CaptionEvent] = []
    for turn in turns:
        scores = _activity_scores(turn, activities, track_ids)
        selected_track, score = _select_track(scores)
        if selected_track is None:
            events.append(_unassigned_event(turn, words, bool(scores)))
            continue
        label = _speaker_label(turn.speaker_id, selected_track, voice_labels, track_labels)
        if label is None:
            events.append(_unassigned_event(turn, words, True))
            continue
        events.append(
            CaptionEvent(
                turn.start,
                turn.end,
                label,
                selected_track,
                _turn_text(turn, words),
                min(turn.confidence, score),
                "visible",
            )
        )
    return _disambiguate_overlaps(events)


def _activity_scores(
    turn: SpeakerTurn,
    activities: list[FaceActivity],
    track_ids: set[str],
) -> dict[str, float]:
    weighted_scores: dict[str, float] = defaultdict(float)
    for activity in activities:
        if activity.track_id not in track_ids:
            continue
        overlap = min(activity.end, turn.end) - max(activity.start, turn.start)
        if overlap <= 0:
            continue
        weighted_scores[activity.track_id] += activity.speaking_probability * overlap
    duration = turn.end - turn.start
    return {track_id: score / duration for track_id, score in weighted_scores.items()}


def _select_track(scores: dict[str, float]) -> tuple[str | None, float]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, 0.0
    selected_track, score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if score < ACTIVE_SPEAKER_THRESHOLD or score - runner_up < ASSIGNMENT_MARGIN:
        return None, score
    return selected_track, score


def _speaker_label(
    voice_id: str,
    track_id: str,
    voice_labels: dict[str, str],
    track_labels: dict[str, str],
) -> str | None:
    existing_voice_label = voice_labels.get(voice_id)
    existing_track_label = track_labels.get(track_id)
    if existing_voice_label and existing_track_label and existing_voice_label != existing_track_label:
        return None
    label = existing_voice_label or existing_track_label or f"Person {len(track_labels) + 1}"
    voice_labels[voice_id] = label
    track_labels[track_id] = label
    return label


def _unassigned_event(turn: SpeakerTurn, words: list[Word], has_candidates: bool) -> CaptionEvent:
    return CaptionEvent(
        turn.start,
        turn.end,
        UNCERTAIN_SPEAKER if has_candidates else OFF_SCREEN_SPEAKER,
        None,
        _turn_text(turn, words),
        turn.confidence,
        "uncertain" if has_candidates else "off_screen",
    )


def _turn_text(turn: SpeakerTurn, words: list[Word]) -> str:
    text = " ".join(
        word.text for word in words if turn.start <= (word.start + word.end) / 2 <= turn.end
    ).strip()
    return re.sub(r"(\w)\s+(['’])\s*(\w)", r"\1\2\3", text) or DEFAULT_SPEECH_TEXT


def _disambiguate_overlaps(events: list[CaptionEvent]) -> list[CaptionEvent]:
    ambiguous_indices = {
        index
        for index, event in enumerate(events)
        if event.state == "visible"
        and any(
            other_index != index
            and other.speaker == event.speaker
            and other.state == "visible"
            and event.start < other.end
            and event.end > other.start
            and (other.confidence, -other_index) > (event.confidence, -index)
            for other_index, other in enumerate(events)
        )
    }
    overlapping_indices = {
        index
        for index, event in enumerate(events)
        if event.state == "visible"
        and index not in ambiguous_indices
        and any(
            other_index != index
            and other.state == "visible"
            and other_index not in ambiguous_indices
            and event.start < other.end
            and event.end > other.start
            for other_index, other in enumerate(events)
        )
    }
    return [
        replace(event, speaker=UNCERTAIN_SPEAKER, face_track=None, state="uncertain")
        if index in ambiguous_indices
        else replace(event, state="overlapping")
        if index in overlapping_indices
        else event
        for index, event in enumerate(events)
    ]
