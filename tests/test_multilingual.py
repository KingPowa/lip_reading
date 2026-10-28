import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.app.service import RecognitionService
from src.core.audio.language import parse_languages
from src.core.audio.realtime import MultilingualTranscriber, TranscriptionUpdate
from src.core.contracts import CaptionEvent, Word
from src.core.tracking.association import merge_word_events
from src.core.tracking.realtime import RealtimeTracker

SECOND = np.full(16000, 800, dtype="<i2").tobytes()
SHORT = np.full(1600, 800, dtype="<i2").tobytes()


class FakeDecoder:
    """Like Vosk, word times follow the samples fed since construction (Reset keeps them)."""

    def __init__(self, words=(), endpoint=False, partial=""):
        self.texts = list(words)
        self.endpoint, self.partial, self.calls = endpoint, partial, []
        self.clock, self.base, self.packets = 0.0, None, 0

    def SetWords(self, enabled):
        pass

    def AcceptWaveform(self, content):
        self.calls.append("accept")
        self.packets += 1
        if self.base is None:
            self.base = self.clock
        self.clock += len(content) / 32000
        return self.endpoint

    def _words(self):
        base, self.base = self.base or 0.0, None
        return [
            {
                "word": text,
                "start": base + index * 0.2,
                "end": base + index * 0.2 + 0.2,
                "conf": 0.9,
            }
            for index, text in enumerate(self.texts)
        ]

    def Reset(self):
        self.calls.append("Reset")
        self.base = None

    def Result(self):
        self.calls.append("Result")
        return json.dumps({"result": self._words()})

    def FinalResult(self):
        self.calls.append("FinalResult")
        return json.dumps({"result": [] if self.calls[-2:-1] == ["Result"] else self._words()})

    def PartialResult(self):
        return json.dumps({"partial": self.partial})


def transcriber(decoders, identify=None, origin=0.0):
    with patch("vosk.KaldiRecognizer", side_effect=lambda model, rate: decoders[model]):
        return MultilingualTranscriber(
            {name: name for name in decoders}, origin, identify
        )


class LanguageParsingTest(unittest.TestCase):
    def test_list_is_deduplicated_in_priority_order(self):
        self.assertEqual(parse_languages("it, en,it"), ("it", "en"))
        self.assertEqual(parse_languages(None), ())
        self.assertEqual(parse_languages("fr"), ("fr",))

    def test_invalid_unknown_and_excess_languages_are_rejected(self):
        for text in ("it,e", "it,xx", "en,it,es,fr", "it;en"):
            with self.assertRaises(ValueError, msg=text):
                parse_languages(text)

    def test_custom_model_requires_exactly_one_language(self):
        self.assertEqual(parse_languages("pt", custom_model=True), ("pt",))
        with self.assertRaises(ValueError):
            parse_languages("it,en", custom_model=True)
        with self.assertRaises(ValueError):
            RecognitionService("it,en", Path("models"), Path("custom"))


LONG = np.full(24000, 800, dtype="<i2").tobytes()  # 1.5 s


def votes(*probabilities):
    """identify() returning the given en/it probabilities in turn."""
    return Mock(side_effect=[{"en": 1 - it, "it": it} for it in probabilities])


def speak(mixed, times=1, content=LONG):
    return [mixed.accept(content, False) for _ in range(times)]


class MultilingualTranscriberTest(unittest.TestCase):
    def build(self, identify, **kwargs):
        decoders = {
            "en": FakeDecoder(["hello"], endpoint=True),
            "it": FakeDecoder(["ciao"], endpoint=True),
        }
        return decoders, transcriber(decoders, identify, **kwargs)

    def test_only_the_active_decoder_receives_packets(self):
        decoders, mixed = self.build(votes(0.1, 0.1))
        speak(mixed, 2)
        self.assertEqual(decoders["en"].packets, 2)
        self.assertEqual(decoders["it"].packets, 0)
        self.assertEqual(mixed.language, "en")

    def test_weak_or_one_off_evidence_keeps_the_language(self):
        decoders, mixed = self.build(votes(0.7, 0.8, 0.1, 0.8, 0.1, 0.89))
        updates = speak(mixed, 6)
        self.assertEqual({u.language for u in updates}, {"en"})
        self.assertEqual(decoders["it"].packets, 0)

    def test_short_utterances_do_not_vote(self):
        identify = Mock(return_value={"it": 1.0})
        _, mixed = self.build(identify)
        speak(mixed, 3, SHORT)
        identify.assert_not_called()
        self.assertEqual(mixed.language, "en")

    def test_a_suspicious_utterance_is_held_and_nothing_is_emitted(self):
        decoders, mixed = self.build(votes(0.8))
        update = speak(mixed)[0]
        self.assertEqual((update.words, update.language), ([], "en"))
        self.assertNotIn("Reset", decoders["it"].calls)

    def test_a_confirmed_switch_decodes_held_and_current_together(self):
        decoders, mixed = self.build(votes(0.1, 0.8, 0.8), origin=2.0)
        _, held, confirmed = speak(mixed, 3)
        italian = decoders["it"]
        self.assertEqual(held.words, [])
        self.assertEqual(italian.calls, ["Reset", "accept", "Result", "FinalResult"])
        self.assertEqual(italian.packets, 1)
        self.assertEqual(italian.clock, 3.0)  # held + current in one decode
        self.assertEqual(confirmed.language, "it")
        # The held utterance began 1.5 s into the stream, after origin 2.0.
        self.assertEqual([(w.text, w.start) for w in confirmed.words], [("ciao", 3.5)])
        self.assertEqual({w.language for w in confirmed.words}, {"it"})
        # Later packets go to Italian only, still on the stream clock.
        italian.texts = ["dopo"]
        later = mixed.accept(LONG, False)
        self.assertEqual(decoders["en"].packets, 3)
        self.assertEqual([(w.text, w.start) for w in later.words], [("dopo", 6.5)])

    def test_a_refuted_hold_releases_both_in_the_old_language(self):
        decoders, mixed = self.build(votes(0.8, 0.1))
        speak(mixed)
        update = speak(mixed)[0]
        self.assertEqual(update.language, "en")
        self.assertEqual([w.text for w in update.words], ["hello", "hello"])
        self.assertEqual({w.language for w in update.words}, {"en"})
        self.assertEqual(decoders["it"].calls, [])

    def test_a_refuted_strong_hold_is_decoded_alone_without_switching(self):
        wide = np.full(40000, 800, dtype="<i2").tobytes()  # 2.5 s
        decoders, mixed = self.build(votes(0.1, 0.95, 0.1), origin=2.0)
        speak(mixed)
        speak(mixed, content=wide)
        update = speak(mixed, content=wide)[0]
        italian = decoders["it"]
        self.assertEqual(italian.calls, ["Reset", "accept", "Result", "FinalResult"])
        self.assertEqual(italian.clock, 2.5)  # the held utterance only
        self.assertEqual(
            [(w.text, w.language, w.start) for w in update.words],
            [("ciao", "it", 3.5), ("hello", "en", 6.0)],
        )
        self.assertEqual(mixed.language, "en")
        self.assertEqual(mixed.shift, 0.0)
        mixed.accept(LONG, False)
        self.assertEqual((decoders["en"].packets, italian.packets), (4, 1))

    def test_a_refuted_weak_or_short_hold_stays_in_the_active_language(self):
        wide = np.full(40000, 800, dtype="<i2").tobytes()
        for probability, content in ((0.8, wide), (0.95, LONG)):
            decoders, mixed = self.build(votes(probability, 0.1))
            speak(mixed, content=content)
            update = speak(mixed, content=content)[0]
            self.assertEqual({w.language for w in update.words}, {"en"})
            self.assertEqual(decoders["it"].calls, [])

    def test_a_short_utterance_after_a_hold_releases_it(self):
        decoders, mixed = self.build(votes(0.8))
        speak(mixed)
        update = mixed.accept(SHORT, False)
        self.assertEqual([w.text for w in update.words], ["hello", "hello"])
        self.assertEqual((update.language, decoders["it"].calls), ("en", []))

    def test_final_flush_releases_a_held_utterance(self):
        decoders, mixed = self.build(votes(0.8))
        speak(mixed)
        update = mixed.accept(b"", True)
        self.assertEqual([w.text for w in update.words], ["hello"])
        self.assertEqual(mixed.held, None)

    def test_final_flush_can_confirm_a_held_utterance(self):
        decoders, mixed = self.build(votes(0.8, 0.8))
        speak(mixed)
        update = mixed.accept(LONG, True)
        self.assertEqual((update.language, [w.text for w in update.words]), ("it", ["ciao"]))

    def test_a_hold_that_would_overflow_the_buffer_is_released(self):
        decoders, mixed = self.build(votes(0.8, 0.8, 0.1))
        with patch("src.core.audio.realtime.MAXIMUM_UTTERANCE_SECONDS", 2.0):
            first, second, third = speak(mixed, 3)
        self.assertEqual([w.text for w in second.words], ["hello"])  # first released, second held
        self.assertEqual(third.language, "en")  # the second one was held and refuted
        self.assertEqual(decoders["it"].calls, [])

    def test_without_language_identification_the_first_language_is_kept(self):
        _, mixed = self.build(None)
        self.assertEqual({u.language for u in speak(mixed, 3)}, {"en"})

    def test_failed_identification_keeps_the_language(self):
        _, mixed = self.build(Mock(side_effect=RuntimeError("model broke")))
        with self.assertLogs("src.core.audio.realtime", level="WARNING"):
            self.assertEqual(speak(mixed)[0].language, "en")

    def test_probabilities_are_restricted_and_renormalised(self):
        # Spanish dominates the raw scores; among en/it Italian holds 0.95.
        identify = Mock(return_value={"es": 0.9, "en": 0.005, "it": 0.095})
        decoders, mixed = self.build(identify)
        self.assertEqual(speak(mixed, 2)[1].language, "it")

    def test_final_flush_ends_the_utterance(self):
        decoders = {"en": FakeDecoder(["a"]), "it": FakeDecoder(["b"])}
        update = transcriber(decoders, votes()).accept(b"", True)
        self.assertEqual((update.language, update.partial_text), ("en", ""))
        self.assertEqual(decoders["en"].calls, ["FinalResult"])

    def test_partials_follow_the_current_language(self):
        decoders = {
            "en": FakeDecoder(["a"], partial="hello wor"),
            "it": FakeDecoder(["b"], partial="ciao mon"),
        }
        mixed = transcriber(decoders)
        update = mixed.accept(SHORT, False)
        self.assertEqual((update.partial_text, update.language), ("hello wor", "en"))

    def test_utterance_audio_is_bounded(self):
        decoders = {"en": FakeDecoder([]), "it": FakeDecoder([])}
        mixed = transcriber(decoders)
        with patch("src.core.audio.realtime.MAXIMUM_UTTERANCE_SECONDS", 1.5):
            for _ in range(3):
                mixed.accept(SECOND, False)
        self.assertEqual(len(mixed.utterance), 48000)


class LanguageContractTest(unittest.TestCase):
    def event(self, start, language):
        return CaptionEvent(
            start, start + 0.2, "Person 1", "t", "w", 0.9, "visible", language
        )

    def test_merge_stops_at_a_language_change(self):
        merged = merge_word_events(
            [self.event(0, "it"), self.event(0.2, "it"), self.event(0.4, "en")]
        )
        self.assertEqual(
            [(e.text, e.language) for e in merged], [("w w", "it"), ("w", "en")]
        )

    def test_json_includes_language_only_when_known(self):
        self.assertNotIn("language", self.event(0, None).as_json())
        self.assertEqual(self.event(0, "it").as_json()["language"], "it")

    def test_tracker_passes_word_language_to_events(self):
        events = RealtimeTracker().caption_events([Word("ciao", 0.0, 0.2, 0.9, "it")])
        self.assertEqual(events[0].language, "it")


class MultilingualServiceTest(unittest.TestCase):
    def setUp(self):
        voice = patch(
            "src.app.service.warm_up_speaker_model", side_effect=RuntimeError("off")
        )
        voice.start()
        self.addCleanup(voice.stop)
        self.service = RecognitionService("it,en", Path("models"))
        self.service.models = Mock()
        self.addCleanup(self.service.close)
        self.pcm = np.full(1600, 1600, dtype="<i2").tobytes()

    def test_session_loads_every_model_and_one_shared_identifier(self):
        with patch("src.app.service.load_language_identifier") as load:
            started = self.service.start_session("camera", "it,en")
            self.service.start_session("other", None)
        load.assert_called_once()
        self.assertEqual(started["languages"], ["it", "en"])
        self.assertEqual(started["language"], "it")
        self.assertEqual(self.service.models.load.call_count, 4)

    def test_single_language_loads_no_identifier_and_keeps_plain_decoder(self):
        with patch("src.app.service.load_language_identifier") as load:
            self.service.start_session("camera", "en")
        load.assert_not_called()
        with self.service.registry.use("camera") as session:
            with patch("vosk.KaldiRecognizer"):
                transcriber = self.service._build_transcriber(session, 0.0)
        self.assertNotIsInstance(transcriber, MultilingualTranscriber)

    def test_changing_the_language_set_is_rejected(self):
        with patch("src.app.service.load_language_identifier"):
            self.service.start_session("camera", "it,en")
            for languages in ("it", "en,it", "it,es"):
                with self.assertRaises(ValueError, msg=languages):
                    self.service.start_session("camera", languages)
                with self.assertRaises(ValueError, msg=languages):
                    self.service.recognize_audio(
                        self.pcm, languages, "camera", 0.0, False, "1"
                    )

    def test_result_reports_language(self):
        with patch("src.app.service.load_language_identifier"):
            self.service.start_session("camera", "it,en")
        decoder = Mock()
        decoder.accept.return_value = TranscriptionUpdate(
            [Word("ciao", 0.0, 0.1, 0.9, "it")], "", "it"
        )
        with self.service.registry.use("camera") as session:
            session.transcriber = decoder
            session.tracker.speaker_at = lambda timestamp: Mock(
                speaker="Person 1", face_track=None, state="off_screen"
            )
        result = self.service.recognize_audio(
            self.pcm, "it,en", "camera", 0.0, False, "1"
        )
        self.assertEqual(result["language"], "it")
        self.assertEqual(result["events"][0]["language"], "it")

    def test_warm_up_loads_all_configured_models(self):
        with (
            patch("src.app.service.StreamingTranscriber") as decoder,
            patch("src.app.service.LiveFaceDetector"),
            patch("src.app.service.load_language_identifier") as load,
        ):
            self.service.warm_up()
        self.assertEqual(decoder.return_value.accept.call_count, 2)
        load.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class PauseCutTest(unittest.TestCase):
    SPEECH = np.full(1600, 8000, dtype="<i2").tobytes()
    SILENCE = np.zeros(1600, dtype="<i2").tobytes()

    def feed(self, built, packets):
        return [built.accept(packet, False).words for packet in packets]

    def test_a_pause_after_enough_speech_forces_the_utterance_end(self):
        decoders = {"en": FakeDecoder(["a"]), "it": FakeDecoder(["b"])}
        built = transcriber(decoders)
        out = self.feed(built, [self.SPEECH] * 13 + [self.SILENCE] * 3)
        self.assertEqual([i for i, words in enumerate(out) if words], [15])
        self.assertEqual(out[15][0].text, "a")
        self.assertEqual(decoders["en"].calls.count("FinalResult"), 1)

    def test_idle_silence_and_short_utterances_are_not_cut(self):
        decoder = FakeDecoder(["a"])
        built = transcriber({"en": decoder})
        self.assertFalse(any(self.feed(built, [self.SILENCE] * 30)))
        self.assertFalse(any(self.feed(built, [self.SPEECH] * 8 + [self.SILENCE] * 3)))
        self.assertNotIn("FinalResult", decoder.calls)

    def test_continuous_speech_is_cut_at_the_hard_cap(self):
        decoder = FakeDecoder(["a"])
        built = transcriber({"en": decoder})
        self.feed(built, [self.SPEECH] * 99)
        self.assertEqual(decoder.calls.count("FinalResult"), 0)
        self.feed(built, [self.SPEECH])
        self.assertEqual(decoder.calls.count("FinalResult"), 1)
