"""
Structured traces for the two-swarm speak path.

Used by the WebUI. Freeze behavior is unchanged: picker G2P → phoneme ids →
speaker flies. Speakers never see KC activity.
"""

from __future__ import annotations

import base64
import io
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .lexicon import get_phonemes, is_known_word
from .more_fly import MoreFlySwarm
from .phonemes import PHONEME_LIST, strip_stress
from .speaker_fly import SpeakerSwarm
from .synth import SAMPLE_RATE
from .two_swarm import (
    COMMA_PAUSE_S,
    PERIOD_PAUSE_S,
    WORD_GAP_S,
    silence_samples,
    tokenize_spoken_text,
)


def _wav_b64(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _letter_context(word: str, letter_pos: int, context_size: int) -> str:
    chars = []
    for offset in range(-context_size, context_size + 1):
        idx = letter_pos + offset
        chars.append(word[idx] if 0 <= idx < len(word) else "_")
    return "".join(chars)


def trace_word(
    picker: MoreFlySwarm,
    speakers: SpeakerSwarm,
    word: str,
) -> Dict[str, Any]:
    word = word.lower()
    ref = [strip_stress(p) for p in get_phonemes(word, allow_g2p=True)]
    known = is_known_word(word)
    n_phonemes = len(ref)
    context_size = picker.config.context_size
    n_letters = len(word)
    slots: List[Dict[str, Any]] = []
    picker_phones: List[str] = []
    crumbs: List[np.ndarray] = []
    previous: Optional[str] = None

    for p_idx in range(n_phonemes):
        if n_phonemes == 1:
            letter_pos = n_letters // 2
            phoneme_pos = 0.5
            position = 0.5
        else:
            letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
            phoneme_pos = p_idx / (n_phonemes - 1)
            position = p_idx / (n_phonemes - 1)
        letter_pos = max(0, min(letter_pos, n_letters - 1))
        letter_context = _letter_context(word, letter_pos, context_size)

        kc = picker.shared.encode_to_kc(
            letter_context, phoneme_pos, n_phonemes, previous
        )
        winner, confidence, scores = picker.predict(
            letter_context, phoneme_pos, n_phonemes, previous
        )
        votes = []
        for phoneme in PHONEME_LIST:
            specialist = picker.specialists[phoneme]
            is_yes, conf, mbon = specialist.forward(kc)
            votes.append(
                {
                    "phoneme": phoneme,
                    "isYes": bool(is_yes),
                    "confidence": float(conf),
                    "score": float(scores[phoneme]),
                    "mbonYes": float(mbon[0]),
                    "mbonNo": float(mbon[1]),
                }
            )
        votes.sort(key=lambda v: -v["score"])

        crumb = speakers.synthesize_phoneme(
            winner, prev_phone=previous, position=position
        )
        crumbs.append(crumb)
        params = speakers.get_speaker_params(winner)
        effective = speakers.speakers[winner]._get_effective_params(previous, position)

        slots.append(
            {
                "index": p_idx,
                "letterContext": letter_context,
                "letterPos": letter_pos,
                "phonemePos": float(phoneme_pos),
                "nPhonemes": n_phonemes,
                "previousPhone": previous,
                "referencePhone": ref[p_idx],
                "pickerPhone": winner,
                "confidence": float(confidence),
                "nKcActive": int(kc.sum()),
                "nKc": int(picker.config.wiring_config.n_kc),
                "kcActive": np.flatnonzero(kc).astype(int).tolist(),
                "votes": votes,
                "speaker": {
                    "phoneme": winner,
                    "f0": float(effective.f0),
                    "f1Shift": float(effective.f1_shift),
                    "f2Shift": float(effective.f2_shift),
                    "f3Shift": float(effective.f3_shift),
                    "noiseLevel": float(effective.noise_level),
                    "durationMs": float(effective.duration_ms),
                    "attackMs": float(effective.attack_ms),
                    "releaseMs": float(effective.release_ms),
                    "harmonicRolloff": float(effective.harmonic_rolloff),
                },
                "crumbWavB64": _wav_b64(crumb),
            }
        )
        picker_phones.append(winner)
        previous = winner
        _ = params

    audio = speakers.synthesize_sequence(picker_phones)
    n_match = sum(1 for a, b in zip(picker_phones, ref) if a == b)
    return {
        "word": word,
        "isKnown": known,
        "slotCountSource": "cmudict" if known else "g2p",
        "referencePhonemes": ref,
        "pickerPhonemes": picker_phones,
        "nMatch": n_match,
        "slots": slots,
        "wavB64": _wav_b64(audio),
    }


def trace_text(
    picker: MoreFlySwarm,
    speakers: SpeakerSwarm,
    text: str,
) -> Dict[str, Any]:
    tokens_in = tokenize_spoken_text(text)
    tokens: List[Dict[str, Any]] = []
    parts: List[np.ndarray] = []
    n_match = 0
    n_total = 0
    for word, pause_s in tokens_in:
        traced = trace_word(picker, speakers, word)
        traced["pauseS"] = float(pause_s)
        tokens.append(traced)
        n_match += traced["nMatch"]
        n_total += len(traced["slots"])
        word_audio = speakers.synthesize_sequence(traced["pickerPhonemes"])
        parts.append(word_audio)
        if pause_s > 0:
            parts.append(silence_samples(pause_s))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return {
        "text": text,
        "tokens": tokens,
        "sampleRate": SAMPLE_RATE,
        "wavB64": _wav_b64(audio),
        "durationS": float(len(audio) / SAMPLE_RATE) if len(audio) else 0.0,
        "nMatch": n_match,
        "nTotal": n_total,
        "pauses": {
            "wordGapS": WORD_GAP_S,
            "commaS": COMMA_PAUSE_S,
            "periodS": PERIOD_PAUSE_S,
        },
        "architecture": (
            "picker G2P → phoneme ids → Marian speaker flies "
            "(speakers never see KC)"
        ),
        "wiring": picker.get_wiring_metadata(),
    }


def load_freeze_models(
    picker_path: str = "model_more_fly_best.npz",
    speaker_path: str = "model_speaker.npz",
) -> Tuple[MoreFlySwarm, SpeakerSwarm]:
    picker = MoreFlySwarm.load(picker_path)
    speakers = SpeakerSwarm.load(speaker_path)
    return picker, speakers
