"""Trace helpers for the two-swarm lab WebUI.

Does not start the HTTP server. Freeze inference only.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PICKER = ROOT / "model_more_fly_best.npz"
SPEAKER = ROOT / "model_speaker.npz"


@pytest.mark.skipif(not PICKER.exists() or not SPEAKER.exists(), reason="freeze models missing")
def test_trace_cat_golden():
    from cursed_tts.trace import load_freeze_models, trace_word

    picker, speakers = load_freeze_models(str(PICKER), str(SPEAKER))
    traced = trace_word(picker, speakers, "cat")
    assert traced["pickerPhonemes"] == ["K", "AE", "T"]
    assert traced["referencePhonemes"] == ["K", "AE", "T"]
    assert traced["nMatch"] == 3
    assert traced["slotCountSource"] == "cmudict"
    assert len(traced["slots"]) == 3
    first = traced["slots"][0]
    assert first["pickerPhone"] == "K"
    assert first["nKc"] == picker.config.wiring_config.n_kc
    assert 0 < first["nKcActive"] <= first["nKc"]
    assert len(first["votes"]) == 39
    assert "kc" not in first["speaker"]
    assert first["crumbWavB64"]


@pytest.mark.skipif(not PICKER.exists() or not SPEAKER.exists(), reason="freeze models missing")
def test_trace_text_pauses_and_architecture():
    from cursed_tts.trace import load_freeze_models, trace_text

    picker, speakers = load_freeze_models(str(PICKER), str(SPEAKER))
    traced = trace_text(picker, speakers, "Hi me.")
    assert "speakers never see KC" in traced["architecture"]
    assert traced["pauses"]["periodS"] == 0.55
    assert traced["sampleRate"] == 22050
    assert traced["wavB64"]
    assert traced["nTotal"] == sum(len(t["slots"]) for t in traced["tokens"])
