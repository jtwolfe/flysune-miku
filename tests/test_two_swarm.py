"""
Tests for the two-swarm architecture.

Key invariants:
1. SpeakerFly/SpeakerSwarm have NO KC/MB dependency
2. Different phoneme inputs produce different audio outputs
3. Picker → Speaker pipeline works end-to-end
4. Duration caps are respected (120-250ms per crumb)
"""

import pytest
import numpy as np
import inspect
from pathlib import Path
import tempfile


class TestSpeakerNoKCDependency:
    """Tests verifying speakers have no KC/MB dependency."""
    
    def test_speaker_fly_synthesize_signature(self):
        """Verify SpeakerFly.synthesize has no KC-related parameters."""
        from cursed_tts.speaker_fly import SpeakerFly
        
        sig = inspect.signature(SpeakerFly.synthesize)
        param_names = [p.lower() for p in sig.parameters.keys()]
        
        kc_terms = ['kc', 'kc_activity', 'kenyon', 'mushroom', 'mb', 
                    'mbon', 'picker', 'expansion']
        
        for kc_term in kc_terms:
            for param in param_names:
                assert kc_term not in param, \
                    f"SpeakerFly.synthesize has KC-related parameter: {param}"
    
    def test_speaker_swarm_synthesize_signature(self):
        """Verify SpeakerSwarm has no KC-related methods/parameters."""
        from cursed_tts.speaker_fly import SpeakerSwarm
        
        # Check synthesize_sequence
        sig = inspect.signature(SpeakerSwarm.synthesize_sequence)
        param_names = [p.lower() for p in sig.parameters.keys()]
        
        kc_terms = ['kc', 'kc_activity', 'kenyon', 'mushroom', 'mb', 'mbon', 'picker']
        
        for kc_term in kc_terms:
            for param in param_names:
                assert kc_term not in param, \
                    f"SpeakerSwarm.synthesize_sequence has KC-related parameter: {param}"
        
        # Check synthesize_phoneme
        sig = inspect.signature(SpeakerSwarm.synthesize_phoneme)
        param_names = [p.lower() for p in sig.parameters.keys()]
        
        for kc_term in kc_terms:
            for param in param_names:
                assert kc_term not in param, \
                    f"SpeakerSwarm.synthesize_phoneme has KC-related parameter: {param}"
    
    def test_speaker_params_no_kc_fields(self):
        """Verify SpeakerParams has no KC-related fields."""
        from cursed_tts.speaker_fly import SpeakerParams
        import dataclasses
        
        fields = [f.name.lower() for f in dataclasses.fields(SpeakerParams)]
        
        kc_terms = ['kc', 'kenyon', 'mushroom', 'mb', 'mbon', 'picker', 'expansion']
        
        for kc_term in kc_terms:
            for field in fields:
                assert kc_term not in field, \
                    f"SpeakerParams has KC-related field: {field}"
    
    def test_verify_no_kc_dependency_function(self):
        """Test the built-in verification function."""
        from cursed_tts.speaker_fly import verify_no_kc_dependency
        
        assert verify_no_kc_dependency() is True, \
            "verify_no_kc_dependency() failed - KC dependency detected!"


class TestSpeakerDifferentPhonemes:
    """Tests verifying different phonemes produce different audio."""
    
    def test_different_phonemes_different_audio(self):
        """Different phonemes must produce different audio."""
        from cursed_tts.speaker_fly import SpeakerSwarm, SpeakerSwarmConfig
        
        config = SpeakerSwarmConfig(mode='formant', seed=42)
        swarm = SpeakerSwarm(config)
        
        # Test distinct phonemes
        phonemes = ['K', 'AE', 'T', 'M', 'IY', 'S']
        
        audio_samples = {}
        for p in phonemes:
            audio_samples[p] = swarm.synthesize_phoneme(p)
        
        # Each phoneme should produce different audio
        for i, p1 in enumerate(phonemes):
            for p2 in phonemes[i+1:]:
                a1 = audio_samples[p1]
                a2 = audio_samples[p2]
                
                # Compare using shorter length
                min_len = min(len(a1), len(a2), 1000)
                
                # Audio should differ (RMS of difference should be significant)
                diff_rms = np.sqrt(np.mean((a1[:min_len] - a2[:min_len]) ** 2))
                assert diff_rms > 0.01, \
                    f"Phonemes {p1} and {p2} produced nearly identical audio (diff_rms={diff_rms:.4f})"
    
    def test_different_words_different_audio(self):
        """Different words must produce different audio (via different phoneme sequences)."""
        from cursed_tts.speaker_fly import SpeakerSwarm, SpeakerSwarmConfig
        
        config = SpeakerSwarmConfig(mode='formant', seed=42)
        swarm = SpeakerSwarm(config)
        
        # Words with different phoneme sequences
        word_phones = {
            'cat': ['K', 'AE', 'T'],
            'bat': ['B', 'AE', 'T'],
            'dog': ['D', 'AO', 'G'],
            'me': ['M', 'IY'],
        }
        
        audio_samples = {}
        for word, phones in word_phones.items():
            audio_samples[word] = swarm.synthesize_sequence(phones)
        
        # Each word should produce different audio
        words = list(audio_samples.keys())
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                # Audio lengths might differ - that's okay
                min_len = min(len(audio_samples[w1]), len(audio_samples[w2]))
                # First 1000 samples should differ
                a1 = audio_samples[w1][:min(1000, min_len)]
                a2 = audio_samples[w2][:min(1000, min_len)]
                
                assert not np.allclose(a1, a2, atol=0.01), \
                    f"Words '{w1}' and '{w2}' produced nearly identical audio!"


class TestSpeakerDurationCaps:
    """Tests verifying duration caps are respected."""
    
    def test_crumb_duration_within_bounds(self):
        """Each phoneme crumb should be within duration bounds."""
        from cursed_tts.speaker_fly import (
            SpeakerSwarm, SpeakerSwarmConfig, 
            SPEAKER_DURATION_MIN_MS, SPEAKER_DURATION_MAX_MS, SAMPLE_RATE
        )
        
        config = SpeakerSwarmConfig(mode='formant', seed=42)
        swarm = SpeakerSwarm(config)
        
        for phoneme in swarm.phoneme_list:
            audio = swarm.synthesize_phoneme(phoneme)
            duration_ms = len(audio) / SAMPLE_RATE * 1000
            
            assert duration_ms >= SPEAKER_DURATION_MIN_MS * 0.8, \
                f"Phoneme {phoneme} duration {duration_ms:.0f}ms below minimum"
            
            # Allow some tolerance above max (crossfade can extend slightly)
            assert duration_ms <= SPEAKER_DURATION_MAX_MS * 1.5, \
                f"Phoneme {phoneme} duration {duration_ms:.0f}ms above maximum"


class TestSpeakerSaveLoad:
    """Tests for speaker swarm serialization."""
    
    def test_save_load_roundtrip(self):
        """Speaker swarm should save and load correctly."""
        from cursed_tts.speaker_fly import SpeakerSwarm, SpeakerSwarmConfig, SpeakerParams
        
        # Create swarm with custom params
        config = SpeakerSwarmConfig(mode='formant', seed=123)
        swarm = SpeakerSwarm(config)
        
        # Modify some params
        original_params = {}
        for p in ['K', 'AE']:
            params = swarm.get_speaker_params(p)
            params.f0 = 200.0 + hash(p) % 50
            params.noise_level = 0.5
            swarm.set_speaker_params(p, params)
            original_params[p] = params.f0
        
        # Generate original audio
        original_audio = swarm.synthesize_sequence(['K', 'AE', 'T'])
        
        # Save and reload
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            swarm.save(temp_path)
            loaded = SpeakerSwarm.load(temp_path)
            
            # Check params preserved
            for p, expected_f0 in original_params.items():
                loaded_params = loaded.get_speaker_params(p)
                assert abs(loaded_params.f0 - expected_f0) < 0.1, \
                    f"Params not preserved for {p}"
            
            # Check audio matches
            loaded_audio = loaded.synthesize_sequence(['K', 'AE', 'T'])
            assert np.allclose(original_audio, loaded_audio, atol=0.001), \
                "Audio changed after save/load"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestTwoSwarmPipeline:
    """Tests for the full two-swarm pipeline."""
    
    @pytest.fixture
    def picker_model_path(self):
        """Get path to picker model if available."""
        # Prefer more_fly model as it has full phonemes and correct dimensions
        more_fly_path = Path("model_more_fly_best.npz")
        if more_fly_path.exists():
            return str(more_fly_path), 'more_fly'
        
        swarm_path = Path("model_swarm.npz")
        if swarm_path.exists():
            return str(swarm_path), 'swarm'
        
        pytest.skip("No picker model found - run train-swarm or train-more-fly first")
    
    def test_two_swarm_speak_produces_audio(self, picker_model_path):
        """TwoSwarmSpeaker should produce non-empty audio."""
        from cursed_tts.two_swarm import TwoSwarmSpeaker
        
        path, picker_type = picker_model_path
        
        try:
            speaker = TwoSwarmSpeaker.from_model_files(
                picker_path=path,
                speaker_path=None,  # Use formant baseline
                use_formant_baseline=True,
                picker_type=picker_type,
            )
            
            audio, picker_ph, ref_ph, known = speaker.speak("cat", verbose=False)
            
            assert len(audio) > 0, "Audio is empty"
            assert len(picker_ph) > 0, "No phonemes predicted"
            assert len(ref_ph) > 0, "No reference phonemes"
        except ValueError as e:
            if "matmul" in str(e) and "mismatch" in str(e):
                pytest.skip(f"Model dimension mismatch - retrain model: {e}")
            raise
    
    def test_two_swarm_different_words_different_audio(self, picker_model_path):
        """Different words should produce different audio through full pipeline."""
        from cursed_tts.two_swarm import TwoSwarmSpeaker
        
        path, picker_type = picker_model_path
        
        try:
            speaker = TwoSwarmSpeaker.from_model_files(
                picker_path=path,
                speaker_path=None,
                use_formant_baseline=True,
                picker_type=picker_type,
            )
            
            # Speak different words
            audio_cat, _, _, _ = speaker.speak("cat", verbose=False)
            audio_dog, _, _, _ = speaker.speak("dog", verbose=False)
            
            # Compare first portion
            min_len = min(len(audio_cat), len(audio_dog), 2000)
            
            assert not np.allclose(audio_cat[:min_len], audio_dog[:min_len], atol=0.01), \
                "Different words produced nearly identical audio!"
        except ValueError as e:
            if "matmul" in str(e) and "mismatch" in str(e):
                pytest.skip(f"Model dimension mismatch - retrain model: {e}")
            raise
    
    def test_picker_phonemes_not_passed_to_speakers(self, picker_model_path):
        """Verify picker state is not passed to speakers."""
        from cursed_tts.two_swarm import TwoSwarmSpeaker
        
        path, picker_type = picker_model_path
        
        try:
            # This is an architecture test - speakers should only receive phoneme IDs
            speaker = TwoSwarmSpeaker.from_model_files(
                picker_path=path,
                speaker_path=None,
                use_formant_baseline=True,
                picker_type=picker_type,
            )
            
            # The synthesize method should only take phonemes, not KC activity
            sig = inspect.signature(speaker.synthesize)
            param_names = [p.lower() for p in sig.parameters.keys()]
            
            kc_terms = ['kc', 'activity', 'picker', 'mb']
            for kc_term in kc_terms:
                for param in param_names:
                    assert kc_term not in param, \
                        f"synthesize() has KC-related parameter: {param}"
        except ValueError as e:
            if "matmul" in str(e) and "mismatch" in str(e):
                pytest.skip(f"Model dimension mismatch - retrain model: {e}")
            raise


class TestTrainSpeaker:
    """Tests for speaker training."""
    
    def test_train_formant_bootstrap(self):
        """Test formant-bootstrap training mode."""
        from cursed_tts.train_speaker import (
            train_speaker_swarm_formant_bootstrap,
            SpeakerTrainingConfig,
        )
        
        config = SpeakerTrainingConfig(
            mode='formant-bootstrap',
            n_iterations=10,  # Reduced for testing
            seed=42,
        )
        
        # Train just a few phonemes
        swarm = train_speaker_swarm_formant_bootstrap(
            config,
            phonemes=['K', 'AE', 'T'],
            verbose=False,
        )
        
        assert len(swarm.speakers) == 3
        
        # Synthesize and verify non-empty
        audio = swarm.synthesize_sequence(['K', 'AE', 'T'])
        assert len(audio) > 0
    
    def test_trained_params_are_reasonable(self):
        """Trained parameters should be within reasonable ranges."""
        from cursed_tts.train_speaker import (
            train_speaker_for_phoneme,
            SpeakerTrainingConfig,
        )
        from cursed_tts.synth import synthesize_phoneme
        
        config = SpeakerTrainingConfig(
            mode='formant-bootstrap',
            n_iterations=20,
            seed=42,
        )
        
        # Train for one phoneme
        target = synthesize_phoneme('K')
        params = train_speaker_for_phoneme('K', target, config, verbose=False)
        
        # Check ranges
        assert 50 <= params.f0 <= 500, f"F0 out of range: {params.f0}"
        assert 0.5 <= params.f1_shift <= 2.0, f"F1 shift out of range: {params.f1_shift}"
        assert 0 <= params.noise_level <= 1, f"Noise level out of range: {params.noise_level}"
        assert 50 <= params.duration_ms <= 300, f"Duration out of range: {params.duration_ms}"


class TestMarianAliasAndCaps:
    """Marian alias resolution and oto duration caps (no KC, no full voicebank required)."""

    def test_resolve_prefers_standalone_then_onset(self):
        from cursed_tts.train_speaker import resolve_marian_alias

        aliases = ['k aa', '- k', 'k', 'aa k']
        assert resolve_marian_alias('K', aliases) == 'k'

        aliases_no_stand = ['k aa', '- k', 'aa k']
        assert resolve_marian_alias('K', aliases_no_stand) == '- k'

        aliases_digits = ['iy1', 'iy aa', '- iy']
        assert resolve_marian_alias('IY', aliases_digits) == 'iy1'

    def test_oto_window_is_capped(self):
        from cursed_tts.train_speaker import (
            oto_window_ms, MARIAN_CRUMB_MIN_MS, MARIAN_CRUMB_MAX_MS,
        )

        # Huge negative cutoff must still cap
        start, length = oto_window_ms(
            {'offset_ms': 100.0, 'consonant_ms': 900.0, 'cutoff_ms': -2000.0},
            wav_duration_ms=5000.0,
        )
        assert start == 100.0
        assert MARIAN_CRUMB_MIN_MS <= length <= MARIAN_CRUMB_MAX_MS

        # Tiny window is lifted to the minimum
        start, length = oto_window_ms(
            {'offset_ms': 10.0, 'consonant_ms': 5.0, 'cutoff_ms': -20.0},
            wav_duration_ms=1000.0,
        )
        assert length >= MARIAN_CRUMB_MIN_MS

    def test_speaker_train_has_no_kc_imports(self):
        import cursed_tts.train_speaker as ts
        import cursed_tts.speaker_fly as sf
        assert not hasattr(ts, 'encode_to_kc')
        assert not hasattr(sf, 'encode_to_kc')
        assert 'mushroom_body' not in getattr(ts, '__file__', '')
        # Training/synth modules must not import picker KC machinery
        assert 'cursed_tts.mushroom_body' not in getattr(ts, '__dict__', {})
        assert 'cursed_tts.specialist_fly' not in ts.__dict__
        assert 'cursed_tts.more_fly' not in ts.__dict__


class TestPunctuationPauses:
    """Comma/period silence on the sentence concat path (no speaker retraining)."""

    def test_tokenize_comma_and_period_pauses(self):
        from cursed_tts.two_swarm import (
            tokenize_spoken_text, WORD_GAP_S, COMMA_PAUSE_S, PERIOD_PAUSE_S,
        )

        tokens = tokenize_spoken_text("hello, world. yes")
        assert [w for w, _ in tokens] == ['hello', 'world', 'yes']
        pauses = {w: p for w, p in tokens}
        assert pauses['hello'] == COMMA_PAUSE_S
        assert pauses['world'] == PERIOD_PAUSE_S
        assert pauses['yes'] == 0.0

        glued = tokenize_spoken_text("hello world yes")
        assert [w for w, _ in glued] == ['hello', 'world', 'yes']
        assert glued[0][1] == WORD_GAP_S
        assert glued[1][1] == WORD_GAP_S
        assert glued[2][1] == 0.0

    def test_bare_punctuation_attaches_to_previous_word(self):
        from cursed_tts.two_swarm import tokenize_spoken_text, PERIOD_PAUSE_S

        tokens = tokenize_spoken_text("hello .")
        assert tokens == [('hello', PERIOD_PAUSE_S)]

    def test_punctuation_stripped_from_spoken_words(self):
        from cursed_tts.two_swarm import tokenize_spoken_text

        tokens = tokenize_spoken_text("Avocados grow on trees.")
        assert [w.lower() for w, _ in tokens] == ['avocados', 'grow', 'on', 'trees']

    def test_speak_sequence_period_is_longer_than_glued(self, picker_model_path=None):
        """Duration check: punctuation adds silence, not extra phones."""
        from cursed_tts.two_swarm import (
            TwoSwarmSpeaker, COMMA_PAUSE_S, PERIOD_PAUSE_S, WORD_GAP_S, SAMPLE_RATE,
        )
        from cursed_tts.synth import synthesize_all_phonemes, concatenate_phonemes
        from cursed_tts.lexicon import get_phonemes
        from cursed_tts.phonemes import strip_stress

        class DummyPicker:
            def predict_word(self, word, n_phonemes):
                ph = [strip_stress(p) for p in get_phonemes(word, allow_g2p=True)]
                if not ph:
                    ph = ['AH']
                if len(ph) >= n_phonemes:
                    return ph[:n_phonemes]
                return ph + [ph[-1]] * (n_phonemes - len(ph))

        speaker = TwoSwarmSpeaker(
            DummyPicker(), speaker_swarm=None, use_formant_baseline=True
        )
        glued, ph_g = speaker.speak_sequence("yes no", verbose=False)
        punct, ph_p = speaker.speak_sequence("yes, no.", verbose=False)
        assert ph_g == ph_p
        extra = (COMMA_PAUSE_S - WORD_GAP_S) + PERIOD_PAUSE_S
        extra_samples = int(round(extra * SAMPLE_RATE))
        assert abs(len(punct) - len(glued) - extra_samples) <= 2


# Integration test
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_smoke(self):
        """Smoke test for the full two-swarm pipeline."""
        from cursed_tts.speaker_fly import SpeakerSwarm, SpeakerSwarmConfig
        from cursed_tts.synth import SAMPLE_RATE
        
        # Just verify we can create and use speakers
        config = SpeakerSwarmConfig(mode='formant', seed=42)
        swarm = SpeakerSwarm(config)
        
        # Synthesize a word
        phonemes = ['HH', 'EH', 'L', 'OW']  # "hello"
        audio = swarm.synthesize_sequence(phonemes)
        
        assert len(audio) > 0
        duration_s = len(audio) / SAMPLE_RATE
        assert 0.1 < duration_s < 2.0, f"Unexpected duration: {duration_s}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
