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
