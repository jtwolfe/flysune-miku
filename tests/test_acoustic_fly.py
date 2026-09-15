"""
Unit tests for acoustic fly head module.

Tests cover:
1. Feature extraction from picker swarm
2. Voice parameter encoding/decoding
3. Training step mechanics
4. Audio synthesis from parameters
5. Save/load roundtrip
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from cursed_tts.acoustic_fly import (
    AcousticConfig,
    AcousticFly,
    AcousticFlySwarm,
    VoiceParams,
    synthesize_from_params,
    create_formant_targets,
    extract_voice_params,
)
from cursed_tts.synth import synthesize_phoneme, SAMPLE_RATE


class TestVoiceParams:
    """Test voice parameter representation."""
    
    def test_to_array_roundtrip(self):
        """Test packing/unpacking voice params."""
        config = AcousticConfig(n_frames=8, n_formants=3, n_harmonics=6)
        
        # Create random params
        f0 = np.random.rand(config.n_frames).astype(np.float32) * 100 + 100
        formants = np.random.rand(config.n_frames, config.n_formants).astype(np.float32) * 1000 + 500
        amplitudes = np.random.rand(config.n_frames, config.n_harmonics).astype(np.float32)
        noise_amp = np.random.rand(config.n_frames).astype(np.float32) * 0.5
        
        params = VoiceParams(f0=f0, formants=formants, amplitudes=amplitudes, noise_amp=noise_amp)
        
        # Pack and unpack
        arr = params.to_array()
        restored = VoiceParams.from_array(arr, config)
        
        assert np.allclose(params.f0, restored.f0)
        assert np.allclose(params.formants, restored.formants)
        assert np.allclose(params.amplitudes, restored.amplitudes)
        assert np.allclose(params.noise_amp, restored.noise_amp)
    
    def test_param_dim_calculation(self):
        """Test parameter dimension calculation."""
        config = AcousticConfig(n_frames=16, n_formants=3, n_harmonics=8)
        expected = 16 + 16*3 + 16*8 + 16  # f0 + formants + amplitudes + noise
        assert VoiceParams.get_param_dim(config) == expected


class TestAcousticFly:
    """Test single acoustic fly."""
    
    def test_forward_pass(self):
        """Test forward pass produces output."""
        config = AcousticConfig(kc_dim=100, cue_dim=48, hidden_dim=64)
        rng = np.random.default_rng(42)
        
        fly = AcousticFly('AA', config, rng)
        
        kc = np.random.rand(config.kc_dim).astype(np.float32)
        cue = np.random.rand(config.cue_dim).astype(np.float32)
        
        output, hidden = fly.forward(kc, cue)
        
        expected_dim = VoiceParams.get_param_dim(config)
        assert output.shape == (expected_dim,)
        assert hidden.shape == (config.hidden_dim,)
    
    def test_training_reduces_loss(self):
        """Test that training reduces loss on same example."""
        config = AcousticConfig(kc_dim=100, cue_dim=48, hidden_dim=64, learning_rate=0.001)
        rng = np.random.default_rng(42)
        
        fly = AcousticFly('AA', config, rng)
        
        kc = np.random.rand(config.kc_dim).astype(np.float32)
        cue = np.random.rand(config.cue_dim).astype(np.float32)
        # Use realistic target values (scaled to ~100-1000 range)
        target = np.random.rand(VoiceParams.get_param_dim(config)).astype(np.float32) * 500 + 100
        
        # Get initial loss
        pred_before, _ = fly.forward(kc, cue)
        loss_before = np.mean((pred_before - target) ** 2)
        
        # Train for several steps
        for _ in range(50):
            fly.train_step(kc, cue, target)
        
        # Check loss decreased
        pred_after, _ = fly.forward(kc, cue)
        loss_after = np.mean((pred_after - target) ** 2)
        
        assert loss_after < loss_before, f"Training should reduce loss: {loss_after} >= {loss_before}"
    
    def test_synthesize_produces_audio(self):
        """Test synthesis produces non-zero audio."""
        config = AcousticConfig(kc_dim=100, cue_dim=48, hidden_dim=64)
        rng = np.random.default_rng(42)
        
        fly = AcousticFly('AA', config, rng)
        
        kc = np.random.rand(config.kc_dim).astype(np.float32)
        cue = np.random.rand(config.cue_dim).astype(np.float32)
        
        audio = fly.synthesize(kc, cue)
        
        assert len(audio) > 0
        assert np.abs(audio).max() > 0, "Audio should not be silent"


class TestAcousticFlySwarm:
    """Test acoustic fly swarm (MoE)."""
    
    def test_creation(self):
        """Test swarm creates flies for all phonemes."""
        config = AcousticConfig(kc_dim=100)
        swarm = AcousticFlySwarm(config)
        
        assert len(swarm.flies) == 39  # Full ARPAbet
    
    def test_phoneme_cue_encoding(self):
        """Test phoneme cue creates different encodings."""
        config = AcousticConfig(kc_dim=100, cue_dim=48)
        swarm = AcousticFlySwarm(config)
        
        cue_aa = swarm.get_phoneme_cue('AA', position=0.5)
        cue_iy = swarm.get_phoneme_cue('IY', position=0.5)
        
        assert not np.allclose(cue_aa, cue_iy), "Different phonemes should have different cues"
    
    def test_position_affects_cue(self):
        """Test that position affects cue encoding."""
        config = AcousticConfig(kc_dim=100, cue_dim=48)
        swarm = AcousticFlySwarm(config)
        
        cue_start = swarm.get_phoneme_cue('AA', position=0.0)
        cue_end = swarm.get_phoneme_cue('AA', position=1.0)
        
        assert not np.allclose(cue_start, cue_end), "Different positions should have different cues"
    
    def test_synthesize_sequence(self):
        """Test sequence synthesis concatenates correctly."""
        config = AcousticConfig(kc_dim=100, cue_dim=48)
        swarm = AcousticFlySwarm(config)
        
        phonemes = ['K', 'AE', 'T']
        kcs = [np.random.rand(config.kc_dim).astype(np.float32) for _ in phonemes]
        
        audio = swarm.synthesize_sequence(phonemes, kcs)
        
        assert len(audio) > 0
        # Should be longer than single phoneme
        single_audio = swarm.synthesize_phoneme('K', kcs[0])
        assert len(audio) > len(single_audio)
    
    def test_save_load_roundtrip(self):
        """Test save/load preserves weights."""
        config = AcousticConfig(kc_dim=100, cue_dim=48, hidden_dim=64)
        swarm = AcousticFlySwarm(config)
        
        # Modify some weights
        swarm.flies['AA'].W1[0, 0] = 12345.0
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            swarm.save(f.name)
            loaded = AcousticFlySwarm.load(f.name)
        
        assert loaded.flies['AA'].W1[0, 0] == 12345.0


class TestTargetExtraction:
    """Test training target extraction."""
    
    def test_formant_targets_created(self):
        """Test formant target creation for all phonemes."""
        config = AcousticConfig()
        targets = create_formant_targets(config.phonemes, config)
        
        assert len(targets) == 39  # All phonemes
        
        for phoneme, target in targets.items():
            expected_dim = VoiceParams.get_param_dim(config)
            assert target.shape == (expected_dim,)
    
    def test_extract_from_audio(self):
        """Test extracting params from synthesized audio."""
        config = AcousticConfig(n_frames=16)
        
        audio = synthesize_phoneme('AA')
        params = extract_voice_params(audio, config)
        
        assert params.f0.shape == (config.n_frames,)
        assert params.formants.shape == (config.n_frames, config.n_formants)


class TestIntegration:
    """Integration tests with MORE FLY picker."""
    
    @pytest.mark.skipif(
        not Path("artifacts/eval/more_fly/swarm__A_B_data.npz").exists(),
        reason="Picker swarm not available"
    )
    def test_kc_activity_extraction(self):
        """Test extracting KC activity from picker swarm."""
        from cursed_tts.more_fly import MoreFlySwarm
        from cursed_tts.acoustic_fly import get_kc_activity_from_more_fly
        
        picker = MoreFlySwarm.load("artifacts/eval/more_fly/swarm__A_B_data.npz")
        
        kc = get_kc_activity_from_more_fly(
            picker,
            letter_context="_ca_t__",
            phoneme_pos=0.33,
            n_phonemes=3,
        )
        
        assert kc.shape == (picker.config.wiring_config.n_kc,)
        assert 0 < kc.sum() < len(kc)  # Sparse but not empty


class TestSynthesis:
    """Test audio synthesis from parameters."""
    
    def test_synthesis_non_silent(self):
        """Test that synthesis produces non-silent audio."""
        config = AcousticConfig(n_frames=16)
        
        params = VoiceParams(
            f0=np.full(config.n_frames, 140.0, dtype=np.float32),
            formants=np.array([[500, 1500, 2800]] * config.n_frames, dtype=np.float32),
            amplitudes=np.full((config.n_frames, config.n_harmonics), 0.3, dtype=np.float32),
            noise_amp=np.full(config.n_frames, 0.1, dtype=np.float32),
        )
        
        audio = synthesize_from_params(params, config)
        
        assert len(audio) > 0
        assert np.abs(audio).max() > 0.1, "Audio should not be too quiet"
    
    def test_synthesis_normalized(self):
        """Test that synthesis normalizes output."""
        config = AcousticConfig(n_frames=16)
        
        # High amplitude parameters
        params = VoiceParams(
            f0=np.full(config.n_frames, 140.0, dtype=np.float32),
            formants=np.array([[500, 1500, 2800]] * config.n_frames, dtype=np.float32),
            amplitudes=np.full((config.n_frames, config.n_harmonics), 1.0, dtype=np.float32),
            noise_amp=np.full(config.n_frames, 0.5, dtype=np.float32),
        )
        
        audio = synthesize_from_params(params, config)
        
        assert np.abs(audio).max() <= 1.0, "Audio should be normalized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
