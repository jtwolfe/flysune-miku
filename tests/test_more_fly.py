"""
Unit tests for MORE FLY features.

Tests:
1. Wiring mode ≠ random when flywire enabled (hash/sparsity check)
2. Only KC→MBON changes under DAN step when PN→KC frozen
3. Previous-phone cue changes encoding
4. Short word slots differ by position features ("me" problem)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cursed_tts.more_fly import (
    CueConfig, WiringConfig, DANConfig, MoreFlyConfig,
    encode_enhanced_context, get_enhanced_input_dim,
    create_wiring_matrix, MoreFlySwarm, MoreFlyExpansion,
    verify_wiring_is_not_random, verify_previous_phone_cue_changes_encoding,
    verify_short_word_position_differs,
)
from cursed_tts.phonemes import PHONEME_LIST


class TestEnhancedCues:
    """Tests for Stage A: Enhanced cue features."""
    
    def test_previous_phone_cue_changes_encoding(self):
        """Previous-phone cue should change the feature encoding."""
        cue_config = CueConfig(
            use_position_features=True,
            use_previous_phone=True,
            use_focus_features=True,
        )
        
        # Same letter context, different previous phones
        enc_none = encode_enhanced_context(
            "_ca_t__",
            phoneme_pos=0.5,
            n_phonemes=3,
            previous_phone=None,
            cue_config=cue_config,
        )
        
        enc_k = encode_enhanced_context(
            "_ca_t__",
            phoneme_pos=0.5,
            n_phonemes=3,
            previous_phone='K',
            cue_config=cue_config,
        )
        
        enc_ae = encode_enhanced_context(
            "_ca_t__",
            phoneme_pos=0.5,
            n_phonemes=3,
            previous_phone='AE',
            cue_config=cue_config,
        )
        
        # Encodings should differ
        assert not np.allclose(enc_none, enc_k), "None vs K should differ"
        assert not np.allclose(enc_k, enc_ae), "K vs AE should differ"
        assert not np.allclose(enc_none, enc_ae), "None vs AE should differ"
    
    def test_previous_phone_cue_disabled(self):
        """Disabling previous-phone should not change encoding based on prev phone."""
        cue_config = CueConfig(
            use_position_features=True,
            use_previous_phone=False,  # Disabled
            use_focus_features=True,
        )
        
        enc_none = encode_enhanced_context(
            "_ca_t__",
            phoneme_pos=0.5,
            n_phonemes=3,
            previous_phone=None,
            cue_config=cue_config,
        )
        
        enc_k = encode_enhanced_context(
            "_ca_t__",
            phoneme_pos=0.5,
            n_phonemes=3,
            previous_phone='K',
            cue_config=cue_config,
        )
        
        # Should be identical when feature is disabled
        assert np.allclose(enc_none, enc_k), "Encodings should match when prev_phone disabled"
    
    def test_short_word_slots_differ(self):
        """Short word slots should have different position encodings."""
        cue_config = CueConfig(
            use_position_features=True,
            use_previous_phone=True,
            use_focus_features=True,
        )
        
        # "me" - 2 phonemes (M, IY)
        # Slot 0: position 0.0, no previous
        # Slot 1: position 1.0, previous M
        
        enc_slot0 = encode_enhanced_context(
            "___me__",
            phoneme_pos=0.0,
            n_phonemes=2,
            previous_phone=None,
            cue_config=cue_config,
        )
        
        enc_slot1 = encode_enhanced_context(
            "__me___",
            phoneme_pos=1.0,
            n_phonemes=2,
            previous_phone='M',
            cue_config=cue_config,
        )
        
        # Slots should differ
        assert not np.allclose(enc_slot0, enc_slot1), \
            "Short word slots should have different encodings"
    
    def test_focus_features_for_short_words(self):
        """Focus features should indicate short words."""
        cue_config = CueConfig(
            use_position_features=True,
            use_previous_phone=False,
            use_focus_features=True,
        )
        
        # Short word (2 phonemes)
        enc_short = encode_enhanced_context(
            "___me__",
            phoneme_pos=0.5,
            n_phonemes=2,
            cue_config=cue_config,
        )
        
        # Long word (10 phonemes)
        enc_long = encode_enhanced_context(
            "mushro_",
            phoneme_pos=0.5,
            n_phonemes=10,
            cue_config=cue_config,
        )
        
        # Should differ due to focus features
        assert not np.allclose(enc_short, enc_long)
    
    def test_input_dimension_calculation(self):
        """Input dimension should be correctly calculated for different configs."""
        # Full features
        cue_full = CueConfig(
            use_position_features=True,
            use_previous_phone=True,
            use_focus_features=True,
        )
        dim_full = get_enhanced_input_dim(3, cue_full)
        
        # No previous phone
        cue_no_prev = CueConfig(
            use_position_features=True,
            use_previous_phone=False,
            use_focus_features=True,
        )
        dim_no_prev = get_enhanced_input_dim(3, cue_no_prev)
        
        # Minimal features
        cue_minimal = CueConfig(
            use_position_features=False,
            use_previous_phone=False,
            use_focus_features=False,
        )
        dim_minimal = get_enhanced_input_dim(3, cue_minimal)
        
        # Check ordering
        assert dim_full > dim_no_prev > dim_minimal
        
        # Check specific dimensions
        # Base: 7*27 + 26*26 + 27 = 189 + 676 + 27 = 892
        assert dim_minimal == 892
        # + position (8) + focus (4) = 904
        assert dim_no_prev == 892 + 8 + 4
        # + previous phone (39)
        assert dim_full == 892 + 8 + 39 + 4


class TestWiring:
    """Tests for Stage C: Real wiring."""
    
    def test_random_wiring_is_deterministic(self):
        """Random wiring with same seed should be identical."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        config = WiringConfig(mode='random', n_pn=100, n_kc=500, seed=42)
        
        _, pn_kc_1, meta1 = create_wiring_matrix(config, 500, rng1)
        _, pn_kc_2, meta2 = create_wiring_matrix(config, 500, rng2)
        
        assert np.allclose(pn_kc_1, pn_kc_2), "Same seed should give same wiring"
        assert meta1['hash'] == meta2['hash'], "Hashes should match"
    
    def test_random_wiring_differs_with_seed(self):
        """Different seeds should give different wiring."""
        config1 = WiringConfig(mode='random', n_pn=100, n_kc=500, seed=42)
        config2 = WiringConfig(mode='random', n_pn=100, n_kc=500, seed=123)
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        
        _, pn_kc_1, meta1 = create_wiring_matrix(config1, 500, rng1)
        _, pn_kc_2, meta2 = create_wiring_matrix(config2, 500, rng2)
        
        assert not np.allclose(pn_kc_1, pn_kc_2), "Different seeds should give different wiring"
        assert meta1['hash'] != meta2['hash'], "Hashes should differ"
    
    def test_wiring_sparsity(self):
        """Wiring matrix should have expected sparsity."""
        config = WiringConfig(mode='random', n_pn=100, n_kc=500, pn_per_kc=7, seed=42)
        rng = np.random.default_rng(42)
        
        _, pn_kc, metadata = create_wiring_matrix(config, 500, rng)
        
        # Check avg PNs per KC
        avg_pn = np.mean(np.sum(pn_kc > 0, axis=0))
        assert abs(avg_pn - 7) < 0.5, f"Expected ~7 PNs per KC, got {avg_pn}"
        
        # Check sparsity
        sparsity = np.mean(pn_kc > 0)
        expected_sparsity = 7 / 100  # 7 PNs out of 100
        assert abs(sparsity - expected_sparsity) < 0.02, \
            f"Expected sparsity ~{expected_sparsity}, got {sparsity}"
    
    def test_wiring_mode_in_metadata(self):
        """Wiring mode should be recorded in metadata."""
        config = WiringConfig(mode='random', seed=42)
        rng = np.random.default_rng(42)
        
        _, _, metadata = create_wiring_matrix(config, 500, rng)
        
        assert 'source' in metadata
        assert 'hash' in metadata
        assert 'sparsity' in metadata
    
    def test_flywire_mode_fallback(self):
        """Flywire mode should fall back gracefully when data not available."""
        config = WiringConfig(mode='flywire', seed=42, connectome_path=None)
        rng = np.random.default_rng(42)
        
        _, pn_kc, metadata = create_wiring_matrix(config, 500, rng)
        
        # Should fall back to random_hemibrain_stats
        assert metadata['source'] in ['random_hemibrain_stats', 'hemibrain']
        assert pn_kc.shape[1] == config.n_kc


class TestDANTeaching:
    """Tests for Stage D: DAN compartment-local teaching."""
    
    def test_pn_kc_frozen_during_training(self):
        """PN→KC weights should not change during DAN training step."""
        config = MoreFlyConfig(
            wiring_config=WiringConfig(mode='random', freeze_pn_kc=True),
            dan_config=DANConfig(enabled=True),
            phonemes=['K', 'AE', 'T'],  # Small subset
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        # Copy PN→KC weights before training
        pn_kc_before = swarm.shared.pn_kc_weights.copy()
        
        # Run several training steps
        for _ in range(10):
            swarm.train_step(
                "_ca_t__",
                "K",
                phoneme_pos=0.33,
                n_phonemes=3,
            )
        
        # PN→KC should be unchanged
        pn_kc_after = swarm.shared.pn_kc_weights
        
        assert np.allclose(pn_kc_before, pn_kc_after), \
            "PN→KC weights should be frozen during training"
    
    def test_kc_mbon_changes_during_training(self):
        """KC→MBON weights should change during training on errors."""
        config = MoreFlyConfig(
            wiring_config=WiringConfig(mode='random'),
            dan_config=DANConfig(enabled=True, learning_rate=0.1),
            phonemes=['K', 'AE', 'T'],
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        # Get initial KC→MBON weights
        weights_before = {
            p: s.kc_mbon_weights.copy()
            for p, s in swarm.specialists.items()
        }
        
        # Run training steps (should cause some errors)
        for i in range(20):
            target = ['K', 'AE', 'T'][i % 3]
            swarm.train_step(
                "_ca_t__" if i % 2 == 0 else "__dog__",
                target,
                phoneme_pos=0.5,
                n_phonemes=3,
            )
        
        # Check that at least one specialist's weights changed
        any_changed = False
        for p, before in weights_before.items():
            after = swarm.specialists[p].kc_mbon_weights
            if not np.allclose(before, after):
                any_changed = True
                break
        
        assert any_changed, "At least one specialist's KC→MBON weights should change"
    
    def test_dan_updates_compartment_specific(self):
        """DAN should only update specialists involved in the error."""
        config = MoreFlyConfig(
            wiring_config=WiringConfig(mode='random'),
            dan_config=DANConfig(enabled=True, compartment_teaching=True),
            phonemes=['K', 'AE', 'T', 'D', 'G'],
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        # Get initial weights
        weights_before = {
            p: s.kc_mbon_weights.copy()
            for p, s in swarm.specialists.items()
        }
        
        # Train on K (only K and predicted should update)
        swarm.train_step("_ka_t__", "K", phoneme_pos=0.33, n_phonemes=3)
        
        # Check DAN stats
        stats = swarm.dan_teacher.get_stats()
        updated_phonemes = set(stats['compartment_updates'].keys())
        
        # At most 2 phonemes should be updated (target + wrong prediction)
        assert len(updated_phonemes) <= 2, \
            f"Expected at most 2 updated compartments, got {len(updated_phonemes)}"


class TestMoreFlySwarm:
    """Integration tests for MoreFlySwarm."""
    
    def test_swarm_creation(self):
        """Swarm should be created with correct configuration."""
        config = MoreFlyConfig(
            phonemes=['K', 'AE', 'T'],
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        assert len(swarm.specialists) == 3
        assert 'K' in swarm.specialists
        assert 'AE' in swarm.specialists
        assert 'T' in swarm.specialists
    
    def test_swarm_prediction(self):
        """Swarm should produce predictions."""
        config = MoreFlyConfig(
            phonemes=['K', 'AE', 'T', 'D', 'AA', 'G'],
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        phoneme, confidence, scores = swarm.predict(
            "_ca_t__",
            phoneme_pos=0.33,
            n_phonemes=3,
        )
        
        assert phoneme in swarm.phoneme_list
        assert 0 <= confidence <= 1
        assert len(scores) == len(swarm.specialists)
    
    def test_swarm_word_prediction(self):
        """Swarm should predict word sequences."""
        config = MoreFlyConfig(
            phonemes=PHONEME_LIST[:20],  # Subset for speed
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        phonemes = swarm.predict_word("cat", n_phonemes=3)
        
        assert len(phonemes) == 3
        assert all(p in swarm.phoneme_list for p in phonemes)
    
    def test_swarm_save_load(self, tmp_path):
        """Swarm should save and load correctly."""
        config = MoreFlyConfig(
            phonemes=['K', 'AE', 'T'],
            seed=42,
        )
        
        swarm = MoreFlySwarm(config)
        
        # Train a bit to change weights
        for _ in range(5):
            swarm.train_step("_ca_t__", "K", phoneme_pos=0.33, n_phonemes=3)
        
        # Save
        save_path = tmp_path / "test_swarm.npz"
        swarm.save(str(save_path))
        
        # Load
        swarm_loaded = MoreFlySwarm.load(str(save_path))
        
        # Check weights match
        for p in swarm.specialists:
            orig = swarm.specialists[p].kc_mbon_weights
            loaded = swarm_loaded.specialists[p].kc_mbon_weights
            assert np.allclose(orig, loaded), f"Weights for {p} should match"
    
    def test_verify_functions(self):
        """Verification utility functions should work."""
        cue_config = CueConfig(use_previous_phone=True)
        
        # Previous phone cue test
        assert verify_previous_phone_cue_changes_encoding(cue_config)
        
        # Short word position test
        assert verify_short_word_position_differs(cue_config)


class TestAblationConfigs:
    """Tests for ablation configurations."""
    
    def test_baseline_config_no_new_features(self):
        """Baseline config should not have new Stage A/D features."""
        from cursed_tts.train_more_fly import get_baseline_config
        
        config = get_baseline_config()
        
        assert config.cue_config.use_previous_phone == False
        assert config.cue_config.use_focus_features == False
        assert config.dan_config.enabled == False
        assert config.wiring_config.mode == 'random'
    
    def test_full_config_has_all_features(self):
        """Full config should have all new features."""
        from cursed_tts.train_more_fly import get_full_config
        
        config = get_full_config()
        
        assert config.cue_config.use_previous_phone == True
        assert config.cue_config.use_focus_features == True
        assert config.dan_config.enabled == True
        assert config.wiring_config.mode == 'flywire'
    
    def test_configs_deterministic(self):
        """Same seed should give identical configs."""
        from cursed_tts.train_more_fly import get_baseline_config
        
        config1 = get_baseline_config(seed=42)
        config2 = get_baseline_config(seed=42)
        
        assert config1.seed == config2.seed
        assert config1.to_dict() == config2.to_dict()


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
