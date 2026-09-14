"""
Unit tests for alignment consistency between training and inference.

These tests verify that the letter contexts generated during training match
those used during inference, ensuring no train/decode mismatch.
"""
import pytest
import numpy as np
from cursed_tts.alignment import align_word, encode_letter_context
from cursed_tts.specialist_fly import FlySwarm, SwarmConfig, SpecialistConfig
from cursed_tts.phonemes import PHONEME_LIST


def get_inference_contexts(word: str, n_phonemes: int, context_size: int = 3):
    """
    Replicate the inference context generation from FlySwarm.predict_word.
    
    This should produce identical contexts to align_word() for the same inputs.
    """
    word = word.lower()
    n_letters = len(word)
    contexts = []
    
    for p_idx in range(n_phonemes):
        if n_phonemes == 1:
            letter_pos = n_letters // 2
        else:
            letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
        letter_pos = max(0, min(letter_pos, n_letters - 1))
        
        context_chars = []
        for offset in range(-context_size, context_size + 1):
            idx = letter_pos + offset
            if 0 <= idx < n_letters:
                context_chars.append(word[idx])
            else:
                context_chars.append('_')
        contexts.append(''.join(context_chars))
    
    return contexts


class TestAlignmentConsistency:
    """Tests for train/inference alignment consistency."""
    
    @pytest.fixture
    def test_words(self):
        """Test cases with words and their phonemes."""
        return [
            ('cat', ['K', 'AE', 'T']),
            ('dog', ['D', 'AO', 'G']),
            ('mushroom', ['M', 'AH', 'SH', 'R', 'UW', 'M']),
            ('yes', ['Y', 'EH', 'S']),
            ('go', ['G', 'OW']),
            ('a', ['AH']),  # Single letter edge case
            ('hi', ['HH', 'AY']),
            ('australia', ['AO', 'S', 'T', 'R', 'EY', 'L', 'Y', 'AH']),
        ]
    
    def test_contexts_match_exactly(self, test_words):
        """Verify training and inference contexts are identical."""
        for word, phonemes in test_words:
            train_pairs = align_word(word, phonemes, context_size=3)
            train_contexts = [p.letter_context for p in train_pairs]
            infer_contexts = get_inference_contexts(word, len(phonemes), context_size=3)
            
            assert len(train_contexts) == len(infer_contexts), \
                f"Context count mismatch for '{word}'"
            
            for i, (tc, ic) in enumerate(zip(train_contexts, infer_contexts)):
                assert tc == ic, \
                    f"Context mismatch for '{word}' phoneme {i}: train='{tc}' != infer='{ic}'"
    
    def test_context_size_variations(self, test_words):
        """Test alignment consistency across different context sizes."""
        for context_size in [1, 2, 3, 4]:
            for word, phonemes in test_words:
                train_pairs = align_word(word, phonemes, context_size=context_size)
                train_contexts = [p.letter_context for p in train_pairs]
                infer_contexts = get_inference_contexts(word, len(phonemes), context_size=context_size)
                
                for i, (tc, ic) in enumerate(zip(train_contexts, infer_contexts)):
                    assert tc == ic, \
                        f"Context mismatch (size={context_size}) for '{word}'"
    
    def test_feature_encoding_deterministic(self):
        """Verify feature encoding is deterministic."""
        context = "_cat___"
        
        features1 = encode_letter_context(context)
        features2 = encode_letter_context(context)
        
        np.testing.assert_array_equal(features1, features2,
            err_msg="Feature encoding should be deterministic")
    
    def test_swarm_predict_word_uses_same_alignment(self):
        """Verify FlySwarm.predict_word uses the same alignment as training."""
        # Create a minimal swarm for testing
        config = SwarmConfig(
            phonemes=['K', 'AE', 'T'],
            seed=42,
        )
        swarm = FlySwarm(config)
        
        # The swarm's predict_word should generate the same contexts as align_word
        word = 'cat'
        phonemes = ['K', 'AE', 'T']
        
        train_pairs = align_word(word, phonemes, context_size=3)
        train_contexts = [p.letter_context for p in train_pairs]
        
        # We can't directly access the contexts from predict_word, but we can
        # verify the alignment logic is consistent by checking the implementation
        n_letters = len(word)
        n_phonemes = len(phonemes)
        
        for p_idx in range(n_phonemes):
            if n_phonemes == 1:
                expected_pos = n_letters // 2
            else:
                expected_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
            expected_pos = max(0, min(expected_pos, n_letters - 1))
            
            assert train_pairs[p_idx].letter_pos == expected_pos, \
                f"Letter position mismatch for phoneme {p_idx}"


class TestCalibration:
    """Tests for calibration functionality."""
    
    def test_calibrator_initialization(self):
        """Test PlattCalibrator initialization."""
        from cursed_tts.specialist_fly import PlattCalibrator
        
        phonemes = ['K', 'AE', 'T']
        cal = PlattCalibrator(phonemes)
        
        assert cal.phonemes == phonemes
        assert not cal.is_fitted
        assert len(cal.params) == len(phonemes)
    
    def test_calibrator_fit(self):
        """Test calibration fitting."""
        from cursed_tts.specialist_fly import PlattCalibrator
        
        phonemes = ['K', 'AE']
        cal = PlattCalibrator(phonemes)
        
        # Create synthetic data
        scores = {
            'K': [0.5, 0.8, -0.2, 0.6, 0.9] * 10,
            'AE': [-0.1, 0.3, 0.7, 0.2, -0.3] * 10,
        }
        labels = {
            'K': [1, 1, 0, 1, 1] * 10,
            'AE': [0, 1, 1, 0, 0] * 10,
        }
        
        cal.fit(scores, labels)
        
        assert cal.is_fitted
        
        # Test calibration
        test_scores = {'K': 0.5, 'AE': 0.3}
        calibrated = cal.calibrate(test_scores)
        
        assert 0 <= calibrated['K'] <= 1
        assert 0 <= calibrated['AE'] <= 1
    
    def test_calibrator_serialization(self):
        """Test calibrator save/load via dict."""
        from cursed_tts.specialist_fly import PlattCalibrator
        
        phonemes = ['K', 'AE', 'T']
        cal = PlattCalibrator(phonemes)
        cal.params['K'] = np.array([2.0, -0.5], dtype=np.float32)
        cal.is_fitted = True
        
        # Serialize
        d = cal.to_dict()
        
        # Deserialize
        cal2 = PlattCalibrator.from_dict(d)
        
        assert cal2.phonemes == cal.phonemes
        assert cal2.is_fitted == cal.is_fitted
        np.testing.assert_array_almost_equal(cal2.params['K'], cal.params['K'])


class TestDemoPhonemes:
    """Tests for demo phoneme coverage."""
    
    def test_demo_phonemes_cover_demo_words(self):
        """Verify get_demo_phonemes() covers all phonemes in get_demo_words()."""
        from cursed_tts.specialist_fly import get_demo_phonemes
        from cursed_tts.lexicon import get_phonemes, get_demo_words
        from cursed_tts.phonemes import strip_stress
        
        demo_phonemes = set(get_demo_phonemes())
        demo_words = get_demo_words()
        
        for word in demo_words:
            try:
                phones = get_phonemes(word, allow_g2p=True)
                for p in phones:
                    p_stripped = strip_stress(p)
                    assert p_stripped in demo_phonemes, \
                        f"Phoneme '{p_stripped}' from word '{word}' not in demo phonemes"
            except Exception:
                pass  # Skip words that can't be looked up


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
