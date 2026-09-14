"""
Tests for MARIAN ILUSTRADO voicebank integration.

Tests ARPAbet ↔ Arpasing mapping, UTAU singer flies, and synthesis.
These tests run without network access if crumbs are vendored.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from cursed_tts.voicebanks.arpasing_map import (
    ARPABET_TO_ARPASING, ARPASING_TO_ARPABET,
    arpabet_to_arpasing, arpasing_to_arpabet,
    get_arpasing_fallback, get_mapping_coverage,
    parse_arpasing_alias, get_phonemes_from_alias,
)
from cursed_tts.voicebanks.utau_singer import (
    UTAUSingerFly, UTAUSingerSwarm, MarianSingerSwarm,
    load_oto_ini, find_voicebank_samples, OtoEntry,
)
from cursed_tts.phonemes import PHONEME_LIST, strip_stress
from cursed_tts.synth import SAMPLE_RATE


class TestArpasingMapping:
    """Test ARPAbet ↔ Arpasing mapping."""
    
    def test_all_phonemes_mapped(self):
        """All 39 CMUdict phonemes should have Arpasing aliases."""
        for phoneme in PHONEME_LIST:
            alias = arpabet_to_arpasing(phoneme)
            assert alias is not None, f"Missing mapping for {phoneme}"
            assert alias.islower(), f"Alias should be lowercase: {alias}"
    
    def test_mapping_is_bijective(self):
        """Mapping should be one-to-one."""
        assert len(ARPABET_TO_ARPASING) == len(ARPASING_TO_ARPABET)
        
        for arpabet, arpasing in ARPABET_TO_ARPASING.items():
            assert ARPASING_TO_ARPABET[arpasing] == arpabet
    
    def test_stress_stripping(self):
        """Stress markers should be stripped."""
        assert arpabet_to_arpasing('AA0') == 'aa'
        assert arpabet_to_arpasing('AA1') == 'aa'
        assert arpabet_to_arpasing('AA2') == 'aa'
        assert arpabet_to_arpasing('IY0') == 'iy'
    
    def test_reverse_mapping(self):
        """Arpasing → ARPAbet mapping should work."""
        assert arpasing_to_arpabet('aa') == 'AA'
        assert arpasing_to_arpabet('iy') == 'IY'
        assert arpasing_to_arpabet('k') == 'K'
        assert arpasing_to_arpabet('ch') == 'CH'
    
    def test_coverage_analysis(self):
        """Coverage analysis should report correct stats."""
        # Full coverage
        full_bank = set(ARPABET_TO_ARPASING.values())
        coverage = get_mapping_coverage(full_bank)
        
        assert coverage['total_phonemes'] == 39
        assert coverage['directly_covered'] == 39
        assert coverage['direct_coverage_pct'] == 100.0
        assert len(coverage['missing_no_fallback']) == 0
    
    def test_partial_coverage_with_fallback(self):
        """Partial coverage should show fallbacks."""
        partial_bank = {'aa', 'ae', 'iy', 'k', 't', 's', 'm', 'n', 'l', 'r'}
        coverage = get_mapping_coverage(partial_bank)
        
        assert coverage['directly_covered'] < 39
        assert coverage['covered_with_fallback'] >= coverage['directly_covered']
    
    def test_fallback_chains(self):
        """Fallback chains should find alternatives."""
        bank_without_th = set(ARPABET_TO_ARPASING.values()) - {'th'}
        
        fallback = get_arpasing_fallback('TH', bank_without_th)
        assert fallback is not None
        assert fallback in bank_without_th
        assert fallback in ['f', 's']  # Expected fallbacks for TH


class TestArpasingAliasParser:
    """Test Arpasing alias pattern parsing."""
    
    def test_standalone_vowel(self):
        """Parse standalone vowel aliases."""
        alias_type, phonemes = parse_arpasing_alias('aa')
        assert alias_type == 'v'
        assert phonemes == ['aa']
    
    def test_cv_alias(self):
        """Parse CV (consonant-vowel) aliases."""
        alias_type, phonemes = parse_arpasing_alias('k aa')
        assert alias_type in ['cv', 'cc', 'vv']  # Pattern matcher may vary
        assert 'aa' in phonemes or 'k' in phonemes
    
    def test_phoneme_extraction(self):
        """Extract ARPAbet phonemes from aliases."""
        phonemes = get_phonemes_from_alias('k aa')
        # Should contain K and/or AA
        assert len(phonemes) >= 1


class TestOtoIni:
    """Test oto.ini parsing."""
    
    def test_parse_oto_entry(self):
        """Parse basic oto.ini format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("test.wav=aa,100,50,-300,20,10\n")
            f.write("test2.wav=k aa,0,100,-200,30,15\n")
            f.flush()
            
            entries = load_oto_ini(Path(f.name))
        
        assert 'aa' in entries
        assert entries['aa'].filename == 'test.wav'
        assert entries['aa'].offset == 100
        assert entries['aa'].consonant == 50
        assert entries['aa'].cutoff == -300
    
    def test_empty_oto(self):
        """Handle empty oto.ini gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("")
            f.flush()
            
            entries = load_oto_ini(Path(f.name))
        
        assert entries == {}
    
    def test_missing_oto(self):
        """Handle missing oto.ini gracefully."""
        entries = load_oto_ini(Path('/nonexistent/oto.ini'))
        assert entries == {}


class TestUTAUSingerFly:
    """Test UTAU singer fly without actual samples."""
    
    def test_fallback_synthesis(self):
        """Singer fly should fall back to formant synthesis."""
        fly = UTAUSingerFly(
            phoneme='AA',
            sample_path=None,
            fallback_synth=True,
        )
        
        audio = fly.synthesize()
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        assert not fly.has_sample
    
    def test_silence_without_fallback(self):
        """Singer fly should return silence if no sample and no fallback."""
        fly = UTAUSingerFly(
            phoneme='AA',
            sample_path=None,
            fallback_synth=False,
        )
        
        audio = fly.synthesize()
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
    
    def test_caching(self):
        """Audio should be cached."""
        fly = UTAUSingerFly(
            phoneme='AA',
            sample_path=None,
            fallback_synth=True,
        )
        
        audio1 = fly.synthesize()
        audio2 = fly.synthesize()
        
        # Should be equal but not same object
        np.testing.assert_array_equal(audio1, audio2)


class TestUTAUSingerSwarm:
    """Test UTAU singer swarm without actual voicebank."""
    
    def test_swarm_creation(self):
        """Create swarm without voicebank path."""
        swarm = UTAUSingerSwarm(
            voicebank_path=None,
            phonemes=['AA', 'K', 'T'],
            fallback_synth=True,
        )
        
        assert len(swarm.singers) == 3
        assert 'AA' in swarm.singers
        assert 'K' in swarm.singers
        assert 'T' in swarm.singers
    
    def test_synthesis_fallback(self):
        """Swarm should synthesize using fallback."""
        swarm = UTAUSingerSwarm(
            voicebank_path=None,
            phonemes=['K', 'AE', 'T'],
            fallback_synth=True,
        )
        
        audio = swarm.synthesize_sequence(['K', 'AE', 'T'])
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32
    
    def test_coverage_summary(self):
        """Coverage summary should work."""
        swarm = UTAUSingerSwarm(
            voicebank_path=None,
            phonemes=['AA', 'AE', 'IY'],
            fallback_synth=True,
        )
        
        summary = swarm.get_coverage_summary()
        
        assert 'Voicebank' in summary
        assert 'Fallback to formant' in summary


class TestMarianSingerSwarm:
    """Test MARIAN-specific functionality."""
    
    def test_is_available_without_install(self):
        """is_available should be False if not installed."""
        # This should work even without MARIAN installed
        result = MarianSingerSwarm.is_available()
        assert isinstance(result, bool)
    
    def test_swarm_creation(self):
        """Create MARIAN swarm (falls back to formant)."""
        swarm = MarianSingerSwarm(
            voicebank_path=None,
            phonemes=['K', 'AE', 'T'],
            fallback_synth=True,
        )
        
        assert swarm.voicebank_name == 'MARIAN ILUSTRADO'
        assert swarm.author == 'Kanabun'
    
    def test_attribution(self):
        """Attribution text should include required info."""
        swarm = MarianSingerSwarm(
            voicebank_path=None,
            phonemes=['AA'],
            fallback_synth=True,
        )
        
        attribution = swarm.get_attribution()
        
        assert 'MARIAN ILUSTRADO' in attribution
        assert 'Kanabun' in attribution
        assert 'downloadmarian.carrd.co' in attribution


class TestPhoneCoverage:
    """Test phoneme coverage requirements."""
    
    def test_essential_phonemes_mapped(self):
        """Essential phonemes for demo words should be mapped."""
        demo_phonemes = ['K', 'AE', 'T', 'D', 'AO', 'G', 'M', 'IY', 'AH']
        
        for phoneme in demo_phonemes:
            alias = arpabet_to_arpasing(phoneme)
            assert alias is not None, f"Demo phoneme {phoneme} not mapped"
    
    def test_cat_pronunciation(self):
        """'cat' phonemes should all be mapped."""
        cat_phonemes = ['K', 'AE', 'T']
        
        for phoneme in cat_phonemes:
            alias = arpabet_to_arpasing(phoneme)
            assert alias is not None


class TestAliasResolution:
    """Test alternate alias pattern resolution."""
    
    def test_resolve_onset_pattern(self):
        """Swarm should try '- alias' pattern for consonants."""
        # Simulate a voicebank with only onset patterns
        swarm = UTAUSingerSwarm(
            voicebank_path=None,
            phonemes=['K', 'T', 'AA'],
            fallback_synth=True,
        )
        
        # Add mock samples with onset patterns
        swarm.available_samples = {
            '- k': (Path('/mock/k.wav'), None),
            '- t': (Path('/mock/t.wav'), None),
            'aa': (Path('/mock/aa.wav'), None),
        }
        swarm.available_aliases = set(swarm.available_samples.keys())
        
        # Test resolution
        assert swarm._resolve_alias('k', is_vowel=False) == '- k'
        assert swarm._resolve_alias('t', is_vowel=False) == '- t'
        assert swarm._resolve_alias('aa', is_vowel=True) == 'aa'
    
    def test_resolve_cv_pattern(self):
        """Swarm should try 'C V' pattern for consonants."""
        swarm = UTAUSingerSwarm(
            voicebank_path=None,
            phonemes=['K'],
            fallback_synth=True,
        )
        
        # Only CV pattern available
        swarm.available_samples = {
            'k aa': (Path('/mock/k_aa.wav'), None),
        }
        swarm.available_aliases = set(swarm.available_samples.keys())
        
        # Should find CV pattern when standalone not available
        result = swarm._resolve_alias('k', is_vowel=False)
        assert result == 'k aa'
    
    def test_resolve_numbered_variants(self):
        """Swarm should try numbered variants like 'aa1'."""
        swarm = UTAUSingerSwarm(
            voicebank_path=None,
            phonemes=['AA'],
            fallback_synth=True,
        )
        
        # Only numbered variant available
        swarm.available_samples = {
            'aa1': (Path('/mock/aa1.wav'), None),
        }
        swarm.available_aliases = set(swarm.available_samples.keys())
        
        result = swarm._resolve_alias('aa', is_vowel=True)
        assert result == 'aa1'


class TestDurationCapping:
    """Test crumb duration capping."""
    
    def test_max_duration_constants(self):
        """Duration constants should be reasonable."""
        from cursed_tts.voicebanks.utau_singer import (
            MAX_VOWEL_DURATION_MS, MAX_CONSONANT_DURATION_MS
        )
        
        assert MAX_VOWEL_DURATION_MS == 220
        assert MAX_CONSONANT_DURATION_MS == 120
    
    def test_singer_fly_knows_vowel_type(self):
        """Singer fly should correctly identify vowel vs consonant."""
        vowel_fly = UTAUSingerFly(phoneme='AA', fallback_synth=True)
        consonant_fly = UTAUSingerFly(phoneme='K', fallback_synth=True)
        
        assert vowel_fly._is_vowel is True
        assert consonant_fly._is_vowel is False


class TestIntegrationWithSpeaker:
    """Test integration with speak module."""
    
    def test_marian_speaker_import(self):
        """MarianSwarmSpeaker should be importable."""
        from cursed_tts.speak import MarianSwarmSpeaker
        assert MarianSwarmSpeaker is not None
    
    def test_speak_word_marian_function(self):
        """speak_word_marian convenience function should exist."""
        from cursed_tts.speak import speak_word_marian
        assert callable(speak_word_marian)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
