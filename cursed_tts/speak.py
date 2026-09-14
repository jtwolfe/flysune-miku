"""
G2P-MB speaker: predict phonemes with mushroom body, render with clean synth.

Default path: MB predicts phonemes → concatenate formant crumbs
--lexicon path: Use CMUdict/G2P reference phonemes → concatenate formant crumbs
--swarm path: Use one-phoneme-per-fly ensemble → concatenate formant crumbs

All paths use the same clean concatenative synthesizer (Stage 1 quality).
The difference is where the phoneme predictions come from.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .mushroom_body import MushroomBody
from .synth import (
    synthesize_phoneme, synthesize_all_phonemes, concatenate_phonemes,
    save_wav, SAMPLE_RATE
)
from .lexicon import get_phonemes, is_known_word, get_demo_words
from .phonemes import strip_stress


class Speaker:
    """
    G2P speaker using mushroom body predictions.
    
    The MB is a phoneme classifier (like OCR hiragana demo).
    Audio comes from clean formant synthesis, not neural trajectory regression.
    """
    
    def __init__(self, mb: MushroomBody):
        self.mb = mb
        
        # Precompute phoneme audio crumbs
        self.phoneme_audio = synthesize_all_phonemes()
    
    @classmethod
    def from_model_file(cls, path: str) -> 'Speaker':
        """Load speaker from model file."""
        mb = MushroomBody.load(path)
        return cls(mb)
    
    def get_reference_phonemes(self, word: str) -> Tuple[List[str], bool]:
        """
        Get reference phonemes from CMUdict or G2P.
        
        Returns:
            phonemes: List of phoneme symbols
            is_known: Whether word is in CMUdict
        """
        phonemes = get_phonemes(word, allow_g2p=True)
        known = is_known_word(word)
        return phonemes, known
    
    def get_mb_phonemes(self, word: str, n_phonemes: int) -> List[str]:
        """
        Predict phonemes using the mushroom body.
        
        Args:
            word: Word to predict
            n_phonemes: Number of phonemes to predict (from reference)
        
        Returns:
            List of predicted phoneme symbols
        """
        return self.mb.predict_word(word, n_phonemes)
    
    def synthesize(self, phonemes: List[str]) -> np.ndarray:
        """
        Synthesize audio from phoneme sequence.
        
        Uses clean concatenative formant synthesis (Stage 1 quality).
        """
        return concatenate_phonemes(phonemes, self.phoneme_audio)
    
    def speak(
        self,
        word: str,
        output_path: Optional[str] = None,
        use_lexicon: bool = False,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[str], bool]:
        """
        Speak a word.
        
        Args:
            word: Word to speak
            output_path: Optional path to save WAV
            use_lexicon: If True, use dictionary phonemes (baseline).
                         If False (default), use MB predictions (cursed G2P).
            verbose: Print details
        
        Returns:
            audio: Audio signal
            audio_phonemes: Phonemes used for synthesis
            reference_phonemes: Reference phonemes from CMUdict/G2P
            is_known: Whether word is in CMUdict
        """
        word = word.lower().strip()
        
        # Get reference phonemes (for comparison and length)
        ref_phonemes, is_known = self.get_reference_phonemes(word)
        ref_phonemes_stripped = [strip_stress(p) for p in ref_phonemes]
        
        if use_lexicon:
            # Baseline: use dictionary phonemes
            audio_phonemes = ref_phonemes_stripped
            source = "lexicon"
        else:
            # Default: use MB predictions (the cursed G2P)
            audio_phonemes = self.get_mb_phonemes(word, len(ref_phonemes))
            audio_phonemes = [strip_stress(p) for p in audio_phonemes]
            source = "MB"
        
        # Synthesize audio
        audio = self.synthesize(audio_phonemes)
        
        if verbose:
            known_str = "CMUdict" if is_known else "G2P"
            print(f"\nSpeaking: '{word}' ({known_str})")
            print(f"  Reference: {' '.join(ref_phonemes_stripped)}")
            print(f"  Audio ({source}): {' '.join(audio_phonemes)}")
            
            # Show comparison
            n_match = sum(1 for r, a in zip(ref_phonemes_stripped, audio_phonemes) if r == a)
            n_total = len(ref_phonemes_stripped)
            match_pct = 100 * n_match / n_total if n_total > 0 else 0
            print(f"  Match: {n_match}/{n_total} ({match_pct:.0f}%)")
        
        if output_path:
            save_wav(audio, output_path, SAMPLE_RATE)
        
        return audio, audio_phonemes, ref_phonemes_stripped, is_known
    
    def speak_demo(
        self,
        output_dir: str = "artifacts/g2p",
        use_lexicon: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak all demo words."""
        demo_words = get_demo_words()
        return self._speak_wordlist(demo_words, output_dir, use_lexicon, verbose)
    
    def speak_oov_examples(
        self,
        output_dir: str = "artifacts/g2p",
        use_lexicon: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak OOV example words."""
        oov_words = [
            'mushroom', 'connectome', 'chaos', 'hatsune', 'australia',
            'neural', 'phoneme', 'cursed', 'flywire', 'kenyon'
        ]
        return self._speak_wordlist(oov_words, output_dir, use_lexicon, verbose)
    
    def _speak_wordlist(
        self,
        words: List[str],
        output_dir: str,
        use_lexicon: bool,
        verbose: bool,
    ) -> Dict[str, Dict]:
        """Speak a list of words."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for word in words:
            word_clean = ''.join(c for c in word.lower() if c.isalnum())
            output_path = output_dir / f"{word_clean}.wav"
            
            try:
                audio, audio_ph, ref_ph, known = self.speak(
                    word, str(output_path), use_lexicon=use_lexicon, verbose=verbose
                )
                results[word] = {
                    'audio_phonemes': audio_ph,
                    'reference_phonemes': ref_ph,
                    'is_known': known,
                    'path': str(output_path),
                }
            except Exception as e:
                print(f"Error on '{word}': {e}")
                results[word] = {'error': str(e)}
        
        return results
    
    def speak_all(
        self,
        output_dir: str = "artifacts/g2p",
        use_lexicon: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak demo words and OOV examples."""
        results = {}
        
        if verbose:
            source = "lexicon" if use_lexicon else "MB (G2P)"
            print(f"\n=== Demo Words ({source}) ===")
        results.update(self.speak_demo(output_dir, use_lexicon, verbose))
        
        if verbose:
            print(f"\n=== OOV Examples ({source}) ===")
        results.update(self.speak_oov_examples(output_dir, use_lexicon, verbose))
        
        if verbose:
            print(f"\nWAVs saved to: {output_dir}")
        
        return results


class SwarmSpeaker:
    """
    Speaker using one-phoneme-per-fly ensemble (picker + singer flies).
    
    Two stages:
    1. Picker flies: Each specialist votes YES/NO for its target phoneme.
       The phoneme with the strongest YES wins.
    2. Singer flies: Each phoneme has its own personalized audio renderer
       with distinct voice characteristics (F0, formants, vibrato, etc.)
    
    This creates a "mixture of flies" (MoE) synthesis where different parts
    of the utterance are sung by different specialized flies.
    """
    
    def __init__(self, picker_swarm, singer_swarm=None):
        from .specialist_fly import FlySwarm
        from .singer_fly import SingerSwarm
        
        self.picker_swarm: FlySwarm = picker_swarm
        
        # Create singer swarm with same phonemes as picker
        if singer_swarm is None:
            self.singer_swarm = SingerSwarm(
                phonemes=list(picker_swarm.specialists.keys()),
                seed=42,
            )
        else:
            self.singer_swarm: SingerSwarm = singer_swarm
        
        # Fallback for phonemes not in singer swarm
        self.phoneme_audio = synthesize_all_phonemes()
    
    @classmethod
    def from_model_file(cls, path: str, singer_path: Optional[str] = None) -> 'SwarmSpeaker':
        """Load swarm speaker from model file."""
        from .specialist_fly import FlySwarm
        from .singer_fly import SingerSwarm
        
        picker_swarm = FlySwarm.load(path)
        
        singer_swarm = None
        if singer_path and Path(singer_path).exists():
            try:
                singer_swarm = SingerSwarm.load(singer_path)
            except Exception:
                pass
        
        return cls(picker_swarm, singer_swarm)
    
    def get_reference_phonemes(self, word: str) -> Tuple[List[str], bool]:
        """Get reference phonemes from CMUdict or G2P."""
        phonemes = get_phonemes(word, allow_g2p=True)
        known = is_known_word(word)
        return phonemes, known
    
    def get_swarm_phonemes(self, word: str, n_phonemes: int) -> List[str]:
        """Predict phonemes using the picker fly swarm."""
        return self.picker_swarm.predict_word(word, n_phonemes)
    
    def synthesize(self, phonemes: List[str]) -> np.ndarray:
        """
        Synthesize audio from phoneme sequence using singer flies.
        
        Each phoneme is rendered by its own singer fly with personalized
        voice characteristics. This is the MoE (mixture-of-flies) synthesis.
        """
        return self.singer_swarm.synthesize_sequence(phonemes)
    
    def speak(
        self,
        word: str,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[str], bool]:
        """
        Speak a word using the fly swarm.
        
        Returns:
            audio: Audio signal
            audio_phonemes: Phonemes used for synthesis (swarm predictions)
            reference_phonemes: Reference phonemes from CMUdict/G2P
            is_known: Whether word is in CMUdict
        """
        word = word.lower().strip()
        
        # Get reference phonemes
        ref_phonemes, is_known = self.get_reference_phonemes(word)
        ref_phonemes_stripped = [strip_stress(p) for p in ref_phonemes]
        
        # Use swarm predictions
        audio_phonemes = self.get_swarm_phonemes(word, len(ref_phonemes))
        audio_phonemes = [strip_stress(p) for p in audio_phonemes]
        
        # Synthesize
        audio = self.synthesize(audio_phonemes)
        
        if verbose:
            known_str = "CMUdict" if is_known else "G2P"
            print(f"\nSpeaking: '{word}' ({known_str}) [SWARM]")
            print(f"  Reference: {' '.join(ref_phonemes_stripped)}")
            print(f"  Swarm:     {' '.join(audio_phonemes)}")
            
            n_match = sum(1 for r, a in zip(ref_phonemes_stripped, audio_phonemes) if r == a)
            n_total = len(ref_phonemes_stripped)
            match_pct = 100 * n_match / n_total if n_total > 0 else 0
            print(f"  Match: {n_match}/{n_total} ({match_pct:.0f}%)")
        
        if output_path:
            save_wav(audio, output_path, SAMPLE_RATE)
        
        return audio, audio_phonemes, ref_phonemes_stripped, is_known
    
    def speak_demo(
        self,
        output_dir: str = "artifacts/swarm",
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak all demo words."""
        demo_words = get_demo_words()
        return self._speak_wordlist(demo_words, output_dir, verbose)
    
    def speak_oov_examples(
        self,
        output_dir: str = "artifacts/swarm",
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak OOV example words (swarm-specific demo words)."""
        oov_words = [
            'cat', 'dog', 'mushroom', 'hatsune', 'australia',
            'kenyon', 'chaos', 'connectome'
        ]
        return self._speak_wordlist(oov_words, output_dir, verbose)
    
    def _speak_wordlist(
        self,
        words: List[str],
        output_dir: str,
        verbose: bool,
    ) -> Dict[str, Dict]:
        """Speak a list of words."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for word in words:
            word_clean = ''.join(c for c in word.lower() if c.isalnum())
            output_path = output_dir / f"{word_clean}.wav"
            
            try:
                audio, audio_ph, ref_ph, known = self.speak(
                    word, str(output_path), verbose=verbose
                )
                results[word] = {
                    'audio_phonemes': audio_ph,
                    'reference_phonemes': ref_ph,
                    'is_known': known,
                    'path': str(output_path),
                }
            except Exception as e:
                print(f"Error on '{word}': {e}")
                results[word] = {'error': str(e)}
        
        return results
    
    def speak_all(
        self,
        output_dir: str = "artifacts/swarm",
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak demo words and swarm-specific examples."""
        results = {}
        
        if verbose:
            print(f"\n=== Demo Words (SWARM) ===")
        results.update(self.speak_demo(output_dir, verbose))
        
        if verbose:
            print(f"\n=== Swarm Examples ===")
        # Add swarm-specific demo words
        swarm_words = [
            'mushroom', 'connectome', 'chaos', 'hatsune', 'australia',
            'kenyon', 'flywire', 'neural'
        ]
        results.update(self._speak_wordlist(swarm_words, output_dir, verbose))
        
        if verbose:
            print(f"\nSwarm WAVs saved to: {output_dir}")
        
        return results


def speak_word(
    word: str,
    model_path: str = "model.npz",
    output_path: Optional[str] = None,
    use_lexicon: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to speak a single word.
    
    Args:
        word: Word to speak
        model_path: Path to trained MB model
        output_path: Optional path to save WAV
        use_lexicon: Use dictionary phonemes instead of MB predictions
        verbose: Print details
    
    Returns:
        audio: Audio signal
        phonemes: Phonemes used for synthesis
    """
    speaker = Speaker.from_model_file(model_path)
    audio, audio_ph, _, _ = speaker.speak(word, output_path, use_lexicon, verbose)
    return audio, audio_ph


def speak_word_swarm(
    word: str,
    model_path: str = "model_swarm.npz",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to speak a single word with swarm.
    """
    speaker = SwarmSpeaker.from_model_file(model_path)
    audio, audio_ph, _, _ = speaker.speak(word, output_path, verbose)
    return audio, audio_ph


if __name__ == "__main__":
    # Test speaking
    import sys
    
    word = sys.argv[1] if len(sys.argv) > 1 else "cat"
    use_lexicon = "--lexicon" in sys.argv
    
    print(f"Testing speak for '{word}' (lexicon={use_lexicon})...")
    
    try:
        speaker = Speaker.from_model_file("model.npz")
        audio, audio_ph, ref_ph, known = speaker.speak(
            word, f"test_{word}.wav", use_lexicon=use_lexicon, verbose=True
        )
    except FileNotFoundError:
        print("Model not found. Train first: python -m cursed_tts train")
