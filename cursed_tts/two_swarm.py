"""
Two-Swarm Pipeline: Picker + Speaker Architecture.

This module implements the clean two-swarm TTS pipeline:

    word → PICKER SWARM → phoneme list → SPEAKER SWARM → audio

Architecture principles:
1. PICKER SWARM (recognition): Uses KC/MB for G2P classification
   - Input: letter context
   - Output: phoneme sequence
   
2. SPEAKER SWARM (synthesis): NO KC/MB dependency
   - Input: phoneme ID + optional context (prev-phone, position)
   - Output: audio crumb

This is the correct architecture where:
- Recognition flies pick (G2P)
- Speaking flies speak (synthesis)

The PR #6 "acoustic flies" approach that stacked generation on picker KC
features is deprecated - it caused words to sound identical because speaker
output was coupled to picker state rather than phoneme identity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .phonemes import PHONEME_LIST, strip_stress
from .lexicon import get_phonemes, is_known_word
from .synth import save_wav, SAMPLE_RATE, synthesize_all_phonemes, concatenate_phonemes


class TwoSwarmSpeaker:
    """
    Two-swarm TTS speaker: Picker → Speakers.
    
    Pipeline:
        word → picker swarm → phoneme list → speaker swarm → audio
    
    The picker and speaker swarms are completely decoupled:
    - Picker uses KC/MB for classification
    - Speakers use ONLY phoneme ID (no KC activity)
    
    This ensures different words produce different audio because:
    - Different words → different phoneme sequences (from picker)
    - Different phonemes → different audio (from speakers)
    
    Contrast with PR #6 which passed KC activity to speakers, causing
    words with similar letter patterns to sound nearly identical.
    """
    
    def __init__(
        self,
        picker_swarm,           # FlySwarm or MoreFlySwarm (G2P)
        speaker_swarm=None,     # SpeakerSwarm (synthesis)
        use_formant_baseline: bool = False,  # For A/B comparison
    ):
        self.picker_swarm = picker_swarm
        self.speaker_swarm = speaker_swarm
        self.use_formant_baseline = use_formant_baseline
        
        # Fallback formant audio for baseline mode
        if use_formant_baseline or speaker_swarm is None:
            self._formant_audio = synthesize_all_phonemes()
        else:
            self._formant_audio = None
        
        mode = "formant-baseline" if use_formant_baseline else "speaker-swarm"
        if speaker_swarm is None:
            mode = "formant-baseline (no speaker swarm)"
        
        print(f"TwoSwarmSpeaker initialized (mode={mode})")
        print("  Architecture: picker → phoneme list → speakers → audio")
        print("  NO KC passed to speakers (clean two-swarm split)")
    
    @classmethod
    def from_model_files(
        cls,
        picker_path: str,
        speaker_path: Optional[str] = None,
        use_formant_baseline: bool = False,
        picker_type: str = 'swarm',  # 'swarm' or 'more_fly'
    ) -> 'TwoSwarmSpeaker':
        """
        Load two-swarm speaker from model files.
        
        Args:
            picker_path: Path to picker swarm model
            speaker_path: Path to speaker swarm model (None = formant baseline)
            use_formant_baseline: Force formant baseline even if speaker available
            picker_type: Type of picker swarm ('swarm' or 'more_fly')
        """
        # Load picker
        if picker_type == 'more_fly':
            from .more_fly import MoreFlySwarm
            picker = MoreFlySwarm.load(picker_path)
        else:
            from .specialist_fly import FlySwarm
            picker = FlySwarm.load(picker_path)
        
        # Load speaker if available
        speaker = None
        if speaker_path and Path(speaker_path).exists() and not use_formant_baseline:
            from .speaker_fly import SpeakerSwarm
            speaker = SpeakerSwarm.load(speaker_path)
        
        return cls(picker, speaker, use_formant_baseline)
    
    def get_reference_phonemes(self, word: str) -> Tuple[List[str], bool]:
        """Get reference phonemes from CMUdict or G2P."""
        phonemes = get_phonemes(word, allow_g2p=True)
        known = is_known_word(word)
        return phonemes, known
    
    def get_picker_phonemes(self, word: str, n_phonemes: int) -> List[str]:
        """
        Get phonemes from picker swarm.
        
        This uses the picker's G2P classification (KC/MB-based).
        The output is a phoneme sequence, NOT audio features.
        """
        return self.picker_swarm.predict_word(word, n_phonemes)
    
    def synthesize(self, phonemes: List[str]) -> np.ndarray:
        """
        Synthesize audio from phoneme sequence.
        
        Uses speaker swarm if available, otherwise formant baseline.
        
        CRITICAL: No KC/MB activity is passed here.
        Speakers receive only phoneme IDs.
        """
        if self.use_formant_baseline or self.speaker_swarm is None:
            # Formant baseline
            return concatenate_phonemes(phonemes, self._formant_audio)
        else:
            # Speaker swarm
            return self.speaker_swarm.synthesize_sequence(phonemes)
    
    def speak(
        self,
        word: str,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[str], bool]:
        """
        Speak a word using the two-swarm pipeline.
        
        Pipeline: word → picker → phonemes → speakers → audio
        
        Args:
            word: Word to speak
            output_path: Optional path to save WAV
            verbose: Print details
        
        Returns:
            audio: Audio signal
            picker_phonemes: Phonemes from picker swarm
            reference_phonemes: Reference phonemes from CMUdict/G2P
            is_known: Whether word is in CMUdict
        """
        word = word.lower().strip()
        
        # Get reference phonemes (for comparison)
        ref_phonemes, is_known = self.get_reference_phonemes(word)
        ref_stripped = [strip_stress(p) for p in ref_phonemes]
        
        # Get picker phonemes (G2P classification)
        picker_phonemes = self.get_picker_phonemes(word, len(ref_phonemes))
        picker_stripped = [strip_stress(p) for p in picker_phonemes]
        
        # Synthesize with speaker swarm (no KC passed!)
        audio = self.synthesize(picker_stripped)
        
        if verbose:
            known_str = "CMUdict" if is_known else "G2P"
            mode_str = "formant" if (self.use_formant_baseline or self.speaker_swarm is None) else "speakers"
            print(f"\nSpeaking: '{word}' ({known_str}) [TWO-SWARM, {mode_str}]")
            print(f"  Reference: {' '.join(ref_stripped)}")
            print(f"  Picker:    {' '.join(picker_stripped)}")
            
            # Show comparison
            n_match = sum(1 for r, p in zip(ref_stripped, picker_stripped) if r == p)
            n_total = len(ref_stripped)
            match_pct = 100 * n_match / n_total if n_total > 0 else 0
            print(f"  Match: {n_match}/{n_total} ({match_pct:.0f}%)")
        
        if output_path:
            save_wav(audio, output_path, SAMPLE_RATE)
        
        return audio, picker_stripped, ref_stripped, is_known
    
    def speak_sequence(
        self,
        text: str,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Speak a sequence of words (sentence/phrase).
        
        Args:
            text: Space-separated words
            output_path: Optional path to save WAV
            verbose: Print details
        
        Returns:
            audio: Concatenated audio
            all_phonemes: List of phoneme sequences per word
        """
        words = text.lower().strip().split()
        
        if verbose:
            print(f"\nSpeaking: '{text}' [TWO-SWARM]")
        
        all_audio = []
        all_phonemes = []
        
        # Inter-word silence
        silence_samples = int(0.15 * SAMPLE_RATE)  # 150ms between words
        silence = np.zeros(silence_samples, dtype=np.float32)
        
        for i, word in enumerate(words):
            # Clean word
            word_clean = ''.join(c for c in word if c.isalpha())
            if not word_clean:
                continue
            
            # Speak word
            audio, picker_ph, ref_ph, _ = self.speak(word_clean, verbose=False)
            
            all_audio.append(audio)
            all_phonemes.append(picker_ph)
            
            # Add silence between words
            if i < len(words) - 1:
                all_audio.append(silence)
            
            if verbose:
                print(f"  {word_clean}: {' '.join(picker_ph)}")
        
        # Concatenate
        if all_audio:
            result = np.concatenate(all_audio)
        else:
            result = np.array([], dtype=np.float32)
        
        if output_path:
            save_wav(result, output_path, SAMPLE_RATE)
            if verbose:
                duration = len(result) / SAMPLE_RATE
                print(f"\nSaved to {output_path} ({duration:.2f}s)")
        
        return result, all_phonemes
    
    TWO_SWARM_WORDS = [
        'cat', 'bat', 'dog', 'go', 'no', 'hi', 'bye', 'yes', 'me', 'you',
        'hello', 'world', 'mushroom', 'connectome', 'chaos', 'hatsune',
        'australia', 'neural', 'phoneme', 'cursed', 'flywire', 'kenyon',
    ]
    TWO_SWARM_SENTENCES = [
        'hello world',
        'the cat sat on the mat',
        'yes me too',
        'dogs go and cats bat',
    ]
    TWO_SWARM_PARAGRAPH = (
        'the cat sat on the mat. yes it did. me too. '
        'hello world this is a test of speaking flies.'
    )

    def speak_demo(
        self,
        output_dir: str = "artifacts/eval/two_swarm",
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Speak demo words using two-swarm pipeline."""
        return self._speak_wordlist(self.TWO_SWARM_WORDS, output_dir, verbose)

    def speak_eval_suite(
        self,
        output_dir: str = "artifacts/eval/two_swarm",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Words + short sentences + one 3–4 sentence paragraph."""
        output_dir = Path(output_dir)
        words_dir = output_dir / "words"
        sent_dir = output_dir / "sentences"
        words_dir.mkdir(parents=True, exist_ok=True)
        sent_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'words': self._speak_wordlist(self.TWO_SWARM_WORDS, str(words_dir), verbose),
            'sentences': {},
        }

        for i, text in enumerate(self.TWO_SWARM_SENTENCES, 1):
            slug = '_'.join(c for c in text.split() if c.isalpha())[:40]
            path = sent_dir / f"s{i}_{slug}.wav"
            audio, phones = self.speak_sequence(text, str(path), verbose=verbose)
            results['sentences'][text] = {
                'path': str(path),
                'phonemes': phones,
            }

        para_path = output_dir / "paragraph_four_sentences.wav"
        audio, phones = self.speak_sequence(
            self.TWO_SWARM_PARAGRAPH, str(para_path), verbose=verbose
        )
        results['paragraph'] = {
            'text': self.TWO_SWARM_PARAGRAPH,
            'path': str(para_path),
            'phonemes': phones,
        }
        return results
    
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
                audio, picker_ph, ref_ph, known = self.speak(
                    word, str(output_path), verbose=verbose
                )
                results[word] = {
                    'picker_phonemes': picker_ph,
                    'reference_phonemes': ref_ph,
                    'is_known': known,
                    'path': str(output_path),
                }
            except Exception as e:
                print(f"Error on '{word}': {e}")
                results[word] = {'error': str(e)}
        
        return results


def speak_two_swarm(
    word: str,
    picker_path: str = "model_swarm.npz",
    speaker_path: Optional[str] = "model_speaker.npz",
    output_path: Optional[str] = None,
    use_formant_baseline: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to speak using two-swarm pipeline.
    
    Pipeline: word → picker → phonemes → speakers → audio
    
    Args:
        word: Word to speak
        picker_path: Path to picker swarm model
        speaker_path: Path to speaker swarm model
        output_path: Optional path to save WAV
        use_formant_baseline: Use formant synthesis instead of speakers
        verbose: Print details
    
    Returns:
        audio: Audio signal
        phonemes: Phonemes from picker
    """
    speaker = TwoSwarmSpeaker.from_model_files(
        picker_path, speaker_path, use_formant_baseline
    )
    audio, picker_ph, _, _ = speaker.speak(word, output_path, verbose)
    return audio, picker_ph


def compare_two_swarm_vs_formant(
    words: List[str],
    picker_path: str,
    speaker_path: str,
    output_dir: str = "artifacts/eval/two_swarm/comparison",
    verbose: bool = True,
    picker_type: str = 'swarm',
) -> Dict[str, Dict]:
    """
    Generate comparison: two-swarm speakers vs formant baseline.
    
    Outputs two WAVs per word:
    - {word}_speakers.wav: Using trained speaker swarm
    - {word}_formant.wav: Using formant baseline
    
    This allows A/B testing of speaker quality.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("Generating two-swarm comparison demos")
        print("=" * 60)
    
    speaker_mode = TwoSwarmSpeaker.from_model_files(
        picker_path, speaker_path, False, picker_type=picker_type
    )
    formant_mode = TwoSwarmSpeaker.from_model_files(
        picker_path, None, True, picker_type=picker_type
    )
    
    results = {}
    
    for word in words:
        word_clean = ''.join(c for c in word.lower() if c.isalnum())
        
        # Speaker version
        speaker_path_out = output_dir / f"{word_clean}_speakers.wav"
        audio_s, ph_s, ref, known = speaker_mode.speak(word, str(speaker_path_out), verbose=False)
        
        # Formant version
        formant_path_out = output_dir / f"{word_clean}_formant.wav"
        audio_f, ph_f, _, _ = formant_mode.speak(word, str(formant_path_out), verbose=False)
        
        results[word] = {
            'phonemes': ph_s,
            'reference': ref,
            'speaker_path': str(speaker_path_out),
            'formant_path': str(formant_path_out),
            'speaker_duration_ms': len(audio_s) / SAMPLE_RATE * 1000,
            'formant_duration_ms': len(audio_f) / SAMPLE_RATE * 1000,
        }
        
        if verbose:
            print(f"  {word}: {' '.join(ph_s)}")
            print(f"    Speaker: {results[word]['speaker_duration_ms']:.0f}ms")
            print(f"    Formant: {results[word]['formant_duration_ms']:.0f}ms")
    
    if verbose:
        print(f"\nComparison demos saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    print("Testing TwoSwarmSpeaker...")
    print("=" * 60)
    
    # Check if models exist
    from pathlib import Path
    
    picker_path = Path("model_swarm.npz")
    speaker_path = Path("model_speaker.npz")
    
    if not picker_path.exists():
        print(f"Picker model not found: {picker_path}")
        print("Train first: python -m cursed_tts train-swarm")
    else:
        # Test with formant baseline (speaker model optional)
        print("\nTesting two-swarm pipeline with formant baseline...")
        
        speaker = TwoSwarmSpeaker.from_model_files(
            str(picker_path),
            str(speaker_path) if speaker_path.exists() else None,
            use_formant_baseline=True,
        )
        
        # Test words
        test_words = ['cat', 'dog', 'mushroom']
        
        for word in test_words:
            audio, picker_ph, ref_ph, known = speaker.speak(word, verbose=True)
            save_wav(audio, f"/tmp/two_swarm_{word}.wav", SAMPLE_RATE)
        
        # Test sentence
        print("\n" + "=" * 60)
        sentence = "hello world this is a test"
        audio, phonemes = speaker.speak_sequence(sentence, "/tmp/two_swarm_sentence.wav")
        
        print("\n" + "=" * 60)
        print("Two-swarm tests complete!")
        print("\nArchitecture verified:")
        print("  word → picker swarm → phoneme list → speakers → audio")
        print("  NO KC passed to speakers (clean split)")
