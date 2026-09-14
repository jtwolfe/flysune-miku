"""
UTAU/OpenUtau voicebank singer flies.

Replaces or supplements synthetic formant synthesis with real recorded
audio samples from open voicebanks like MARIAN ILUSTRADO.

This module provides:
- UTAUSingerFly: Single phoneme renderer using UTAU samples
- UTAUSingerSwarm: Swarm of UTAU singer flies for full phoneme coverage
- MarianSingerSwarm: Pre-configured swarm for MARIAN ILUSTRADO voicebank

The architecture maintains the picker/singer fly separation:
- Picker flies (from specialist_fly.py) choose the phoneme sequence
- UTAU singer flies render each phoneme with real samples

Attribution:
    MARIAN ILUSTRADO voicebank by Kanabun
    https://downloadmarian.carrd.co/
    License: Free to use with attribution (see README/NOTICE)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import wave
import struct
import re
import configparser
import io

from ..synth import SAMPLE_RATE, _make_envelope, save_wav
from ..phonemes import PHONEME_LIST, strip_stress, PHONEME_INVENTORY
from .arpasing_map import (
    ARPABET_TO_ARPASING, arpabet_to_arpasing, get_arpasing_fallback,
    get_mapping_coverage, ARPASING_TO_ARPABET,
)


# Default locations for voicebank data
DEFAULT_MARIAN_PATH = Path(__file__).parent.parent.parent / "data" / "marian_crumbs"
VOICEBANK_SEARCH_PATHS = [
    DEFAULT_MARIAN_PATH,
    Path.home() / ".cursed_tts" / "voicebanks" / "marian",
    Path("/opt/voicebanks/marian"),
]


@dataclass
class OtoEntry:
    """Parsed entry from oto.ini (UTAU timing configuration)."""
    filename: str          # WAV filename
    alias: str             # Phoneme alias (e.g., "aa", "k aa")
    offset: float          # Offset from start (ms)
    consonant: float       # Fixed consonant region (ms)
    cutoff: float          # Cutoff from end (ms, negative = from end)
    preutterance: float    # Pre-utterance timing (ms)
    overlap: float         # Overlap with previous note (ms)


def load_oto_ini(oto_path: Path) -> Dict[str, OtoEntry]:
    """
    Load and parse UTAU oto.ini file.
    
    oto.ini format:
        filename.wav=alias,offset,consonant,cutoff,preutterance,overlap
    
    Args:
        oto_path: Path to oto.ini file
    
    Returns:
        Dictionary mapping alias → OtoEntry
    """
    entries = {}
    
    if not oto_path.exists():
        return entries
    
    try:
        # oto.ini can be in various encodings
        for encoding in ['utf-8', 'shift_jis', 'cp932', 'latin-1']:
            try:
                content = oto_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return entries
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
            
            # Parse: filename.wav=alias,offset,consonant,cutoff,preutterance,overlap
            match = re.match(r'^(.+\.wav)=(.+?),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)$', 
                           line, re.IGNORECASE)
            if match:
                filename, alias, offset, consonant, cutoff, preutter, overlap = match.groups()
                
                # Convert numeric values (empty = 0)
                def to_float(s):
                    try:
                        return float(s) if s.strip() else 0.0
                    except ValueError:
                        return 0.0
                
                entry = OtoEntry(
                    filename=filename.strip(),
                    alias=alias.strip().lower(),
                    offset=to_float(offset),
                    consonant=to_float(consonant),
                    cutoff=to_float(cutoff),
                    preutterance=to_float(preutter),
                    overlap=to_float(overlap),
                )
                entries[entry.alias] = entry
    
    except Exception as e:
        print(f"Warning: Error parsing {oto_path}: {e}")
    
    return entries


# Maximum crumb durations (ms) to prevent multi-second samples
MAX_VOWEL_DURATION_MS = 220
MAX_CONSONANT_DURATION_MS = 120
VOWEL_BODY_MS = 80  # Extra body after consonant region for vowels


def load_utau_sample(
    wav_path: Path,
    target_sr: int = SAMPLE_RATE,
    oto_entry: Optional[OtoEntry] = None,
    max_duration_ms: Optional[float] = None,
    is_vowel: bool = True,
) -> np.ndarray:
    """
    Load a UTAU sample WAV and resample to target rate.
    
    Applies oto.ini timing if provided (offset, cutoff), with duration capping
    to prevent multi-second crumbs from full phrase recordings.
    
    Args:
        wav_path: Path to WAV file
        target_sr: Target sample rate
        oto_entry: Optional OtoEntry for timing
        max_duration_ms: Maximum crumb duration (None = use defaults)
        is_vowel: Whether this is a vowel (affects default max duration)
    
    Returns:
        Audio signal as float32 numpy array
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Sample not found: {wav_path}")
    
    with wave.open(str(wav_path), 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        
        # Read raw data
        raw = wav.readframes(n_frames)
        
        # Convert to numpy
        if sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        
        # Convert to mono if stereo
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Normalize to -1..1
        if sample_width == 1:
            audio = (audio - 128) / max_val
        else:
            audio = audio / max_val
        
        # Apply oto.ini timing
        if oto_entry:
            offset_samples = int(oto_entry.offset / 1000 * framerate)
            
            if oto_entry.cutoff < 0:
                # Negative cutoff = from end of file
                cutoff_samples = int(-oto_entry.cutoff / 1000 * framerate)
                end_sample = len(audio) - cutoff_samples
            elif oto_entry.cutoff > 0:
                # Positive cutoff = from offset
                end_sample = offset_samples + int(oto_entry.cutoff / 1000 * framerate)
            else:
                end_sample = len(audio)
            
            # Clamp to valid range
            offset_samples = max(0, min(offset_samples, len(audio) - 1))
            end_sample = max(offset_samples + 1, min(end_sample, len(audio)))
            
            audio = audio[offset_samples:end_sample]
            
            # Cap duration based on oto.consonant + body
            # ARPAsing WAVs can be full phrase recordings; we only want the crumb
            if max_duration_ms is None:
                if is_vowel:
                    # For vowels: consonant region + vowel body
                    if oto_entry.consonant > 0:
                        max_duration_ms = oto_entry.consonant + VOWEL_BODY_MS
                    else:
                        max_duration_ms = MAX_VOWEL_DURATION_MS
                else:
                    # For consonants: consonant region + small tail
                    if oto_entry.consonant > 0:
                        max_duration_ms = oto_entry.consonant + 30
                    else:
                        max_duration_ms = MAX_CONSONANT_DURATION_MS
            
            # Apply duration cap
            if max_duration_ms:
                max_samples = int(max_duration_ms / 1000 * framerate)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
        
        # Resample if needed
        if framerate != target_sr:
            # Simple linear interpolation resampling
            old_len = len(audio)
            new_len = int(old_len * target_sr / framerate)
            old_indices = np.arange(old_len)
            new_indices = np.linspace(0, old_len - 1, new_len)
            audio = np.interp(new_indices, old_indices, audio)
        
        return audio.astype(np.float32)


def find_voicebank_samples(voicebank_path: Path) -> Dict[str, Tuple[Path, Optional[OtoEntry]]]:
    """
    Scan a voicebank directory for available samples.
    
    Args:
        voicebank_path: Path to voicebank root directory
    
    Returns:
        Dictionary mapping Arpasing alias → (wav_path, oto_entry)
    """
    samples = {}
    
    if not voicebank_path.exists():
        return samples
    
    # Load oto.ini entries
    oto_entries = {}
    for oto_file in voicebank_path.rglob('oto.ini'):
        entries = load_oto_ini(oto_file)
        oto_dir = oto_file.parent
        
        for alias, entry in entries.items():
            wav_path = oto_dir / entry.filename
            if wav_path.exists():
                samples[alias] = (wav_path, entry)
                oto_entries[alias] = entry
    
    # Also scan for standalone WAV files not in oto.ini
    for wav_file in voicebank_path.rglob('*.wav'):
        # Use filename (minus extension) as alias
        alias = wav_file.stem.lower()
        if alias not in samples:
            samples[alias] = (wav_file, None)
    
    return samples


def pitch_shift_simple(audio: np.ndarray, semitones: float) -> np.ndarray:
    """
    Simple pitch shift by resampling (changes duration).
    
    For proper pitch-shifting without duration change, use a library
    like librosa or rubberband. This is a quick approximation.
    """
    if abs(semitones) < 0.01:
        return audio
    
    ratio = 2 ** (semitones / 12)
    old_len = len(audio)
    new_len = int(old_len / ratio)
    
    if new_len < 2:
        return audio
    
    old_indices = np.arange(old_len)
    new_indices = np.linspace(0, old_len - 1, new_len)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


class UTAUSingerFly:
    """
    A singer fly that uses real UTAU samples instead of formant synthesis.
    
    Loads WAV samples from an Arpasing voicebank and provides them as
    phoneme audio crumbs, with optional pitch shifting for expressiveness.
    """
    
    def __init__(
        self,
        phoneme: str,
        sample_path: Optional[Path] = None,
        oto_entry: Optional[OtoEntry] = None,
        fallback_synth: bool = True,
        pitch_shift: float = 0.0,
        max_duration_ms: Optional[float] = None,
    ):
        """
        Initialize UTAU singer fly for a phoneme.
        
        Args:
            phoneme: ARPAbet phoneme (e.g., 'AA', 'K')
            sample_path: Path to WAV sample
            oto_entry: OtoEntry for timing configuration
            fallback_synth: If True, use formant synthesis when sample unavailable
            pitch_shift: Pitch shift in semitones
            max_duration_ms: Maximum crumb duration (None = auto based on phoneme type)
        """
        self.phoneme = strip_stress(phoneme)
        self.sample_path = sample_path
        self.oto_entry = oto_entry
        self.fallback_synth = fallback_synth
        self.pitch_shift = pitch_shift
        self.max_duration_ms = max_duration_ms
        
        # Determine if vowel for duration capping
        self._is_vowel = self._check_is_vowel()
        
        # Lazy-loaded audio cache
        self._audio_cache: Optional[np.ndarray] = None
        self._has_sample = sample_path is not None and sample_path.exists()
    
    def _check_is_vowel(self) -> bool:
        """Check if this phoneme is a vowel."""
        if self.phoneme in PHONEME_INVENTORY:
            return PHONEME_INVENTORY[self.phoneme].phoneme_type == 'vowel'
        # Default guess based on common patterns
        return len(self.phoneme) == 2 and self.phoneme[0] in 'AEIOU'
    
    def _load_sample(self) -> Optional[np.ndarray]:
        """Load and process the UTAU sample."""
        if not self._has_sample:
            return None
        
        try:
            audio = load_utau_sample(
                self.sample_path,
                target_sr=SAMPLE_RATE,
                oto_entry=self.oto_entry,
                max_duration_ms=self.max_duration_ms,
                is_vowel=self._is_vowel,
            )
            
            # Apply pitch shift if specified
            if self.pitch_shift != 0:
                audio = pitch_shift_simple(audio, self.pitch_shift)
            
            # Apply envelope to smooth edges
            audio *= _make_envelope(len(audio), attack=0.01, release=0.02)
            
            # Normalize
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.75
            
            return audio
        
        except Exception as e:
            print(f"Warning: Failed to load {self.sample_path}: {e}")
            return None
    
    def _synthesize_fallback(self) -> np.ndarray:
        """Fallback to formant synthesis when sample unavailable."""
        from ..synth import synthesize_phoneme
        try:
            return synthesize_phoneme(self.phoneme)
        except ValueError:
            # Unknown phoneme - return silence
            return np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
    
    def synthesize(self) -> np.ndarray:
        """
        Get audio for this phoneme.
        
        Returns UTAU sample if available, otherwise falls back to formant synthesis.
        """
        if self._audio_cache is not None:
            return self._audio_cache.copy()
        
        audio = self._load_sample()
        
        if audio is None and self.fallback_synth:
            audio = self._synthesize_fallback()
        elif audio is None:
            audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        
        self._audio_cache = audio
        return audio.copy()
    
    def get_audio(self) -> np.ndarray:
        """Alias for synthesize() for API compatibility."""
        return self.synthesize()
    
    @property
    def has_sample(self) -> bool:
        """Whether this fly has a real UTAU sample."""
        return self._has_sample


class UTAUSingerSwarm:
    """
    A swarm of UTAU singer flies for voicebank-based synthesis.
    
    Similar to SingerSwarm but uses real UTAU samples instead of
    formant synthesis. Falls back to formant synth for missing phonemes.
    """
    
    def __init__(
        self,
        voicebank_path: Optional[Path] = None,
        phonemes: Optional[List[str]] = None,
        fallback_synth: bool = True,
    ):
        """
        Initialize UTAU singer swarm.
        
        Args:
            voicebank_path: Path to voicebank directory
            phonemes: List of phonemes to support (None = all 39)
            fallback_synth: Fall back to formant synthesis for missing samples
        """
        self.voicebank_path = voicebank_path
        self.fallback_synth = fallback_synth
        
        if phonemes is None:
            phonemes = PHONEME_LIST.copy()
        
        self.phoneme_list = [strip_stress(p) for p in phonemes]
        
        # Discover available samples
        self.available_samples: Dict[str, Tuple[Path, Optional[OtoEntry]]] = {}
        if voicebank_path and voicebank_path.exists():
            self.available_samples = find_voicebank_samples(voicebank_path)
        
        # Build alias set for coverage analysis
        self.available_aliases = set(self.available_samples.keys())
        
        # Create singer flies
        self.singers: Dict[str, UTAUSingerFly] = {}
        self._create_singers()
        
        # Coverage stats
        self.coverage = get_mapping_coverage(self.available_aliases)
    
    def _resolve_alias(self, arpasing: str, is_vowel: bool) -> Optional[str]:
        """
        Resolve an Arpasing alias to an available sample, trying alternate patterns.
        
        ARPAsing voicebanks typically have aliases like:
        - Standalone: "aa", "k" (may not exist for consonants)
        - Onset: "- aa", "- k" (vowel/consonant onset)
        - CV patterns: "k aa", "t iy" (consonant + vowel)
        - VC patterns: "aa k", "iy t" (vowel + consonant)
        - Numbered: "aa1", "- aa1" (pitch variants)
        
        For consonants especially, the standalone alias may not exist;
        we need to try "- C" or "C V" patterns.
        """
        # Common vowels to try for CV patterns
        common_vowels = ['aa', 'ae', 'ah', 'iy', 'uw', 'eh', 'ow']
        
        # Patterns to try, in order of preference
        patterns_to_try = []
        
        # 1. Direct alias
        patterns_to_try.append(arpasing)
        
        # 2. Onset pattern "- alias"
        patterns_to_try.append(f"- {arpasing}")
        
        # 3. Numbered variants
        for i in range(1, 4):
            patterns_to_try.append(f"{arpasing}{i}")
            patterns_to_try.append(f"- {arpasing}{i}")
        
        # 4. For consonants, try CV patterns (consonant + common vowel)
        if not is_vowel:
            for vowel in common_vowels:
                patterns_to_try.append(f"{arpasing} {vowel}")
        
        # 5. For vowels, try some common onset patterns
        if is_vowel:
            patterns_to_try.append(f"_{arpasing}")  # Alternate onset notation
        
        # Try each pattern
        for pattern in patterns_to_try:
            if pattern in self.available_samples:
                return pattern
        
        return None
    
    def _create_singers(self):
        """Create singer flies for each phoneme."""
        for phoneme in self.phoneme_list:
            arpasing = arpabet_to_arpasing(phoneme)
            
            # Determine if vowel
            is_vowel = False
            if phoneme in PHONEME_INVENTORY:
                is_vowel = PHONEME_INVENTORY[phoneme].phoneme_type == 'vowel'
            
            sample_path = None
            oto_entry = None
            resolved_alias = None
            
            # Try to resolve alias with alternate patterns
            resolved_alias = self._resolve_alias(arpasing, is_vowel)
            
            if resolved_alias and resolved_alias in self.available_samples:
                sample_path, oto_entry = self.available_samples[resolved_alias]
            else:
                # Try ARPAbet fallback chain
                fallback = get_arpasing_fallback(phoneme, self.available_aliases)
                if fallback:
                    # Check if fallback is vowel for pattern resolution
                    fallback_arpabet = ARPASING_TO_ARPABET.get(fallback, phoneme)
                    fb_is_vowel = False
                    if fallback_arpabet in PHONEME_INVENTORY:
                        fb_is_vowel = PHONEME_INVENTORY[fallback_arpabet].phoneme_type == 'vowel'
                    
                    resolved_fallback = self._resolve_alias(fallback, fb_is_vowel)
                    if resolved_fallback and resolved_fallback in self.available_samples:
                        sample_path, oto_entry = self.available_samples[resolved_fallback]
            
            self.singers[phoneme] = UTAUSingerFly(
                phoneme=phoneme,
                sample_path=sample_path,
                oto_entry=oto_entry,
                fallback_synth=self.fallback_synth,
            )
    
    def get_phoneme_audio(self) -> Dict[str, np.ndarray]:
        """Get audio crumbs for all phonemes."""
        return {p: singer.get_audio() for p, singer in self.singers.items()}
    
    def synthesize_phoneme(self, phoneme: str) -> np.ndarray:
        """Synthesize a single phoneme."""
        p = strip_stress(phoneme)
        if p in self.singers:
            return self.singers[p].get_audio()
        else:
            # Create fallback singer
            return UTAUSingerFly(
                phoneme=p,
                fallback_synth=self.fallback_synth,
            ).synthesize()
    
    def synthesize_sequence(
        self,
        phoneme_sequence: List[str],
        crossfade_ms: float = 15,
    ) -> np.ndarray:
        """
        Synthesize a phoneme sequence using UTAU samples.
        
        Args:
            phoneme_sequence: List of ARPAbet phonemes
            crossfade_ms: Crossfade duration between phonemes
        
        Returns:
            Concatenated audio signal
        """
        if not phoneme_sequence:
            return np.array([], dtype=np.float32)
        
        crossfade_samples = int(crossfade_ms / 1000 * SAMPLE_RATE)
        
        # Get first phoneme
        first_p = strip_stress(phoneme_sequence[0])
        result = self.synthesize_phoneme(first_p).copy()
        
        for phoneme in phoneme_sequence[1:]:
            p = strip_stress(phoneme)
            audio = self.synthesize_phoneme(p)
            
            if (crossfade_samples > 0 and 
                len(result) >= crossfade_samples and 
                len(audio) >= crossfade_samples):
                
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                result[-crossfade_samples:] *= fade_out
                audio_copy = audio.copy()
                audio_copy[:crossfade_samples] *= fade_in
                
                result[-crossfade_samples:] += audio_copy[:crossfade_samples]
                result = np.concatenate([result, audio_copy[crossfade_samples:]])
            else:
                result = np.concatenate([result, audio])
        
        return result
    
    def get_coverage_summary(self) -> str:
        """Get a human-readable coverage summary."""
        lines = [
            f"UTAU Singer Swarm Coverage",
            f"=" * 40,
            f"Voicebank: {self.voicebank_path or 'Not loaded'}",
            f"Available aliases: {len(self.available_aliases)}",
            f"Phonemes covered: {self.coverage['directly_covered']}/{self.coverage['total_phonemes']} "
            f"({self.coverage['direct_coverage_pct']:.1f}%)",
            f"With fallbacks: {self.coverage['covered_with_fallback']}/{self.coverage['total_phonemes']} "
            f"({self.coverage['total_coverage_pct']:.1f}%)",
        ]
        
        # List which phonemes have real samples
        has_sample = [p for p, s in self.singers.items() if s.has_sample]
        no_sample = [p for p, s in self.singers.items() if not s.has_sample]
        
        if has_sample:
            lines.append(f"\nWith UTAU samples ({len(has_sample)}):")
            lines.append(f"  {' '.join(sorted(has_sample))}")
        
        if no_sample:
            lines.append(f"\nFallback to formant ({len(no_sample)}):")
            lines.append(f"  {' '.join(sorted(no_sample))}")
        
        return '\n'.join(lines)


class MarianSingerSwarm(UTAUSingerSwarm):
    """
    Pre-configured UTAU singer swarm for MARIAN ILUSTRADO voicebank.
    
    MARIAN ILUSTRADO is an English ARPAsing voicebank by Kanabun.
    https://downloadmarian.carrd.co/
    
    Terms of use: Free to use with attribution.
    See README/NOTICE for full license terms.
    """
    
    VOICEBANK_NAME = "MARIAN ILUSTRADO"
    AUTHOR = "Kanabun"
    DOWNLOAD_URL = "https://downloadmarian.carrd.co/"
    
    def __init__(
        self,
        voicebank_path: Optional[Path] = None,
        phonemes: Optional[List[str]] = None,
        fallback_synth: bool = True,
    ):
        """
        Initialize MARIAN singer swarm.
        
        If voicebank_path is None, searches standard locations.
        """
        if voicebank_path is None:
            voicebank_path = self._find_marian_path()
        
        super().__init__(
            voicebank_path=voicebank_path,
            phonemes=phonemes,
            fallback_synth=fallback_synth,
        )
        
        self.voicebank_name = self.VOICEBANK_NAME
        self.author = self.AUTHOR
    
    @classmethod
    def _find_marian_path(cls) -> Optional[Path]:
        """Search standard locations for MARIAN voicebank."""
        for search_path in VOICEBANK_SEARCH_PATHS:
            if search_path.exists():
                # Check for characteristic files
                if list(search_path.glob('*.wav')) or list(search_path.glob('**/oto.ini')):
                    return search_path
        return None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if MARIAN voicebank is installed."""
        return cls._find_marian_path() is not None
    
    def get_attribution(self) -> str:
        """Get attribution text for MARIAN voicebank."""
        return f"""
Voice samples: {self.VOICEBANK_NAME} by {self.AUTHOR}
Download: {self.DOWNLOAD_URL}
License: Free to use with attribution (see NOTICE file)
"""


def create_marian_swarm(
    voicebank_path: Optional[Path] = None,
    phonemes: Optional[List[str]] = None,
) -> MarianSingerSwarm:
    """
    Create a MARIAN singer swarm.
    
    Convenience function for creating a MarianSingerSwarm.
    Falls back to formant synthesis if voicebank not found.
    """
    return MarianSingerSwarm(
        voicebank_path=voicebank_path,
        phonemes=phonemes,
        fallback_synth=True,
    )


# ============================================================================
# Module Self-Test
# ============================================================================

if __name__ == "__main__":
    print("UTAU Singer Module Test")
    print("=" * 60)
    
    # Check if MARIAN is available
    print(f"\nMARIAN available: {MarianSingerSwarm.is_available()}")
    
    # Create swarm
    swarm = MarianSingerSwarm()
    print(f"\n{swarm.get_coverage_summary()}")
    
    # Test synthesis
    test_sequence = ['K', 'AE', 'T']
    print(f"\nSynthesizing: {' '.join(test_sequence)}")
    
    audio = swarm.synthesize_sequence(test_sequence)
    print(f"Audio: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.3f}s")
    
    # Save test audio
    test_path = Path("/tmp/marian_test_cat.wav")
    save_wav(audio, str(test_path), SAMPLE_RATE)
    print(f"\nSaved test audio to: {test_path}")
