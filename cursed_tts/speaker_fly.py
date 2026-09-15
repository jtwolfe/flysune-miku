"""
Speaker Flies: Per-phoneme audio renderers WITHOUT KC/MB dependency.

The Two-Swarm Architecture:
1. PICKER SWARM: Grapheme-to-phoneme classification (letter context → phoneme)
   - Uses KC/MB for sparse expansion and classification
   - Outputs: phoneme sequence

2. SPEAKER SWARM: Phoneme-to-audio rendering (phoneme ID → audio crumb)
   - Does NOT use KC/MB
   - Each fly specializes in producing audio for ONE phoneme
   - Input: phoneme ID + optional context (prev-phone, position)
   - Output: audio waveform

This clean separation ensures:
- Recognition flies pick (G2P classification)
- Speaking flies speak (audio synthesis)
- No KC/voice coupling that caused PR #6 issues

Speaker modes:
- 'formant': Baseline formant synthesis (existing clean synth params)
- 'trained': Learned per-phoneme parametric voice (F0/formants/noise/duration)
- 'sample': Plays carefully sliced Marian crumbs (optional, when available)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .phonemes import (
    PHONEME_LIST, PHONEME_INVENTORY, VOWELS, CONSONANTS, SEMIVOWELS,
    STOPS, AFFRICATES, FRICATIVES, NASALS, LIQUIDS, VOICED_CONSONANTS,
    DIPHTHONGS, strip_stress, phoneme_to_index, NUM_PHONEMES
)
from .synth import (
    SAMPLE_RATE, VOWEL_FORMANTS, DIPHTHONG_END, CONSONANT_PARAMS,
    SEMIVOWEL_PARAMS, _make_envelope, save_wav, synthesize_phoneme
)


SPEAKER_DURATION_MIN_MS = 60
SPEAKER_DURATION_MAX_MS = 250


@dataclass
class SpeakerParams:
    """
    Trainable parameters for a speaker fly.
    
    These parameters control how a phoneme sounds when spoken.
    Crucially: NO KC activity, NO picker features - just phoneme identity.
    """
    f0: float = 140.0               # Fundamental frequency (Hz)
    f1_shift: float = 1.0           # F1 formant multiplier
    f2_shift: float = 1.0           # F2 formant multiplier
    f3_shift: float = 1.0           # F3 formant multiplier (for liquids)
    noise_level: float = 0.0        # Noise/breathiness (0-1)
    duration_ms: float = 100.0      # Duration in milliseconds
    attack_ms: float = 10.0         # Attack time
    release_ms: float = 15.0        # Release time
    harmonic_rolloff: float = 0.7   # Harmonic amplitude decay
    
    # Optional: context-dependent adjustments (prev-phone one-hot weights)
    prev_phone_f0_delta: Optional[np.ndarray] = None  # (NUM_PHONEMES,) F0 adjustment
    prev_phone_dur_delta: Optional[np.ndarray] = None  # (NUM_PHONEMES,) duration adjustment
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'f0': self.f0,
            'f1_shift': self.f1_shift,
            'f2_shift': self.f2_shift,
            'f3_shift': self.f3_shift,
            'noise_level': self.noise_level,
            'duration_ms': self.duration_ms,
            'attack_ms': self.attack_ms,
            'release_ms': self.release_ms,
            'harmonic_rolloff': self.harmonic_rolloff,
        }
        if self.prev_phone_f0_delta is not None:
            result['prev_phone_f0_delta'] = self.prev_phone_f0_delta.tolist()
        if self.prev_phone_dur_delta is not None:
            result['prev_phone_dur_delta'] = self.prev_phone_dur_delta.tolist()
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpeakerParams':
        params = cls(
            f0=d.get('f0', 140.0),
            f1_shift=d.get('f1_shift', 1.0),
            f2_shift=d.get('f2_shift', 1.0),
            f3_shift=d.get('f3_shift', 1.0),
            noise_level=d.get('noise_level', 0.0),
            duration_ms=d.get('duration_ms', 100.0),
            attack_ms=d.get('attack_ms', 10.0),
            release_ms=d.get('release_ms', 15.0),
            harmonic_rolloff=d.get('harmonic_rolloff', 0.7),
        )
        if 'prev_phone_f0_delta' in d:
            params.prev_phone_f0_delta = np.array(d['prev_phone_f0_delta'], dtype=np.float32)
        if 'prev_phone_dur_delta' in d:
            params.prev_phone_dur_delta = np.array(d['prev_phone_dur_delta'], dtype=np.float32)
        return params
    
    def clone(self) -> 'SpeakerParams':
        """Create a copy of these parameters."""
        p = SpeakerParams(
            f0=self.f0,
            f1_shift=self.f1_shift,
            f2_shift=self.f2_shift,
            f3_shift=self.f3_shift,
            noise_level=self.noise_level,
            duration_ms=self.duration_ms,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
            harmonic_rolloff=self.harmonic_rolloff,
        )
        if self.prev_phone_f0_delta is not None:
            p.prev_phone_f0_delta = self.prev_phone_f0_delta.copy()
        if self.prev_phone_dur_delta is not None:
            p.prev_phone_dur_delta = self.prev_phone_dur_delta.copy()
        return p


def _get_default_params_for_phoneme(phoneme: str, seed: int = 42) -> SpeakerParams:
    """
    Get reasonable default synthesis parameters for a phoneme.
    
    These are derived from phoneme acoustic properties, not from any
    recognition/classification state.
    """
    p = strip_stress(phoneme)
    rng = np.random.default_rng(seed + hash(p) % 10000)
    info = PHONEME_INVENTORY.get(p)
    
    if info is None:
        return SpeakerParams()
    
    params = SpeakerParams()
    
    if info.phoneme_type == 'vowel':
        # Vowels: longer, voiced, distinct F0 per vowel
        vowel_idx = list(VOWELS).index(p) if p in VOWELS else 0
        n_vowels = len(VOWELS)
        params.f0 = 120 + (vowel_idx / max(n_vowels - 1, 1)) * 60  # 120-180 Hz
        params.duration_ms = 150.0 if p in DIPHTHONGS else 120.0
        params.noise_level = rng.uniform(0.0, 0.1)
        params.harmonic_rolloff = rng.uniform(0.6, 0.8)
        
    elif info.phoneme_type == 'consonant':
        if p in STOPS:
            params.f0 = 100 + rng.uniform(0, 20)
            params.duration_ms = 80.0
            params.noise_level = 0.4 if p not in VOICED_CONSONANTS else 0.2
        elif p in FRICATIVES:
            params.f0 = 130 + rng.uniform(0, 30)
            params.duration_ms = 90.0
            params.noise_level = 0.5 + rng.uniform(0, 0.3)
        elif p in NASALS:
            params.f0 = 125 + rng.uniform(0, 15)
            params.duration_ms = 100.0
            params.noise_level = 0.05
        elif p in LIQUIDS:
            params.f0 = 130 + rng.uniform(0, 20)
            params.duration_ms = 110.0
            params.noise_level = 0.1
        elif p in AFFRICATES:
            params.f0 = 115 + rng.uniform(0, 25)
            params.duration_ms = 120.0
            params.noise_level = 0.35
        else:
            params.duration_ms = 80.0
            params.noise_level = 0.3
            
    else:  # semivowels
        params.f0 = 140 + rng.uniform(0, 20)
        params.duration_ms = 100.0
        params.noise_level = 0.05 + rng.uniform(0, 0.1)
    
    return params


class SpeakerFly:
    """
    A speaker fly that produces audio for ONE specific phoneme.
    
    CRITICAL: This fly does NOT receive any KC/MB activity.
    Input: phoneme ID (implicit - one fly per phoneme) + optional context
    Output: audio waveform
    
    This is in contrast to the picker fly which DOES use KC for classification.
    """
    
    def __init__(
        self,
        phoneme: str,
        params: Optional[SpeakerParams] = None,
        mode: str = 'formant',
        sample_path: Optional[Path] = None,
        seed: int = 42,
    ):
        self.phoneme = strip_stress(phoneme)
        self.mode = mode
        self.sample_path = sample_path
        self.seed = seed
        self.rng = np.random.default_rng(seed + hash(self.phoneme) % 10000)
        
        # Initialize parameters
        if params is not None:
            self.params = params
        else:
            self.params = _get_default_params_for_phoneme(self.phoneme, seed)
        
        # Phoneme info
        self.info = PHONEME_INVENTORY.get(self.phoneme)
        
        # Sample cache for sample mode
        self._sample_cache: Optional[np.ndarray] = None
        if mode == 'sample' and sample_path and Path(sample_path).exists():
            self._load_sample()
    
    def _load_sample(self):
        """Load sample from file (for sample mode)."""
        try:
            import wave
            with wave.open(str(self.sample_path), 'r') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample if needed
                src_rate = wf.getframerate()
                if src_rate != SAMPLE_RATE:
                    ratio = SAMPLE_RATE / src_rate
                    new_len = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, len(audio), new_len),
                        np.arange(len(audio)),
                        audio
                    )
                
                # Cap duration
                max_samples = int(SPEAKER_DURATION_MAX_MS / 1000 * SAMPLE_RATE)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                    audio *= _make_envelope(len(audio), attack=0.01, release=0.02)
                
                self._sample_cache = audio.astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load sample for {self.phoneme}: {e}")
            self._sample_cache = None
    
    def synthesize(
        self,
        prev_phone: Optional[str] = None,
        position: float = 0.5,  # 0=start, 1=end of utterance
    ) -> np.ndarray:
        """
        Synthesize audio for this phoneme.
        
        Args:
            prev_phone: Previous phoneme (optional context, NOT KC activity!)
            position: Position in utterance (0-1, optional context)
        
        Returns:
            Audio waveform as float32 numpy array
        """
        if self.mode == 'sample' and self._sample_cache is not None:
            return self._sample_cache.copy()
        
        # Get effective parameters (possibly adjusted by context)
        params = self._get_effective_params(prev_phone, position)
        
        # Synthesize based on phoneme type
        if self.phoneme in VOWELS:
            return self._synth_vowel(params)
        elif self.phoneme in SEMIVOWELS:
            return self._synth_semivowel(params)
        elif self.phoneme in CONSONANTS:
            return self._synth_consonant(params)
        else:
            # Fallback
            return self._synth_fallback(params)
    
    def _get_effective_params(
        self,
        prev_phone: Optional[str],
        position: float,
    ) -> SpeakerParams:
        """Get parameters adjusted for context."""
        params = self.params.clone()
        
        # Apply previous-phone adjustments if available
        if prev_phone is not None and params.prev_phone_f0_delta is not None:
            try:
                prev_idx = phoneme_to_index(strip_stress(prev_phone))
                params.f0 += params.prev_phone_f0_delta[prev_idx]
            except (ValueError, KeyError):
                pass
        
        if prev_phone is not None and params.prev_phone_dur_delta is not None:
            try:
                prev_idx = phoneme_to_index(strip_stress(prev_phone))
                params.duration_ms += params.prev_phone_dur_delta[prev_idx]
            except (ValueError, KeyError):
                pass
        
        # Slight position-based pitch contour (natural prosody)
        # Higher pitch at start, lower at end
        params.f0 *= (1.0 + 0.05 * (1.0 - position))
        
        # Clamp duration
        params.duration_ms = max(SPEAKER_DURATION_MIN_MS, 
                                  min(SPEAKER_DURATION_MAX_MS, params.duration_ms))
        
        return params
    
    def _synth_vowel(self, params: SpeakerParams) -> np.ndarray:
        """Synthesize a vowel with given parameters."""
        duration = params.duration_ms / 1000.0
        n_samples = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n_samples)
        
        # Get base formants and apply shifts
        f1_base, f2_base = VOWEL_FORMANTS.get(self.phoneme, (500, 1500))
        f1 = f1_base * params.f1_shift
        f2 = f2_base * params.f2_shift
        
        # Handle diphthongs
        if self.phoneme in DIPHTHONG_END:
            f1_end, f2_end = DIPHTHONG_END[self.phoneme]
            f1_end *= params.f1_shift
            f2_end *= params.f2_shift
            f1 = np.linspace(f1, f1_end, n_samples)
            f2 = np.linspace(f2, f2_end, n_samples)
        else:
            f1 = np.full(n_samples, f1)
            f2 = np.full(n_samples, f2)
        
        # Generate harmonics with formant shaping
        signal = np.zeros(n_samples, dtype=np.float32)
        f0 = params.f0
        
        for harmonic in range(1, 12):
            freq = f0 * harmonic
            amp1 = np.exp(-((freq - f1) / 150) ** 2)
            amp2 = np.exp(-((freq - f2) / 200) ** 2)
            amp = (amp1 + amp2 * 0.6) / (harmonic ** params.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        # Apply envelope
        attack = params.attack_ms / 1000.0
        release = params.release_ms / 1000.0
        signal *= _make_envelope(n_samples, attack=attack, release=release)
        
        # Add breathiness/noise
        if params.noise_level > 0:
            noise = self.rng.standard_normal(n_samples).astype(np.float32)
            noise *= params.noise_level * 0.2
            signal += noise
        
        # Normalize
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max() * 0.75
        
        return signal.astype(np.float32)
    
    def _synth_consonant(self, params: SpeakerParams) -> np.ndarray:
        """Synthesize a consonant with given parameters."""
        duration = params.duration_ms / 1000.0
        n_samples = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n_samples)
        
        cons_params = CONSONANT_PARAMS.get(self.phoneme, {
            'type': 'fricative', 'freq': 1000, 'noise': 0.5, 'voiced': False
        })
        ctype = cons_params.get('type', 'fricative')
        
        # Apply formant shift to characteristic frequency
        base_freq = cons_params.get('freq', 1000)
        freq = base_freq * params.f1_shift
        
        if ctype == 'stop':
            signal = self._synth_stop_core(n_samples, t, freq, cons_params, params)
        elif ctype == 'affricate':
            signal = self._synth_affricate_core(n_samples, t, freq, cons_params, params)
        elif ctype == 'fricative':
            signal = self._synth_fricative_core(n_samples, t, freq, cons_params, params)
        elif ctype == 'nasal':
            signal = self._synth_nasal_core(n_samples, t, freq, cons_params, params)
        elif ctype == 'liquid':
            signal = self._synth_liquid_core(n_samples, t, freq, cons_params, params)
        else:
            signal = self.rng.standard_normal(n_samples).astype(np.float32) * 0.3
        
        # Normalize
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max() * 0.65
        
        return signal.astype(np.float32)
    
    def _synth_stop_core(self, n_samples, t, freq, cons_params, params):
        """Core stop synthesis."""
        signal = np.zeros(n_samples, dtype=np.float32)
        
        burst_start = int(n_samples * 0.35)
        burst_len = int(n_samples * cons_params.get('burst', 0.4))
        if burst_start + burst_len > n_samples:
            burst_len = n_samples - burst_start
        
        if burst_len > 0:
            noise = self.rng.standard_normal(burst_len).astype(np.float32) * cons_params['noise']
            t_burst = np.linspace(0, burst_len / SAMPLE_RATE, burst_len)
            carrier = np.sin(2 * np.pi * freq * t_burst)
            filtered = noise * 0.6 + carrier * 0.4 * np.abs(noise)
            
            burst_env = np.exp(-np.linspace(0, 6, burst_len))
            filtered *= burst_env
            
            signal[burst_start:burst_start + burst_len] = filtered
        
        if cons_params.get('voiced', False):
            voice = np.sin(2 * np.pi * params.f0 * t) * 0.25
            voice *= _make_envelope(n_samples, attack=0.15, release=0.15)
            signal += voice
        
        return signal
    
    def _synth_affricate_core(self, n_samples, t, freq, cons_params, params):
        """Core affricate synthesis."""
        signal = np.zeros(n_samples, dtype=np.float32)
        
        stop_len = int(n_samples * 0.4)
        stop_params_local = {'freq': freq, 'noise': cons_params['noise'] * 0.5,
                             'voiced': cons_params['voiced'], 'burst': 0.6}
        t_stop = np.linspace(0, stop_len / SAMPLE_RATE, stop_len)
        signal[:stop_len] = self._synth_stop_core(stop_len, t_stop, freq, stop_params_local, params)
        
        fric_len = n_samples - stop_len
        if fric_len > 0:
            t_fric = np.linspace(0, fric_len / SAMPLE_RATE, fric_len)
            
            noise = self.rng.standard_normal(fric_len).astype(np.float32) * cons_params['noise']
            carrier = np.sin(2 * np.pi * freq * t_fric)
            fric_signal = noise * 0.5 + carrier * 0.3 * noise
            
            fric_env = _make_envelope(fric_len, attack=0.05, release=0.1)
            signal[stop_len:] = fric_signal * fric_env
        
        if cons_params.get('voiced', False):
            voice = np.sin(2 * np.pi * params.f0 * t) * 0.2
            signal += voice * _make_envelope(n_samples, attack=0.1, release=0.1)
        
        return signal
    
    def _synth_fricative_core(self, n_samples, t, freq, cons_params, params):
        """Core fricative synthesis."""
        noise = self.rng.standard_normal(n_samples).astype(np.float32)
        noise *= (cons_params.get('noise', 0.5) + params.noise_level) * 0.5
        
        if cons_params.get('aspirate', False):
            signal = noise * 0.5
        else:
            carrier = np.sin(2 * np.pi * freq * t)
            signal = noise * 0.4 + carrier * 0.2 * np.abs(noise)
        
        attack = params.attack_ms / 1000.0
        release = params.release_ms / 1000.0
        signal *= _make_envelope(n_samples, attack=max(0.05, attack), release=max(0.08, release))
        
        if cons_params.get('voiced', False):
            voice = np.sin(2 * np.pi * params.f0 * t) * 0.3
            signal += voice * _make_envelope(n_samples, attack=0.1, release=0.1)
        
        return signal
    
    def _synth_nasal_core(self, n_samples, t, freq, cons_params, params):
        """Core nasal synthesis."""
        f1 = freq * params.f1_shift
        f2 = cons_params.get('f2', 1000) * params.f2_shift
        
        signal = np.zeros(n_samples, dtype=np.float32)
        
        for harmonic in range(1, 8):
            h_freq = params.f0 * harmonic
            amp1 = np.exp(-((h_freq - f1) / 100) ** 2)
            amp2 = np.exp(-((h_freq - f2) / 150) ** 2) * 0.5
            amp = (amp1 + amp2) / (harmonic ** params.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * h_freq * t)
        
        signal += np.sin(2 * np.pi * f1 * t) * 0.3
        
        attack = params.attack_ms / 1000.0
        release = params.release_ms / 1000.0
        signal *= _make_envelope(n_samples, attack=max(0.08, attack), release=max(0.08, release))
        
        return signal
    
    def _synth_liquid_core(self, n_samples, t, freq, cons_params, params):
        """Core liquid synthesis."""
        f1 = freq * params.f1_shift
        f2 = cons_params.get('f2', 1200) * params.f2_shift
        f3 = cons_params.get('f3', 2800) * params.f3_shift
        
        signal = np.zeros(n_samples, dtype=np.float32)
        
        for harmonic in range(1, 10):
            h_freq = params.f0 * harmonic
            amp1 = np.exp(-((h_freq - f1) / 100) ** 2)
            amp2 = np.exp(-((h_freq - f2) / 150) ** 2) * 0.6
            amp3 = np.exp(-((h_freq - f3) / 200) ** 2) * 0.4
            amp = (amp1 + amp2 + amp3) / (harmonic ** params.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * h_freq * t)
        
        attack = params.attack_ms / 1000.0
        release = params.release_ms / 1000.0
        signal *= _make_envelope(n_samples, attack=max(0.08, attack), release=max(0.1, release))
        
        return signal
    
    def _synth_semivowel(self, params: SpeakerParams) -> np.ndarray:
        """Synthesize a semivowel with given parameters."""
        duration = params.duration_ms / 1000.0
        n_samples = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n_samples)
        
        sv_params = SEMIVOWEL_PARAMS.get(self.phoneme, {
            'start': (350, 700), 'end': (400, 1200), 'voiced': True
        })
        
        f1_start, f2_start = sv_params['start']
        f1_end, f2_end = sv_params['end']
        
        # Apply formant shifts
        f1 = np.linspace(f1_start * params.f1_shift, f1_end * params.f1_shift, n_samples)
        f2 = np.linspace(f2_start * params.f2_shift, f2_end * params.f2_shift, n_samples)
        
        signal = np.zeros(n_samples, dtype=np.float32)
        
        for harmonic in range(1, 10):
            freq = params.f0 * harmonic
            amp1 = np.exp(-((freq - f1) / 120) ** 2)
            amp2 = np.exp(-((freq - f2) / 180) ** 2) * 0.7
            amp = (amp1 + amp2) / (harmonic ** params.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        attack = params.attack_ms / 1000.0
        release = params.release_ms / 1000.0
        signal *= _make_envelope(n_samples, attack=max(0.06, attack), release=max(0.08, release))
        
        # Add noise
        if params.noise_level > 0:
            noise = self.rng.standard_normal(n_samples).astype(np.float32)
            noise *= params.noise_level * 0.15
            signal += noise
        
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max() * 0.7
        
        return signal.astype(np.float32)
    
    def _synth_fallback(self, params: SpeakerParams) -> np.ndarray:
        """Fallback synthesis for unknown phonemes."""
        duration = params.duration_ms / 1000.0
        n_samples = int(duration * SAMPLE_RATE)
        
        # Simple tone + noise
        t = np.linspace(0, duration, n_samples)
        signal = np.sin(2 * np.pi * params.f0 * t) * 0.5
        signal += self.rng.standard_normal(n_samples).astype(np.float32) * 0.1
        signal *= _make_envelope(n_samples, attack=0.05, release=0.05)
        
        return signal.astype(np.float32)


@dataclass
class SpeakerSwarmConfig:
    """Configuration for the speaker swarm."""
    mode: str = 'formant'           # 'formant', 'trained', or 'sample'
    seed: int = 42
    crossfade_ms: float = 15.0      # Crossfade between phonemes
    phonemes: List[str] = field(default_factory=lambda: PHONEME_LIST.copy())
    sample_dir: Optional[str] = None  # Directory for sample mode
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'seed': self.seed,
            'crossfade_ms': self.crossfade_ms,
            'phonemes': self.phonemes,
            'sample_dir': self.sample_dir,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpeakerSwarmConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SpeakerSwarm:
    """
    Swarm of speaker flies - one per phoneme.
    
    ARCHITECTURE:
    This is the SPEAKING half of the two-swarm system:
    - Input: Phoneme sequence (from picker swarm)
    - Output: Concatenated audio crumbs
    
    CRITICAL: No KC/MB activity involved. Speakers receive only:
    - Phoneme ID (implicit - one fly per phoneme)
    - Optional context (prev-phone, position)
    
    This ensures clean separation from picker swarm.
    """
    
    def __init__(self, config: Optional[SpeakerSwarmConfig] = None):
        self.config = config or SpeakerSwarmConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Create speaker flies
        self.speakers: Dict[str, SpeakerFly] = {}
        for phoneme in self.config.phonemes:
            p = strip_stress(phoneme)
            if p not in self.speakers:
                # Determine sample path if in sample mode
                sample_path = None
                if self.config.mode == 'sample' and self.config.sample_dir:
                    sample_dir = Path(self.config.sample_dir)
                    # Try various naming conventions
                    for suffix in ['.wav', '_crumb.wav', f'_{p}.wav']:
                        candidate = sample_dir / f"{p}{suffix}"
                        if candidate.exists():
                            sample_path = candidate
                            break
                
                self.speakers[p] = SpeakerFly(
                    phoneme=p,
                    mode=self.config.mode,
                    sample_path=sample_path,
                    seed=self.config.seed,
                )
        
        self.phoneme_list = list(self.speakers.keys())
        print(f"SpeakerSwarm: {len(self.speakers)} speaker flies (mode={self.config.mode})")
        print("  Architecture: phoneme ID → speaker fly → audio (NO KC dependency)")
    
    def synthesize_phoneme(
        self,
        phoneme: str,
        prev_phone: Optional[str] = None,
        position: float = 0.5,
    ) -> np.ndarray:
        """
        Synthesize audio for a single phoneme.
        
        Args:
            phoneme: Phoneme to synthesize
            prev_phone: Previous phoneme (optional context)
            position: Position in utterance (0-1)
        
        Returns:
            Audio waveform
        """
        p = strip_stress(phoneme)
        
        if p in self.speakers:
            return self.speakers[p].synthesize(prev_phone=prev_phone, position=position)
        else:
            # Create temporary speaker for unknown phoneme
            temp_speaker = SpeakerFly(
                phoneme=p,
                mode='formant',
                seed=self.config.seed,
            )
            return temp_speaker.synthesize(prev_phone=prev_phone, position=position)
    
    def synthesize_sequence(
        self,
        phoneme_sequence: List[str],
        crossfade_ms: Optional[float] = None,
    ) -> np.ndarray:
        """
        Synthesize audio for a phoneme sequence.
        
        Args:
            phoneme_sequence: List of phonemes
            crossfade_ms: Crossfade duration (None = use config)
        
        Returns:
            Concatenated audio waveform
        """
        if not phoneme_sequence:
            return np.array([], dtype=np.float32)
        
        if crossfade_ms is None:
            crossfade_ms = self.config.crossfade_ms
        
        crossfade_samples = int(crossfade_ms / 1000 * SAMPLE_RATE)
        n_phones = len(phoneme_sequence)
        
        # Synthesize first phoneme
        result = self.synthesize_phoneme(
            phoneme_sequence[0],
            prev_phone=None,
            position=0.0 if n_phones > 1 else 0.5,
        ).copy()
        
        for i, phoneme in enumerate(phoneme_sequence[1:], 1):
            position = i / (n_phones - 1) if n_phones > 1 else 0.5
            prev_phone = phoneme_sequence[i - 1]
            
            audio = self.synthesize_phoneme(
                phoneme,
                prev_phone=prev_phone,
                position=position,
            )
            
            # Crossfade
            if crossfade_samples > 0 and len(result) >= crossfade_samples and len(audio) >= crossfade_samples:
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
    
    def set_speaker_params(self, phoneme: str, params: SpeakerParams):
        """Set parameters for a specific speaker fly."""
        p = strip_stress(phoneme)
        if p in self.speakers:
            self.speakers[p].params = params
    
    def get_speaker_params(self, phoneme: str) -> Optional[SpeakerParams]:
        """Get parameters for a specific speaker fly."""
        p = strip_stress(phoneme)
        if p in self.speakers:
            return self.speakers[p].params
        return None
    
    def get_all_params(self) -> Dict[str, SpeakerParams]:
        """Get parameters for all speaker flies."""
        return {p: s.params for p, s in self.speakers.items()}
    
    def save(self, path: str):
        """Save speaker swarm to file."""
        path = Path(path)
        
        params_dict = {p: s.params.to_dict() for p, s in self.speakers.items()}
        
        np.savez(
            path,
            config=json.dumps(self.config.to_dict()),
            params=json.dumps(params_dict),
        )
        print(f"SpeakerSwarm saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SpeakerSwarm':
        """Load speaker swarm from file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = SpeakerSwarmConfig.from_dict(json.loads(str(data['config'])))
        swarm = cls(config)
        
        # Load trained parameters
        params_dict = json.loads(str(data['params']))
        for phoneme, p_dict in params_dict.items():
            params = SpeakerParams.from_dict(p_dict)
            swarm.set_speaker_params(phoneme, params)
        
        print(f"SpeakerSwarm loaded from {path} ({len(swarm.speakers)} speakers)")
        return swarm


def verify_no_kc_dependency() -> bool:
    """
    Verify that SpeakerSwarm has no KC/MB dependency.
    
    This is a critical architectural invariant:
    - Speakers receive: phoneme ID, prev_phone, position
    - Speakers do NOT receive: KC activity, MB features, picker state
    
    Returns True if architecture is correct.
    """
    # Check that SpeakerFly.synthesize doesn't accept KC-related args
    import inspect
    sig = inspect.signature(SpeakerFly.synthesize)
    param_names = list(sig.parameters.keys())
    
    kc_related = ['kc', 'kc_activity', 'mb', 'mushroom', 'picker', 'mbon']
    for kc_name in kc_related:
        for param in param_names:
            if kc_name in param.lower():
                return False  # Found KC dependency!
    
    # Check SpeakerSwarm.synthesize_sequence
    sig = inspect.signature(SpeakerSwarm.synthesize_sequence)
    param_names = list(sig.parameters.keys())
    for kc_name in kc_related:
        for param in param_names:
            if kc_name in param.lower():
                return False
    
    return True


if __name__ == "__main__":
    print("Testing SpeakerSwarm (two-swarm architecture)...")
    print("=" * 60)
    
    # Verify no KC dependency
    print("\n1. Verifying no KC/MB dependency...")
    if verify_no_kc_dependency():
        print("   ✓ SpeakerSwarm has no KC/MB dependency")
    else:
        print("   ✗ ERROR: KC dependency detected!")
    
    # Create swarm
    print("\n2. Creating SpeakerSwarm...")
    config = SpeakerSwarmConfig(mode='formant', seed=42)
    swarm = SpeakerSwarm(config)
    
    # Test single phoneme synthesis
    print("\n3. Testing single phoneme synthesis...")
    for phoneme in ['K', 'AE', 'T', 'M', 'IY']:
        audio = swarm.synthesize_phoneme(phoneme)
        duration_ms = len(audio) / SAMPLE_RATE * 1000
        print(f"   {phoneme}: {len(audio)} samples, {duration_ms:.1f}ms")
    
    # Test sequence synthesis
    print("\n4. Testing sequence synthesis (cat = K AE T)...")
    sequence = ['K', 'AE', 'T']
    audio = swarm.synthesize_sequence(sequence)
    duration_ms = len(audio) / SAMPLE_RATE * 1000
    print(f"   Result: {len(audio)} samples, {duration_ms:.1f}ms")
    
    # Save test
    save_wav(audio, "/tmp/speaker_test_cat.wav", SAMPLE_RATE)
    print(f"   Saved to /tmp/speaker_test_cat.wav")
    
    # Test save/load
    print("\n5. Testing save/load...")
    swarm.save("/tmp/test_speaker_swarm.npz")
    loaded = SpeakerSwarm.load("/tmp/test_speaker_swarm.npz")
    print(f"   Loaded {len(loaded.speakers)} speakers")
    
    print("\n" + "=" * 60)
    print("SpeakerSwarm tests complete!")
    print("\nArchitecture summary:")
    print("  - Recognition flies pick (G2P via KC/MB)")
    print("  - Speaking flies speak (audio synthesis, NO KC)")
