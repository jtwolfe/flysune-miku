"""
Singer flies: Per-phoneme audio crumb renderers.

Each phoneme owns a personal "singer fly" with its own voice characteristics:
- Distinct F0 (fundamental frequency / pitch)
- Formant shifts (voice color / timbre)
- Vibrato/tremolo characteristics
- Breathiness / noise mix
- Duration scaling

When speaking with swarm mode, the picker flies choose the phoneme sequence,
then each phoneme is rendered by that phoneme's singer fly. The result is a
"mixture of flies" (MoE) where different parts of the utterance are sung by
different specialized flies.

Biology analogy: In real insects, different neurons/compartments can modulate
motor output differently. Here, each phoneme "compartment" has its own voice.

## UTAU/OpenUTAU Voicebank Hook (Future)

This architecture is designed to later support loading real phoneme audio
samples from open voicebanks instead of synthesized formants. To integrate:

1. Download an open UTAU voicebank (e.g., from utau.wiki or similar)
2. Extract .wav files for each phoneme (typically organized by CV/VC)
3. Create a SingerFly subclass that loads samples instead of synthesizing:

   class UTAUSingerFly(SingerFly):
       def __init__(self, phoneme: str, sample_path: Path, ...):
           self.sample = load_and_resample(sample_path)
       
       def synthesize(self) -> np.ndarray:
           return self.sample.copy()

4. The MoE concatenation logic remains the same — just swap the audio source.

Note: Do NOT include copyrighted Vocaloid/Crypton voicebank samples.
Only use freely-licensed UTAU banks or create original recordings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .phonemes import (
    PHONEME_LIST, PHONEME_INVENTORY, VOWELS, CONSONANTS, SEMIVOWELS,
    STOPS, AFFRICATES, FRICATIVES, NASALS, LIQUIDS, VOICED_CONSONANTS,
    DIPHTHONGS, strip_stress
)
from .synth import (
    SAMPLE_RATE, VOWEL_FORMANTS, DIPHTHONG_END, CONSONANT_PARAMS,
    SEMIVOWEL_PARAMS, _make_envelope, save_wav
)


@dataclass
class SingerVoice:
    """Voice characteristics for a singer fly."""
    f0_base: float = 140.0        # Base fundamental frequency (Hz)
    f0_variation: float = 0.0     # Random pitch variation (+/- cents)
    formant_shift: float = 1.0    # Formant frequency multiplier (>1 = brighter)
    vibrato_rate: float = 0.0     # Vibrato frequency (Hz), 0 = no vibrato
    vibrato_depth: float = 0.0    # Vibrato depth (semitones)
    breathiness: float = 0.0      # Noise mix (0-1)
    duration_scale: float = 1.0   # Duration multiplier
    attack_scale: float = 1.0     # Attack time multiplier
    harmonic_rolloff: float = 0.7 # Harmonic amplitude rolloff (higher = brighter)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'f0_base': self.f0_base,
            'f0_variation': self.f0_variation,
            'formant_shift': self.formant_shift,
            'vibrato_rate': self.vibrato_rate,
            'vibrato_depth': self.vibrato_depth,
            'breathiness': self.breathiness,
            'duration_scale': self.duration_scale,
            'attack_scale': self.attack_scale,
            'harmonic_rolloff': self.harmonic_rolloff,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'SingerVoice':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _generate_phoneme_voices(seed: int = 42) -> Dict[str, SingerVoice]:
    """
    Generate distinct voice presets for each phoneme.
    
    Each phoneme gets a personal "singer fly" with unique characteristics.
    The voices are deterministically generated from the phoneme's properties
    to ensure consistency while maintaining distinctiveness.
    """
    rng = np.random.default_rng(seed)
    voices = {}
    
    for i, phoneme in enumerate(PHONEME_LIST):
        # Use phoneme index to seed unique characteristics
        p_seed = seed + i * 137  # Prime multiplier for spread
        p_rng = np.random.default_rng(p_seed)
        
        # Base characteristics depend on phoneme type
        info = PHONEME_INVENTORY[phoneme]
        
        if info.phoneme_type == 'vowel':
            # Vowels get distinct pitches spread across a range
            # Map vowel index to pitch range (100-200 Hz for variety)
            vowel_idx = list(VOWELS).index(phoneme) if phoneme in VOWELS else 0
            n_vowels = len(VOWELS)
            f0 = 120 + (vowel_idx / max(n_vowels - 1, 1)) * 80  # 120-200 Hz
            
            # Front vowels (high F2) get brighter formants
            _, f2 = VOWEL_FORMANTS.get(phoneme, (500, 1500))
            formant_shift = 0.9 + (f2 / 2400) * 0.3  # 0.9-1.2 based on F2
            
            # Add slight vibrato to some vowels
            vibrato_rate = p_rng.choice([0, 4, 5, 6]) 
            vibrato_depth = 0.3 if vibrato_rate > 0 else 0
            
            # Diphthongs get slightly longer duration
            duration_scale = 1.2 if phoneme in DIPHTHONGS else 1.0
            
            voices[phoneme] = SingerVoice(
                f0_base=f0,
                f0_variation=p_rng.uniform(0, 10),
                formant_shift=formant_shift,
                vibrato_rate=vibrato_rate,
                vibrato_depth=vibrato_depth,
                breathiness=p_rng.uniform(0, 0.15),
                duration_scale=duration_scale,
                attack_scale=p_rng.uniform(0.8, 1.2),
                harmonic_rolloff=p_rng.uniform(0.6, 0.8),
            )
            
        elif info.phoneme_type == 'consonant':
            # Consonants get voice characteristics based on type
            if phoneme in STOPS:
                # Stops: short, punchy
                f0 = 100 + p_rng.uniform(0, 30)
                duration_scale = 0.8 + p_rng.uniform(0, 0.2)
                breathiness = 0.1 if phoneme in VOICED_CONSONANTS else 0.3
            elif phoneme in FRICATIVES:
                # Fricatives: breathy, noisy
                f0 = 130 + p_rng.uniform(0, 40)
                duration_scale = 1.0 + p_rng.uniform(0, 0.3)
                breathiness = 0.4 + p_rng.uniform(0, 0.3)
            elif phoneme in NASALS:
                # Nasals: warm, resonant
                f0 = 120 + p_rng.uniform(0, 20)
                duration_scale = 1.0
                breathiness = 0.05 + p_rng.uniform(0, 0.1)
            elif phoneme in LIQUIDS:
                # Liquids: smooth, vowel-like
                f0 = 135 + p_rng.uniform(0, 25)
                duration_scale = 1.1
                breathiness = 0.1 + p_rng.uniform(0, 0.1)
            elif phoneme in AFFRICATES:
                # Affricates: complex, two-part
                f0 = 110 + p_rng.uniform(0, 30)
                duration_scale = 1.2
                breathiness = 0.25 + p_rng.uniform(0, 0.2)
            else:
                f0 = 130
                duration_scale = 1.0
                breathiness = 0.2
            
            voices[phoneme] = SingerVoice(
                f0_base=f0,
                f0_variation=p_rng.uniform(0, 5),
                formant_shift=p_rng.uniform(0.95, 1.1),
                vibrato_rate=0,  # No vibrato for consonants
                vibrato_depth=0,
                breathiness=breathiness,
                duration_scale=duration_scale,
                attack_scale=p_rng.uniform(0.7, 1.0),
                harmonic_rolloff=p_rng.uniform(0.5, 0.7),
            )
            
        else:  # Semivowels
            # Semivowels: gliding, transitional
            f0 = 145 + p_rng.uniform(0, 20)
            voices[phoneme] = SingerVoice(
                f0_base=f0,
                f0_variation=p_rng.uniform(5, 15),
                formant_shift=p_rng.uniform(0.95, 1.05),
                vibrato_rate=p_rng.choice([0, 3, 4]),
                vibrato_depth=0.2,
                breathiness=p_rng.uniform(0.05, 0.15),
                duration_scale=1.0,
                attack_scale=p_rng.uniform(0.9, 1.1),
                harmonic_rolloff=p_rng.uniform(0.65, 0.75),
            )
    
    return voices


class SingerFly:
    """
    A singer fly that renders audio for one specific phoneme.
    
    Each singer has its own voice characteristics (F0, formants, vibrato, etc.)
    making its phoneme crumb sound distinctly different from other singers.
    """
    
    def __init__(self, phoneme: str, voice: SingerVoice, rng: np.random.Generator):
        self.phoneme = strip_stress(phoneme)
        self.voice = voice
        self.rng = rng
        self.info = PHONEME_INVENTORY[self.phoneme]
        
        # Pre-render the phoneme audio crumb
        self._audio_cache: Optional[np.ndarray] = None
    
    def _apply_vibrato(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply vibrato (pitch modulation) to a signal."""
        if self.voice.vibrato_rate <= 0:
            return signal
        
        # Vibrato as amplitude modulation (simplified)
        depth = self.voice.vibrato_depth * 0.1  # Convert to amplitude scale
        vibrato = 1 + depth * np.sin(2 * np.pi * self.voice.vibrato_rate * t)
        return signal * vibrato
    
    def _add_breathiness(self, signal: np.ndarray) -> np.ndarray:
        """Add breathy noise to a signal."""
        if self.voice.breathiness <= 0:
            return signal
        
        noise = self.rng.standard_normal(len(signal)).astype(np.float32)
        noise *= self.voice.breathiness * 0.3
        return signal + noise
    
    def _synthesize_vowel(self) -> np.ndarray:
        """Synthesize a vowel with this singer's voice."""
        base_duration = 0.15 * self.voice.duration_scale
        n_samples = int(base_duration * SAMPLE_RATE)
        t = np.linspace(0, base_duration, n_samples)
        
        # Get formants and apply shift
        f1_base, f2_base = VOWEL_FORMANTS.get(self.phoneme, (500, 1500))
        f1 = f1_base * self.voice.formant_shift
        f2 = f2_base * self.voice.formant_shift
        
        # Handle diphthongs
        if self.phoneme in DIPHTHONG_END:
            f1_end, f2_end = DIPHTHONG_END[self.phoneme]
            f1_end *= self.voice.formant_shift
            f2_end *= self.voice.formant_shift
            f1 = np.linspace(f1, f1_end, n_samples)
            f2 = np.linspace(f2, f2_end, n_samples)
        else:
            f1 = np.full(n_samples, f1)
            f2 = np.full(n_samples, f2)
        
        # Apply F0 variation
        f0 = self.voice.f0_base
        if self.voice.f0_variation > 0:
            f0 += self.rng.uniform(-self.voice.f0_variation, self.voice.f0_variation)
        
        # Generate harmonics with formant shaping
        signal = np.zeros(n_samples, dtype=np.float32)
        for harmonic in range(1, 12):
            freq = f0 * harmonic
            
            # Formant shaping
            amp1 = np.exp(-((freq - f1) / 150) ** 2)
            amp2 = np.exp(-((freq - f2) / 200) ** 2)
            amp = (amp1 + amp2 * 0.6) / (harmonic ** self.voice.harmonic_rolloff)
            
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        # Apply vibrato
        signal = self._apply_vibrato(signal, t)
        
        # Apply envelope
        attack = 0.03 * self.voice.attack_scale
        signal *= _make_envelope(n_samples, attack=attack, release=0.05)
        
        # Add breathiness
        signal = self._add_breathiness(signal)
        
        # Normalize
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max() * 0.75
        
        return signal.astype(np.float32)
    
    def _synthesize_consonant(self) -> np.ndarray:
        """Synthesize a consonant with this singer's voice."""
        base_duration = 0.08 * self.voice.duration_scale
        n_samples = int(base_duration * SAMPLE_RATE)
        t = np.linspace(0, base_duration, n_samples)
        
        params = CONSONANT_PARAMS.get(self.phoneme, {
            'type': 'fricative', 'freq': 1000, 'noise': 0.5, 'voiced': False
        })
        ctype = params.get('type', 'fricative')
        
        # Apply formant shift to characteristic frequency
        base_freq = params.get('freq', 1000)
        freq = base_freq * self.voice.formant_shift
        
        # Get F0 for voicing
        f0 = self.voice.f0_base
        if self.voice.f0_variation > 0:
            f0 += self.rng.uniform(-self.voice.f0_variation, self.voice.f0_variation)
        
        if ctype == 'stop':
            signal = self._synth_stop(n_samples, t, freq, params, f0)
        elif ctype == 'affricate':
            signal = self._synth_affricate(n_samples, t, freq, params, f0)
        elif ctype == 'fricative':
            signal = self._synth_fricative(n_samples, t, freq, params, f0)
        elif ctype == 'nasal':
            signal = self._synth_nasal(n_samples, t, freq, params, f0)
        elif ctype == 'liquid':
            signal = self._synth_liquid(n_samples, t, freq, params, f0)
        else:
            signal = self.rng.standard_normal(n_samples).astype(np.float32) * 0.3
        
        # Add extra breathiness for this singer
        signal = self._add_breathiness(signal)
        
        # Normalize
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max() * 0.65
        
        return signal.astype(np.float32)
    
    def _synth_stop(self, n_samples, t, freq, params, f0):
        """Synthesize a stop with personalized voice."""
        signal = np.zeros(n_samples, dtype=np.float32)
        
        burst_start = int(n_samples * 0.35)
        burst_len = int(n_samples * params.get('burst', 0.4))
        if burst_start + burst_len > n_samples:
            burst_len = n_samples - burst_start
        
        # Personalized noise burst
        noise = self.rng.standard_normal(burst_len).astype(np.float32) * params['noise']
        t_burst = np.linspace(0, burst_len / SAMPLE_RATE, burst_len)
        carrier = np.sin(2 * np.pi * freq * t_burst)
        filtered = noise * 0.6 + carrier * 0.4 * np.abs(noise)
        
        burst_env = np.exp(-np.linspace(0, 6, burst_len))
        filtered *= burst_env
        
        signal[burst_start:burst_start + burst_len] = filtered
        
        if params.get('voiced', False):
            voice = np.sin(2 * np.pi * f0 * t) * 0.25
            voice *= _make_envelope(n_samples, attack=0.15, release=0.15)
            signal += voice
        
        return signal
    
    def _synth_affricate(self, n_samples, t, freq, params, f0):
        """Synthesize an affricate with personalized voice."""
        signal = np.zeros(n_samples, dtype=np.float32)
        
        stop_len = int(n_samples * 0.4)
        stop_params = {'freq': freq, 'noise': params['noise'] * 0.5,
                       'voiced': params['voiced'], 'burst': 0.6}
        t_stop = np.linspace(0, stop_len / SAMPLE_RATE, stop_len)
        signal[:stop_len] = self._synth_stop(stop_len, t_stop, freq, stop_params, f0)
        
        fric_len = n_samples - stop_len
        t_fric = np.linspace(0, fric_len / SAMPLE_RATE, fric_len)
        
        noise = self.rng.standard_normal(fric_len).astype(np.float32) * params['noise']
        carrier = np.sin(2 * np.pi * freq * t_fric)
        fric_signal = noise * 0.5 + carrier * 0.3 * noise
        
        fric_env = _make_envelope(fric_len, attack=0.05, release=0.1)
        signal[stop_len:] = fric_signal * fric_env
        
        if params.get('voiced', False):
            voice = np.sin(2 * np.pi * f0 * t) * 0.2
            signal += voice * _make_envelope(n_samples, attack=0.1, release=0.1)
        
        return signal
    
    def _synth_fricative(self, n_samples, t, freq, params, f0):
        """Synthesize a fricative with personalized voice."""
        noise = self.rng.standard_normal(n_samples).astype(np.float32) * params['noise']
        
        if params.get('aspirate', False):
            signal = noise * 0.5
        else:
            carrier = np.sin(2 * np.pi * freq * t)
            signal = noise * 0.4 + carrier * 0.2 * np.abs(noise)
        
        signal *= _make_envelope(n_samples, attack=0.05, release=0.08)
        
        if params.get('voiced', False):
            voice = np.sin(2 * np.pi * f0 * t) * 0.3
            signal += voice * _make_envelope(n_samples, attack=0.1, release=0.1)
        
        return signal
    
    def _synth_nasal(self, n_samples, t, freq, params, f0):
        """Synthesize a nasal with personalized voice."""
        f1 = freq
        f2 = params.get('f2', 1000) * self.voice.formant_shift
        
        signal = np.zeros(n_samples, dtype=np.float32)
        
        for harmonic in range(1, 8):
            h_freq = f0 * harmonic
            amp1 = np.exp(-((h_freq - f1) / 100) ** 2)
            amp2 = np.exp(-((h_freq - f2) / 150) ** 2) * 0.5
            amp = (amp1 + amp2) / (harmonic ** self.voice.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * h_freq * t)
        
        signal += np.sin(2 * np.pi * f1 * t) * 0.3
        signal *= _make_envelope(n_samples, attack=0.08, release=0.08)
        
        return signal
    
    def _synth_liquid(self, n_samples, t, freq, params, f0):
        """Synthesize a liquid with personalized voice."""
        f1 = freq
        f2 = params.get('f2', 1200) * self.voice.formant_shift
        f3 = params.get('f3', 2800) * self.voice.formant_shift
        
        signal = np.zeros(n_samples, dtype=np.float32)
        
        for harmonic in range(1, 10):
            h_freq = f0 * harmonic
            amp1 = np.exp(-((h_freq - f1) / 100) ** 2)
            amp2 = np.exp(-((h_freq - f2) / 150) ** 2) * 0.6
            amp3 = np.exp(-((h_freq - f3) / 200) ** 2) * 0.4
            amp = (amp1 + amp2 + amp3) / (harmonic ** self.voice.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * h_freq * t)
        
        signal *= _make_envelope(n_samples, attack=0.08, release=0.1)
        
        return signal
    
    def _synthesize_semivowel(self) -> np.ndarray:
        """Synthesize a semivowel with this singer's voice."""
        base_duration = 0.10 * self.voice.duration_scale
        n_samples = int(base_duration * SAMPLE_RATE)
        t = np.linspace(0, base_duration, n_samples)
        
        params = SEMIVOWEL_PARAMS.get(self.phoneme, {
            'start': (350, 700), 'end': (400, 1200), 'voiced': True
        })
        
        f1_start, f2_start = params['start']
        f1_end, f2_end = params['end']
        
        # Apply formant shift
        f1 = np.linspace(f1_start * self.voice.formant_shift,
                         f1_end * self.voice.formant_shift, n_samples)
        f2 = np.linspace(f2_start * self.voice.formant_shift,
                         f2_end * self.voice.formant_shift, n_samples)
        
        f0 = self.voice.f0_base
        if self.voice.f0_variation > 0:
            f0 += self.rng.uniform(-self.voice.f0_variation, self.voice.f0_variation)
        
        signal = np.zeros(n_samples, dtype=np.float32)
        
        for harmonic in range(1, 10):
            freq = f0 * harmonic
            amp1 = np.exp(-((freq - f1) / 120) ** 2)
            amp2 = np.exp(-((freq - f2) / 180) ** 2) * 0.7
            amp = (amp1 + amp2) / (harmonic ** self.voice.harmonic_rolloff)
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        signal = self._apply_vibrato(signal, t)
        signal *= _make_envelope(n_samples, attack=0.06, release=0.08)
        signal = self._add_breathiness(signal)
        
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max() * 0.7
        
        return signal.astype(np.float32)
    
    def synthesize(self) -> np.ndarray:
        """Synthesize this phoneme with the singer's personalized voice."""
        if self._audio_cache is not None:
            return self._audio_cache.copy()
        
        if self.phoneme in VOWELS:
            audio = self._synthesize_vowel()
        elif self.phoneme in SEMIVOWELS:
            audio = self._synthesize_semivowel()
        elif self.phoneme in CONSONANTS:
            audio = self._synthesize_consonant()
        else:
            # Fallback
            audio = self.rng.standard_normal(int(0.1 * SAMPLE_RATE)).astype(np.float32) * 0.3
        
        self._audio_cache = audio
        return audio.copy()
    
    def get_audio(self) -> np.ndarray:
        """Get cached or synthesized audio for this phoneme."""
        return self.synthesize()


class SingerSwarm:
    """
    A swarm of singer flies, one per phoneme.
    
    Each phoneme has its own singer fly with personalized voice characteristics.
    When rendering audio, each phoneme in the sequence is synthesized by its
    corresponding singer, creating a "mixture of flies" effect.
    """
    
    def __init__(self, phonemes: Optional[List[str]] = None, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Generate voice presets for all phonemes
        self.voices = _generate_phoneme_voices(seed)
        
        # Create singer flies for specified phonemes (or all)
        if phonemes is None:
            phonemes = PHONEME_LIST.copy()
        
        self.singers: Dict[str, SingerFly] = {}
        for phoneme in phonemes:
            p = strip_stress(phoneme)
            if p not in self.singers and p in self.voices:
                self.singers[p] = SingerFly(
                    phoneme=p,
                    voice=self.voices[p],
                    rng=np.random.default_rng(seed + hash(p) % 10000),
                )
        
        self.phoneme_list = list(self.singers.keys())
        print(f"SingerSwarm: {len(self.singers)} singer flies with personalized voices")
    
    def get_phoneme_audio(self) -> Dict[str, np.ndarray]:
        """Get audio crumbs for all phonemes in the swarm."""
        return {p: singer.get_audio() for p, singer in self.singers.items()}
    
    def synthesize_phoneme(self, phoneme: str) -> np.ndarray:
        """Synthesize a single phoneme using its singer fly."""
        p = strip_stress(phoneme)
        if p in self.singers:
            return self.singers[p].get_audio()
        else:
            # Fallback to a generic voice
            fallback_singer = SingerFly(
                phoneme=p,
                voice=SingerVoice(),  # Default voice
                rng=self.rng,
            )
            return fallback_singer.synthesize()
    
    def synthesize_sequence(
        self,
        phoneme_sequence: List[str],
        crossfade_ms: float = 15,
    ) -> np.ndarray:
        """
        Synthesize a phoneme sequence using each phoneme's singer fly.
        
        This is the MoE (mixture-of-flies) synthesis: each phoneme segment
        is rendered by its specialized singer.
        """
        if not phoneme_sequence:
            return np.array([], dtype=np.float32)
        
        crossfade_samples = int(crossfade_ms / 1000 * SAMPLE_RATE)
        
        # Get first phoneme audio
        first_p = strip_stress(phoneme_sequence[0])
        result = self.synthesize_phoneme(first_p).copy()
        
        for phoneme in phoneme_sequence[1:]:
            p = strip_stress(phoneme)
            audio = self.synthesize_phoneme(p)
            
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
    
    def get_voice_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of voice characteristics for each phoneme."""
        return {p: singer.voice.to_dict() for p, singer in self.singers.items()}
    
    def save(self, path: str):
        """Save singer swarm configuration."""
        path = Path(path)
        
        voice_data = {p: v.to_dict() for p, v in self.voices.items()}
        
        np.savez(
            path,
            voices=json.dumps(voice_data),
            phonemes=json.dumps(self.phoneme_list),
            seed=self.seed,
        )
        print(f"SingerSwarm saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SingerSwarm':
        """Load singer swarm from file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        phonemes = json.loads(str(data['phonemes']))
        seed = int(data['seed'])
        
        swarm = cls(phonemes=phonemes, seed=seed)
        
        # Restore saved voices
        voice_data = json.loads(str(data['voices']))
        for p, v_dict in voice_data.items():
            if p in swarm.voices:
                swarm.voices[p] = SingerVoice.from_dict(v_dict)
                if p in swarm.singers:
                    swarm.singers[p].voice = swarm.voices[p]
                    swarm.singers[p]._audio_cache = None  # Regenerate
        
        print(f"SingerSwarm loaded from {path}")
        return swarm


def create_default_singer_swarm(
    phonemes: Optional[List[str]] = None,
    seed: int = 42,
) -> SingerSwarm:
    """Create a singer swarm with default voice presets."""
    return SingerSwarm(phonemes=phonemes, seed=seed)


if __name__ == "__main__":
    print("Testing SingerSwarm...")
    
    # Create swarm with subset of phonemes
    test_phonemes = ['K', 'AE', 'T', 'D', 'AO', 'G', 'M', 'IY']
    swarm = SingerSwarm(phonemes=test_phonemes, seed=42)
    
    # Show voice characteristics
    print("\nVoice characteristics:")
    for p, singer in swarm.singers.items():
        v = singer.voice
        print(f"  {p}: F0={v.f0_base:.0f}Hz, shift={v.formant_shift:.2f}, "
              f"vibrato={v.vibrato_rate:.0f}Hz, breath={v.breathiness:.2f}")
    
    # Synthesize a test sequence
    print("\nSynthesizing 'cat' (K AE T)...")
    audio = swarm.synthesize_sequence(['K', 'AE', 'T'])
    print(f"  Audio: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.3f}s")
    
    # Save test
    save_wav(audio, "/tmp/singer_test_cat.wav", SAMPLE_RATE)
