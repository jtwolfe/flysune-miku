"""
Acoustic Fly Head: Trained speech production from fly activity.

This module implements TRAINED acoustic flies that produce voice from 
fly-native features (KC activations, cues, specialist YES scores).

Unlike concatenative UTAU playback, these flies LEARN to map:
    (KC activity, phoneme cue, position) → audio parameters → waveform

Architecture follows fly-faithful principles:
- Per-phoneme acoustic flies: Small MoE of specialist renderers
- KC→MBON-style readout: Sparse KC pattern → voice parameters
- Compartment teaching: Only update responsible acoustic fly

Targets come from:
1. MARIAN ILUSTRADO oto-sliced crumbs (preferred)
2. Current formant crumbs (fallback for bootstrapping)

References:
- Hige et al. 2015: Mushroom body output neurons encode valence
- Handler et al. 2019: Timing-dependent dopamine plasticity
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json

from .phonemes import (
    PHONEME_LIST, PHONEME_INVENTORY, strip_stress, VOWELS, CONSONANTS,
    SEMIVOWELS, NUM_PHONEMES
)
from .synth import (
    SAMPLE_RATE, VOWEL_FORMANTS, DIPHTHONG_END, synthesize_phoneme,
    _make_envelope, save_wav
)


# =============================================================================
# Acoustic Feature Configuration
# =============================================================================

@dataclass
class AcousticConfig:
    """Configuration for acoustic fly head."""
    # Feature dimensions
    kc_dim: int = 2000              # Input KC activity dimension
    cue_dim: int = 64               # Additional cue features (phoneme, position)
    hidden_dim: int = 128           # Hidden layer dimension
    
    # Audio parameters
    sample_rate: int = SAMPLE_RATE  # Output sample rate (22050)
    frame_size: int = 512           # Frame size for parametric synthesis
    hop_size: int = 256             # Hop size between frames
    n_frames: int = 16              # Number of output frames per phoneme
    
    # Parametric voice dimensions
    n_formants: int = 3             # F1, F2, F3 formant frequencies
    n_harmonics: int = 8            # Harmonic amplitudes
    
    # Learning
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    # Phoneme set
    phonemes: List[str] = field(default_factory=lambda: PHONEME_LIST.copy())
    
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'kc_dim': self.kc_dim,
            'cue_dim': self.cue_dim,
            'hidden_dim': self.hidden_dim,
            'sample_rate': self.sample_rate,
            'frame_size': self.frame_size,
            'hop_size': self.hop_size,
            'n_frames': self.n_frames,
            'n_formants': self.n_formants,
            'n_harmonics': self.n_harmonics,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'phonemes': self.phonemes,
            'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AcousticConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Voice Parameter Representation
# =============================================================================

@dataclass
class VoiceParams:
    """Parametric voice representation for one phoneme crumb."""
    f0: np.ndarray              # Fundamental frequency over frames (n_frames,)
    formants: np.ndarray        # Formant frequencies (n_frames, n_formants)
    amplitudes: np.ndarray      # Harmonic amplitudes (n_frames, n_harmonics)
    noise_amp: np.ndarray       # Noise amplitude over frames (n_frames,)
    
    @staticmethod
    def from_array(arr: np.ndarray, config: AcousticConfig) -> 'VoiceParams':
        """Unpack flat array into VoiceParams."""
        n_frames = config.n_frames
        n_formants = config.n_formants
        n_harmonics = config.n_harmonics
        
        idx = 0
        f0 = arr[idx:idx + n_frames]
        idx += n_frames
        formants = arr[idx:idx + n_frames * n_formants].reshape(n_frames, n_formants)
        idx += n_frames * n_formants
        amplitudes = arr[idx:idx + n_frames * n_harmonics].reshape(n_frames, n_harmonics)
        idx += n_frames * n_harmonics
        noise_amp = arr[idx:idx + n_frames]
        
        return VoiceParams(f0=f0, formants=formants, amplitudes=amplitudes, noise_amp=noise_amp)
    
    def to_array(self) -> np.ndarray:
        """Pack VoiceParams into flat array."""
        return np.concatenate([
            self.f0.flatten(),
            self.formants.flatten(),
            self.amplitudes.flatten(),
            self.noise_amp.flatten(),
        ])
    
    @staticmethod
    def get_param_dim(config: AcousticConfig) -> int:
        """Get total parameter dimension."""
        n_frames = config.n_frames
        return (n_frames +                                    # f0
                n_frames * config.n_formants +                # formants
                n_frames * config.n_harmonics +               # amplitudes
                n_frames)                                     # noise_amp


def synthesize_from_params(
    params: VoiceParams,
    config: AcousticConfig,
) -> np.ndarray:
    """
    Synthesize audio from voice parameters using additive synthesis.
    
    This is a clean oscillator bank - NOT Griffin-Lim vocoding.
    """
    n_samples = config.n_frames * config.hop_size
    audio = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples) / config.sample_rate
    
    for frame_idx in range(config.n_frames):
        start_sample = frame_idx * config.hop_size
        end_sample = start_sample + config.frame_size
        if end_sample > n_samples:
            end_sample = n_samples
        
        frame_len = end_sample - start_sample
        t_frame = t[start_sample:end_sample]
        
        f0 = np.clip(params.f0[frame_idx], 50, 500)
        formants = params.formants[frame_idx]
        amps = params.amplitudes[frame_idx]
        noise = params.noise_amp[frame_idx]
        
        frame_audio = np.zeros(frame_len, dtype=np.float32)
        
        for h in range(config.n_harmonics):
            freq = f0 * (h + 1)
            amp = np.clip(amps[h], 0, 1)
            
            for f_idx, f_freq in enumerate(formants):
                f_freq = np.clip(f_freq, 200, 4000)
                formant_gain = np.exp(-((freq - f_freq) / 150) ** 2)
                amp *= (1 + 0.5 * formant_gain)
            
            frame_audio += amp * np.sin(2 * np.pi * freq * t_frame)
        
        if noise > 0.01:
            frame_audio += noise * np.random.randn(frame_len).astype(np.float32) * 0.3
        
        window = _make_envelope(frame_len, attack=0.02, release=0.02)
        audio[start_sample:end_sample] += frame_audio * window
    
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max() * 0.75
    
    return audio.astype(np.float32)


# =============================================================================
# Per-Phoneme Acoustic Fly
# =============================================================================

class AcousticFly:
    """
    A single acoustic fly that learns to produce voice for one phoneme.
    
    Architecture:
        KC activity (sparse) + phoneme cue → hidden → voice parameters
        
    This is a KC→MBON-style readout where the "MBON" output is not
    a binary classification but parametric voice features.
    
    Learning:
        Supervised: MSE loss between predicted and target audio parameters
        Teaching signal: compartment-local (only this fly updates on its phoneme)
    """
    
    def __init__(
        self,
        target_phoneme: str,
        config: AcousticConfig,
        rng: np.random.Generator,
    ):
        self.phoneme = strip_stress(target_phoneme)
        self.config = config
        self.rng = rng
        
        input_dim = config.kc_dim + config.cue_dim
        output_dim = VoiceParams.get_param_dim(config)
        
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = rng.standard_normal((input_dim, config.hidden_dim)).astype(np.float32) * scale
        self.b1 = np.zeros(config.hidden_dim, dtype=np.float32)
        
        scale2 = np.sqrt(2.0 / config.hidden_dim)
        self.W2 = rng.standard_normal((config.hidden_dim, output_dim)).astype(np.float32) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        
        self._init_output_bias()
        
        self.training_steps = 0
        self.total_loss = 0.0
    
    def _init_output_bias(self):
        """Initialize output biases to reasonable voice parameter defaults."""
        config = self.config
        n_frames = config.n_frames
        idx = 0
        
        info = PHONEME_INVENTORY.get(self.phoneme)
        is_vowel = info and info.phoneme_type == 'vowel'
        
        default_f0 = 140 if is_vowel else 120
        self.b2[idx:idx + n_frames] = default_f0
        idx += n_frames
        
        if self.phoneme in VOWEL_FORMANTS:
            f1, f2 = VOWEL_FORMANTS[self.phoneme]
            f3 = 2800
        else:
            f1, f2, f3 = 500, 1500, 2800
        
        for frame in range(n_frames):
            self.b2[idx + frame * config.n_formants] = f1
            self.b2[idx + frame * config.n_formants + 1] = f2
            if config.n_formants > 2:
                self.b2[idx + frame * config.n_formants + 2] = f3
        idx += n_frames * config.n_formants
        
        for frame in range(n_frames):
            for h in range(config.n_harmonics):
                self.b2[idx + frame * config.n_harmonics + h] = 0.5 / (h + 1)
        idx += n_frames * config.n_harmonics
        
        noise_default = 0.3 if info and info.phoneme_type == 'consonant' else 0.05
        self.b2[idx:idx + n_frames] = noise_default
    
    def forward(self, kc_activity: np.ndarray, cue: np.ndarray) -> np.ndarray:
        """Forward pass: KC + cue → voice parameters."""
        x = np.concatenate([kc_activity.flatten(), cue.flatten()])
        
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        
        out = h @ self.W2 + self.b2
        
        return out, h
    
    def predict_params(
        self,
        kc_activity: np.ndarray,
        cue: np.ndarray,
    ) -> VoiceParams:
        """Predict voice parameters from fly features."""
        out, _ = self.forward(kc_activity, cue)
        return VoiceParams.from_array(out, self.config)
    
    def synthesize(
        self,
        kc_activity: np.ndarray,
        cue: np.ndarray,
    ) -> np.ndarray:
        """Synthesize audio from fly features."""
        params = self.predict_params(kc_activity, cue)
        return synthesize_from_params(params, self.config)
    
    def train_step(
        self,
        kc_activity: np.ndarray,
        cue: np.ndarray,
        target_params: np.ndarray,
    ) -> float:
        """
        Train on one example using gradient descent.
        
        Returns MSE loss.
        """
        predicted, hidden = self.forward(kc_activity, cue)
        
        error = predicted - target_params
        loss = np.mean(error ** 2)
        
        lr = self.config.learning_rate
        
        d_out = error * (2.0 / len(error))
        d_W2 = np.outer(hidden, d_out)
        d_b2 = d_out
        
        d_hidden = d_out @ self.W2.T
        d_hidden = d_hidden * (hidden > 0)
        
        x = np.concatenate([kc_activity.flatten(), cue.flatten()])
        d_W1 = np.outer(x, d_hidden)
        d_b1 = d_hidden
        
        wd = self.config.weight_decay
        self.W2 -= lr * (d_W2 + wd * self.W2)
        self.b2 -= lr * d_b2
        self.W1 -= lr * (d_W1 + wd * self.W1)
        self.b1 -= lr * d_b1
        
        self.training_steps += 1
        self.total_loss += loss
        
        return loss
    
    def get_avg_loss(self) -> float:
        if self.training_steps == 0:
            return 0.0
        return self.total_loss / self.training_steps
    
    def reset_stats(self):
        self.training_steps = 0
        self.total_loss = 0.0


# =============================================================================
# Acoustic Fly Swarm (MoE)
# =============================================================================

class AcousticFlySwarm:
    """
    Mixture-of-Experts swarm of acoustic flies.
    
    Each phoneme has its own acoustic fly that learns to produce that sound.
    At synthesis time, the picker chooses phonemes and the acoustic swarm
    renders each one.
    
    This is analogous to:
    - Per-compartment MBON readouts (each compartment = one acoustic fly)
    - The "teaching signal" is compartment-local: only the responsible
      acoustic fly gets updated for each training example.
    """
    
    def __init__(self, config: Optional[AcousticConfig] = None):
        self.config = config or AcousticConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        self.flies: Dict[str, AcousticFly] = {}
        for phoneme in self.config.phonemes:
            p = strip_stress(phoneme)
            if p not in self.flies:
                self.flies[p] = AcousticFly(
                    target_phoneme=p,
                    config=self.config,
                    rng=np.random.default_rng(self.config.seed + hash(p) % 10000),
                )
        
        self.phoneme_list = list(self.flies.keys())
        print(f"AcousticFlySwarm: {len(self.flies)} acoustic flies")
    
    def get_phoneme_cue(self, phoneme: str, position: float = 0.5) -> np.ndarray:
        """Create cue vector for phoneme + position."""
        cue = np.zeros(self.config.cue_dim, dtype=np.float32)
        
        p = strip_stress(phoneme)
        if p in self.phoneme_list:
            idx = self.phoneme_list.index(p)
            if idx < self.config.cue_dim - 4:
                cue[idx] = 1.0
        
        cue[-4] = position
        cue[-3] = np.sin(np.pi * position)
        cue[-2] = np.cos(np.pi * position)
        cue[-1] = 1.0 if p in VOWELS else 0.0
        
        return cue
    
    def synthesize_phoneme(
        self,
        phoneme: str,
        kc_activity: np.ndarray,
        position: float = 0.5,
    ) -> np.ndarray:
        """Synthesize one phoneme using its acoustic fly."""
        p = strip_stress(phoneme)
        cue = self.get_phoneme_cue(p, position)
        
        if p in self.flies:
            return self.flies[p].synthesize(kc_activity, cue)
        else:
            return synthesize_phoneme(p)
    
    def synthesize_sequence(
        self,
        phonemes: List[str],
        kc_activities: List[np.ndarray],
        crossfade_ms: float = 15,
    ) -> np.ndarray:
        """
        Synthesize a sequence of phonemes.
        
        Args:
            phonemes: List of phoneme symbols
            kc_activities: List of KC activity patterns (one per phoneme)
            crossfade_ms: Crossfade between phonemes
        """
        if not phonemes:
            return np.array([], dtype=np.float32)
        
        if len(kc_activities) < len(phonemes):
            kc_activities = list(kc_activities) + [kc_activities[-1]] * (len(phonemes) - len(kc_activities))
        
        crossfade_samples = int(crossfade_ms / 1000 * self.config.sample_rate)
        
        n_phonemes = len(phonemes)
        result = self.synthesize_phoneme(
            phonemes[0],
            kc_activities[0],
            position=0.0 if n_phonemes > 1 else 0.5,
        )
        
        for i, phoneme in enumerate(phonemes[1:], 1):
            position = i / (n_phonemes - 1) if n_phonemes > 1 else 0.5
            audio = self.synthesize_phoneme(phoneme, kc_activities[i], position)
            
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
    
    def train_phoneme(
        self,
        phoneme: str,
        kc_activity: np.ndarray,
        target_params: np.ndarray,
        position: float = 0.5,
    ) -> float:
        """Train one acoustic fly on its phoneme."""
        p = strip_stress(phoneme)
        if p not in self.flies:
            return 0.0
        
        cue = self.get_phoneme_cue(p, position)
        return self.flies[p].train_step(kc_activity, cue, target_params)
    
    def get_training_stats(self) -> Dict[str, Dict[str, float]]:
        """Get training statistics for all flies."""
        return {
            p: {
                'steps': fly.training_steps,
                'avg_loss': fly.get_avg_loss(),
            }
            for p, fly in self.flies.items()
        }
    
    def save(self, path: str):
        """Save acoustic swarm to file."""
        path = Path(path)
        
        fly_data = {}
        for p, fly in self.flies.items():
            fly_data[f'{p}_W1'] = fly.W1
            fly_data[f'{p}_b1'] = fly.b1
            fly_data[f'{p}_W2'] = fly.W2
            fly_data[f'{p}_b2'] = fly.b2
        
        np.savez(
            path,
            config=json.dumps(self.config.to_dict()),
            **fly_data,
        )
        print(f"AcousticFlySwarm saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AcousticFlySwarm':
        """Load acoustic swarm from file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = AcousticConfig.from_dict(json.loads(str(data['config'])))
        swarm = cls(config)
        
        for p in swarm.flies:
            if f'{p}_W1' in data:
                swarm.flies[p].W1 = data[f'{p}_W1']
                swarm.flies[p].b1 = data[f'{p}_b1']
                swarm.flies[p].W2 = data[f'{p}_W2']
                swarm.flies[p].b2 = data[f'{p}_b2']
        
        print(f"AcousticFlySwarm loaded from {path}")
        return swarm


# =============================================================================
# Target Extraction from Audio
# =============================================================================

def extract_voice_params(
    audio: np.ndarray,
    config: AcousticConfig,
) -> VoiceParams:
    """
    Extract voice parameters from audio for training targets.
    
    This creates synthetic targets from the formant synth output.
    For MARIAN targets, we'd do spectral analysis instead.
    """
    n_frames = config.n_frames
    n_formants = config.n_formants
    n_harmonics = config.n_harmonics
    
    f0 = np.full(n_frames, 140.0, dtype=np.float32)
    formants = np.zeros((n_frames, n_formants), dtype=np.float32)
    formants[:, 0] = 500
    formants[:, 1] = 1500
    if n_formants > 2:
        formants[:, 2] = 2800
    
    amplitudes = np.zeros((n_frames, n_harmonics), dtype=np.float32)
    for h in range(n_harmonics):
        amplitudes[:, h] = 0.5 / (h + 1)
    
    audio_power = np.abs(audio).mean() if len(audio) > 0 else 0.1
    noise_amp = np.full(n_frames, 0.1 * audio_power, dtype=np.float32)
    
    if len(audio) > 0:
        fft_result = np.fft.rfft(audio[:min(len(audio), 2048)])
        power = np.abs(fft_result) ** 2
        
        if len(power) > 10:
            freqs = np.fft.rfftfreq(min(len(audio), 2048), 1/config.sample_rate)
            
            for f_idx in range(n_formants):
                if f_idx == 0:
                    f_range = (200, 900)
                elif f_idx == 1:
                    f_range = (800, 2500)
                else:
                    f_range = (2000, 4000)
                
                mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
                if mask.any():
                    peak_idx = np.argmax(power[mask])
                    formants[:, f_idx] = freqs[mask][peak_idx]
    
    return VoiceParams(f0=f0, formants=formants, amplitudes=amplitudes, noise_amp=noise_amp)


def create_formant_targets(
    phonemes: List[str],
    config: AcousticConfig,
) -> Dict[str, np.ndarray]:
    """
    Create training targets from formant synthesizer.
    
    This is the FALLBACK when MARIAN voicebank is not available.
    Uses current formant crumbs as bootstrap targets.
    """
    targets = {}
    
    for phoneme in phonemes:
        p = strip_stress(phoneme)
        if p in targets:
            continue
        
        audio = synthesize_phoneme(p)
        params = extract_voice_params(audio, config)
        
        if p in VOWEL_FORMANTS:
            f1, f2 = VOWEL_FORMANTS[p]
            params.formants[:, 0] = f1
            params.formants[:, 1] = f2
            if config.n_formants > 2:
                params.formants[:, 2] = 2800
        
        targets[p] = params.to_array()
    
    return targets


# =============================================================================
# Integration with Picker Swarm
# =============================================================================

def get_kc_activity_from_more_fly(
    swarm,  # MoreFlySwarm
    letter_context: str,
    phoneme_pos: float = 0.5,
    n_phonemes: int = 3,
    previous_phone: str = None,
) -> np.ndarray:
    """
    Extract KC activity from a MORE FLY swarm.
    
    This provides the fly-native features for acoustic synthesis.
    """
    return swarm.shared.encode_to_kc(
        letter_context,
        phoneme_pos=phoneme_pos,
        n_phonemes=n_phonemes,
        previous_phone=previous_phone,
    )


def get_specialist_scores(
    swarm,  # MoreFlySwarm
    letter_context: str,
    phoneme_pos: float = 0.5,
    n_phonemes: int = 3,
    previous_phone: str = None,
) -> Dict[str, float]:
    """
    Get specialist YES scores from a MORE FLY swarm.
    
    These scores can be additional features for acoustic synthesis.
    """
    return swarm._get_raw_scores(
        letter_context,
        phoneme_pos=phoneme_pos,
        n_phonemes=n_phonemes,
        previous_phone=previous_phone,
    )


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("Testing Acoustic Fly module...")
    
    config = AcousticConfig(kc_dim=100, cue_dim=48)
    swarm = AcousticFlySwarm(config)
    
    print("\n1. Testing single phoneme synthesis:")
    kc = np.random.rand(config.kc_dim).astype(np.float32)
    kc = (kc > 0.9).astype(np.float32)
    
    audio = swarm.synthesize_phoneme('AA', kc, position=0.5)
    print(f"   AA: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.3f}s")
    
    print("\n2. Testing sequence synthesis:")
    phonemes = ['K', 'AE', 'T']
    kcs = [kc.copy() for _ in phonemes]
    audio = swarm.synthesize_sequence(phonemes, kcs)
    print(f"   'cat' (K AE T): {len(audio)} samples, {len(audio)/SAMPLE_RATE:.3f}s")
    
    print("\n3. Testing training step:")
    targets = create_formant_targets(['AA'], config)
    loss = swarm.train_phoneme('AA', kc, targets['AA'])
    print(f"   AA training loss: {loss:.4f}")
    
    print("\n4. Testing save/load:")
    swarm.save("/tmp/test_acoustic_swarm.npz")
    loaded = AcousticFlySwarm.load("/tmp/test_acoustic_swarm.npz")
    print(f"   Loaded {len(loaded.flies)} flies")
    
    print("\n✓ Acoustic Fly module tests complete")
