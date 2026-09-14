"""
Stage 2b: Formant track synthesis from mushroom body trajectories.

Instead of mel spectrograms + Griffin-Lim (Stage 2), we:
1. Run the MB for T frames with time-varying word/position cues
2. Record KC/MBON activity as a trajectory A[t]
3. Learn a linear map A[t] → formant_tracks[t] (F1, F2, F3, voiced, noise, pitch)
4. Render audio using the same formant synthesis as Stage 1 phoneme crumbs

This produces cleaner audio than Stage 2 Griffin-Lim while still being
brain-driven (activity → sound, not just phoneme selection).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import json

from .mushroom_body import MushroomBody, MushroomBodyConfig
from .lexicon import get_phonemes, get_lexicon, create_training_data, is_known_word
from .synth import (
    SAMPLE_RATE, VOWEL_FORMANTS, CONSONANT_PARAMS, SEMIVOWEL_PARAMS,
    DIPHTHONG_END, save_wav
)
from .phonemes import strip_stress, VOWELS, SEMIVOWELS, CONSONANTS, DIPHTHONGS
from .stage2 import phoneme_to_frames, word_to_total_frames


STAGE2B_SAMPLE_RATE = 22050
FRAME_HOP_SAMPLES = 512  # ~23ms per frame at 22050 Hz
FRAME_DURATION_S = FRAME_HOP_SAMPLES / STAGE2B_SAMPLE_RATE

N_FORMANT_TRACKS = 6  # F1, F2, F3, voiced_gain, noise_gain, pitch


@dataclass
class Stage2bConfig:
    """Configuration for Stage 2b synthesis."""
    sample_rate: int = STAGE2B_SAMPLE_RATE
    frame_hop: int = FRAME_HOP_SAMPLES
    n_tracks: int = N_FORMANT_TRACKS
    
    use_kc: bool = True
    use_mbon: bool = True
    use_pn: bool = False
    
    ridge_alpha: float = 10.0  # Higher regularization for smoother outputs
    
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_rate': self.sample_rate, 'frame_hop': self.frame_hop,
            'n_tracks': self.n_tracks, 'use_kc': self.use_kc,
            'use_mbon': self.use_mbon, 'use_pn': self.use_pn,
            'ridge_alpha': self.ridge_alpha, 'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Stage2bConfig':
        return cls(**d)


class FormantTargetGenerator:
    """
    Generate smooth formant/amplitude target tracks from phoneme sequences.
    
    Target vector per frame: [F1, F2, F3, voiced_gain, noise_gain, pitch]
    - F1, F2, F3: formant frequencies in Hz (normalized 0-1 for 0-5000 Hz)
    - voiced_gain: 0-1 voicing amplitude
    - noise_gain: 0-1 noise amplitude
    - pitch: 0-1 fundamental frequency (normalized)
    """
    
    MAX_FREQ = 5000.0  # Max formant frequency for normalization
    MAX_PITCH = 200.0  # Max F0 for normalization
    
    def __init__(self, config: Stage2bConfig):
        self.config = config
    
    def _get_phoneme_formants(self, phoneme: str, t: float) -> Tuple[float, float, float]:
        """Get formants for a phoneme at time t (0 to 1 within phoneme)."""
        p = strip_stress(phoneme)
        
        if p in VOWELS:
            f1, f2 = VOWEL_FORMANTS.get(p, (500, 1500))
            f3 = 2500.0
            
            if p in DIPHTHONGS:
                f1_end, f2_end = DIPHTHONG_END.get(p, (f1, f2))
                f1 = f1 * (1 - t) + f1_end * t
                f2 = f2 * (1 - t) + f2_end * t
            
            return float(f1), float(f2), f3
        
        elif p in SEMIVOWELS:
            params = SEMIVOWEL_PARAMS.get(p, {'start': (350, 1000), 'end': (400, 1500)})
            f1_start, f2_start = params['start']
            f1_end, f2_end = params['end']
            f1 = f1_start * (1 - t) + f1_end * t
            f2 = f2_start * (1 - t) + f2_end * t
            return float(f1), float(f2), 2500.0
        
        else:
            params = CONSONANT_PARAMS.get(p, {'freq': 1000, 'type': 'fricative'})
            ctype = params.get('type', 'fricative')
            
            if ctype in ['nasal', 'liquid']:
                f1 = float(params.get('freq', 300))
                f2 = float(params.get('f2', 1200))
                f3 = float(params.get('f3', 2500))
            else:
                f1 = float(params.get('freq', 500))
                f2 = 1500.0
                f3 = 2500.0
            
            return f1, f2, f3
    
    def _get_phoneme_gains(self, phoneme: str, t: float) -> Tuple[float, float]:
        """Get voiced and noise gains for a phoneme at time t."""
        p = strip_stress(phoneme)
        
        if p in VOWELS:
            return 0.9, 0.05
        
        elif p in SEMIVOWELS:
            return 0.8, 0.1
        
        else:
            params = CONSONANT_PARAMS.get(p, {'type': 'fricative', 'voiced': False, 'noise': 0.5})
            ctype = params.get('type', 'fricative')
            voiced = params.get('voiced', False)
            noise_level = params.get('noise', 0.5)
            
            if ctype == 'stop':
                if 0.3 < t < 0.7:
                    return (0.3 if voiced else 0.0), min(noise_level * 1.5, 1.0)
                else:
                    return (0.2 if voiced else 0.0), 0.1
            
            elif ctype in ['fricative', 'affricate']:
                return (0.3 if voiced else 0.0), noise_level
            
            elif ctype == 'nasal':
                return 0.8, 0.05
            
            elif ctype == 'liquid':
                return 0.7, 0.1
            
            else:
                return (0.3 if voiced else 0.0), noise_level
    
    def _get_phoneme_pitch(self, phoneme: str) -> float:
        """Get pitch (F0) for a phoneme."""
        p = strip_stress(phoneme)
        
        if p in VOWELS or p in SEMIVOWELS:
            return 140.0
        
        params = CONSONANT_PARAMS.get(p, {'voiced': False})
        if params.get('voiced', False):
            return 130.0
        return 0.0
    
    def phoneme_to_tracks(self, phoneme: str) -> np.ndarray:
        """Generate formant track frames for a single phoneme."""
        n_frames = phoneme_to_frames(phoneme)
        tracks = np.zeros((n_frames, self.config.n_tracks), dtype=np.float32)
        
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            
            f1, f2, f3 = self._get_phoneme_formants(phoneme, t)
            voiced, noise = self._get_phoneme_gains(phoneme, t)
            pitch = self._get_phoneme_pitch(phoneme)
            
            tracks[i, 0] = f1 / self.MAX_FREQ
            tracks[i, 1] = f2 / self.MAX_FREQ
            tracks[i, 2] = f3 / self.MAX_FREQ
            tracks[i, 3] = voiced
            tracks[i, 4] = noise
            tracks[i, 5] = pitch / self.MAX_PITCH
        
        return tracks
    
    def word_to_tracks(self, phonemes: List[str]) -> np.ndarray:
        """Generate formant track frames for a word's phoneme sequence."""
        all_tracks = []
        for phoneme in phonemes:
            tracks = self.phoneme_to_tracks(phoneme)
            all_tracks.append(tracks)
        
        if all_tracks:
            return np.concatenate(all_tracks, axis=0)
        else:
            return np.zeros((1, self.config.n_tracks), dtype=np.float32)


class FormantRenderer:
    """
    Render formant tracks to audio using additive synthesis.
    
    Uses the same harmonic + noise approach as Stage 1 phoneme synthesis,
    but driven by continuous tracks rather than fixed phoneme parameters.
    """
    
    MAX_FREQ = FormantTargetGenerator.MAX_FREQ
    MAX_PITCH = FormantTargetGenerator.MAX_PITCH
    
    def __init__(self, config: Stage2bConfig):
        self.config = config
        self.sr = config.sample_rate
        self.hop = config.frame_hop
    
    def _render_frame(self, f1: float, f2: float, f3: float,
                      voiced: float, noise: float, pitch: float,
                      n_samples: int, phase_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Render a single frame of audio."""
        t = np.arange(n_samples) / self.sr
        signal = np.zeros(n_samples, dtype=np.float32)
        
        if voiced > 0.01 and pitch > 0.01:
            f0 = pitch * self.MAX_PITCH
            
            for harmonic in range(1, 12):
                freq = f0 * harmonic
                if freq > self.sr / 2:
                    break
                
                amp1 = np.exp(-((freq - f1 * self.MAX_FREQ) / 150) ** 2)
                amp2 = np.exp(-((freq - f2 * self.MAX_FREQ) / 180) ** 2) * 0.7
                amp3 = np.exp(-((freq - f3 * self.MAX_FREQ) / 200) ** 2) * 0.4
                amp = (amp1 + amp2 + amp3) / (harmonic ** 0.6) * voiced
                
                phase = phase_state[harmonic - 1]
                signal += amp * np.sin(2 * np.pi * freq * t + phase)
                phase_state[harmonic - 1] = (phase + 2 * np.pi * freq * n_samples / self.sr) % (2 * np.pi)
        
        if noise > 0.01:
            freq_center = max(f2 * self.MAX_FREQ, 1000)
            noise_sig = np.random.randn(n_samples).astype(np.float32) * noise * 0.5
            
            carrier = np.sin(2 * np.pi * freq_center * t)
            noise_sig = noise_sig * 0.6 + carrier * noise_sig * 0.4
            signal += noise_sig
        
        return signal, phase_state
    
    def render(self, tracks: np.ndarray) -> np.ndarray:
        """
        Render formant tracks to audio.
        
        Args:
            tracks: [T, 6] array of [F1, F2, F3, voiced, noise, pitch] (normalized)
        
        Returns:
            Audio signal
        """
        n_frames = len(tracks)
        total_samples = n_frames * self.hop + self.hop
        audio = np.zeros(total_samples, dtype=np.float32)
        
        phase_state = np.zeros(12, dtype=np.float32)
        
        window = np.hanning(self.hop * 2)
        
        for i in range(n_frames):
            f1, f2, f3, voiced, noise, pitch = tracks[i]
            
            f1 = np.clip(f1, 0.05, 0.95)
            f2 = np.clip(f2, 0.1, 0.95)
            f3 = np.clip(f3, 0.2, 0.95)
            voiced = np.clip(voiced, 0, 1)
            noise = np.clip(noise, 0, 1)
            pitch = np.clip(pitch, 0, 1)
            
            frame_audio, phase_state = self._render_frame(
                f1, f2, f3, voiced, noise, pitch,
                self.hop * 2, phase_state
            )
            
            frame_audio *= window
            
            start = i * self.hop
            end = start + self.hop * 2
            if end <= total_samples:
                audio[start:end] += frame_audio
        
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.75
        
        return audio.astype(np.float32)


class Stage2bModel:
    """
    Stage 2b synthesis model.
    
    Uses the mushroom body to generate temporal trajectories,
    then maps those to formant tracks via a learned linear readout.
    Audio is rendered using formant synthesis (not Griffin-Lim).
    """
    
    def __init__(self, mb: MushroomBody, config: Optional[Stage2bConfig] = None):
        self.mb = mb
        self.config = config or Stage2bConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        self.activity_dim = 0
        if self.config.use_kc:
            self.activity_dim += mb.config.n_kc
        if self.config.use_mbon:
            self.activity_dim += mb.config.n_mbon
        if self.config.use_pn:
            self.activity_dim += mb.config.n_pn
        
        self.readout_weights: Optional[np.ndarray] = None
        self.readout_bias: Optional[np.ndarray] = None
        
        self.target_gen = FormantTargetGenerator(self.config)
        self.renderer = FormantRenderer(self.config)
        
        self.trained = False
        self.train_mse = 0.0
    
    def _get_activity_vector(self, activations: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract activity vector from MB activations."""
        parts = []
        if self.config.use_kc:
            parts.append(activations['kc_activity'])
        if self.config.use_mbon:
            parts.append(activations['mbon_input'])
        if self.config.use_pn:
            parts.append(activations['pn_activity'])
        return np.concatenate(parts)
    
    def rollout(self, word: str, phonemes: List[str]) -> np.ndarray:
        """
        Generate activity trajectory for a word.
        
        Runs the MB with time-varying cues (fractional position along word).
        
        Returns:
            Activity matrix [T, activity_dim]
        """
        total_frames = word_to_total_frames(phonemes)
        activity_trajectory = []
        
        cumulative = 0
        phoneme_boundaries = [0]
        for p in phonemes:
            cumulative += phoneme_to_frames(p)
            phoneme_boundaries.append(cumulative)
        
        for frame_idx in range(total_frames):
            phoneme_idx = 0
            for i, boundary in enumerate(phoneme_boundaries[1:]):
                if frame_idx < boundary:
                    phoneme_idx = i
                    break
            
            local_start = phoneme_boundaries[phoneme_idx]
            local_end = phoneme_boundaries[phoneme_idx + 1]
            local_frac = (frame_idx - local_start) / max(local_end - local_start - 1, 1)
            
            pos = phoneme_idx + local_frac * 0.5
            
            _, activations = self.mb.forward(word, int(pos))
            
            activity = self._get_activity_vector(activations)
            activity_trajectory.append(activity)
        
        return np.array(activity_trajectory, dtype=np.float32)
    
    def train(self, words: List[str], lexicon: Dict[str, List[str]],
              verbose: bool = True) -> float:
        """
        Train the readout weights using ridge regression.
        
        Args:
            words: List of training words
            lexicon: Word -> phoneme mapping
            verbose: Print progress
        
        Returns:
            Training MSE
        """
        if verbose:
            print(f"Stage 2b training on {len(words)} words...")
        
        all_activities = []
        all_targets = []
        
        for i, word in enumerate(words):
            if word not in lexicon:
                continue
            
            phonemes = lexicon[word]
            
            activity = self.rollout(word, phonemes)
            target_tracks = self.target_gen.word_to_tracks(phonemes)
            
            min_len = min(len(activity), len(target_tracks))
            activity = activity[:min_len]
            target_tracks = target_tracks[:min_len]
            
            all_activities.append(activity)
            all_targets.append(target_tracks)
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(words)} words")
        
        if not all_activities:
            print("No training data!")
            return float('inf')
        
        A = np.vstack(all_activities)
        Y = np.vstack(all_targets)
        
        if verbose:
            print(f"  Total frames: {A.shape[0]}")
            print(f"  Activity dim: {A.shape[1]}")
            print(f"  Target dim: {Y.shape[1]} (F1, F2, F3, voiced, noise, pitch)")
        
        alpha = self.config.ridge_alpha
        n_features = A.shape[1]
        
        AtA = A.T @ A + alpha * np.eye(n_features)
        AtY = A.T @ Y
        
        self.readout_weights = np.linalg.solve(AtA, AtY).astype(np.float32)
        
        predictions = A @ self.readout_weights
        self.readout_bias = np.mean(Y - predictions, axis=0).astype(np.float32)
        
        predictions_biased = predictions + self.readout_bias
        mse = np.mean((Y - predictions_biased) ** 2)
        
        track_mse = np.mean((Y - predictions_biased) ** 2, axis=0)
        if verbose:
            track_names = ['F1', 'F2', 'F3', 'Voiced', 'Noise', 'Pitch']
            for name, err in zip(track_names, track_mse):
                print(f"    {name}: MSE={err:.4f}")
        
        self.trained = True
        self.train_mse = float(mse)
        
        if verbose:
            print(f"  Overall Train MSE: {mse:.4f}")
        
        return mse
    
    def synthesize(self, word: str, phonemes: Optional[List[str]] = None) -> np.ndarray:
        """
        Synthesize audio for a word using Stage 2b pipeline.
        
        Args:
            word: Word to synthesize
            phonemes: Optional phoneme sequence (uses G2P if not provided)
        
        Returns:
            Audio signal
        """
        if not self.trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        if phonemes is None:
            phonemes = get_phonemes(word, allow_g2p=True)
        
        activity = self.rollout(word, phonemes)
        
        tracks = activity @ self.readout_weights + self.readout_bias
        
        tracks[:, :3] = np.clip(tracks[:, :3], 0.05, 0.95)
        tracks[:, 3:5] = np.clip(tracks[:, 3:5], 0, 1)
        tracks[:, 5] = np.clip(tracks[:, 5], 0, 1)
        
        audio = self.renderer.render(tracks)
        
        return audio
    
    def save(self, path: str):
        """Save Stage 2b model weights."""
        if not self.trained:
            raise RuntimeError("Model not trained!")
        
        path = Path(path)
        np.savez(
            path,
            readout_weights=self.readout_weights,
            readout_bias=self.readout_bias,
            config=json.dumps(self.config.to_dict()),
            train_mse=self.train_mse,
            activity_dim=self.activity_dim,
        )
        print(f"Stage 2b model saved to {path}")
    
    @classmethod
    def load(cls, mb: MushroomBody, path: str) -> 'Stage2bModel':
        """Load Stage 2b model weights."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = Stage2bConfig.from_dict(json.loads(str(data['config'])))
        model = cls(mb, config)
        
        model.readout_weights = data['readout_weights']
        model.readout_bias = data['readout_bias']
        model.train_mse = float(data['train_mse'])
        model.trained = True
        
        print(f"Stage 2b model loaded from {path} (train MSE: {model.train_mse:.4f})")
        return model


def train_stage2b(
    mb_path: str = "model.npz",
    output_path: str = "model_stage2b.npz",
    max_words: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> Stage2bModel:
    """
    Train Stage 2b model.
    
    Args:
        mb_path: Path to trained Stage 1 mushroom body
        output_path: Where to save Stage 2b weights
        max_words: Maximum training words
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Trained Stage2bModel
    """
    mb = MushroomBody.load(mb_path)
    
    config = Stage2bConfig(seed=seed)
    model = Stage2bModel(mb, config)
    
    train_lexicon, train_words, _ = create_training_data(
        max_words=max_words, seed=seed, include_demo=True
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("STAGE 2b TRAINING (Formant Tracks)")
        print(f"{'='*60}")
        print(f"Words: {len(train_words)}")
        print(f"Activity dim: {model.activity_dim}")
        print(f"Track dim: {config.n_tracks} (F1, F2, F3, voiced, noise, pitch)")
        print(f"Ridge alpha: {config.ridge_alpha}")
        print(f"{'='*60}\n")
    
    mse = model.train(train_words, train_lexicon, verbose=verbose)
    
    model.save(output_path)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage 2b training complete")
        print(f"Final MSE: {mse:.4f}")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}\n")
    
    return model


def speak_stage2b(
    word: str,
    mb_path: str = "model.npz",
    stage2b_path: str = "model_stage2b.npz",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Speak a word using Stage 2b synthesis (formant tracks).
    
    Args:
        word: Word to speak
        mb_path: Path to MB model
        stage2b_path: Path to Stage 2b model
        output_path: Optional WAV output path
        verbose: Print details
    
    Returns:
        audio: Audio signal
        phonemes: Reference phonemes
    """
    mb = MushroomBody.load(mb_path)
    model = Stage2bModel.load(mb, stage2b_path)
    
    phonemes = get_phonemes(word, allow_g2p=True)
    known = is_known_word(word)
    
    if verbose:
        source = "CMUdict" if known else "G2P"
        print(f"\nStage 2b speaking: '{word}' ({source})")
        print(f"Phonemes: {' '.join(phonemes)}")
        print(f"Total frames: {word_to_total_frames(phonemes)}")
    
    audio = model.synthesize(word, phonemes)
    
    if verbose:
        duration = len(audio) / STAGE2B_SAMPLE_RATE
        print(f"Audio duration: {duration:.2f}s")
    
    if output_path:
        save_wav(audio, output_path, STAGE2B_SAMPLE_RATE)
    
    return audio, phonemes


if __name__ == "__main__":
    print("Testing Stage 2b components...")
    
    config = Stage2bConfig()
    target_gen = FormantTargetGenerator(config)
    
    test_phonemes = ['K', 'AE', 'T']
    tracks = target_gen.word_to_tracks(test_phonemes)
    print(f"Tracks shape for 'cat': {tracks.shape}")
    print(f"Track ranges: F1={tracks[:,0].min():.2f}-{tracks[:,0].max():.2f}, "
          f"F2={tracks[:,1].min():.2f}-{tracks[:,1].max():.2f}, "
          f"voiced={tracks[:,3].min():.2f}-{tracks[:,3].max():.2f}")
    
    renderer = FormantRenderer(config)
    audio = renderer.render(tracks)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / config.sample_rate:.2f}s")
