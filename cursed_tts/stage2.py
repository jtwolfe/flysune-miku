"""
Stage 2: Temporal signal generation from mushroom body trajectories.

Instead of selecting canned phoneme WAVs, we:
1. Run the MB for T frames with time-varying word/position cues
2. Record KC/MBON activity as a trajectory A[t]
3. Learn a linear map A[t] → acoustic_frame[t] (mel spectrogram)
4. Invert mel to audio via Griffin-Lim

This produces truly "generated" speech from neural activity patterns.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import json

from .mushroom_body import MushroomBody, MushroomBodyConfig
from .lexicon import get_phonemes, get_lexicon, create_training_data, is_known_word
from .synth import SAMPLE_RATE, VOWEL_FORMANTS, CONSONANT_PARAMS, SEMIVOWEL_PARAMS
from .phonemes import strip_stress, VOWELS, SEMIVOWELS, CONSONANTS


# Stage 2 audio parameters
STAGE2_SAMPLE_RATE = 22050
HOP_LENGTH = 256  # ~11.6ms per frame at 22050 Hz
N_MELS = 40  # Mel spectrogram bins
N_FFT = 1024
FRAME_DURATION_MS = HOP_LENGTH / STAGE2_SAMPLE_RATE * 1000  # ~11.6ms

# Phoneme durations in frames
PHONEME_FRAMES = {
    'vowel': 6,      # ~70ms
    'consonant': 4,  # ~46ms
    'semivowel': 5,  # ~58ms
}


@dataclass
class Stage2Config:
    """Configuration for Stage 2 synthesis."""
    n_mels: int = N_MELS
    hop_length: int = HOP_LENGTH
    sample_rate: int = STAGE2_SAMPLE_RATE
    
    # Activity feature dimensions
    use_kc: bool = True       # Include KC activity
    use_mbon: bool = True     # Include MBON inputs
    use_pn: bool = False      # Include PN activity (optional)
    
    # Readout
    ridge_alpha: float = 1.0  # Ridge regularization
    
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_mels': self.n_mels, 'hop_length': self.hop_length,
            'sample_rate': self.sample_rate, 'use_kc': self.use_kc,
            'use_mbon': self.use_mbon, 'use_pn': self.use_pn,
            'ridge_alpha': self.ridge_alpha, 'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Stage2Config':
        return cls(**d)


def phoneme_to_frames(phoneme: str) -> int:
    """Get number of frames for a phoneme."""
    p = strip_stress(phoneme)
    if p in VOWELS:
        return PHONEME_FRAMES['vowel']
    elif p in SEMIVOWELS:
        return PHONEME_FRAMES['semivowel']
    else:
        return PHONEME_FRAMES['consonant']


def word_to_total_frames(phonemes: List[str]) -> int:
    """Get total frames for a word's phoneme sequence."""
    return sum(phoneme_to_frames(p) for p in phonemes)


class MelSynthesizer:
    """
    Generate mel spectrograms from phoneme sequences.
    
    Uses formant synthesis to create target spectrograms for training.
    """
    
    def __init__(self, config: Stage2Config, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        
        # Precompute mel filterbank
        self.mel_filterbank = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix."""
        n_fft = N_FFT
        n_mels = self.config.n_mels
        sr = self.config.sample_rate
        
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Mel points
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        return filterbank.astype(np.float32)
    
    def _formants_to_spectrum(self, f1: float, f2: float, f3: float,
                              voiced: bool, noise: float) -> np.ndarray:
        """Convert formants to a spectral envelope."""
        freqs = np.linspace(0, self.config.sample_rate / 2, N_FFT // 2 + 1)
        
        spectrum = np.zeros_like(freqs)
        
        if voiced:
            # Add formant peaks
            for f, bw in [(f1, 100), (f2, 150), (f3, 200)]:
                if f > 0:
                    peak = np.exp(-((freqs - f) / bw) ** 2)
                    spectrum += peak
        
        if noise > 0:
            # Add noise component (high frequency)
            noise_spectrum = self.rng.random(len(freqs)) * noise
            noise_spectrum *= np.linspace(0, 1, len(freqs))  # More HF
            spectrum += noise_spectrum
        
        # Normalize
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        return spectrum.astype(np.float32)
    
    def _spectrum_to_mel(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert linear spectrum to mel spectrum."""
        mel = self.mel_filterbank @ spectrum
        # Log compression
        mel = np.log(mel + 1e-6)
        return mel.astype(np.float32)
    
    def phoneme_to_mel_frames(self, phoneme: str) -> np.ndarray:
        """Generate mel frames for a single phoneme."""
        p = strip_stress(phoneme)
        n_frames = phoneme_to_frames(phoneme)
        
        mel_frames = []
        
        if p in VOWELS:
            # Vowel: formant synthesis
            f1, f2 = VOWEL_FORMANTS.get(p, (500, 1500))
            f3 = 2500
            
            for i in range(n_frames):
                # Slight variation over time
                t = i / max(n_frames - 1, 1)
                f1_t = f1 * (1 + 0.1 * (t - 0.5))
                f2_t = f2 * (1 + 0.05 * (t - 0.5))
                
                spectrum = self._formants_to_spectrum(f1_t, f2_t, f3, voiced=True, noise=0.05)
                mel = self._spectrum_to_mel(spectrum)
                mel_frames.append(mel)
        
        elif p in SEMIVOWELS:
            # Semivowel: gliding formants
            params = SEMIVOWEL_PARAMS.get(p, {'start': (350, 1000), 'end': (400, 1500)})
            f1_start, f2_start = params['start']
            f1_end, f2_end = params['end']
            
            for i in range(n_frames):
                t = i / max(n_frames - 1, 1)
                f1 = f1_start + (f1_end - f1_start) * t
                f2 = f2_start + (f2_end - f2_start) * t
                
                spectrum = self._formants_to_spectrum(f1, f2, 2500, voiced=True, noise=0.1)
                mel = self._spectrum_to_mel(spectrum)
                mel_frames.append(mel)
        
        else:
            # Consonant
            params = CONSONANT_PARAMS.get(p, {'type': 'fricative', 'freq': 2000, 'noise': 0.5})
            ctype = params.get('type', 'fricative')
            voiced = params.get('voiced', False)
            
            for i in range(n_frames):
                t = i / max(n_frames - 1, 1)
                
                if ctype == 'stop':
                    # Burst in middle
                    if 0.3 < t < 0.7:
                        noise = params.get('noise', 0.5) * 2
                    else:
                        noise = 0.1
                    f1 = params.get('freq', 1000) if voiced else 0
                    spectrum = self._formants_to_spectrum(f1, 0, 0, voiced, noise)
                
                elif ctype in ['fricative', 'affricate']:
                    noise = params.get('noise', 0.8)
                    f1 = 200 if voiced else 0
                    spectrum = self._formants_to_spectrum(f1, 0, 0, voiced, noise)
                
                elif ctype == 'nasal':
                    f1 = params.get('freq', 250)
                    f2 = params.get('f2', 1000)
                    spectrum = self._formants_to_spectrum(f1, f2, 2500, voiced=True, noise=0.05)
                
                elif ctype == 'liquid':
                    f1 = params.get('freq', 350)
                    f2 = params.get('f2', 1200)
                    f3 = params.get('f3', 2800)
                    spectrum = self._formants_to_spectrum(f1, f2, f3, voiced=True, noise=0.1)
                
                else:
                    spectrum = self._formants_to_spectrum(500, 1500, 2500, voiced, 0.3)
                
                mel = self._spectrum_to_mel(spectrum)
                mel_frames.append(mel)
        
        return np.array(mel_frames, dtype=np.float32)
    
    def word_to_mel(self, phonemes: List[str]) -> np.ndarray:
        """Generate mel spectrogram for a word's phoneme sequence."""
        all_frames = []
        for phoneme in phonemes:
            frames = self.phoneme_to_mel_frames(phoneme)
            all_frames.append(frames)
        
        if all_frames:
            return np.concatenate(all_frames, axis=0)
        else:
            return np.zeros((1, self.config.n_mels), dtype=np.float32)


class GriffinLim:
    """
    Griffin-Lim algorithm for mel spectrogram inversion.
    """
    
    def __init__(self, config: Stage2Config, mel_filterbank: np.ndarray):
        self.config = config
        self.mel_filterbank = mel_filterbank
        
        # Pseudo-inverse for mel to linear
        self.mel_to_linear = np.linalg.pinv(mel_filterbank)
    
    def mel_to_audio(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """
        Convert mel spectrogram to audio using Griffin-Lim.
        
        Args:
            mel: Mel spectrogram [T, n_mels] (log scale)
            n_iter: Number of Griffin-Lim iterations
        
        Returns:
            Audio signal
        """
        # Exp to undo log
        mel_linear = np.exp(mel)
        
        # Mel to linear spectrogram
        linear = (self.mel_to_linear @ mel_linear.T).T  # [T, n_fft//2+1]
        linear = np.maximum(linear, 1e-6)
        
        n_frames = linear.shape[0]
        n_fft = N_FFT
        hop = self.config.hop_length
        
        # Initialize with random phase
        rng = np.random.default_rng(42)
        phase = rng.uniform(-np.pi, np.pi, linear.shape)
        
        # Griffin-Lim iterations
        for _ in range(n_iter):
            # Construct complex spectrogram
            stft = linear * np.exp(1j * phase)
            
            # Inverse STFT
            audio = self._istft(stft, hop, n_fft)
            
            # Forward STFT
            stft_new = self._stft(audio, hop, n_fft)
            
            # Update phase
            phase = np.angle(stft_new)
        
        # Final synthesis
        stft = linear * np.exp(1j * phase)
        audio = self._istft(stft, hop, n_fft)
        
        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.8
        
        return audio.astype(np.float32)
    
    def _stft(self, audio: np.ndarray, hop: int, n_fft: int) -> np.ndarray:
        """Short-time Fourier transform."""
        # Pad audio
        pad_len = n_fft // 2
        audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
        
        # Window
        window = np.hanning(n_fft)
        
        # Number of frames
        n_frames = 1 + (len(audio_padded) - n_fft) // hop
        
        stft = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex64)
        
        for i in range(n_frames):
            start = i * hop
            frame = audio_padded[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft[i] = spectrum
        
        return stft
    
    def _istft(self, stft: np.ndarray, hop: int, n_fft: int) -> np.ndarray:
        """Inverse short-time Fourier transform."""
        n_frames = stft.shape[0]
        window = np.hanning(n_fft)
        
        # Output length
        out_len = (n_frames - 1) * hop + n_fft
        audio = np.zeros(out_len, dtype=np.float32)
        window_sum = np.zeros(out_len, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop
            frame = np.fft.irfft(stft[i], n_fft).real * window
            audio[start:start + n_fft] += frame
            window_sum[start:start + n_fft] += window ** 2
        
        # Normalize by window sum
        window_sum = np.maximum(window_sum, 1e-6)
        audio = audio / window_sum
        
        # Remove padding
        pad_len = n_fft // 2
        audio = audio[pad_len:-pad_len]
        
        return audio


class Stage2Model:
    """
    Stage 2 synthesis model.
    
    Uses the mushroom body to generate temporal trajectories,
    then maps those to mel spectrograms via a learned linear readout.
    """
    
    def __init__(self, mb: MushroomBody, config: Optional[Stage2Config] = None):
        self.mb = mb
        self.config = config or Stage2Config()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Activity dimension
        self.activity_dim = 0
        if self.config.use_kc:
            self.activity_dim += mb.config.n_kc
        if self.config.use_mbon:
            self.activity_dim += mb.config.n_mbon
        if self.config.use_pn:
            self.activity_dim += mb.config.n_pn
        
        # Readout weights (learned)
        self.readout_weights: Optional[np.ndarray] = None
        self.readout_bias: Optional[np.ndarray] = None
        
        # Mel synthesizer and vocoder
        self.mel_synth = MelSynthesizer(self.config, self.rng)
        self.vocoder = GriffinLim(self.config, self.mel_synth.mel_filterbank)
        
        # Stats
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
        
        # Track cumulative frames per phoneme
        cumulative = 0
        phoneme_boundaries = [0]
        for p in phonemes:
            cumulative += phoneme_to_frames(p)
            phoneme_boundaries.append(cumulative)
        
        for frame_idx in range(total_frames):
            # Find which phoneme this frame belongs to
            phoneme_idx = 0
            for i, boundary in enumerate(phoneme_boundaries[1:]):
                if frame_idx < boundary:
                    phoneme_idx = i
                    break
            
            # Fractional position within word (0 to 1)
            frac_pos = frame_idx / max(total_frames - 1, 1)
            
            # Use phoneme index as position for MB (character-aligned)
            # But interpolate based on frame position within phoneme
            local_start = phoneme_boundaries[phoneme_idx]
            local_end = phoneme_boundaries[phoneme_idx + 1]
            local_frac = (frame_idx - local_start) / max(local_end - local_start - 1, 1)
            
            # Interpolated position for MB encoding
            pos = phoneme_idx + local_frac * 0.5  # Slight sub-phoneme variation
            
            # Run MB forward
            _, activations = self.mb.forward(word, int(pos))
            
            # Extract activity
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
            print(f"Stage 2 training on {len(words)} words...")
        
        # Collect all activity-target pairs
        all_activities = []
        all_targets = []
        
        for i, word in enumerate(words):
            if word not in lexicon:
                continue
            
            phonemes = lexicon[word]
            
            # Get activity trajectory
            activity = self.rollout(word, phonemes)
            
            # Get target mel spectrogram
            target_mel = self.mel_synth.word_to_mel(phonemes)
            
            # Ensure same length
            min_len = min(len(activity), len(target_mel))
            activity = activity[:min_len]
            target_mel = target_mel[:min_len]
            
            all_activities.append(activity)
            all_targets.append(target_mel)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(words)} words")
        
        if not all_activities:
            print("No training data!")
            return float('inf')
        
        # Stack all data
        A = np.vstack(all_activities)  # [total_frames, activity_dim]
        Y = np.vstack(all_targets)     # [total_frames, n_mels]
        
        if verbose:
            print(f"  Total frames: {A.shape[0]}")
            print(f"  Activity dim: {A.shape[1]}")
            print(f"  Target dim: {Y.shape[1]}")
        
        # Ridge regression: W = (A^T A + λI)^-1 A^T Y
        alpha = self.config.ridge_alpha
        n_features = A.shape[1]
        
        AtA = A.T @ A + alpha * np.eye(n_features)
        AtY = A.T @ Y
        
        self.readout_weights = np.linalg.solve(AtA, AtY).astype(np.float32)
        
        # Compute bias (mean of residuals)
        predictions = A @ self.readout_weights
        self.readout_bias = np.mean(Y - predictions, axis=0).astype(np.float32)
        
        # Compute MSE
        predictions_biased = predictions + self.readout_bias
        mse = np.mean((Y - predictions_biased) ** 2)
        
        self.trained = True
        self.train_mse = float(mse)
        
        if verbose:
            print(f"  Train MSE: {mse:.4f}")
        
        return mse
    
    def synthesize(self, word: str, phonemes: Optional[List[str]] = None) -> np.ndarray:
        """
        Synthesize audio for a word using Stage 2 pipeline.
        
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
        
        # Get activity trajectory
        activity = self.rollout(word, phonemes)
        
        # Apply readout
        mel = activity @ self.readout_weights + self.readout_bias
        
        # Clip to reasonable range
        mel = np.clip(mel, -10, 2)
        
        # Invert to audio
        audio = self.vocoder.mel_to_audio(mel)
        
        return audio
    
    def save(self, path: str):
        """Save Stage 2 model weights."""
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
        print(f"Stage 2 model saved to {path}")
    
    @classmethod
    def load(cls, mb: MushroomBody, path: str) -> 'Stage2Model':
        """Load Stage 2 model weights."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = Stage2Config.from_dict(json.loads(str(data['config'])))
        model = cls(mb, config)
        
        model.readout_weights = data['readout_weights']
        model.readout_bias = data['readout_bias']
        model.train_mse = float(data['train_mse'])
        model.trained = True
        
        print(f"Stage 2 model loaded from {path} (train MSE: {model.train_mse:.4f})")
        return model


def train_stage2(
    mb_path: str = "model.npz",
    output_path: str = "model_stage2.npz",
    max_words: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> Stage2Model:
    """
    Train Stage 2 model.
    
    Args:
        mb_path: Path to trained Stage 1 mushroom body
        output_path: Where to save Stage 2 weights
        max_words: Maximum training words
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Trained Stage2Model
    """
    # Load MB
    mb = MushroomBody.load(mb_path)
    
    # Create Stage 2 model
    config = Stage2Config(seed=seed)
    model = Stage2Model(mb, config)
    
    # Get training data
    train_lexicon, train_words, _ = create_training_data(
        max_words=max_words, seed=seed, include_demo=True
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("STAGE 2 TRAINING")
        print(f"{'='*60}")
        print(f"Words: {len(train_words)}")
        print(f"Activity dim: {model.activity_dim}")
        print(f"Mel bins: {config.n_mels}")
        print(f"{'='*60}\n")
    
    # Train
    mse = model.train(train_words, train_lexicon, verbose=verbose)
    
    # Save
    model.save(output_path)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage 2 training complete")
        print(f"Final MSE: {mse:.4f}")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}\n")
    
    return model


def speak_stage2(
    word: str,
    mb_path: str = "model.npz",
    stage2_path: str = "model_stage2.npz",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Speak a word using Stage 2 synthesis.
    
    Args:
        word: Word to speak
        mb_path: Path to MB model
        stage2_path: Path to Stage 2 model
        output_path: Optional WAV output path
        verbose: Print details
    
    Returns:
        audio: Audio signal
        phonemes: Reference phonemes
    """
    # Load models
    mb = MushroomBody.load(mb_path)
    model = Stage2Model.load(mb, stage2_path)
    
    # Get phonemes
    phonemes = get_phonemes(word, allow_g2p=True)
    known = is_known_word(word)
    
    if verbose:
        source = "CMUdict" if known else "G2P"
        print(f"\nStage 2 speaking: '{word}' ({source})")
        print(f"Phonemes: {' '.join(phonemes)}")
        print(f"Total frames: {word_to_total_frames(phonemes)}")
    
    # Synthesize
    audio = model.synthesize(word, phonemes)
    
    if verbose:
        duration = len(audio) / STAGE2_SAMPLE_RATE
        print(f"Audio duration: {duration:.2f}s")
    
    # Save if requested
    if output_path:
        from .synth import save_wav
        save_wav(audio, output_path, STAGE2_SAMPLE_RATE)
    
    return audio, phonemes


if __name__ == "__main__":
    # Quick test
    print("Testing Stage 2 components...")
    
    # Test mel synthesizer
    config = Stage2Config()
    rng = np.random.default_rng(42)
    mel_synth = MelSynthesizer(config, rng)
    
    test_phonemes = ['K', 'AE', 'T']  # "cat"
    mel = mel_synth.word_to_mel(test_phonemes)
    print(f"Mel shape for 'cat': {mel.shape}")
    
    # Test vocoder
    vocoder = GriffinLim(config, mel_synth.mel_filterbank)
    audio = vocoder.mel_to_audio(mel)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / config.sample_rate:.2f}s")
