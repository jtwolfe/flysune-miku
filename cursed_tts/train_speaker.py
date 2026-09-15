"""
Train Speaker Flies: Fit per-phoneme parametric voices to target audio.

The speaker training is completely decoupled from picker training:
- Picker swarm learns: letter context → phoneme (G2P classification)
- Speaker swarm learns: phoneme → audio crumb (synthesis parameters)

Training modes:
1. formant-bootstrap: Use existing formant synth as targets (self-supervised)
2. marian-fit: Fit to Marian voicebank crumbs (when data/marian_crumbs exists)

The trained parameters are:
- F0 (fundamental frequency)
- Formant shifts (F1, F2, F3)
- Noise level
- Duration
- Attack/release times

Crucially: NO KC activity is used. Speakers are conditioned only on:
- Phoneme ID (which speaker fly)
- Optional: previous phoneme (one-hot context)
- Optional: position in utterance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass

from .phonemes import PHONEME_LIST, strip_stress, phoneme_to_index, NUM_PHONEMES
from .speaker_fly import (
    SpeakerSwarm, SpeakerSwarmConfig, SpeakerParams, SpeakerFly,
    SPEAKER_DURATION_MIN_MS, SPEAKER_DURATION_MAX_MS, SAMPLE_RATE,
    _get_default_params_for_phoneme
)
from .synth import save_wav


@dataclass
class SpeakerTrainingConfig:
    """Configuration for speaker training."""
    mode: str = 'formant-bootstrap'  # 'formant-bootstrap' or 'marian-fit'
    n_iterations: int = 50           # Training iterations per phoneme
    learning_rate: float = 0.01      # Parameter update rate
    seed: int = 42
    marian_dir: str = 'data/marian_crumbs'  # Directory for Marian samples
    duration_cap_ms: float = 250.0   # Max crumb duration
    fit_prev_phone: bool = True      # Learn prev-phone adjustments
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'n_iterations': self.n_iterations,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'marian_dir': self.marian_dir,
            'duration_cap_ms': self.duration_cap_ms,
            'fit_prev_phone': self.fit_prev_phone,
        }


def _load_marian_crumb(phoneme: str, marian_dir: Path) -> Optional[np.ndarray]:
    """
    Load Marian voicebank crumb for a phoneme.
    
    Handles various alias conventions:
    - Direct: K.wav, AE.wav
    - With dash: K-.wav, -K.wav (C V / V C context markers)
    - Lowercase: k.wav
    
    Returns audio array or None if not found.
    """
    p = strip_stress(phoneme)
    
    # Try various naming conventions
    candidates = [
        marian_dir / f"{p}.wav",
        marian_dir / f"{p.lower()}.wav",
        marian_dir / f"{p}-.wav",
        marian_dir / f"-{p}.wav",
        marian_dir / f"{p}_crumb.wav",
        marian_dir / f"crumb_{p}.wav",
    ]
    
    for path in candidates:
        if path.exists():
            try:
                import wave
                with wave.open(str(path), 'r') as wf:
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
                    
                    return audio.astype(np.float32)
            except Exception:
                continue
    
    return None


def _estimate_f0(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Estimate fundamental frequency from audio using autocorrelation.
    
    Simple pitch detection for training speaker parameters.
    """
    if len(audio) < sample_rate // 50:  # Need at least 20ms
        return 140.0  # Default
    
    # Autocorrelation
    n = len(audio)
    corr = np.correlate(audio, audio, mode='full')
    corr = corr[n-1:]  # Keep positive lags only
    
    # Find first peak after lag corresponding to max expected F0 (500 Hz)
    min_lag = sample_rate // 500  # ~500 Hz max
    max_lag = sample_rate // 50   # ~50 Hz min
    
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return 140.0
    
    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0:
        return 140.0
    
    peak_idx = np.argmax(search_region) + min_lag
    
    if peak_idx > 0:
        f0 = sample_rate / peak_idx
        # Sanity check
        if 50 <= f0 <= 500:
            return float(f0)
    
    return 140.0


def _estimate_duration(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Estimate effective duration in milliseconds."""
    # Find where energy drops below threshold
    threshold = 0.05 * np.abs(audio).max()
    above_threshold = np.abs(audio) > threshold
    
    if not above_threshold.any():
        return len(audio) / sample_rate * 1000
    
    first = np.argmax(above_threshold)
    last = len(above_threshold) - np.argmax(above_threshold[::-1])
    
    duration_samples = last - first
    return duration_samples / sample_rate * 1000


def _compute_spectral_centroid(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Compute spectral centroid as a formant brightness indicator."""
    if len(audio) < 256:
        return 1500.0
    
    # Simple FFT
    n_fft = min(2048, len(audio))
    spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    
    if spectrum.sum() > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        return float(centroid)
    
    return 1500.0


def _mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute MSE loss between two audio signals."""
    # Align lengths
    min_len = min(len(pred), len(target))
    if min_len == 0:
        return 1e6
    
    pred = pred[:min_len]
    target = target[:min_len]
    
    return float(np.mean((pred - target) ** 2))


def _spectral_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute spectral loss between two audio signals."""
    min_len = min(len(pred), len(target))
    if min_len < 256:
        return 1e6
    
    pred = pred[:min_len]
    target = target[:min_len]
    
    n_fft = min(1024, min_len)
    pred_spec = np.abs(np.fft.rfft(pred[:n_fft]))
    target_spec = np.abs(np.fft.rfft(target[:n_fft]))
    
    # Normalize
    pred_spec = pred_spec / (np.linalg.norm(pred_spec) + 1e-6)
    target_spec = target_spec / (np.linalg.norm(target_spec) + 1e-6)
    
    return float(np.mean((pred_spec - target_spec) ** 2))


def train_speaker_for_phoneme(
    phoneme: str,
    target_audio: np.ndarray,
    config: SpeakerTrainingConfig,
    verbose: bool = False,
) -> SpeakerParams:
    """
    Train speaker parameters for one phoneme to match target audio.
    
    Uses simple gradient-free optimization (random search + local refinement).
    
    Args:
        phoneme: Phoneme to train
        target_audio: Target audio crumb
        config: Training configuration
        verbose: Print progress
    
    Returns:
        Trained SpeakerParams
    """
    rng = np.random.default_rng(config.seed + hash(phoneme) % 10000)
    
    # Initialize with defaults
    params = _get_default_params_for_phoneme(phoneme, config.seed)
    
    # Extract target features for guidance
    target_f0 = _estimate_f0(target_audio)
    target_duration = _estimate_duration(target_audio)
    target_centroid = _compute_spectral_centroid(target_audio)
    
    # Cap duration
    target_duration = min(target_duration, config.duration_cap_ms)
    
    # Initial guess from target features
    params.f0 = target_f0
    params.duration_ms = target_duration
    
    # Formant shift based on spectral centroid (rough heuristic)
    if target_centroid > 2000:
        params.f1_shift = 1.1
        params.f2_shift = 1.05
    elif target_centroid < 1000:
        params.f1_shift = 0.9
        params.f2_shift = 0.95
    
    # Create speaker for evaluation
    speaker = SpeakerFly(phoneme, params=params, mode='formant', seed=config.seed)
    
    best_params = params.clone()
    best_loss = float('inf')
    
    # Simple random search + refinement
    for iteration in range(config.n_iterations):
        # Generate candidate parameters
        candidate = params.clone()
        
        # Exploration: random perturbations
        if iteration < config.n_iterations // 2:
            # Broader search
            candidate.f0 = params.f0 * (1 + rng.uniform(-0.1, 0.1))
            candidate.f1_shift = params.f1_shift * (1 + rng.uniform(-0.1, 0.1))
            candidate.f2_shift = params.f2_shift * (1 + rng.uniform(-0.1, 0.1))
            candidate.noise_level = max(0, min(1, params.noise_level + rng.uniform(-0.1, 0.1)))
            candidate.duration_ms = max(SPEAKER_DURATION_MIN_MS, 
                                        min(SPEAKER_DURATION_MAX_MS,
                                            params.duration_ms * (1 + rng.uniform(-0.15, 0.15))))
        else:
            # Local refinement
            lr = config.learning_rate * (1 - iteration / config.n_iterations)
            candidate.f0 = params.f0 * (1 + rng.uniform(-lr, lr))
            candidate.f1_shift = params.f1_shift * (1 + rng.uniform(-lr, lr))
            candidate.f2_shift = params.f2_shift * (1 + rng.uniform(-lr, lr))
            candidate.noise_level = max(0, min(1, params.noise_level + rng.uniform(-lr, lr)))
            candidate.duration_ms = max(SPEAKER_DURATION_MIN_MS,
                                        min(SPEAKER_DURATION_MAX_MS,
                                            params.duration_ms * (1 + rng.uniform(-lr, lr))))
        
        # Evaluate
        speaker.params = candidate
        pred_audio = speaker.synthesize()
        
        # Combined loss
        loss = 0.3 * _mse_loss(pred_audio, target_audio) + 0.7 * _spectral_loss(pred_audio, target_audio)
        
        if loss < best_loss:
            best_loss = loss
            best_params = candidate.clone()
            params = candidate.clone()  # Accept improvement
    
    if verbose:
        print(f"  {phoneme}: F0={best_params.f0:.1f}Hz, dur={best_params.duration_ms:.0f}ms, "
              f"loss={best_loss:.4f}")
    
    return best_params


def train_speaker_swarm_formant_bootstrap(
    config: SpeakerTrainingConfig,
    phonemes: Optional[List[str]] = None,
    verbose: bool = True,
) -> SpeakerSwarm:
    """
    Train speaker swarm using formant synthesis as bootstrap targets.
    
    This is self-supervised: we generate targets from the default formant
    synth, then train speakers to reproduce them. This establishes baseline
    parameters before optional fine-tuning on real samples.
    """
    if phonemes is None:
        phonemes = PHONEME_LIST.copy()
    
    if verbose:
        print("Training SpeakerSwarm (formant-bootstrap mode)")
        print("=" * 60)
        print("  Mode: Self-supervised bootstrap from formant synthesis")
        print("  NO KC dependency: speakers learn from phoneme ID only")
        print()
    
    # Create swarm
    swarm_config = SpeakerSwarmConfig(
        mode='trained',
        seed=config.seed,
        phonemes=phonemes,
    )
    swarm = SpeakerSwarm(swarm_config)
    
    # For each phoneme, train to match default formant output
    from .synth import synthesize_phoneme as synth_formant
    
    if verbose:
        print("Training per-phoneme speakers:")
    
    for phoneme in phonemes:
        p = strip_stress(phoneme)
        if p not in swarm.speakers:
            continue
        
        # Generate target from default formant synth
        target = synth_formant(p)
        
        # Cap duration
        max_samples = int(config.duration_cap_ms / 1000 * SAMPLE_RATE)
        if len(target) > max_samples:
            target = target[:max_samples]
        
        # Train
        trained_params = train_speaker_for_phoneme(
            p, target, config, verbose=verbose
        )
        
        swarm.set_speaker_params(p, trained_params)
    
    if verbose:
        print()
        print("Speaker swarm trained (formant-bootstrap)")
    
    return swarm


def train_speaker_swarm_marian_fit(
    config: SpeakerTrainingConfig,
    phonemes: Optional[List[str]] = None,
    verbose: bool = True,
) -> SpeakerSwarm:
    """
    Train speaker swarm to match Marian voicebank crumbs.
    
    This fits parametric speakers to real recorded phoneme samples.
    For phonemes without Marian data, falls back to formant bootstrap.
    
    Attribution: Marian voicebank by [attribution per NOTICE file].
    """
    if phonemes is None:
        phonemes = PHONEME_LIST.copy()
    
    marian_dir = Path(config.marian_dir)
    has_marian = marian_dir.exists()
    
    if verbose:
        print("Training SpeakerSwarm (marian-fit mode)")
        print("=" * 60)
        if has_marian:
            print(f"  Marian directory: {marian_dir}")
        else:
            print(f"  Marian directory not found: {marian_dir}")
            print("  Falling back to formant-bootstrap for all phonemes")
        print("  NO KC dependency: speakers learn from phoneme ID only")
        print()
    
    # Create swarm
    swarm_config = SpeakerSwarmConfig(
        mode='trained',
        seed=config.seed,
        phonemes=phonemes,
    )
    swarm = SpeakerSwarm(swarm_config)
    
    # Track which phonemes use Marian vs formant
    marian_count = 0
    formant_count = 0
    
    from .synth import synthesize_phoneme as synth_formant
    
    if verbose:
        print("Training per-phoneme speakers:")
    
    for phoneme in phonemes:
        p = strip_stress(phoneme)
        if p not in swarm.speakers:
            continue
        
        # Try to load Marian crumb
        target = None
        source = 'formant'
        
        if has_marian:
            marian_audio = _load_marian_crumb(p, marian_dir)
            if marian_audio is not None:
                target = marian_audio
                source = 'marian'
                marian_count += 1
        
        if target is None:
            # Fallback to formant
            target = synth_formant(p)
            formant_count += 1
        
        # Cap duration
        max_samples = int(config.duration_cap_ms / 1000 * SAMPLE_RATE)
        if len(target) > max_samples:
            target = target[:max_samples]
        
        # Train
        trained_params = train_speaker_for_phoneme(
            p, target, config, verbose=verbose
        )
        
        if verbose and source == 'marian':
            print(f"    ^ trained from Marian sample")
        
        swarm.set_speaker_params(p, trained_params)
    
    if verbose:
        print()
        print(f"Speaker swarm trained: {marian_count} Marian, {formant_count} formant-bootstrap")
    
    return swarm


def train_speaker_swarm(
    output_path: str = "model_speaker.npz",
    config: Optional[SpeakerTrainingConfig] = None,
    phonemes: Optional[List[str]] = None,
    verbose: bool = True,
) -> SpeakerSwarm:
    """
    Train and save speaker swarm.
    
    Args:
        output_path: Path to save trained swarm
        config: Training configuration
        phonemes: Phonemes to train (None = all)
        verbose: Print progress
    
    Returns:
        Trained SpeakerSwarm
    """
    if config is None:
        config = SpeakerTrainingConfig()
    
    # Check for Marian data
    marian_dir = Path(config.marian_dir)
    if config.mode == 'marian-fit' or (config.mode == 'auto' and marian_dir.exists()):
        swarm = train_speaker_swarm_marian_fit(config, phonemes, verbose)
    else:
        swarm = train_speaker_swarm_formant_bootstrap(config, phonemes, verbose)
    
    # Save
    swarm.save(output_path)
    
    return swarm


def generate_speaker_demo(
    swarm: SpeakerSwarm,
    output_dir: str = "artifacts/eval/two_swarm",
    words: Optional[List[str]] = None,
    picker_swarm=None,  # Optional picker for full pipeline demo
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Generate demo audio files using speaker swarm.
    
    If picker_swarm is provided, runs full two-swarm pipeline:
        word → picker → phonemes → speakers → audio
    
    Otherwise, uses reference phonemes from lexicon.
    """
    from .lexicon import get_phonemes
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if words is None:
        words = ['cat', 'bat', 'dog', 'me', 'yes', 'mushroom', 'hello', 'world']
    
    results = {}
    
    if verbose:
        print(f"\nGenerating speaker demos to {output_dir}")
        print("=" * 60)
    
    for word in words:
        word_clean = ''.join(c for c in word.lower() if c.isalnum())
        
        # Get phonemes
        if picker_swarm is not None:
            # Full two-swarm pipeline
            ref_phonemes = get_phonemes(word, allow_g2p=True)
            phonemes = picker_swarm.predict_word(word, len(ref_phonemes))
            source = 'two-swarm'
        else:
            # Reference phonemes only
            phonemes = get_phonemes(word, allow_g2p=True)
            phonemes = [strip_stress(p) for p in phonemes]
            source = 'reference'
        
        # Synthesize
        audio = swarm.synthesize_sequence(phonemes)
        
        # Save
        output_path = output_dir / f"{word_clean}_speaker.wav"
        save_wav(audio, str(output_path), SAMPLE_RATE)
        
        results[word] = str(output_path)
        
        if verbose:
            duration_ms = len(audio) / SAMPLE_RATE * 1000
            print(f"  {word}: {' '.join(phonemes)} [{source}] → {duration_ms:.0f}ms")
    
    if verbose:
        print(f"\nDemos saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    print("Testing speaker training...")
    print("=" * 60)
    
    # Test formant bootstrap training
    config = SpeakerTrainingConfig(
        mode='formant-bootstrap',
        n_iterations=20,  # Reduced for testing
        seed=42,
    )
    
    # Train subset
    test_phonemes = ['K', 'AE', 'T', 'M', 'IY', 'D', 'AO', 'G']
    
    swarm = train_speaker_swarm(
        output_path="/tmp/test_speaker_swarm.npz",
        config=config,
        phonemes=test_phonemes,
        verbose=True,
    )
    
    # Test synthesis
    print("\nTesting synthesis:")
    for word, phones in [('cat', ['K', 'AE', 'T']), ('dog', ['D', 'AO', 'G'])]:
        audio = swarm.synthesize_sequence(phones)
        duration_ms = len(audio) / SAMPLE_RATE * 1000
        print(f"  {word}: {' '.join(phones)} → {len(audio)} samples ({duration_ms:.0f}ms)")
        save_wav(audio, f"/tmp/speaker_test_{word}.wav", SAMPLE_RATE)
    
    print("\n" + "=" * 60)
    print("Speaker training tests complete!")
