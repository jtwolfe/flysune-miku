"""
Training loop for Acoustic Fly Head.

This module trains acoustic flies to produce voice from fly-native features.

Workflow:
1. Load or train a picker swarm (MORE FLY)
2. Build training targets from MARIAN or formant crumbs
3. For each (word, phoneme) pair:
   - Get KC activity from picker
   - Get target audio parameters from crumb
   - Train the responsible acoustic fly
4. Evaluate on demo words, save best checkpoint

The training is compartment-local: each acoustic fly only learns
from its target phoneme's examples.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time

from .phonemes import PHONEME_LIST, strip_stress, NUM_PHONEMES
from .lexicon import get_lexicon, get_phonemes, get_demo_words
from .acoustic_fly import (
    AcousticConfig, AcousticFlySwarm, VoiceParams,
    create_formant_targets, extract_voice_params, synthesize_from_params,
)
from .more_fly import MoreFlySwarm, MoreFlyConfig
from .synth import synthesize_phoneme, save_wav, SAMPLE_RATE


# =============================================================================
# Training Dataset Construction
# =============================================================================

def build_training_dataset(
    picker_swarm: MoreFlySwarm,
    max_words: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build training dataset: (kc_activity, phoneme, position) tuples.
    
    Args:
        picker_swarm: Trained MORE FLY picker swarm
        max_words: Maximum words to use
        seed: Random seed for word selection
    
    Returns:
        List of training examples with:
        - kc_activity: KC activity pattern from picker
        - phoneme: Target phoneme
        - position: Normalized position in word
        - letter_context: Letter context string
    """
    rng = np.random.default_rng(seed)
    lexicon = get_lexicon()
    
    all_words = list(lexicon.keys())
    rng.shuffle(all_words)
    
    if max_words > 0 and max_words < len(all_words):
        words = all_words[:max_words]
    else:
        words = all_words
    
    dataset = []
    phoneme_counts = {p: 0 for p in PHONEME_LIST}
    
    context_size = picker_swarm.config.context_size
    
    for word in words:
        phonemes = lexicon[word]
        phonemes = [strip_stress(p) for p in phonemes]
        n_phonemes = len(phonemes)
        n_letters = len(word)
        
        if n_phonemes == 0 or n_letters == 0:
            continue
        
        for p_idx, phoneme in enumerate(phonemes):
            if n_phonemes == 1:
                letter_pos = n_letters // 2
                phoneme_pos = 0.5
            else:
                letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
                phoneme_pos = p_idx / (n_phonemes - 1)
            letter_pos = max(0, min(letter_pos, n_letters - 1))
            
            context_chars = []
            for offset in range(-context_size, context_size + 1):
                idx = letter_pos + offset
                if 0 <= idx < n_letters:
                    context_chars.append(word[idx])
                else:
                    context_chars.append('_')
            letter_context = ''.join(context_chars)
            
            previous_phone = phonemes[p_idx - 1] if p_idx > 0 else None
            
            kc_activity = picker_swarm.shared.encode_to_kc(
                letter_context,
                phoneme_pos=phoneme_pos,
                n_phonemes=n_phonemes,
                previous_phone=previous_phone,
            )
            
            dataset.append({
                'kc_activity': kc_activity,
                'phoneme': phoneme,
                'position': phoneme_pos,
                'letter_context': letter_context,
                'word': word,
            })
            phoneme_counts[phoneme] = phoneme_counts.get(phoneme, 0) + 1
    
    if verbose:
        print(f"Built training dataset: {len(dataset)} examples from {len(words)} words")
        print(f"Phoneme distribution: min={min(phoneme_counts.values())}, "
              f"max={max(phoneme_counts.values())}, "
              f"mean={np.mean(list(phoneme_counts.values())):.1f}")
    
    return dataset


def try_load_marian_targets(
    config: AcousticConfig,
    verbose: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Try to load MARIAN voicebank targets.
    
    Returns None if MARIAN not available, in which case
    we fall back to formant targets.
    """
    try:
        from .voicebanks.utau_singer import (
            UTAUSingerSwarm, find_voicebank_path, load_utau_sample
        )
        from .voicebanks.arpasing_map import arpabet_to_arpasing
        
        marian_path = find_voicebank_path()
        if marian_path is None:
            if verbose:
                print("MARIAN voicebank not found, using formant targets")
            return None
        
        if verbose:
            print(f"Loading MARIAN targets from {marian_path}")
        
        from .voicebanks.utau_singer import find_voicebank_samples, load_utau_sample
        
        samples = find_voicebank_samples(marian_path)
        if not samples:
            if verbose:
                print("No samples found in MARIAN, using formant targets")
            return None
        
        targets = {}
        for phoneme in config.phonemes:
            p = strip_stress(phoneme)
            arpasing = arpabet_to_arpasing(p)
            
            if arpasing in samples:
                wav_path, oto_entry = samples[arpasing]
                try:
                    audio = load_utau_sample(
                        wav_path,
                        target_sr=config.sample_rate,
                        oto_entry=oto_entry,
                    )
                    params = extract_voice_params(audio, config)
                    targets[p] = params.to_array()
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not load {p}: {e}")
        
        if len(targets) < 10:
            if verbose:
                print(f"Only {len(targets)} MARIAN samples loaded, using formant fallback")
            return None
        
        if verbose:
            print(f"Loaded {len(targets)} MARIAN targets")
        return targets
        
    except ImportError as e:
        if verbose:
            print(f"Voicebank module not available: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"Error loading MARIAN: {e}")
        return None


def get_targets(
    config: AcousticConfig,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Get training targets, preferring MARIAN over formant.
    
    Returns:
        targets: Dict mapping phoneme → target parameter array
        source: 'marian' or 'formant'
    """
    marian_targets = try_load_marian_targets(config, verbose)
    
    if marian_targets is not None:
        formant_targets = create_formant_targets(config.phonemes, config)
        for p in config.phonemes:
            ps = strip_stress(p)
            if ps not in marian_targets and ps in formant_targets:
                marian_targets[ps] = formant_targets[ps]
        return marian_targets, 'marian'
    
    if verbose:
        print("Using formant crumb targets (bootstrap mode)")
    targets = create_formant_targets(config.phonemes, config)
    return targets, 'formant'


# =============================================================================
# Training Loop
# =============================================================================

def train_acoustic_flies(
    acoustic_swarm: AcousticFlySwarm,
    dataset: List[Dict[str, Any]],
    targets: Dict[str, np.ndarray],
    n_epochs: int = 10,
    early_stop_patience: int = 3,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train acoustic flies on dataset.
    
    Args:
        acoustic_swarm: Acoustic fly swarm to train
        dataset: Training dataset from build_training_dataset
        targets: Target parameters from get_targets
        n_epochs: Number of training epochs
        early_stop_patience: Stop if no improvement for N epochs
    
    Returns:
        history: Training history with per-epoch metrics
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'demo_loss': [],
    }
    
    best_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    rng = np.random.default_rng(42)
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        for fly in acoustic_swarm.flies.values():
            fly.reset_stats()
        
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        
        total_loss = 0.0
        n_samples = 0
        
        for idx in indices:
            example = dataset[idx]
            phoneme = example['phoneme']
            
            if phoneme not in targets:
                continue
            
            loss = acoustic_swarm.train_phoneme(
                phoneme=phoneme,
                kc_activity=example['kc_activity'],
                target_params=targets[phoneme],
                position=example['position'],
            )
            total_loss += loss
            n_samples += 1
        
        avg_loss = total_loss / max(n_samples, 1)
        
        demo_loss = evaluate_demo_loss(acoustic_swarm, targets)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['demo_loss'].append(demo_loss)
        
        epoch_time = time.time() - epoch_start
        
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={avg_loss:.4f}, demo_loss={demo_loss:.4f} "
                  f"({epoch_time:.1f}s)")
        
        if demo_loss < best_loss:
            best_loss = demo_loss
            patience_counter = 0
            best_weights = save_weights(acoustic_swarm)
        else:
            patience_counter += 1
        
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break
    
    if best_weights is not None:
        restore_weights(acoustic_swarm, best_weights)
        if verbose:
            print(f"Restored best weights (demo_loss={best_loss:.4f})")
    
    return history


def evaluate_demo_loss(
    acoustic_swarm: AcousticFlySwarm,
    targets: Dict[str, np.ndarray],
) -> float:
    """Evaluate loss on demo phonemes."""
    demo_phonemes = ['K', 'AE', 'T', 'D', 'AO', 'G', 'M', 'IY', 'AH', 'S']
    
    total_loss = 0.0
    n_samples = 0
    
    kc_demo = np.zeros(acoustic_swarm.config.kc_dim, dtype=np.float32)
    kc_demo[::10] = 1.0
    
    for p in demo_phonemes:
        ps = strip_stress(p)
        if ps not in targets:
            continue
        
        cue = acoustic_swarm.get_phoneme_cue(ps, position=0.5)
        predicted, _ = acoustic_swarm.flies[ps].forward(kc_demo, cue)
        
        error = predicted - targets[ps]
        loss = np.mean(error ** 2)
        total_loss += loss
        n_samples += 1
    
    return total_loss / max(n_samples, 1)


def save_weights(swarm: AcousticFlySwarm) -> Dict[str, Dict[str, np.ndarray]]:
    """Save current weights for checkpointing."""
    weights = {}
    for p, fly in swarm.flies.items():
        weights[p] = {
            'W1': fly.W1.copy(),
            'b1': fly.b1.copy(),
            'W2': fly.W2.copy(),
            'b2': fly.b2.copy(),
        }
    return weights


def restore_weights(swarm: AcousticFlySwarm, weights: Dict[str, Dict[str, np.ndarray]]):
    """Restore saved weights."""
    for p, w in weights.items():
        if p in swarm.flies:
            swarm.flies[p].W1 = w['W1']
            swarm.flies[p].b1 = w['b1']
            swarm.flies[p].W2 = w['W2']
            swarm.flies[p].b2 = w['b2']


# =============================================================================
# Full Training Pipeline
# =============================================================================

def train_and_save_acoustic_flies(
    output_path: str = "model_acoustic.npz",
    picker_path: str = "artifacts/eval/more_fly/swarm__A_B_data.npz",
    max_words: int = 5000,
    n_epochs: int = 10,
    early_stop_patience: int = 3,
    kc_dim: int = 2000,
    hidden_dim: int = 128,
    learning_rate: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[AcousticFlySwarm, Dict[str, Any]]:
    """
    Full training pipeline for acoustic flies.
    
    1. Load picker swarm
    2. Build dataset
    3. Get targets (MARIAN or formant fallback)
    4. Train acoustic flies
    5. Save model
    
    Returns:
        swarm: Trained acoustic fly swarm
        info: Training info dict
    """
    if verbose:
        print("="*60)
        print("ACOUSTIC FLY HEAD TRAINING")
        print("="*60)
    
    if verbose:
        print(f"\n1. Loading picker swarm from {picker_path}...")
    
    picker_path = Path(picker_path)
    if not picker_path.exists():
        default_paths = [
            Path("artifacts/eval/more_fly/swarm__A_B_data.npz"),
            Path("model_more_fly.npz"),
            Path("model_swarm.npz"),
        ]
        for p in default_paths:
            if p.exists():
                picker_path = p
                break
        else:
            if verbose:
                print("No picker swarm found. Training minimal picker first...")
            picker_swarm = train_minimal_picker(max_words=min(max_words, 2000), seed=seed)
    
    if picker_path.exists():
        picker_swarm = MoreFlySwarm.load(str(picker_path))
    
    picker_kc_dim = picker_swarm.config.wiring_config.n_kc
    
    acoustic_config = AcousticConfig(
        kc_dim=picker_kc_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    
    if verbose:
        print(f"\n2. Building training dataset ({max_words} words)...")
    dataset = build_training_dataset(picker_swarm, max_words=max_words, seed=seed, verbose=verbose)
    
    if verbose:
        print(f"\n3. Loading training targets...")
    targets, target_source = get_targets(acoustic_config, verbose=verbose)
    
    if verbose:
        print(f"\n4. Creating acoustic fly swarm...")
    acoustic_swarm = AcousticFlySwarm(acoustic_config)
    
    if verbose:
        print(f"\n5. Training acoustic flies ({n_epochs} epochs)...")
    history = train_acoustic_flies(
        acoustic_swarm,
        dataset,
        targets,
        n_epochs=n_epochs,
        early_stop_patience=early_stop_patience,
        verbose=verbose,
    )
    
    acoustic_swarm.save(output_path)
    
    info = {
        'picker_path': str(picker_path),
        'max_words': max_words,
        'n_epochs': n_epochs,
        'target_source': target_source,
        'n_training_examples': len(dataset),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_demo_loss': history['demo_loss'][-1] if history['demo_loss'] else 0,
        'kc_dim': picker_kc_dim,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"  Model saved to: {output_path}")
        print(f"  Target source: {target_source}")
        print(f"  Training examples: {len(dataset)}")
        print(f"  Final demo loss: {info['final_demo_loss']:.4f}")
    
    return acoustic_swarm, info


def train_minimal_picker(max_words: int = 2000, seed: int = 42) -> MoreFlySwarm:
    """Train a minimal picker swarm if none exists."""
    from .train_more_fly import train_more_fly, get_stage_ab_config
    
    config = get_stage_ab_config(seed)
    swarm, _ = train_more_fly(
        config=config,
        max_words=max_words,
        n_epochs=4,
        seed=seed,
        verbose=False,
    )
    return swarm


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for train-acoustic-flies."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train acoustic fly head for speech production"
    )
    parser.add_argument('--output', '-o', type=str, default='model_acoustic.npz',
                        help='Output model path')
    parser.add_argument('--picker', type=str, 
                        default='artifacts/eval/more_fly/swarm__A_B_data.npz',
                        help='Path to picker swarm model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--words', type=int, default=5000,
                        help='Max training words')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stop patience')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    train_and_save_acoustic_flies(
        output_path=args.output,
        picker_path=args.picker,
        max_words=args.words,
        n_epochs=args.epochs,
        early_stop_patience=args.patience,
        hidden_dim=args.hidden,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
