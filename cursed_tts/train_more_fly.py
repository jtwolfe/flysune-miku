"""
Training for MORE FLY enhanced swarm.

Implements:
- Stage B: Full CMUdict support, hard-word curriculum with fixed oversample
- Integration with Stage A cues, Stage C wiring, Stage D DAN teaching
- Ablation configurations for evaluation
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import time
import json

from .more_fly import (
    MoreFlySwarm, MoreFlyConfig, CueConfig, WiringConfig, DANConfig,
    verify_wiring_is_not_random, verify_kc_mbon_only_plasticity,
)
from .alignment import create_g2p_training_data, AlignedPair
from .phonemes import PHONEME_LIST, strip_stress
from .lexicon import get_phonemes, get_demo_words, get_lexicon
from .confusion_mining import (
    ConfusionMiner, get_known_hard_targets, KNOWN_HARD_PAIRS,
)


# =============================================================================
# Stage B: Full CMUdict and Hard-Word Curriculum
# =============================================================================

def create_full_training_data(
    max_words: int = 0,  # 0 or 'all' = full CMUdict
    context_size: int = 3,
    seed: int = 42,
    test_split: float = 0.1,
    verbose: bool = True,
) -> Tuple[List[AlignedPair], List[AlignedPair], List[str], List[str]]:
    """
    Create training data with support for full CMUdict.
    
    Args:
        max_words: Max words (0 = all available)
        context_size: Letter context window
        seed: Random seed for reproducibility
        test_split: Fraction for test set
        verbose: Print progress
    
    Returns:
        train_pairs, test_pairs, train_words, test_words
    """
    lexicon = get_lexicon()
    
    # Filter to valid words
    valid_words = {
        w: p for w, p in lexicon.items()
        if w.isalpha() and 2 <= len(w) <= 15 and 1 <= len(p) <= 15
    }
    
    total_available = len(valid_words)
    
    if max_words == 0 or max_words >= total_available:
        # Use all available words
        max_words = total_available
        if verbose:
            print(f"Using full CMUdict: {max_words} words")
    
    return create_g2p_training_data(
        max_words=max_words,
        context_size=context_size,
        seed=seed,
        test_split=test_split,
        include_demo=True,
        verbose=verbose,
    )


def apply_hard_word_curriculum(
    pairs: List[AlignedPair],
    swarm: MoreFlySwarm = None,
    use_known_hard: bool = True,
    max_oversample_ratio: float = 0.15,  # Fixed from PR#3
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> List[AlignedPair]:
    """
    Apply hard-word curriculum with fixed oversample from PR#3.
    
    This uses the target-only oversampling fix from confusion_mining.py
    that avoids flooding with common phonemes.
    
    Args:
        pairs: Original training pairs
        swarm: Optional swarm for prediction-based filtering
        use_known_hard: Use known hard pairs (IY↔EH, AE↔AA, IY↔UW, etc.)
        max_oversample_ratio: Cap on oversample ratio (default 15%)
        rng: Random generator
        verbose: Print progress
    
    Returns:
        Augmented training pairs
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    target_to_confused = get_known_hard_targets() if use_known_hard else {}
    
    if not target_to_confused:
        return pairs
    
    # Identify hard samples (target phoneme is in known hard pairs)
    hard_indices = []
    for i, pair in enumerate(pairs):
        target = strip_stress(pair.phoneme)
        if target in target_to_confused:
            hard_indices.append(i)
    
    hard_fraction = len(hard_indices) / len(pairs) if pairs else 0
    
    if verbose:
        print(f"Hard-word curriculum:")
        print(f"  Known hard pairs: {len(KNOWN_HARD_PAIRS)}")
        print(f"  Hard samples: {len(hard_indices)}/{len(pairs)} ({100*hard_fraction:.1f}%)")
    
    # Cap oversample
    max_duplicates = int(len(pairs) * max_oversample_ratio)
    n_duplicates = min(len(hard_indices), max_duplicates)
    
    if n_duplicates > 0:
        dup_indices = rng.choice(hard_indices, size=n_duplicates, replace=False)
        duplicates = [pairs[i] for i in dup_indices]
    else:
        duplicates = []
    
    if verbose:
        oversample_ratio = len(duplicates) / len(pairs) if pairs else 0
        print(f"  Duplicates added: {len(duplicates)} ({100*oversample_ratio:.1f}%)")
    
    # Combine and shuffle
    augmented = list(pairs) + duplicates
    indices = rng.permutation(len(augmented))
    augmented = [augmented[i] for i in indices]
    
    return augmented


def apply_short_word_curriculum(
    pairs: List[AlignedPair],
    max_phonemes: int = 4,  # Increased to capture more short words
    oversample_ratio: float = 0.15,  # 15% extra short words
    target_phonemes: Set[str] = None,  # Specific phonemes to boost (e.g., IY for "me")
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> List[AlignedPair]:
    """
    Apply short-word curriculum to boost short words in training.
    
    This helps recover accuracy on short words like "me" when using
    non-random wiring (hemibrain) which can have different expansion
    geometry that hurts short-word discrimination.
    
    Args:
        pairs: Original training pairs
        max_phonemes: Maximum phoneme count to consider "short" (default: 4)
        oversample_ratio: Fraction of corpus to add as short-word duplicates
        target_phonemes: If set, only oversample short words with these phonemes
        rng: Random generator
        verbose: Print progress
    
    Returns:
        Augmented training pairs
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Default: all common short-word vowels plus schwa (null if broad coverage)
    if target_phonemes is None:
        target_phonemes = None  # None = all phonemes in short words
    
    # Identify short-word samples
    short_indices = []
    for i, pair in enumerate(pairs):
        if pair.n_phonemes <= max_phonemes:
            if target_phonemes is None:
                short_indices.append(i)
            else:
                target = strip_stress(pair.phoneme)
                if target in target_phonemes:
                    short_indices.append(i)
    
    short_fraction = len(short_indices) / len(pairs) if pairs else 0
    
    if verbose:
        print(f"Short-word curriculum:")
        print(f"  Max phonemes: {max_phonemes}")
        print(f"  Target phonemes: {sorted(target_phonemes) if target_phonemes else 'all'}")
        print(f"  Short-word samples: {len(short_indices)}/{len(pairs)} ({100*short_fraction:.1f}%)")
    
    # Calculate duplicates
    max_duplicates = int(len(pairs) * oversample_ratio)
    n_duplicates = min(len(short_indices), max_duplicates)
    
    if n_duplicates > 0 and short_indices:
        # Sample with replacement if needed
        replace = n_duplicates > len(short_indices)
        dup_indices = rng.choice(short_indices, size=n_duplicates, replace=replace)
        duplicates = [pairs[i] for i in dup_indices]
    else:
        duplicates = []
    
    if verbose:
        actual_ratio = len(duplicates) / len(pairs) if pairs else 0
        print(f"  Duplicates added: {len(duplicates)} ({100*actual_ratio:.1f}%)")
    
    # Combine and shuffle
    augmented = list(pairs) + duplicates
    indices = rng.permutation(len(augmented))
    augmented = [augmented[i] for i in indices]
    
    return augmented


# =============================================================================
# Ablation Configurations
# =============================================================================

def get_baseline_config(seed: int = 42) -> MoreFlyConfig:
    """Get baseline configuration (equivalent to main branch)."""
    return MoreFlyConfig(
        cue_config=CueConfig(
            use_position_features=True,   # From PR#3
            use_previous_phone=False,     # Baseline: no prev phone
            use_focus_features=False,     # Baseline: no focus
        ),
        wiring_config=WiringConfig(
            mode='random',                # Baseline: random wiring
            seed=seed,
        ),
        dan_config=DANConfig(
            enabled=False,                # Baseline: standard learning
        ),
        seed=seed,
    )


def get_stage_a_config(seed: int = 42) -> MoreFlyConfig:
    """Get Stage A configuration (+cues)."""
    return MoreFlyConfig(
        cue_config=CueConfig(
            use_position_features=True,
            use_previous_phone=True,      # Stage A: add prev phone
            use_focus_features=True,      # Stage A: add focus
        ),
        wiring_config=WiringConfig(
            mode='random',
            seed=seed,
        ),
        dan_config=DANConfig(
            enabled=False,
        ),
        seed=seed,
    )


def get_stage_ab_config(seed: int = 42) -> MoreFlyConfig:
    """Get Stage A+B configuration (+cues, +data via curriculum)."""
    # Same as Stage A, data curriculum applied in training
    return get_stage_a_config(seed)


def get_recommended_config(seed: int = 42) -> MoreFlyConfig:
    """Get the recommended default configuration.
    
    This is Stage A+B with random wiring:
    - Stage A cues: previous-phone, focus features, enhanced position
    - Stage B curriculum: hard-word oversampling (IY↔EH, AE↔AA, IY↔UW)
    - Random wiring: simpler and matches or exceeds hemibrain on held-out
    
    Why not hemibrain wiring?
    - Hemibrain wiring is experimental (requires init_seed tuning)
    - Current ablations show random+data ≥ hemibrain on held-out accuracy
    - Stats-matched fallback, not real synapses (extraction impractical)
    
    Use +A+B+C_wiring or +A+B+C+D_full with --wiring flywire/hemibrain
    to experiment with bio-faithful wiring once real synapse data is available.
    """
    return get_stage_a_config(seed)


def get_stage_abc_config(seed: int = 42, wiring_mode: str = 'flywire') -> MoreFlyConfig:
    """Get Stage A+B+C configuration (+cues, +data, +wiring).
    
    Note: For real hemibrain wiring, we use n_pn=400 to preserve connectivity structure.
    Subsampling to 180 PNs loses too much of the real connectivity.
    """
    # For real hemibrain wiring, use more PNs to preserve structure
    is_real_wiring = wiring_mode in ('flywire', 'hemibrain')
    n_pn = 400 if is_real_wiring else 180
    
    return MoreFlyConfig(
        cue_config=CueConfig(
            use_position_features=True,
            use_previous_phone=True,
            use_focus_features=True,
        ),
        wiring_config=WiringConfig(
            mode=wiring_mode,             # Stage C: real wiring
            n_pn=n_pn,                    # Use 400 PNs for real hemibrain
            seed=seed,                    # PN→KC seed
        ),
        dan_config=DANConfig(
            enabled=False,
        ),
        seed=seed,                        # KC→MBON init seed
    )


def get_full_config(seed: int = 42, wiring_mode: str = 'flywire') -> MoreFlyConfig:
    """Get full Stage A+B+C+D configuration.
    
    Note: For real hemibrain wiring, we use n_pn=400 to preserve connectivity structure.
    """
    is_real_wiring = wiring_mode in ('flywire', 'hemibrain')
    n_pn = 400 if is_real_wiring else 180
    
    return MoreFlyConfig(
        cue_config=CueConfig(
            use_position_features=True,
            use_previous_phone=True,
            use_focus_features=True,
        ),
        wiring_config=WiringConfig(
            mode=wiring_mode,
            n_pn=n_pn,                    # Use 400 PNs for real hemibrain
            seed=seed,                    # PN→KC seed
        ),
        dan_config=DANConfig(
            enabled=True,                 # Stage D: DAN teaching
            compartment_teaching=True,
            word_reward_modulation=False,
        ),
        seed=seed,                        # KC→MBON init seed
    )


ABLATION_CONFIGS = {
    'baseline': get_baseline_config,
    '+A_cues': get_stage_a_config,
    '+A+B_data': get_stage_ab_config,
    '+A+B+C_wiring': get_stage_abc_config,
    '+A+B+C+D_full': get_full_config,
}


# =============================================================================
# Training Functions
# =============================================================================

def train_more_fly_epoch(
    swarm: MoreFlySwarm,
    train_pairs: List[AlignedPair],
    rng: np.random.Generator,
    use_teacher_forcing: bool = True,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Train MORE FLY swarm for one epoch.
    
    Args:
        swarm: MoreFlySwarm instance
        train_pairs: Training pairs
        rng: Random generator
        use_teacher_forcing: Use reference phonemes for prev-phone cue
        verbose: Print progress
    
    Returns:
        ensemble_accuracy, specialist_accuracies
    """
    indices = rng.permutation(len(train_pairs))
    
    # Reset specialist stats
    for specialist in swarm.specialists.values():
        specialist.reset_stats()
    
    correct = 0
    total = 0
    
    # Group by word for teacher-forcing
    word_to_pairs: Dict[str, List[Tuple[int, AlignedPair]]] = {}
    for i, idx in enumerate(indices):
        pair = train_pairs[idx]
        if pair.word not in word_to_pairs:
            word_to_pairs[pair.word] = []
        word_to_pairs[pair.word].append((i, pair))
    
    # Sort pairs within each word by phoneme position
    for word in word_to_pairs:
        word_to_pairs[word].sort(key=lambda x: x[1].phoneme_idx)
    
    # Flatten back with word grouping intact
    ordered_pairs = []
    for word, wpairs in word_to_pairs.items():
        ordered_pairs.extend(wpairs)
    
    # Train with teacher forcing (previous phoneme from reference)
    prev_word = None
    prev_phoneme = None
    
    for i, (orig_idx, pair) in enumerate(ordered_pairs):
        # Get previous phoneme
        if pair.word != prev_word:
            prev_phoneme = None  # New word, no previous
        
        # Train step with previous phone cue
        is_correct, predicted = swarm.train_step(
            pair.letter_context,
            pair.phoneme,
            phoneme_pos=pair.phoneme_pos,
            n_phonemes=pair.n_phonemes,
            previous_phone=prev_phoneme if use_teacher_forcing else None,
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        # Update previous
        prev_word = pair.word
        prev_phoneme = pair.phoneme  # Teacher forcing: use reference
        
        if verbose and (total % 5000 == 0):
            print(f"  {total}/{len(train_pairs)}: acc={100*correct/total:.1f}%")
    
    ensemble_acc = correct / max(total, 1)
    specialist_accs = {p: s.get_accuracy() for p, s in swarm.specialists.items()}
    
    return ensemble_acc, specialist_accs


def evaluate_more_fly_pairs(
    swarm: MoreFlySwarm,
    pairs: List[AlignedPair],
) -> Tuple[float, Dict[str, Tuple[int, int]]]:
    """Evaluate ensemble accuracy on aligned pairs."""
    correct = 0
    total = 0
    per_phoneme: Dict[str, List[int]] = {p: [0, 0] for p in swarm.phoneme_list}
    
    # Group by word for autoregressive decoding
    word_to_pairs: Dict[str, List[AlignedPair]] = {}
    for pair in pairs:
        if pair.word not in word_to_pairs:
            word_to_pairs[pair.word] = []
        word_to_pairs[pair.word].append(pair)
    
    for word, wpairs in word_to_pairs.items():
        wpairs.sort(key=lambda x: x.phoneme_idx)
        
        prev_predicted = None
        for pair in wpairs:
            target = strip_stress(pair.phoneme)
            
            predicted, _, _ = swarm.predict(
                pair.letter_context,
                phoneme_pos=pair.phoneme_pos,
                n_phonemes=pair.n_phonemes,
                previous_phone=prev_predicted,  # Autoregressive
            )
            
            is_correct = (predicted == target)
            if is_correct:
                correct += 1
            total += 1
            
            if target in per_phoneme:
                per_phoneme[target][1] += 1
                if is_correct:
                    per_phoneme[target][0] += 1
            
            prev_predicted = predicted
    
    return correct / max(total, 1), {k: tuple(v) for k, v in per_phoneme.items()}


def evaluate_more_fly_words(
    swarm: MoreFlySwarm,
    words: List[str],
    lexicon: Dict[str, List[str]],
    verbose: bool = False,
) -> Tuple[float, float, List[Dict]]:
    """Evaluate on whole words."""
    total_phonemes = 0
    correct_phonemes = 0
    correct_words = 0
    details = []
    
    for word in words:
        if word not in lexicon:
            continue
        
        ref_phonemes = [strip_stress(p) for p in lexicon[word]]
        n_phonemes = len(ref_phonemes)
        
        # Predict with autoregressive previous-phone cue
        pred_phonemes = swarm.predict_word(word, n_phonemes)
        pred_phonemes = [strip_stress(p) for p in pred_phonemes]
        
        word_correct = 0
        for ref, pred in zip(ref_phonemes, pred_phonemes):
            total_phonemes += 1
            if ref == pred:
                correct_phonemes += 1
                word_correct += 1
        
        is_word_correct = (word_correct == n_phonemes)
        if is_word_correct:
            correct_words += 1
        
        details.append({
            'word': word,
            'reference': ref_phonemes,
            'predicted': pred_phonemes,
            'phoneme_acc': word_correct / n_phonemes if n_phonemes > 0 else 0,
            'is_correct': is_word_correct,
        })
        
        if verbose:
            ref_str = ' '.join(ref_phonemes)
            pred_str = ' '.join(pred_phonemes)
            status = "✓" if is_word_correct else ("~" if word_correct > 0 else "✗")
            print(f"{word:15} {ref_str:25} {pred_str:25} {status}")
    
    n_words = len(details)
    phoneme_acc = correct_phonemes / max(total_phonemes, 1)
    word_acc = correct_words / max(n_words, 1)
    
    return phoneme_acc, word_acc, details


def evaluate_confusion_pairs(
    swarm: MoreFlySwarm,
    pairs: List[AlignedPair],
    known_pairs: List[Tuple[str, str]] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Evaluate confusion rates for specific phoneme pairs.
    
    Returns dict mapping (target, predicted) to confusion rate.
    """
    if known_pairs is None:
        known_pairs = KNOWN_HARD_PAIRS
    
    # Count confusions
    confusion_counts = {(t, p): 0 for t, p in known_pairs}
    target_counts = {t: 0 for t, _ in known_pairs}
    
    # Group by word for autoregressive
    word_to_pairs: Dict[str, List[AlignedPair]] = {}
    for pair in pairs:
        if pair.word not in word_to_pairs:
            word_to_pairs[pair.word] = []
        word_to_pairs[pair.word].append(pair)
    
    for word, wpairs in word_to_pairs.items():
        wpairs.sort(key=lambda x: x.phoneme_idx)
        
        prev_predicted = None
        for pair in wpairs:
            target = strip_stress(pair.phoneme)
            
            predicted, _, _ = swarm.predict(
                pair.letter_context,
                phoneme_pos=pair.phoneme_pos,
                n_phonemes=pair.n_phonemes,
                previous_phone=prev_predicted,
            )
            
            if target in target_counts:
                target_counts[target] += 1
                if (target, predicted) in confusion_counts:
                    confusion_counts[(target, predicted)] += 1
            
            prev_predicted = predicted
    
    # Compute rates
    rates = {}
    for (target, pred), count in confusion_counts.items():
        total = target_counts.get(target, 0)
        rates[(target, pred)] = count / max(total, 1)
    
    return rates


def train_more_fly(
    config: MoreFlyConfig = None,
    max_words: int = 10000,
    n_epochs: int = 10,
    seed: int = 42,
    use_hard_curriculum: bool = False,  # Stage B: hard phoneme pairs
    use_short_word_curriculum: bool = None,  # Short-word boost (auto for non-random wiring)
    early_stop_patience: int = 0,
    min_epochs_before_stop: int = 4,  # Don't early-stop before this epoch
    save_best: bool = True,
    best_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[MoreFlySwarm, Dict]:
    """
    Train MORE FLY swarm.
    
    Args:
        config: MoreFlyConfig (default: full config)
        max_words: Max training words (0 = all)
        n_epochs: Training epochs
        seed: Random seed
        use_hard_curriculum: Apply Stage B hard-word curriculum (IY↔EH, AE↔AA, IY↔UW, etc.)
        use_short_word_curriculum: Boost short words (auto-enabled for non-random wiring)
        early_stop_patience: Early stop patience (0 = disabled)
        min_epochs_before_stop: Don't early-stop before this epoch (helps IY recover)
        save_best: Save best checkpoint
        best_path: Path for best checkpoint
        verbose: Print progress
    
    Returns:
        Trained swarm and history
    """
    rng = np.random.default_rng(seed)
    
    if config is None:
        config = get_full_config(seed)
    
    # Auto-enable short-word curriculum for non-random wiring
    # (hemibrain geometry can hurt short-word discrimination, e.g., "me" IY→UW)
    is_real_wiring = config.wiring_config.mode in ('flywire', 'hemibrain')
    if use_short_word_curriculum is None:
        use_short_word_curriculum = is_real_wiring
    
    # Create training data
    if verbose:
        print("Creating training data...")
    
    train_pairs, test_pairs, train_words, test_words = create_full_training_data(
        max_words=max_words,
        context_size=config.context_size,
        seed=seed,
        verbose=verbose,
    )
    
    # Apply hard-word curriculum if enabled (Stage B: hard phoneme pairs)
    if use_hard_curriculum:
        train_pairs = apply_hard_word_curriculum(
            train_pairs,
            use_known_hard=True,
            max_oversample_ratio=0.15,
            rng=rng,
            verbose=verbose,
        )
    
    # Apply short-word curriculum for non-random wiring
    # This helps recover accuracy on short words under hemibrain expansion geometry
    if use_short_word_curriculum:
        train_pairs = apply_short_word_curriculum(
            train_pairs,
            max_phonemes=4,  # Include more short words
            oversample_ratio=0.10,  # 10% extra short words
            target_phonemes=None,  # All phonemes in short words
            rng=rng,
            verbose=verbose,
        )
    
    # Create swarm
    swarm = MoreFlySwarm(config)
    
    # Get evaluation data
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    if verbose:
        print(f"\n{'='*60}")
        print("MORE FLY TRAINING")
        print(f"{'='*60}")
        print(f"Specialists: {len(swarm.specialists)}")
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Epochs: {n_epochs}")
        print(f"Wiring: {swarm.get_wiring_metadata()['source']}")
        print(f"DAN teaching: {config.dan_config.enabled}")
        print(f"{'='*60}\n")
    
    history = {
        'train_acc': [],
        'test_acc': [],
        'demo_phoneme_acc': [],
        'demo_word_acc': [],
        'best_epoch': None,
        'best_demo_phoneme': 0.0,
        'wiring_metadata': swarm.get_wiring_metadata(),
    }
    
    best_demo_phoneme = 0.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    
    start_time = time.time()
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_acc, _ = train_more_fly_epoch(
            swarm, train_pairs, rng,
            use_teacher_forcing=True,
            verbose=verbose,
        )
        
        # Evaluate on test pairs
        test_acc, _ = evaluate_more_fly_pairs(swarm, test_pairs)
        
        # Evaluate on demo words
        demo_phoneme_acc, demo_word_acc, _ = evaluate_more_fly_words(
            swarm, demo_in_lex, lexicon
        )
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['demo_phoneme_acc'].append(demo_phoneme_acc)
        history['demo_word_acc'].append(demo_word_acc)
        
        # Check for best
        is_best = False
        if demo_phoneme_acc > best_demo_phoneme:
            best_demo_phoneme = demo_phoneme_acc
            best_epoch = epoch
            best_weights = {
                p: s.kc_mbon_weights.copy()
                for p, s in swarm.specialists.items()
            }
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        if verbose:
            best_marker = " ★ BEST" if is_best else ""
            print(f"\nEpoch {epoch:2d}/{n_epochs}: "
                  f"train={100*train_acc:.1f}%, "
                  f"test={100*test_acc:.1f}%, "
                  f"demo_ph={100*demo_phoneme_acc:.1f}%, "
                  f"demo_w={100*demo_word_acc:.1f}% "
                  f"[{epoch_time:.0f}s]{best_marker}")
        
        # Early stopping (with minimum epoch threshold to let IY recover)
        if (early_stop_patience > 0 
            and epochs_without_improvement >= early_stop_patience
            and epoch >= min_epochs_before_stop):
            if verbose:
                print(f"\n⚠ Early stopping: no improvement for {early_stop_patience} epochs")
            break
    
    # Record best
    history['best_epoch'] = best_epoch
    history['best_demo_phoneme'] = best_demo_phoneme
    
    # Restore best weights
    if best_weights is not None and best_epoch < n_epochs:
        if verbose:
            print(f"\n✓ Restoring best weights from epoch {best_epoch}")
        for p, w in best_weights.items():
            if p in swarm.specialists:
                swarm.specialists[p].kc_mbon_weights = w.copy()
    
    # Save best checkpoint
    if save_best and best_weights is not None:
        checkpoint_path = best_path or "model_more_fly_best.npz"
        swarm.save(checkpoint_path)
        if verbose:
            print(f"✓ Best checkpoint saved to {checkpoint_path}")
    
    return swarm, history


def run_ablation(
    config_name: str = 'baseline',
    max_words: int = 5000,
    n_epochs: int = 6,
    seed: int = 42,
    output_dir: str = 'artifacts/eval/more_fly',
    verbose: bool = True,
) -> Dict:
    """
    Run a single ablation configuration.
    
    Returns evaluation results.
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    
    # Get config
    config_fn = ABLATION_CONFIGS[config_name]
    if config_name in ['+A+B+C_wiring', '+A+B+C+D_full']:
        config = config_fn(seed, wiring_mode='flywire')
    else:
        config = config_fn(seed)
    
    # Determine if we should use hard curriculum (Stage B)
    use_hard = config_name in ['+A+B_data', '+A+B+C_wiring', '+A+B+C+D_full']
    
    # Determine if we should use short-word curriculum
    # Auto-enabled for non-random wiring configs to help recover "me" (IY)
    is_real_wiring = config_name in ['+A+B+C_wiring', '+A+B+C+D_full']
    use_short = is_real_wiring  # Auto for hemibrain wiring
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ABLATION: {config_name}")
        print(f"{'='*60}")
    
    # For hemibrain wiring: train longer to let short-word discrimination develop
    # The different expansion geometry needs more epochs to stabilize IY-vs-UW
    actual_epochs = n_epochs + 2 if is_real_wiring else n_epochs
    
    # Disable early stopping for hemibrain wiring to ensure full training
    patience = 0 if is_real_wiring else 3  # 0 = no early stop
    min_epochs = actual_epochs  # Train full
    
    swarm, history = train_more_fly(
        config=config,
        max_words=max_words,
        n_epochs=actual_epochs,
        seed=seed,
        use_hard_curriculum=use_hard,
        use_short_word_curriculum=use_short,
        early_stop_patience=patience,
        min_epochs_before_stop=min_epochs,
        verbose=verbose,
    )
    
    # Final evaluation
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    # Build held-out test sample
    all_words = list(lexicon.keys())
    rng = np.random.default_rng(123)
    test_sample = [
        w for w in rng.choice(all_words, size=500, replace=False)
        if w not in demo_words and w.isalpha() and 2 <= len(w) <= 12
    ][:250]
    
    # Get training data for confusion evaluation
    train_pairs, test_pairs, _, _ = create_full_training_data(
        max_words=max_words,
        context_size=config.context_size,
        seed=seed,
        verbose=False,
    )
    
    # Evaluate
    demo_ph, demo_w, demo_details = evaluate_more_fly_words(swarm, demo_in_lex, lexicon)
    test_ph, test_w, _ = evaluate_more_fly_words(swarm, test_sample, lexicon)
    
    # Specific word checks
    me_pred = swarm.predict_word('me', 2) if 'me' in lexicon else ['?', '?']
    yes_pred = swarm.predict_word('yes', 3) if 'yes' in lexicon else ['?', '?', '?']
    hi_pred = swarm.predict_word('hi', 2) if 'hi' in lexicon else ['?', '?']
    
    # Confusion rates
    confusion_rates = evaluate_confusion_pairs(swarm, test_pairs)
    iy_eh_rate = confusion_rates.get(('IY', 'EH'), 0)
    ae_aa_rate = confusion_rates.get(('AE', 'AA'), 0)
    
    results = {
        'config': config_name,
        'demo_phoneme': demo_ph,
        'demo_word': demo_w,
        'test_phoneme': test_ph,
        'test_word': test_w,
        'me': ' '.join(me_pred),
        'yes': ' '.join(yes_pred),
        'hi': ' '.join(hi_pred),
        'IY_EH': iy_eh_rate,
        'AE_AA': ae_aa_rate,
        'wiring': swarm.get_wiring_metadata()['source'],
        'best_epoch': history['best_epoch'],
    }
    
    # Save swarm
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    swarm_path = f"{output_dir}/swarm_{config_name.replace('+', '_').replace(' ', '_')}.npz"
    swarm.save(swarm_path)
    
    # Save results
    results_path = f"{output_dir}/results_{config_name.replace('+', '_').replace(' ', '_')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_all_ablations(
    max_words: int = 5000,
    n_epochs: int = 6,
    seed: int = 42,
    output_dir: str = 'artifacts/eval/more_fly',
    verbose: bool = True,
) -> List[Dict]:
    """
    Run all ablation configurations and produce comparison table.
    """
    results = []
    
    for config_name in ABLATION_CONFIGS:
        try:
            r = run_ablation(
                config_name=config_name,
                max_words=max_words,
                n_epochs=n_epochs,
                seed=seed,
                output_dir=output_dir,
                verbose=verbose,
            )
            results.append(r)
        except Exception as e:
            print(f"Error in {config_name}: {e}")
            results.append({'config': config_name, 'error': str(e)})
    
    # Print table
    print("\n" + "="*100)
    print("ABLATION RESULTS")
    print("="*100)
    print(f"{'Config':<20} {'Demo Ph':>8} {'Demo W':>8} {'Test Ph':>8} {'Test W':>8} "
          f"{'me':>8} {'yes':>10} {'hi':>8} {'IY→EH':>8} {'AE→AA':>8}")
    print("-"*100)
    
    for r in results:
        if 'error' in r:
            print(f"{r['config']:<20} ERROR: {r['error']}")
        else:
            print(f"{r['config']:<20} "
                  f"{100*r['demo_phoneme']:>7.1f}% "
                  f"{100*r['demo_word']:>7.1f}% "
                  f"{100*r['test_phoneme']:>7.1f}% "
                  f"{100*r['test_word']:>7.1f}% "
                  f"{r['me']:>8} "
                  f"{r['yes']:>10} "
                  f"{r['hi']:>8} "
                  f"{100*r['IY_EH']:>7.1f}% "
                  f"{100*r['AE_AA']:>7.1f}%")
    
    print("="*100)
    
    # Save combined results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined_path = f"{output_dir}/ablation_results.json"
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {combined_path}")
    
    return results


def format_ablation_table(results: List[Dict]) -> str:
    """Format ablation results as markdown table."""
    lines = [
        "| Config | Demo Ph | Demo W | Test Ph | Test W | me | yes | hi | IY→EH | AE→AA |",
        "|--------|---------|--------|---------|--------|----|----|-----|-------|-------|",
    ]
    
    for r in results:
        if 'error' in r:
            lines.append(f"| {r['config']} | ERROR | - | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {r['config']} | "
                f"{100*r['demo_phoneme']:.1f}% | "
                f"{100*r['demo_word']:.1f}% | "
                f"{100*r['test_phoneme']:.1f}% | "
                f"{100*r['test_word']:.1f}% | "
                f"{r['me']} | "
                f"{r['yes']} | "
                f"{r['hi']} | "
                f"{100*r['IY_EH']:.1f}% | "
                f"{100*r['AE_AA']:.1f}% |"
            )
    
    return '\n'.join(lines)


if __name__ == "__main__":
    print("Testing MORE FLY training...")
    
    # Quick test with small data
    config = get_baseline_config(seed=42)
    swarm, history = train_more_fly(
        config=config,
        max_words=1000,
        n_epochs=2,
        verbose=True,
    )
    
    print(f"\nFinal demo_phoneme: {100*history['demo_phoneme_acc'][-1]:.1f}%")
    print(f"Best epoch: {history['best_epoch']}")
