"""
Training for the one-phoneme-per-fly specialist ensemble.

Each specialist fly learns to answer "is the phoneme /P/?" for its target phoneme.
Training data comes from aligned CMUdict, same as the single multi-class MB.

For specialist P:
- Positive examples: contexts aligned to phoneme P
- Negative examples: contexts aligned to other phonemes

The dopamine-when-wrong learning rule is the same: only update when wrong.

Confusion mining (Step 4):
- Mine confusions from predictions to identify hard phoneme pairs
- Oversample hard pairs during training (learn more when wrong on hard cases)
- Stay dopamine/error-driven - this is selective attention to frequent errors
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import time

from .specialist_fly import (
    FlySwarm, SwarmConfig, SpecialistConfig, get_demo_phonemes
)
from .alignment import create_g2p_training_data, AlignedPair
from .phonemes import PHONEME_LIST, strip_stress, NUM_PHONEMES
from .lexicon import get_phonemes, get_demo_words, get_lexicon
from .confusion_mining import (
    ConfusionMiner, ConfusionMatrix, build_confusion_matrix_from_swarm,
    get_known_hard_targets, KNOWN_HARD_PAIRS,
)


def train_swarm_epoch(
    swarm: FlySwarm,
    train_pairs: List[AlignedPair],
    rng: np.random.Generator,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Train swarm for one epoch.
    
    Returns:
        ensemble_accuracy: Fraction correct by ensemble vote
        specialist_accuracies: Per-phoneme binary accuracy
    """
    indices = rng.permutation(len(train_pairs))
    swarm.reset_stats()
    
    correct = 0
    total = 0
    
    for i, idx in enumerate(indices):
        pair = train_pairs[idx]
        # Pass phoneme position for short-word disambiguation
        is_correct, _ = swarm.train_step(
            pair.letter_context, pair.phoneme,
            phoneme_pos=pair.phoneme_pos,
            n_phonemes=pair.n_phonemes,
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(train_pairs)}: ensemble_acc={100*correct/total:.1f}%")
    
    ensemble_acc = correct / max(total, 1)
    specialist_accs = swarm.get_phoneme_accuracies()
    
    return ensemble_acc, specialist_accs


def evaluate_swarm_pairs(
    swarm: FlySwarm, pairs: List[AlignedPair]
) -> Tuple[float, Dict[str, Tuple[int, int]]]:
    """
    Evaluate ensemble accuracy on aligned pairs.
    
    Returns:
        accuracy: Overall accuracy
        per_phoneme: Dict of (correct, total) per target phoneme
    """
    correct = 0
    total = 0
    per_phoneme: Dict[str, List[int]] = {p: [0, 0] for p in swarm.phoneme_list}
    
    for pair in pairs:
        target = strip_stress(pair.phoneme)
        predicted, _, _ = swarm.predict(
            pair.letter_context,
            phoneme_pos=pair.phoneme_pos,
            n_phonemes=pair.n_phonemes,
        )
        
        is_correct = (predicted == target)
        if is_correct:
            correct += 1
        total += 1
        
        if target in per_phoneme:
            per_phoneme[target][1] += 1
            if is_correct:
                per_phoneme[target][0] += 1
    
    return correct / max(total, 1), {k: tuple(v) for k, v in per_phoneme.items()}


def evaluate_swarm_words(
    swarm: FlySwarm,
    words: List[str],
    lexicon: Dict[str, List[str]],
    verbose: bool = False,
    lm: Optional['PhonemeNGramLM'] = None,
    beam_width: int = 5,
    lm_weight: float = 0.3,
    top_k: int = 5,
    vote_strategy: Optional[str] = None,
) -> Tuple[float, float, List[Dict]]:
    """
    Evaluate swarm on whole words.
    
    Args:
        swarm: FlySwarm instance
        words: Words to evaluate
        lexicon: Word to phoneme mapping
        verbose: Print details
        lm: Optional PhonemeNGramLM for beam search
        beam_width: Beam width (if using LM)
        lm_weight: LM weight (if using LM)
        top_k: Top-k candidates per slot (if using LM)
        vote_strategy: Voting strategy for base scores
    
    Returns:
        phoneme_accuracy
        word_accuracy  
        details
    """
    total_phonemes = 0
    correct_phonemes = 0
    correct_words = 0
    details = []
    
    for word in words:
        if word not in lexicon:
            continue
        
        ref_phonemes = [strip_stress(p) for p in lexicon[word]]
        n_phonemes = len(ref_phonemes)
        
        # Predict - use beam search if LM provided
        if lm is not None:
            pred_phonemes = swarm.predict_word_beam(
                word, n_phonemes,
                lm=lm,
                beam_width=beam_width,
                lm_weight=lm_weight,
                top_k=top_k,
                vote_strategy=vote_strategy,
            )
        else:
            pred_phonemes = swarm.predict_word(word, n_phonemes, vote_strategy=vote_strategy)
        pred_phonemes = [strip_stress(p) for p in pred_phonemes]
        
        # Score
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


def _copy_swarm_weights(swarm: FlySwarm) -> Dict[str, np.ndarray]:
    """Copy all specialist weights for checkpointing."""
    return {p: s.kc_mbon_weights.copy() for p, s in swarm.specialists.items()}


def _restore_swarm_weights(swarm: FlySwarm, weights: Dict[str, np.ndarray]):
    """Restore specialist weights from checkpoint."""
    for p, w in weights.items():
        if p in swarm.specialists:
            swarm.specialists[p].kc_mbon_weights = w.copy()


def train_swarm(
    phonemes: Optional[List[str]] = None,
    max_words: int = 10000,
    n_epochs: int = 10,
    context_size: int = 3,
    seed: int = 42,
    vote_strategy: str = 'softmax',
    vote_temperature: float = 0.5,
    vote_margin: float = 0.1,
    early_stop_patience: int = 0,
    save_best: bool = True,
    best_path: Optional[str] = None,
    fit_calibration: bool = True,
    confusion_mine: bool = False,
    confusion_mine_epoch: int = 2,
    hard_negative_weight: float = 2.0,
    oversample_factor: float = 1.5,
    confusion_report_path: Optional[str] = None,
    use_known_hard_pairs: bool = False,
    verbose: bool = True,
) -> Tuple[FlySwarm, Dict]:
    """
    Train fly swarm on CMUdict alignments.
    
    Args:
        phonemes: List of phonemes to train (None = all 39)
        max_words: Maximum training words
        n_epochs: Training epochs
        context_size: Letter context window size
        seed: Random seed
        vote_strategy: Voting strategy ('argmax', 'softmax', 'margin', 'calibrated')
        vote_temperature: Temperature for softmax voting
        vote_margin: Minimum margin for margin voting
        early_stop_patience: Stop if demo_phoneme doesn't improve for N epochs (0=disabled)
        save_best: Save best checkpoint by demo_phoneme accuracy
        best_path: Path for best checkpoint (default: model_swarm_best.npz)
        fit_calibration: Fit Platt calibration on test pairs after training
        confusion_mine: Enable confusion mining (mine after epoch N, then oversample)
        confusion_mine_epoch: Epoch after which to mine confusions (default: 2)
        hard_negative_weight: Weight for hard negative pairs (default: 2.0)
        oversample_factor: How much to oversample hard pairs (default: 1.5 = 50% more)
        confusion_report_path: Path to save confusion report (default: None)
        use_known_hard_pairs: Use pre-defined known hard pairs without mining
        verbose: Print progress
    
    Returns:
        Trained FlySwarm and training history
    """
    rng = np.random.default_rng(seed)
    
    # Default to all phonemes
    if phonemes is None:
        phonemes = PHONEME_LIST.copy()
    
    # Create training data
    if verbose:
        print("Creating aligned G2P training data...")
    
    train_pairs, test_pairs, train_words, test_words = create_g2p_training_data(
        max_words=max_words,
        context_size=context_size,
        seed=seed,
        test_split=0.1,
        include_demo=True,
        verbose=verbose,
    )
    
    # Filter training pairs to only include phonemes we're training
    phoneme_set = set(strip_stress(p) for p in phonemes)
    train_pairs_filtered = [
        p for p in train_pairs if strip_stress(p.phoneme) in phoneme_set
    ]
    test_pairs_filtered = [
        p for p in test_pairs if strip_stress(p.phoneme) in phoneme_set
    ]
    
    if verbose:
        print(f"Training pairs (filtered): {len(train_pairs_filtered)}/{len(train_pairs)}")
        print(f"Test pairs (filtered): {len(test_pairs_filtered)}/{len(test_pairs)}")
    
    # Initialize confusion miner if enabled
    confusion_miner: Optional[ConfusionMiner] = None
    if confusion_mine or use_known_hard_pairs:
        confusion_miner = ConfusionMiner(
            phonemes=list(phoneme_set),
            max_oversample_ratio=0.15,  # Cap at 15% of corpus
            min_confusion_rate=0.05,    # Only include >5% confusion rate
        )
        
        # Pre-populate with known hard pairs if requested
        if use_known_hard_pairs:
            if verbose:
                print(f"Using {len(KNOWN_HARD_PAIRS)} known hard pairs for oversampling")
            confusion_miner.target_to_confused = get_known_hard_targets()
    
    # Track original training pairs for mining
    original_train_pairs = train_pairs_filtered.copy()
    confusion_mined = False  # Track if we've already mined
    
    # Create swarm with voting config
    spec_config = SpecialistConfig(context_size=context_size, seed=seed)
    swarm_config = SwarmConfig(
        specialist_config=spec_config,
        phonemes=phonemes,
        seed=seed,
        vote_strategy=vote_strategy,
        vote_temperature=vote_temperature,
        vote_margin=vote_margin,
    )
    swarm = FlySwarm(swarm_config)
    
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    if verbose:
        print(f"\n{'='*60}")
        print("FLY SWARM TRAINING (One Phoneme Per Fly)")
        print(f"{'='*60}")
        print(f"Specialists: {len(swarm.specialists)} flies")
        print(f"Phonemes: {', '.join(sorted(phonemes)[:10])}{'...' if len(phonemes) > 10 else ''}")
        print(f"Training pairs: {len(train_pairs_filtered)}")
        print(f"Epochs: {n_epochs}")
        print(f"Vote strategy: {vote_strategy} (temp={vote_temperature}, margin={vote_margin})")
        if early_stop_patience > 0:
            print(f"Early stop patience: {early_stop_patience} epochs")
        if save_best:
            print(f"Best checkpoint: enabled")
        if confusion_mine:
            print(f"Confusion mining: enabled (after epoch {confusion_mine_epoch}, "
                  f"weight={hard_negative_weight}, oversample={oversample_factor}x)")
        elif use_known_hard_pairs:
            print(f"Known hard pairs: enabled (weight={hard_negative_weight}, "
                  f"oversample={oversample_factor}x)")
        print(f"{'='*60}\n")
    
    history = {
        'ensemble_train_acc': [],
        'ensemble_test_acc': [],
        'demo_phoneme_acc': [],
        'demo_word_acc': [],
        'specialist_accs': [],
        'best_epoch': None,
        'best_demo_phoneme': 0.0,
        'confusion_mined': False,
        'confusion_mine_epoch': None,
        'hard_pairs_count': 0,
    }
    
    # Best checkpoint tracking
    best_demo_phoneme = 0.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    
    start_time = time.time()
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        # Confusion mining: mine after specified epoch, augment for remaining
        if confusion_mine and epoch == confusion_mine_epoch + 1 and not confusion_mined:
            if verbose:
                print(f"\n{'='*40}")
                print(f"CONFUSION MINING (after epoch {confusion_mine_epoch})")
                print(f"{'='*40}")
            
            # Mine confusions from test pairs (held-out data)
            confusion_miner.mine_confusions(
                swarm, test_pairs_filtered, vote_strategy=vote_strategy, verbose=verbose
            )
            
            # Check hard fraction BEFORE augmentation
            hard_frac_before = confusion_miner.get_hard_fraction(original_train_pairs)
            if verbose:
                print(f"  Hard fraction in train: {100*hard_frac_before:.1f}%")
            
            # Augment training data with TRUE hard negatives (capped at 15%)
            train_pairs_filtered = confusion_miner.augment_training_data(
                original_train_pairs, swarm=swarm, rng=rng, verbose=verbose
            )
            
            # Save confusion report if requested
            if confusion_report_path:
                confusion_miner.save_report(confusion_report_path)
            
            confusion_mined = True
            
            if verbose:
                print(f"{'='*40}\n")
        
        # Apply known hard pair oversampling if enabled (on first epoch)
        if use_known_hard_pairs and epoch == 1 and confusion_miner and not confusion_mined:
            if verbose:
                print("Applying known hard pair oversampling...")
                hard_frac = confusion_miner.get_hard_fraction(original_train_pairs)
                print(f"  Hard fraction in train: {100*hard_frac:.1f}%")
            train_pairs_filtered = confusion_miner.augment_training_data(
                original_train_pairs, rng=rng, verbose=verbose
            )
            confusion_mined = True
        
        # Train
        train_acc, spec_accs = train_swarm_epoch(
            swarm, train_pairs_filtered, rng, verbose=verbose
        )
        
        # Evaluate on test pairs
        test_acc, _ = evaluate_swarm_pairs(swarm, test_pairs_filtered)
        
        # Evaluate on demo words
        demo_phoneme_acc, demo_word_acc, _ = evaluate_swarm_words(
            swarm, demo_in_lex, lexicon
        )
        
        history['ensemble_train_acc'].append(train_acc)
        history['ensemble_test_acc'].append(test_acc)
        history['demo_phoneme_acc'].append(demo_phoneme_acc)
        history['demo_word_acc'].append(demo_word_acc)
        history['specialist_accs'].append(spec_accs)
        
        # Check for best demo performance
        is_best = False
        if demo_phoneme_acc > best_demo_phoneme:
            best_demo_phoneme = demo_phoneme_acc
            best_epoch = epoch
            best_weights = _copy_swarm_weights(swarm)
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        if verbose:
            avg_spec = np.mean(list(spec_accs.values()))
            best_marker = " ★ BEST" if is_best else ""
            print(f"\nEpoch {epoch:2d}/{n_epochs}: "
                  f"ensemble_train={100*train_acc:.1f}%, "
                  f"test={100*test_acc:.1f}%, "
                  f"demo_phoneme={100*demo_phoneme_acc:.1f}%, "
                  f"demo_word={100*demo_word_acc:.1f}%, "
                  f"avg_specialist={100*avg_spec:.1f}% "
                  f"[{epoch_time:.0f}s, total {total_time:.0f}s]{best_marker}")
        
        # Early stopping check
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            if verbose:
                print(f"\n⚠ Early stopping: no improvement in demo_phoneme for {early_stop_patience} epochs")
                print(f"  Best was epoch {best_epoch} with demo_phoneme={100*best_demo_phoneme:.1f}%")
            break
    
    # Record best epoch info
    history['best_epoch'] = best_epoch
    history['best_demo_phoneme'] = best_demo_phoneme
    
    # Record confusion mining info
    if confusion_mined and confusion_miner:
        history['confusion_mined'] = True
        history['confusion_mine_epoch'] = confusion_mine_epoch if confusion_mine else 0
        history['hard_pairs_count'] = len(confusion_miner.target_to_confused)
        if confusion_miner.last_stats:
            history['hard_fraction'] = confusion_miner.last_stats.get('hard_fraction', 0)
            history['oversample_ratio'] = confusion_miner.last_stats.get('oversample_ratio', 0)
        
        # Final confusion report
        if confusion_report_path and confusion_miner.confusion_matrix:
            # Build final confusion matrix on test pairs
            if verbose:
                print("\n" + "-" * 40)
                print("FINAL CONFUSION ANALYSIS")
                print("-" * 40)
            final_cm = build_confusion_matrix_from_swarm(
                swarm, test_pairs_filtered, vote_strategy=vote_strategy, verbose=verbose
            )
            # Append to report
            with open(confusion_report_path, 'a') as f:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("POST-TRAINING CONFUSION ANALYSIS\n")
                f.write("=" * 60 + "\n")
                f.write(final_cm.print_report(top_k=20, min_count=3))
            if verbose:
                print(f"Updated confusion report at {confusion_report_path}")
    
    # Restore best weights if we have them and training continued past best
    if best_weights is not None and best_epoch < n_epochs:
        if verbose:
            print(f"\n✓ Restoring best weights from epoch {best_epoch} "
                  f"(demo_phoneme={100*best_demo_phoneme:.1f}%)")
        _restore_swarm_weights(swarm, best_weights)
    
    # Fit calibration on held-out test pairs (per-MBON bias/gain normalization)
    if fit_calibration and len(test_pairs_filtered) > 0:
        if verbose:
            print("\n" + "-" * 40)
            print("CALIBRATION (per-phoneme Platt scaling)")
            print("-" * 40)
        swarm.fit_calibration(test_pairs_filtered, verbose=verbose)
        history['calibration_fitted'] = True
        
        # Evaluate with calibrated voting
        if verbose:
            cal_phoneme_acc, cal_word_acc, _ = evaluate_swarm_words(
                swarm, demo_in_lex, lexicon
            )
            print(f"  Post-calibration demo: phoneme={100*cal_phoneme_acc:.1f}%, "
                  f"word={100*cal_word_acc:.1f}%")
    else:
        history['calibration_fitted'] = False
    
    # Save best checkpoint if requested
    if save_best and best_weights is not None:
        checkpoint_path = best_path or "model_swarm_best.npz"
        swarm.save(checkpoint_path)
        if verbose:
            print(f"✓ Best checkpoint saved to {checkpoint_path} (epoch {best_epoch})")
    
    return swarm, history


def train_swarm_demo(
    n_epochs: int = 8,
    max_words: int = 5000,
    verbose: bool = True,
) -> Tuple[FlySwarm, Dict]:
    """
    Train a swarm on only the phonemes needed for demo words.
    
    This is faster for quick demos - trains only ~20 specialists instead of 39.
    """
    demo_phonemes = get_demo_phonemes()
    
    if verbose:
        print(f"Demo phonemes ({len(demo_phonemes)}): {', '.join(demo_phonemes)}")
    
    return train_swarm(
        phonemes=demo_phonemes,
        max_words=max_words,
        n_epochs=n_epochs,
        verbose=verbose,
    )


def evaluate_swarm(swarm: FlySwarm, verbose: bool = True) -> Dict[str, float]:
    """Evaluate trained swarm vs demo and test words."""
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    if verbose:
        print("\n" + "="*60)
        print("SWARM EVALUATION")
        print("="*60)
        print("\n--- Demo Words ---\n")
        print(f"{'Word':15} {'Expected':25} {'Predicted':25}")
        print("-" * 70)
    
    demo_phoneme, demo_word, demo_details = evaluate_swarm_words(
        swarm, demo_in_lex, lexicon, verbose=verbose
    )
    
    if verbose:
        print(f"\nDemo: phoneme={100*demo_phoneme:.1f}%, word={100*demo_word:.1f}%")
    
    # Test sample
    all_words = list(lexicon.keys())
    rng = np.random.default_rng(123)
    test_sample = [
        w for w in rng.choice(all_words, size=min(500, len(all_words)), replace=False)
        if w not in demo_words and w.isalpha() and 2 <= len(w) <= 12
    ][:100]
    
    test_phoneme, test_word, _ = evaluate_swarm_words(swarm, test_sample, lexicon)
    
    if verbose:
        print(f"Test sample (100 words): phoneme={100*test_phoneme:.1f}%, word={100*test_word:.1f}%")
        print("="*60)
    
    return {
        'demo_phoneme_acc': demo_phoneme,
        'demo_word_acc': demo_word,
        'test_phoneme_acc': test_phoneme,
        'test_word_acc': test_word,
    }


def compare_to_baseline(
    swarm: FlySwarm,
    baseline_path: str = "model.npz",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compare swarm accuracy to single multi-class baseline."""
    from .mushroom_body import MushroomBody
    from .train import evaluate_words
    
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    # Test sample
    all_words = list(lexicon.keys())
    rng = np.random.default_rng(123)
    test_sample = [
        w for w in rng.choice(all_words, size=min(500, len(all_words)), replace=False)
        if w not in demo_words and w.isalpha() and 2 <= len(w) <= 12
    ][:100]
    
    results = {'swarm': {}, 'baseline': {}}
    
    # Swarm results
    demo_ph, demo_w, _ = evaluate_swarm_words(swarm, demo_in_lex, lexicon)
    test_ph, test_w, _ = evaluate_swarm_words(swarm, test_sample, lexicon)
    results['swarm'] = {
        'demo_phoneme': demo_ph,
        'demo_word': demo_w,
        'test_phoneme': test_ph,
        'test_word': test_w,
    }
    
    # Baseline results
    try:
        baseline = MushroomBody.load(baseline_path)
        demo_ph_b, demo_w_b, _ = evaluate_words(baseline, demo_in_lex, lexicon)
        test_ph_b, test_w_b, _ = evaluate_words(baseline, test_sample, lexicon)
        results['baseline'] = {
            'demo_phoneme': demo_ph_b,
            'demo_word': demo_w_b,
            'test_phoneme': test_ph_b,
            'test_word': test_w_b,
        }
    except FileNotFoundError:
        if verbose:
            print("Baseline model not found at", baseline_path)
        results['baseline'] = None
    
    if verbose:
        print("\n" + "="*60)
        print("SWARM vs BASELINE COMPARISON")
        print("="*60)
        print(f"\n{'Metric':<25} {'Swarm':>12} {'Baseline':>12} {'Delta':>10}")
        print("-" * 60)
        
        for metric in ['demo_phoneme', 'demo_word', 'test_phoneme', 'test_word']:
            swarm_val = results['swarm'].get(metric, 0)
            base_val = results['baseline'].get(metric, 0) if results['baseline'] else 0
            delta = swarm_val - base_val
            delta_str = f"{100*delta:+.1f}%" if results['baseline'] else "N/A"
            print(f"{metric:<25} {100*swarm_val:>11.1f}% {100*base_val:>11.1f}% {delta_str:>10}")
        
        print("="*60)
    
    return results


def train_and_save_swarm(
    output_path: str = "model_swarm.npz",
    phonemes: Optional[List[str]] = None,
    n_epochs: int = 10,
    max_words: int = 10000,
    seed: int = 42,
    vote_strategy: str = 'softmax',
    vote_temperature: float = 0.5,
    vote_margin: float = 0.1,
    early_stop_patience: int = 0,
    save_best: bool = True,
    confusion_mine: bool = False,
    confusion_mine_epoch: int = 2,
    hard_negative_weight: float = 2.0,
    oversample_factor: float = 1.5,
    confusion_report_path: Optional[str] = None,
    use_known_hard_pairs: bool = False,
    verbose: bool = True,
) -> FlySwarm:
    """Train and save a fly swarm."""
    # Best checkpoint path derived from output path
    best_path = output_path.replace('.npz', '_best.npz')
    if best_path == output_path:
        best_path = output_path + '_best'
    
    # Default confusion report path
    if confusion_mine and confusion_report_path is None:
        confusion_report_path = output_path.replace('.npz', '_confusion.txt')
    
    swarm, history = train_swarm(
        phonemes=phonemes,
        max_words=max_words,
        n_epochs=n_epochs,
        seed=seed,
        vote_strategy=vote_strategy,
        vote_temperature=vote_temperature,
        vote_margin=vote_margin,
        early_stop_patience=early_stop_patience,
        save_best=save_best,
        best_path=best_path,
        confusion_mine=confusion_mine,
        confusion_mine_epoch=confusion_mine_epoch,
        hard_negative_weight=hard_negative_weight,
        oversample_factor=oversample_factor,
        confusion_report_path=confusion_report_path,
        use_known_hard_pairs=use_known_hard_pairs,
        verbose=verbose,
    )
    
    if verbose:
        evaluate_swarm(swarm, verbose=True)
        compare_to_baseline(swarm, verbose=True)
        
        # Print best epoch info
        if history.get('best_epoch'):
            print(f"\n★ Best epoch: {history['best_epoch']} "
                  f"(demo_phoneme={100*history['best_demo_phoneme']:.1f}%)")
        
        # Print confusion mining info
        if history.get('confusion_mined'):
            print(f"★ Confusion mining: {history['hard_pairs_count']} hard pair entries, "
                  f"mined after epoch {history.get('confusion_mine_epoch', 'N/A')}")
            if confusion_report_path:
                print(f"★ Confusion report saved to: {confusion_report_path}")
    
    swarm.save(output_path)
    return swarm


if __name__ == "__main__":
    # Quick test with demo phonemes
    swarm, history = train_swarm_demo(n_epochs=3, max_words=2000)
    evaluate_swarm(swarm)
