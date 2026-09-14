"""
G2P training for the mushroom body classifier.

Trains the MB to classify letter contexts → phonemes using aligned CMUdict data.
This is the correct architecture: MB = classifier, synth = renderer.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time

from .mushroom_body import MushroomBody, MushroomBodyConfig
from .alignment import create_g2p_training_data, AlignedPair
from .phonemes import PHONEME_LIST, strip_stress
from .lexicon import get_phonemes, get_demo_words


def train_epoch(
    mb: MushroomBody,
    train_pairs: List[AlignedPair],
    rng: np.random.Generator,
    verbose: bool = True,
) -> float:
    """
    Train for one epoch over aligned pairs.
    
    Returns:
        Epoch accuracy
    """
    # Shuffle pairs
    indices = rng.permutation(len(train_pairs))
    
    mb.reset_stats()
    
    for i, idx in enumerate(indices):
        pair = train_pairs[idx]
        mb.train_step(pair.letter_context, pair.phoneme)
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(train_pairs)}: acc={100*mb.get_accuracy():.1f}%")
    
    return mb.get_accuracy()


def evaluate_pairs(mb: MushroomBody, pairs: List[AlignedPair]) -> Tuple[float, Dict]:
    """Evaluate phoneme accuracy on aligned pairs."""
    correct = 0
    total = 0
    
    for pair in pairs:
        pred, _ = mb.predict(pair.letter_context)
        if strip_stress(pred) == strip_stress(pair.phoneme):
            correct += 1
        total += 1
    
    return correct / max(total, 1), {'correct': correct, 'total': total}


def evaluate_words(
    mb: MushroomBody,
    words: List[str],
    lexicon: Dict[str, List[str]],
    verbose: bool = False,
) -> Tuple[float, float, List[Dict]]:
    """
    Evaluate on whole words.
    
    Returns:
        phoneme_accuracy: Fraction of phonemes correct
        word_accuracy: Fraction of words with all phonemes correct
        details: Per-word results
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
        
        # Predict
        pred_phonemes = mb.predict_word(word, n_phonemes)
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
            print(f"{word:15} {ref_str:25} {pred_str:25} {100*word_correct/n_phonemes:.0f}% {status}")
    
    n_words = len(details)
    phoneme_acc = correct_phonemes / max(total_phonemes, 1)
    word_acc = correct_words / max(n_words, 1)
    
    return phoneme_acc, word_acc, details


def train(
    max_words: int = 10000,
    n_epochs: int = 15,
    context_size: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[MushroomBody, Dict]:
    """
    Train G2P mushroom body on CMUdict alignments.
    
    Args:
        max_words: Maximum training words
        n_epochs: Number of training epochs
        context_size: Letter context window size
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Trained MushroomBody and training history
    """
    rng = np.random.default_rng(seed)
    
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
    
    # Create lexicon dict for evaluation
    from .lexicon import get_lexicon
    lexicon = get_lexicon()
    
    # Create model
    config = MushroomBodyConfig(
        context_size=context_size,
        seed=seed,
    )
    mb = MushroomBody(config)
    
    if verbose:
        print(f"\n{'='*60}")
        print("G2P MUSHROOM BODY TRAINING")
        print(f"{'='*60}")
        print(f"Model: {config.n_pn} PNs, {config.n_kc} KCs, {config.n_mbon} MBONs")
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
        print(f"Epochs: {n_epochs}")
        print(f"{'='*60}\n")
    
    # Get demo words for evaluation
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    # Training history
    history = {
        'train_acc': [],
        'test_acc': [],
        'demo_phoneme_acc': [],
        'demo_word_acc': [],
    }
    
    start_time = time.time()
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_acc = train_epoch(mb, train_pairs, rng, verbose=verbose)
        
        # Evaluate on test pairs
        test_acc, _ = evaluate_pairs(mb, test_pairs)
        
        # Evaluate on demo words
        demo_phoneme_acc, demo_word_acc, _ = evaluate_words(mb, demo_in_lex, lexicon)
        
        # Record history
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['demo_phoneme_acc'].append(demo_phoneme_acc)
        history['demo_word_acc'].append(demo_word_acc)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nEpoch {epoch:2d}/{n_epochs}: "
                  f"train={100*train_acc:.1f}%, test={100*test_acc:.1f}%, "
                  f"demo_phoneme={100*demo_phoneme_acc:.1f}%, demo_word={100*demo_word_acc:.1f}% "
                  f"[{epoch_time:.0f}s, total {total_time:.0f}s]")
    
    return mb, history


def evaluate(mb: MushroomBody, verbose: bool = True) -> Tuple[float, float, float, float]:
    """
    Evaluate trained model on demo and test words.
    
    Returns:
        demo_phoneme_acc, demo_word_acc, test_phoneme_acc, test_word_acc
    """
    from .lexicon import get_lexicon
    
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    demo_in_lex = [w for w in demo_words if w in lexicon]
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        print("\n--- Demo Words ---\n")
        print(f"{'Word':15} {'Expected':25} {'Predicted':25} {'Acc'}")
        print("-" * 75)
    
    demo_phoneme, demo_word, demo_details = evaluate_words(
        mb, demo_in_lex, lexicon, verbose=verbose
    )
    
    if verbose:
        print(f"\nDemo: phoneme={100*demo_phoneme:.1f}%, word={100*demo_word:.1f}%")
    
    # Evaluate on some test words
    all_words = list(lexicon.keys())
    rng = np.random.default_rng(123)
    test_sample = [w for w in rng.choice(all_words, size=min(500, len(all_words)), replace=False)
                   if w not in demo_words and w.isalpha() and 2 <= len(w) <= 12][:100]
    
    if verbose:
        print("\n--- Test Sample (100 words) ---\n")
    
    test_phoneme, test_word, _ = evaluate_words(mb, test_sample, lexicon, verbose=False)
    
    if verbose:
        print(f"Test sample: phoneme={100*test_phoneme:.1f}%, word={100*test_word:.1f}%")
        print("\n" + "="*60)
        print(f"Summary: Demo phoneme={100*demo_phoneme:.1f}%, Test phoneme={100*test_phoneme:.1f}%")
        print("="*60)
    
    return demo_phoneme, demo_word, test_phoneme, test_word


def train_and_save(
    output_path: str = "model.npz",
    n_epochs: int = 15,
    max_words: int = 10000,
    context_size: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> MushroomBody:
    """Train and save the G2P model."""
    
    mb, history = train(
        max_words=max_words,
        n_epochs=n_epochs,
        context_size=context_size,
        seed=seed,
        verbose=verbose,
    )
    
    # Final evaluation
    if verbose:
        evaluate(mb, verbose=True)
    
    # Save
    mb.save(output_path)
    
    return mb


if __name__ == "__main__":
    train_and_save()
