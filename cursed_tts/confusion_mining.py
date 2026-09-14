"""
Confusion mining and hard-negative training for fly swarm.

Identifies confusable phoneme pairs from prediction errors and uses them
to oversample hard cases during training - learning more when wrong on
confusing pairs, staying dopamine/error-driven.

Biology analogy: Selective attention to errors. When a fly brain repeatedly
confuses two similar odors, more learning happens on those specific cases.
This is error-driven plasticity focused on the hardest distinctions.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path

from .phonemes import PHONEME_LIST, strip_stress
from .alignment import AlignedPair


@dataclass
class ConfusionPair:
    """A confusable phoneme pair with statistics."""
    target: str        # What it should be
    predicted: str     # What was predicted instead
    count: int         # How many times this confusion happened
    target_total: int  # Total examples with this target
    confusion_rate: float  # count / target_total


class ConfusionMatrix:
    """
    Confusion matrix for tracking phoneme prediction errors.
    
    Rows = target (true) phonemes, Columns = predicted phonemes.
    Off-diagonal entries represent confusions.
    """
    
    def __init__(self, phonemes: Optional[List[str]] = None):
        self.phonemes = phonemes or PHONEME_LIST.copy()
        self.phoneme_to_idx = {p: i for i, p in enumerate(self.phonemes)}
        self.n_phonemes = len(self.phonemes)
        
        # Initialize confusion counts: matrix[target_idx, pred_idx] = count
        self.matrix = np.zeros((self.n_phonemes, self.n_phonemes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, target: str, predicted: str):
        """Record a single prediction."""
        target = strip_stress(target)
        predicted = strip_stress(predicted)
        
        if target not in self.phoneme_to_idx or predicted not in self.phoneme_to_idx:
            return
        
        t_idx = self.phoneme_to_idx[target]
        p_idx = self.phoneme_to_idx[predicted]
        self.matrix[t_idx, p_idx] += 1
        self.total_samples += 1
    
    def update_batch(self, targets: List[str], predictions: List[str]):
        """Record a batch of predictions."""
        for t, p in zip(targets, predictions):
            self.update(t, p)
    
    def get_confusion_rate(self, target: str, predicted: str) -> float:
        """Get confusion rate: how often target was predicted as predicted."""
        target = strip_stress(target)
        predicted = strip_stress(predicted)
        
        if target not in self.phoneme_to_idx or predicted not in self.phoneme_to_idx:
            return 0.0
        
        t_idx = self.phoneme_to_idx[target]
        p_idx = self.phoneme_to_idx[predicted]
        
        target_total = self.matrix[t_idx, :].sum()
        if target_total == 0:
            return 0.0
        
        return self.matrix[t_idx, p_idx] / target_total
    
    def get_top_confusions(
        self, 
        top_k: int = 20,
        min_count: int = 5,
        exclude_correct: bool = True,
    ) -> List[ConfusionPair]:
        """
        Get the top confusable phoneme pairs.
        
        Args:
            top_k: Number of pairs to return
            min_count: Minimum confusion count to include
            exclude_correct: Exclude diagonal (correct predictions)
        
        Returns:
            List of ConfusionPair sorted by confusion count
        """
        pairs = []
        
        for t_idx, target in enumerate(self.phonemes):
            target_total = int(self.matrix[t_idx, :].sum())
            if target_total == 0:
                continue
            
            for p_idx, predicted in enumerate(self.phonemes):
                if exclude_correct and t_idx == p_idx:
                    continue
                
                count = int(self.matrix[t_idx, p_idx])
                if count < min_count:
                    continue
                
                confusion_rate = count / target_total
                pairs.append(ConfusionPair(
                    target=target,
                    predicted=predicted,
                    count=count,
                    target_total=target_total,
                    confusion_rate=confusion_rate,
                ))
        
        # Sort by count (most frequent confusions first)
        pairs.sort(key=lambda p: -p.count)
        return pairs[:top_k]
    
    def get_hard_pairs_for_phoneme(
        self, 
        phoneme: str, 
        min_rate: float = 0.05,
    ) -> List[str]:
        """
        Get phonemes that are frequently confused with the given phoneme.
        
        Returns phonemes that:
        1. Are often predicted when phoneme is the target (false positives for other specialists)
        2. Are often the target when phoneme is predicted (false negatives for this specialist)
        """
        phoneme = strip_stress(phoneme)
        if phoneme not in self.phoneme_to_idx:
            return []
        
        p_idx = self.phoneme_to_idx[phoneme]
        hard_set = set()
        
        # Type 1: What is predicted when this phoneme is the target?
        target_total = self.matrix[p_idx, :].sum()
        if target_total > 0:
            for other_idx, count in enumerate(self.matrix[p_idx, :]):
                if other_idx != p_idx and count / target_total >= min_rate:
                    hard_set.add(self.phonemes[other_idx])
        
        # Type 2: When this phoneme is predicted, what was the actual target?
        pred_total = self.matrix[:, p_idx].sum()
        if pred_total > 0:
            for other_idx, count in enumerate(self.matrix[:, p_idx]):
                if other_idx != p_idx and count / pred_total >= min_rate:
                    hard_set.add(self.phonemes[other_idx])
        
        return list(hard_set)
    
    def get_accuracy(self) -> float:
        """Get overall accuracy (diagonal / total)."""
        correct = np.trace(self.matrix)
        total = self.matrix.sum()
        return correct / max(total, 1)
    
    def get_per_phoneme_accuracy(self) -> Dict[str, float]:
        """Get per-phoneme accuracy."""
        accuracies = {}
        for t_idx, phoneme in enumerate(self.phonemes):
            total = self.matrix[t_idx, :].sum()
            if total > 0:
                accuracies[phoneme] = float(self.matrix[t_idx, t_idx] / total)
            else:
                accuracies[phoneme] = 0.0
        return accuracies
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            'phonemes': self.phonemes,
            'matrix': self.matrix.tolist(),
            'total_samples': int(self.total_samples),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ConfusionMatrix':
        """Deserialize from dict."""
        cm = cls(phonemes=d.get('phonemes'))
        cm.matrix = np.array(d.get('matrix', []), dtype=np.int64)
        cm.total_samples = d.get('total_samples', 0)
        return cm
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Confusion matrix saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ConfusionMatrix':
        """Load from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        cm = cls.from_dict(d)
        print(f"Confusion matrix loaded from {path} ({cm.total_samples} samples)")
        return cm
    
    def print_report(
        self, 
        top_k: int = 20,
        min_count: int = 3,
        include_matrix: bool = False,
    ) -> str:
        """
        Generate a human-readable confusion report.
        
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CONFUSION MATRIX REPORT")
        lines.append("=" * 60)
        lines.append(f"Total samples: {self.total_samples}")
        lines.append(f"Overall accuracy: {100 * self.get_accuracy():.1f}%")
        lines.append("")
        
        # Top confusions
        lines.append(f"TOP {top_k} CONFUSIONS (target → predicted):")
        lines.append("-" * 60)
        lines.append(f"{'Target':>8} → {'Pred':8} {'Count':>8} {'Rate':>8} {'Target Total':>12}")
        lines.append("-" * 60)
        
        confusions = self.get_top_confusions(top_k=top_k, min_count=min_count)
        for cp in confusions:
            lines.append(
                f"{cp.target:>8} → {cp.predicted:8} {cp.count:>8} "
                f"{100*cp.confusion_rate:>7.1f}% {cp.target_total:>12}"
            )
        
        if not confusions:
            lines.append("  (no significant confusions found)")
        
        lines.append("")
        
        # Per-phoneme accuracy (worst performers)
        per_phone_acc = self.get_per_phoneme_accuracy()
        sorted_acc = sorted(per_phone_acc.items(), key=lambda x: x[1])
        
        lines.append("WORST PERFORMING PHONEMES:")
        lines.append("-" * 40)
        for phoneme, acc in sorted_acc[:10]:
            if phoneme in self.phoneme_to_idx:
                total = int(self.matrix[self.phoneme_to_idx[phoneme], :].sum())
                if total > 0:
                    lines.append(f"  {phoneme:>6}: {100*acc:>5.1f}% accuracy ({total} samples)")
        
        lines.append("")
        
        # Known hard pairs from the task description
        lines.append("KNOWN HARD PAIRS (from task spec):")
        lines.append("-" * 40)
        known_pairs = [
            ('IY', 'EH', "me: M IY → M EH"),
            ('AE', 'AA', "vowel confusion"),
            ('AY', 'IY', "diphthong/vowel"),
            ('AY', 'EH', "diphthong/vowel"),
        ]
        for target, pred, note in known_pairs:
            rate = self.get_confusion_rate(target, pred)
            rev_rate = self.get_confusion_rate(pred, target)
            lines.append(f"  {target} ↔ {pred}: {100*rate:.1f}% / {100*rev_rate:.1f}% ({note})")
        
        lines.append("=" * 60)
        
        # Full matrix if requested
        if include_matrix and self.n_phonemes <= 20:
            lines.append("\nFULL CONFUSION MATRIX:")
            lines.append("-" * 40)
            # Header
            header = "     " + " ".join(f"{p:>4}" for p in self.phonemes)
            lines.append(header)
            for t_idx, target in enumerate(self.phonemes):
                row = f"{target:>4} "
                for p_idx in range(self.n_phonemes):
                    count = self.matrix[t_idx, p_idx]
                    if count > 0:
                        row += f"{count:>4} "
                    else:
                        row += "   . "
                lines.append(row)
        
        return "\n".join(lines)


def build_confusion_matrix_from_swarm(
    swarm,  # FlySwarm instance
    pairs: List[AlignedPair],
    vote_strategy: Optional[str] = None,
    verbose: bool = True,
) -> ConfusionMatrix:
    """
    Build confusion matrix by evaluating swarm on aligned pairs.
    
    Args:
        swarm: Trained FlySwarm
        pairs: AlignedPair examples to evaluate
        vote_strategy: Voting strategy for predictions
        verbose: Print progress
    
    Returns:
        ConfusionMatrix with prediction statistics
    """
    cm = ConfusionMatrix(phonemes=swarm.phoneme_list)
    
    if verbose:
        print(f"Building confusion matrix from {len(pairs)} pairs...")
    
    for i, pair in enumerate(pairs):
        target = strip_stress(pair.phoneme)
        predicted, _, _ = swarm.predict(pair.letter_context, vote_strategy=vote_strategy)
        cm.update(target, predicted)
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(pairs)}: accuracy={100*cm.get_accuracy():.1f}%")
    
    if verbose:
        print(f"  Final accuracy: {100*cm.get_accuracy():.1f}%")
    
    return cm


def get_hard_negative_weights(
    confusion_matrix: ConfusionMatrix,
    phonemes: List[str],
    hard_weight: float = 2.0,
    top_k_confusions: int = 30,
    min_confusion_count: int = 3,
) -> Dict[Tuple[str, str], float]:
    """
    Generate sample weights based on confusion pairs.
    
    Returns a dict mapping (target_phoneme, any_phoneme_in_confusion_pair) to weight.
    Samples involving confusable pairs get higher weight.
    
    Args:
        confusion_matrix: Pre-built confusion matrix
        phonemes: List of phonemes in the swarm
        hard_weight: Weight multiplier for hard pairs
        top_k_confusions: Number of top confusions to consider
        min_confusion_count: Minimum confusion count to include
    
    Returns:
        Dict mapping (target, context_phoneme) -> weight
    """
    weights: Dict[Tuple[str, str], float] = {}
    
    # Get top confusions
    confusions = confusion_matrix.get_top_confusions(
        top_k=top_k_confusions, 
        min_count=min_confusion_count,
    )
    
    # For each confusion pair, upweight both directions
    for cp in confusions:
        target = cp.target
        predicted = cp.predicted
        
        # Scale weight by confusion rate (more confused = higher weight)
        # Use sqrt to moderate the effect
        scaled_weight = hard_weight * (1.0 + np.sqrt(cp.confusion_rate))
        
        # When target is actual target, upweight
        weights[(target, predicted)] = scaled_weight
        weights[(predicted, target)] = scaled_weight
    
    return weights


def compute_sample_weights(
    pairs: List[AlignedPair],
    hard_weights: Dict[Tuple[str, str], float],
    base_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute per-sample weights based on hard negatives.
    
    A sample's weight is increased if its target phoneme is involved
    in a known confusable pair with any of its neighboring phonemes
    or common confusion targets.
    
    Args:
        pairs: Training pairs
        hard_weights: Dict from get_hard_negative_weights
        base_weight: Base weight for non-hard samples
    
    Returns:
        Array of weights, one per pair
    """
    weights = np.full(len(pairs), base_weight, dtype=np.float32)
    
    # Build index of which phonemes each sample involves
    for i, pair in enumerate(pairs):
        target = strip_stress(pair.phoneme)
        
        # Check if this target is in any hard pair
        max_weight = base_weight
        for (t, p), w in hard_weights.items():
            if t == target or p == target:
                max_weight = max(max_weight, w)
        
        weights[i] = max_weight
    
    return weights


def oversample_hard_pairs(
    pairs: List[AlignedPair],
    hard_weights: Dict[Tuple[str, str], float],
    oversample_factor: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> List[AlignedPair]:
    """
    Create oversampled training data by duplicating hard samples.
    
    Instead of weighting, this physically duplicates samples involving
    hard pairs. The dopamine-when-wrong learning will then see these
    cases more often.
    
    Args:
        pairs: Original training pairs
        hard_weights: Dict from get_hard_negative_weights
        oversample_factor: How much to oversample (1.5 = 50% more hard samples)
        rng: Random generator for shuffling duplicates
    
    Returns:
        Augmented list of pairs with hard samples duplicated
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Find hard samples
    hard_pairs = []
    normal_pairs = []
    
    hard_phonemes = set()
    for (t, p), _ in hard_weights.items():
        hard_phonemes.add(t)
        hard_phonemes.add(p)
    
    for pair in pairs:
        target = strip_stress(pair.phoneme)
        if target in hard_phonemes:
            hard_pairs.append(pair)
        else:
            normal_pairs.append(pair)
    
    # Compute how many duplicates to add
    n_duplicates = int(len(hard_pairs) * (oversample_factor - 1.0))
    if n_duplicates > 0:
        duplicate_indices = rng.choice(len(hard_pairs), size=n_duplicates, replace=True)
        duplicates = [hard_pairs[i] for i in duplicate_indices]
    else:
        duplicates = []
    
    # Combine and shuffle
    augmented = normal_pairs + hard_pairs + duplicates
    indices = rng.permutation(len(augmented))
    augmented = [augmented[i] for i in indices]
    
    return augmented


class ConfusionMiner:
    """
    End-to-end confusion mining for swarm training.
    
    Workflow:
    1. Train initial swarm (or load checkpoint)
    2. Build confusion matrix from predictions
    3. Identify hard pairs
    4. Re-train or continue training with hard-negative oversampling
    """
    
    def __init__(
        self,
        phonemes: Optional[List[str]] = None,
        hard_weight: float = 2.0,
        oversample_factor: float = 1.5,
        top_k_confusions: int = 30,
        min_confusion_count: int = 3,
    ):
        self.phonemes = phonemes or PHONEME_LIST.copy()
        self.hard_weight = hard_weight
        self.oversample_factor = oversample_factor
        self.top_k_confusions = top_k_confusions
        self.min_confusion_count = min_confusion_count
        
        self.confusion_matrix: Optional[ConfusionMatrix] = None
        self.hard_weights: Dict[Tuple[str, str], float] = {}
    
    def mine_confusions(
        self,
        swarm,  # FlySwarm instance
        pairs: List[AlignedPair],
        vote_strategy: Optional[str] = None,
        verbose: bool = True,
    ) -> ConfusionMatrix:
        """
        Mine confusions from swarm predictions.
        
        Args:
            swarm: Trained FlySwarm
            pairs: Pairs to evaluate for confusions
            vote_strategy: Voting strategy
            verbose: Print progress
        
        Returns:
            ConfusionMatrix
        """
        self.confusion_matrix = build_confusion_matrix_from_swarm(
            swarm, pairs, vote_strategy=vote_strategy, verbose=verbose
        )
        
        # Compute hard weights
        self.hard_weights = get_hard_negative_weights(
            self.confusion_matrix,
            self.phonemes,
            hard_weight=self.hard_weight,
            top_k_confusions=self.top_k_confusions,
            min_confusion_count=self.min_confusion_count,
        )
        
        if verbose:
            print(f"\nIdentified {len(self.hard_weights)} hard phoneme pair entries")
            top_confusions = self.confusion_matrix.get_top_confusions(top_k=10)
            if top_confusions:
                print("Top confusions:")
                for cp in top_confusions[:5]:
                    print(f"  {cp.target} → {cp.predicted}: {cp.count} times "
                          f"({100*cp.confusion_rate:.1f}%)")
        
        return self.confusion_matrix
    
    def augment_training_data(
        self,
        pairs: List[AlignedPair],
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
    ) -> List[AlignedPair]:
        """
        Augment training data by oversampling hard pairs.
        
        Args:
            pairs: Original training pairs
            rng: Random generator
            verbose: Print progress
        
        Returns:
            Augmented training pairs
        """
        if not self.hard_weights:
            if verbose:
                print("No hard weights computed yet - returning original pairs")
            return pairs
        
        augmented = oversample_hard_pairs(
            pairs,
            self.hard_weights,
            oversample_factor=self.oversample_factor,
            rng=rng,
        )
        
        if verbose:
            print(f"Augmented training data: {len(pairs)} → {len(augmented)} pairs "
                  f"(+{len(augmented) - len(pairs)})")
        
        return augmented
    
    def get_sample_weights(
        self, 
        pairs: List[AlignedPair],
    ) -> np.ndarray:
        """Get per-sample weights for weighted training."""
        return compute_sample_weights(pairs, self.hard_weights)
    
    def save_report(self, path: str):
        """Save confusion report to file."""
        if self.confusion_matrix is None:
            print("No confusion matrix - run mine_confusions first")
            return
        
        report = self.confusion_matrix.print_report(
            top_k=30, 
            min_count=self.min_confusion_count,
        )
        
        with open(path, 'w') as f:
            f.write(report)
        print(f"Confusion report saved to {path}")
    
    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            'phonemes': self.phonemes,
            'hard_weight': self.hard_weight,
            'oversample_factor': self.oversample_factor,
            'top_k_confusions': self.top_k_confusions,
            'min_confusion_count': self.min_confusion_count,
            'confusion_matrix': self.confusion_matrix.to_dict() if self.confusion_matrix else None,
            'hard_weights': {f"{t}|{p}": w for (t, p), w in self.hard_weights.items()},
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ConfusionMiner':
        """Deserialize from dict."""
        miner = cls(
            phonemes=d.get('phonemes'),
            hard_weight=d.get('hard_weight', 2.0),
            oversample_factor=d.get('oversample_factor', 1.5),
            top_k_confusions=d.get('top_k_confusions', 30),
            min_confusion_count=d.get('min_confusion_count', 3),
        )
        if d.get('confusion_matrix'):
            miner.confusion_matrix = ConfusionMatrix.from_dict(d['confusion_matrix'])
        for key, w in d.get('hard_weights', {}).items():
            t, p = key.split('|')
            miner.hard_weights[(t, p)] = w
        return miner


# Pre-defined known hard pairs from the task description
KNOWN_HARD_PAIRS = [
    ('IY', 'EH'),   # me: M IY → M EH
    ('AE', 'AA'),   # vowel confusion
    ('AY', 'IY'),   # diphthong vs vowel
    ('AY', 'EH'),   # diphthong vs vowel
    # Common vowel confusions
    ('IH', 'EH'),   # short i vs short e
    ('AH', 'UH'),   # schwa area
    ('AO', 'AA'),   # open back vowels
]


def get_known_hard_weights(
    hard_weight: float = 2.0,
) -> Dict[Tuple[str, str], float]:
    """
    Get hard weights for known confusable pairs (from task spec).
    
    Can be used as a starting point before mining or as fallback.
    """
    weights = {}
    for p1, p2 in KNOWN_HARD_PAIRS:
        weights[(p1, p2)] = hard_weight
        weights[(p2, p1)] = hard_weight
    return weights


if __name__ == "__main__":
    # Test confusion matrix
    print("Testing confusion mining...")
    
    cm = ConfusionMatrix(phonemes=['K', 'AE', 'T', 'AA', 'IY', 'EH'])
    
    # Simulate some confusions
    test_data = [
        ('K', 'K'), ('K', 'K'), ('K', 'K'), ('K', 'T'),  # K mostly correct
        ('AE', 'AE'), ('AE', 'AA'), ('AE', 'AA'), ('AE', 'AA'),  # AE confused with AA
        ('T', 'T'), ('T', 'T'), ('T', 'K'),  # T mostly correct
        ('IY', 'IY'), ('IY', 'EH'), ('IY', 'EH'),  # IY confused with EH
    ]
    
    for target, pred in test_data:
        cm.update(target, pred)
    
    print(cm.print_report(top_k=5, min_count=1))
    
    # Test hard weights
    print("\nHard weights:")
    weights = get_hard_negative_weights(cm, cm.phonemes, hard_weight=2.0, min_confusion_count=1)
    for (t, p), w in sorted(weights.items()):
        print(f"  {t} ↔ {p}: {w:.2f}")
