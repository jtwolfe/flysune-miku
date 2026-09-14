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
        predicted, _, _ = swarm.predict(
            pair.letter_context, 
            vote_strategy=vote_strategy,
            phoneme_pos=pair.phoneme_pos,
            n_phonemes=pair.n_phonemes,
        )
        cm.update(target, predicted)
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(pairs)}: accuracy={100*cm.get_accuracy():.1f}%")
    
    if verbose:
        print(f"  Final accuracy: {100*cm.get_accuracy():.1f}%")
    
    return cm


def get_confusion_targets(
    confusion_matrix: ConfusionMatrix,
    top_k_confusions: int = 30,
    min_confusion_count: int = 3,
    min_confusion_rate: float = 0.05,
) -> Dict[str, Set[str]]:
    """
    Get mapping of target phonemes to their confused predictions.
    
    Returns dict: target -> {predicted phonemes it's confused with}
    Only includes the TARGET side of confusions (T→P), not the reverse.
    
    Args:
        confusion_matrix: Pre-built confusion matrix
        top_k_confusions: Number of top confusions to consider
        min_confusion_count: Minimum confusion count to include
        min_confusion_rate: Minimum confusion rate (fraction) to include
    
    Returns:
        Dict mapping target -> set of confused predictions
    """
    target_to_confused: Dict[str, Set[str]] = defaultdict(set)
    
    confusions = confusion_matrix.get_top_confusions(
        top_k=top_k_confusions, 
        min_count=min_confusion_count,
    )
    
    for cp in confusions:
        if cp.confusion_rate >= min_confusion_rate:
            # Only add TARGET → PREDICTED, not reverse
            # This means: when target=T, it's confused as P
            target_to_confused[cp.target].add(cp.predicted)
    
    return dict(target_to_confused)


def identify_hard_samples(
    pairs: List[AlignedPair],
    target_to_confused: Dict[str, Set[str]],
    swarm=None,  # Optional FlySwarm for prediction-based filtering
    require_prediction_match: bool = False,
    top_k_predictions: int = 3,
) -> List[int]:
    """
    Identify indices of hard samples that should be oversampled.
    
    A sample is hard if:
    1. Its target phoneme T is in target_to_confused
    2. (Optional) The swarm's current prediction is one of the confused partners
    
    Args:
        pairs: Training pairs
        target_to_confused: Dict from get_confusion_targets
        swarm: Optional FlySwarm for prediction-based filtering
        require_prediction_match: If True, only mark hard if swarm predicts confused partner
        top_k_predictions: Check if confused partner is in top-k predictions
    
    Returns:
        List of indices into pairs that are hard
    """
    hard_indices = []
    
    for i, pair in enumerate(pairs):
        target = strip_stress(pair.phoneme)
        
        # Check if target has known confusions
        if target not in target_to_confused:
            continue
        
        confused_partners = target_to_confused[target]
        
        if require_prediction_match and swarm is not None:
            # Only mark as hard if swarm actually predicts a confused partner
            scores = swarm._get_raw_scores(
                pair.letter_context,
                phoneme_pos=pair.phoneme_pos,
                n_phonemes=pair.n_phonemes,
            )
            sorted_preds = sorted(scores.items(), key=lambda x: -x[1])[:top_k_predictions]
            top_k_phones = {p for p, s in sorted_preds}
            
            # Check if any confused partner is in top-k
            if not (confused_partners & top_k_phones):
                continue
        
        hard_indices.append(i)
    
    return hard_indices


def oversample_hard_negatives(
    pairs: List[AlignedPair],
    target_to_confused: Dict[str, Set[str]],
    swarm=None,
    require_prediction_match: bool = False,
    max_oversample_ratio: float = 0.15,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[List[AlignedPair], Dict[str, int]]:
    """
    Create oversampled training data by duplicating TRUE hard negatives.
    
    Only oversamples pairs where:
    - target=T for a mined confusion (T→P)
    - NOT when target=P (that would flood with common phonemes)
    
    Caps oversample at max_oversample_ratio of original corpus size.
    
    Args:
        pairs: Original training pairs
        target_to_confused: Dict from get_confusion_targets
        swarm: Optional FlySwarm for prediction-based filtering
        require_prediction_match: If True, only oversample if swarm predicts confused partner
        max_oversample_ratio: Max fraction of corpus to add as duplicates (default 15%)
        rng: Random generator
        verbose: Print statistics
    
    Returns:
        (augmented_pairs, stats_dict)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Identify hard samples
    hard_indices = identify_hard_samples(
        pairs, target_to_confused, swarm, require_prediction_match
    )
    
    stats = {
        'original_count': len(pairs),
        'hard_count': len(hard_indices),
        'hard_fraction': len(hard_indices) / len(pairs) if pairs else 0,
    }
    
    if verbose:
        print(f"  Hard samples: {len(hard_indices)}/{len(pairs)} "
              f"({100*stats['hard_fraction']:.1f}%)")
    
    # Cap the number of duplicates
    max_duplicates = int(len(pairs) * max_oversample_ratio)
    n_duplicates = min(len(hard_indices), max_duplicates)
    
    if n_duplicates > 0:
        # Sample from hard indices (with replacement if needed)
        if n_duplicates <= len(hard_indices):
            dup_indices = rng.choice(hard_indices, size=n_duplicates, replace=False)
        else:
            dup_indices = rng.choice(hard_indices, size=n_duplicates, replace=True)
        
        duplicates = [pairs[i] for i in dup_indices]
    else:
        duplicates = []
    
    stats['duplicates_added'] = len(duplicates)
    stats['final_count'] = len(pairs) + len(duplicates)
    stats['oversample_ratio'] = len(duplicates) / len(pairs) if pairs else 0
    
    if verbose:
        print(f"  Duplicates added: {len(duplicates)} "
              f"({100*stats['oversample_ratio']:.1f}% of corpus)")
    
    # Combine and shuffle
    augmented = list(pairs) + duplicates
    indices = rng.permutation(len(augmented))
    augmented = [augmented[i] for i in indices]
    
    return augmented, stats


# Legacy compatibility - kept but deprecated
def get_hard_negative_weights(
    confusion_matrix: ConfusionMatrix,
    phonemes: List[str],
    hard_weight: float = 2.0,
    top_k_confusions: int = 30,
    min_confusion_count: int = 3,
) -> Dict[Tuple[str, str], float]:
    """
    DEPRECATED: Use get_confusion_targets + oversample_hard_negatives instead.
    
    This function builds a weight dict but the weights are not properly used.
    Kept for backward compatibility with known_hard_pairs preset.
    """
    weights: Dict[Tuple[str, str], float] = {}
    
    confusions = confusion_matrix.get_top_confusions(
        top_k=top_k_confusions, 
        min_count=min_confusion_count,
    )
    
    for cp in confusions:
        target = cp.target
        predicted = cp.predicted
        scaled_weight = hard_weight * (1.0 + np.sqrt(cp.confusion_rate))
        # Only store target->predicted direction now
        weights[(target, predicted)] = scaled_weight
    
    return weights


def convert_weights_to_targets(
    hard_weights: Dict[Tuple[str, str], float]
) -> Dict[str, Set[str]]:
    """Convert legacy hard_weights dict to target_to_confused format."""
    target_to_confused: Dict[str, Set[str]] = defaultdict(set)
    for (target, predicted), _ in hard_weights.items():
        target_to_confused[target].add(predicted)
    return dict(target_to_confused)


class ConfusionMiner:
    """
    End-to-end confusion mining for swarm training.
    
    Workflow:
    1. Train initial swarm (or load checkpoint)
    2. Build confusion matrix from predictions
    3. Identify hard pairs (target→predicted confusions only)
    4. Re-train with hard-negative oversampling (capped at ~15% of corpus)
    
    Key fix from v1: Only oversample when target=T for confusion (T→P),
    NOT when target=P. This prevents flooding with common phonemes like AH.
    """
    
    def __init__(
        self,
        phonemes: Optional[List[str]] = None,
        top_k_confusions: int = 30,
        min_confusion_count: int = 3,
        min_confusion_rate: float = 0.05,
        max_oversample_ratio: float = 0.15,
        require_prediction_match: bool = False,
    ):
        self.phonemes = phonemes or PHONEME_LIST.copy()
        self.top_k_confusions = top_k_confusions
        self.min_confusion_count = min_confusion_count
        self.min_confusion_rate = min_confusion_rate
        self.max_oversample_ratio = max_oversample_ratio
        self.require_prediction_match = require_prediction_match
        
        self.confusion_matrix: Optional[ConfusionMatrix] = None
        self.target_to_confused: Dict[str, Set[str]] = {}
        self.last_stats: Dict[str, Any] = {}
    
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
        
        # Get target→confused mappings (only TARGET side, not reverse)
        self.target_to_confused = get_confusion_targets(
            self.confusion_matrix,
            top_k_confusions=self.top_k_confusions,
            min_confusion_count=self.min_confusion_count,
            min_confusion_rate=self.min_confusion_rate,
        )
        
        if verbose:
            n_targets = len(self.target_to_confused)
            n_pairs = sum(len(v) for v in self.target_to_confused.values())
            print(f"\nIdentified {n_targets} target phonemes with {n_pairs} confusion pairs")
            print("Target phonemes being confused:")
            for target, confused in sorted(self.target_to_confused.items())[:10]:
                confused_str = ', '.join(sorted(confused)[:5])
                if len(confused) > 5:
                    confused_str += f"... (+{len(confused)-5} more)"
                print(f"  {target} → {{{confused_str}}}")
        
        return self.confusion_matrix
    
    def augment_training_data(
        self,
        pairs: List[AlignedPair],
        swarm=None,  # For prediction-based filtering
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
    ) -> List[AlignedPair]:
        """
        Augment training data by oversampling TRUE hard negatives.
        
        Only oversamples pairs where target=T for a confusion (T→P),
        NOT when target=P. Capped at max_oversample_ratio.
        
        Args:
            pairs: Original training pairs
            swarm: Optional FlySwarm for prediction-based filtering
            rng: Random generator
            verbose: Print progress
        
        Returns:
            Augmented training pairs
        """
        if not self.target_to_confused:
            if verbose:
                print("No confusions mined yet - returning original pairs")
            return pairs
        
        augmented, stats = oversample_hard_negatives(
            pairs,
            self.target_to_confused,
            swarm=swarm if self.require_prediction_match else None,
            require_prediction_match=self.require_prediction_match,
            max_oversample_ratio=self.max_oversample_ratio,
            rng=rng,
            verbose=verbose,
        )
        
        self.last_stats = stats
        
        if verbose:
            print(f"Augmented: {stats['original_count']} → {stats['final_count']} pairs")
        
        return augmented
    
    def get_hard_fraction(self, pairs: List[AlignedPair]) -> float:
        """Get fraction of pairs that would be marked as hard."""
        if not self.target_to_confused:
            return 0.0
        hard_indices = identify_hard_samples(pairs, self.target_to_confused)
        return len(hard_indices) / len(pairs) if pairs else 0.0
    
    def save_report(self, path: str):
        """Save confusion report to file."""
        if self.confusion_matrix is None:
            print("No confusion matrix - run mine_confusions first")
            return
        
        report = self.confusion_matrix.print_report(
            top_k=30, 
            min_count=self.min_confusion_count,
        )
        
        # Add mining stats
        lines = [report, "", "MINING CONFIGURATION:", "-" * 40]
        lines.append(f"  max_oversample_ratio: {self.max_oversample_ratio}")
        lines.append(f"  min_confusion_rate: {self.min_confusion_rate}")
        lines.append(f"  require_prediction_match: {self.require_prediction_match}")
        lines.append(f"  target phonemes with confusions: {len(self.target_to_confused)}")
        
        if self.last_stats:
            lines.append("")
            lines.append("LAST AUGMENTATION STATS:")
            for k, v in self.last_stats.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
        
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        print(f"Confusion report saved to {path}")
    
    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            'phonemes': self.phonemes,
            'top_k_confusions': self.top_k_confusions,
            'min_confusion_count': self.min_confusion_count,
            'min_confusion_rate': self.min_confusion_rate,
            'max_oversample_ratio': self.max_oversample_ratio,
            'require_prediction_match': self.require_prediction_match,
            'confusion_matrix': self.confusion_matrix.to_dict() if self.confusion_matrix else None,
            'target_to_confused': {t: list(ps) for t, ps in self.target_to_confused.items()},
            'last_stats': self.last_stats,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ConfusionMiner':
        """Deserialize from dict."""
        miner = cls(
            phonemes=d.get('phonemes'),
            top_k_confusions=d.get('top_k_confusions', 30),
            min_confusion_count=d.get('min_confusion_count', 3),
            min_confusion_rate=d.get('min_confusion_rate', 0.05),
            max_oversample_ratio=d.get('max_oversample_ratio', 0.15),
            require_prediction_match=d.get('require_prediction_match', False),
        )
        if d.get('confusion_matrix'):
            miner.confusion_matrix = ConfusionMatrix.from_dict(d['confusion_matrix'])
        for target, confused_list in d.get('target_to_confused', {}).items():
            miner.target_to_confused[target] = set(confused_list)
        miner.last_stats = d.get('last_stats', {})
        return miner


# Pre-defined known hard pairs from the task description
# Format: (target, predicted) - target is what it SHOULD be, predicted is the error
KNOWN_HARD_PAIRS = [
    ('IY', 'EH'),   # me: M IY → M EH
    ('IY', 'IH'),   # common IY confusion
    ('AE', 'AA'),   # vowel confusion
    ('AE', 'AH'),   # vowel confusion (very common)
    ('AA', 'AH'),   # vowel confusion (very common)
    ('AY', 'IY'),   # diphthong vs vowel
    ('AY', 'EH'),   # diphthong vs vowel
    ('IH', 'EH'),   # short i vs short e
    ('IH', 'AH'),   # common schwa confusion
    ('AO', 'AA'),   # open back vowels
    ('Z', 'S'),     # voicing confusion
]


def get_known_hard_targets() -> Dict[str, Set[str]]:
    """
    Get target→confused mappings for known hard pairs (from task spec).
    
    Can be used as a starting point before mining or as fallback.
    Returns dict: target -> {predicted phonemes it's confused with}
    """
    target_to_confused: Dict[str, Set[str]] = defaultdict(set)
    for target, predicted in KNOWN_HARD_PAIRS:
        target_to_confused[target].add(predicted)
    return dict(target_to_confused)


# Legacy compatibility
def get_known_hard_weights(
    hard_weight: float = 2.0,
) -> Dict[Tuple[str, str], float]:
    """DEPRECATED: Use get_known_hard_targets instead."""
    weights = {}
    for target, predicted in KNOWN_HARD_PAIRS:
        weights[(target, predicted)] = hard_weight
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
