"""
One-phoneme-per-fly specialist ensemble.

Instead of a single multi-class MB with 39 MBON phoneme bins, this implements
an ensemble of specialist flies: each fly is a small mushroom body that answers
a binary question like "is the next phoneme /K/?" given letter context.

At inference, all specialists score the context; pick the phoneme with the
strongest "yes" (MBON margin / confidence).

Biology analogy: compartment specialization in the real mushroom body. Different
MBON compartments respond to different learned associations. Here we make that
extreme - one "fly" per phoneme class.

Design:
- SpecialistFly: Binary classifier for one phoneme (YES/NO outputs)
- FlySwarm: Ensemble of 39 specialists (or subset for demo)
- Shared PN→KC expansion for efficiency, separate KC→MBON weights per specialist
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
import json
from pathlib import Path

from .phonemes import (
    NUM_PHONEMES, index_to_phoneme, phoneme_to_index, PHONEME_LIST, strip_stress
)
from .mushroom_body import (
    MushroomBodyConfig, encode_letter_context, get_input_dim, CHAR_VOCAB
)


@dataclass
class SpecialistConfig:
    """Configuration for a specialist fly."""
    n_pn: int = 300            # Projection neurons (smaller than full MB)
    n_kc: int = 2000           # Kenyon cells (smaller for binary task)
    pn_per_kc: int = 8         # PNs randomly connected to each KC
    kc_sparsity: float = 0.10  # Fraction of KCs active
    learning_rate: float = 0.08
    weight_init_mean: float = 1.0
    weight_clip_min: float = 0.01
    weight_clip_max: float = 10.0
    context_size: int = 3
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_pn': self.n_pn, 'n_kc': self.n_kc,
            'pn_per_kc': self.pn_per_kc, 'kc_sparsity': self.kc_sparsity,
            'learning_rate': self.learning_rate,
            'weight_init_mean': self.weight_init_mean,
            'weight_clip_min': self.weight_clip_min,
            'weight_clip_max': self.weight_clip_max,
            'context_size': self.context_size, 'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpecialistConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SharedExpansion:
    """
    Shared PN→KC expansion layer used by all specialists in a swarm.
    
    This mimics the biological reality where projection neurons and Kenyon cells
    are shared infrastructure, while each "compartment" (specialist) has its own
    MBON output weights.
    """
    
    def __init__(self, config: SpecialistConfig, rng: np.random.Generator):
        self.config = config
        self.input_dim = get_input_dim(config.context_size, include_phone_pos=True)
        
        # Input → PN projection (random, normalized)
        self.input_to_pn = rng.standard_normal(
            (self.input_dim, config.n_pn)
        ).astype(np.float32)
        norms = np.linalg.norm(self.input_to_pn, axis=0, keepdims=True)
        self.input_to_pn /= (norms + 1e-6)
        
        # PN → KC sparse random connectivity
        self.pn_kc_weights = np.zeros(
            (config.n_pn, config.n_kc), dtype=np.float32
        )
        for kc in range(config.n_kc):
            connected_pns = rng.choice(
                config.n_pn, size=config.pn_per_kc, replace=False
            )
            self.pn_kc_weights[connected_pns, kc] = 1.0 / config.pn_per_kc
    
    def encode_to_kc(
        self, 
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
    ) -> np.ndarray:
        """
        Transform letter context to sparse KC activity.
        
        Args:
            letter_context: Letter context string
            phoneme_pos: Normalized phoneme position [0, 1] for short-word disambiguation
            n_phonemes: Total phonemes in word
        """
        input_features = encode_letter_context(
            letter_context, 
            phoneme_pos=phoneme_pos,
            n_phonemes=n_phonemes,
        )
        
        # PN activity
        pn_raw = input_features @ self.input_to_pn
        pn_activity = np.maximum(0, pn_raw)
        if pn_activity.max() > 0:
            pn_activity = pn_activity / pn_activity.max()
        
        # KC activity (sparse winner-take-all)
        kc_input = pn_activity @ self.pn_kc_weights
        n_active = max(1, int(self.config.n_kc * self.config.kc_sparsity))
        threshold = np.partition(kc_input, -n_active)[-n_active]
        kc_activity = (kc_input >= threshold).astype(np.float32)
        
        return kc_activity


class SpecialistFly:
    """
    A specialist fly that answers: "Is the phoneme /P/?" (binary YES/NO).
    
    Uses a 2-output MBON structure:
    - MBON 0: YES (target phoneme matches)
    - MBON 1: NO (target phoneme doesn't match)
    
    Winner-take-all with inhibitory logic: minimum input wins.
    Learning: same anti-Hebbian dopamine rule as full MB.
    """
    
    def __init__(
        self,
        target_phoneme: str,
        shared_expansion: SharedExpansion,
        config: SpecialistConfig,
        rng: np.random.Generator,
    ):
        self.target_phoneme = strip_stress(target_phoneme)
        self.target_idx = phoneme_to_index(self.target_phoneme)
        self.shared = shared_expansion
        self.config = config
        self.rng = rng
        
        # KC → MBON weights (2 outputs: YES=0, NO=1)
        self.kc_mbon_weights = np.full(
            (config.n_kc, 2),
            config.weight_init_mean,
            dtype=np.float32
        )
        
        # Training stats
        self.training_steps = 0
        self.correct_count = 0
        self.total_count = 0
    
    def forward(self, kc_activity: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Run forward pass with pre-computed KC activity.
        
        Returns:
            is_yes: Whether the specialist votes YES
            confidence: Margin between YES and NO
            mbon_input: Raw MBON inputs
        """
        mbon_input = kc_activity @ self.kc_mbon_weights
        
        # Winner: minimum input wins (inhibitory logic)
        is_yes = mbon_input[0] < mbon_input[1]  # YES wins if YES input < NO input
        
        # Confidence: margin between winner and loser
        margin = abs(mbon_input[1] - mbon_input[0])
        baseline = abs(mbon_input[0]) + abs(mbon_input[1]) + 1e-6
        confidence = margin / baseline
        
        return is_yes, confidence, mbon_input
    
    def predict(
        self, 
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
    ) -> Tuple[bool, float]:
        """Predict YES/NO for this context."""
        kc_activity = self.shared.encode_to_kc(
            letter_context, phoneme_pos=phoneme_pos, n_phonemes=n_phonemes
        )
        is_yes, confidence, _ = self.forward(kc_activity)
        return is_yes, confidence
    
    def learn(self, kc_activity: np.ndarray, predicted_yes: bool, target_yes: bool):
        """Apply dopamine-modulated learning when wrong."""
        if predicted_yes == target_yes:
            return  # No learning when correct
        
        lr = self.config.learning_rate
        active_kc_mask = kc_activity > 0
        
        # Correct class index
        correct_idx = 0 if target_yes else 1
        wrong_idx = 1 if target_yes else 0
        
        # Depression: weaken connections to correct MBON
        self.kc_mbon_weights[active_kc_mask, correct_idx] *= (1 - lr)
        
        # Potentiation: strengthen connections to wrong MBON
        self.kc_mbon_weights[active_kc_mask, wrong_idx] *= (1 + lr)
        
        # Clip weights
        np.clip(
            self.kc_mbon_weights,
            self.config.weight_clip_min,
            self.config.weight_clip_max,
            out=self.kc_mbon_weights
        )
    
    def train_step(
        self, 
        letter_context: str, 
        actual_phoneme: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
    ) -> Tuple[bool, bool, float]:
        """
        Train on one example.
        
        Args:
            letter_context: Input context
            actual_phoneme: The actual phoneme for this context
            phoneme_pos: Normalized phoneme position [0, 1]
            n_phonemes: Total phonemes in word
        
        Returns:
            is_correct: Whether prediction was correct
            predicted_yes: What the fly predicted
            confidence: Prediction confidence
        """
        actual = strip_stress(actual_phoneme)
        target_yes = (actual == self.target_phoneme)
        
        kc_activity = self.shared.encode_to_kc(
            letter_context, phoneme_pos=phoneme_pos, n_phonemes=n_phonemes
        )
        predicted_yes, confidence, _ = self.forward(kc_activity)
        
        is_correct = (predicted_yes == target_yes)
        
        if not is_correct:
            self.learn(kc_activity, predicted_yes, target_yes)
        
        self.training_steps += 1
        self.total_count += 1
        if is_correct:
            self.correct_count += 1
        
        return is_correct, predicted_yes, confidence
    
    def get_accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    def reset_stats(self):
        self.correct_count = 0
        self.total_count = 0


# Voting strategies for swarm arbitration
VOTE_ARGMAX = 'argmax'      # Raw argmax over YES scores (original, can thrash)
VOTE_SOFTMAX = 'softmax'    # Softmax with temperature over YES scores
VOTE_MARGIN = 'margin'      # Require margin between best and second-best
VOTE_CALIBRATED = 'calibrated'  # Use calibrated probabilities (requires fit_calibration)

VALID_VOTE_STRATEGIES = [VOTE_ARGMAX, VOTE_SOFTMAX, VOTE_MARGIN, VOTE_CALIBRATED]
DEFAULT_VOTE_STRATEGY = VOTE_SOFTMAX


class PlattCalibrator:
    """
    Per-phoneme score calibration for comparable voting.
    
    Uses affine normalization: calibrated = (score - mean) / std * scale + shift
    This normalizes each phoneme's raw YES scores to a common range, enabling
    meaningful comparison across specialists with different response distributions.
    
    Biology analogy: per-MBON bias/gain adjustment, like neuromodulation
    that calibrates different compartments' output ranges.
    """
    
    def __init__(self, phonemes: List[str]):
        self.phonemes = list(phonemes)
        self.n_phonemes = len(phonemes)
        # Parameters: [mean, std] for z-score normalization
        self.params = {p: np.array([0.0, 1.0], dtype=np.float32) for p in phonemes}
        self.is_fitted = False
    
    def fit(
        self, 
        scores_by_phoneme: Dict[str, List[float]], 
        labels_by_phoneme: Dict[str, List[int]],
        max_iter: int = 100,
    ):
        """
        Fit calibration parameters from score distributions.
        
        For each phoneme, learns the mean and std of scores when that phoneme
        is the CORRECT answer, then uses these to normalize scores.
        
        Args:
            scores_by_phoneme: Dict mapping phoneme to list of raw YES scores
            labels_by_phoneme: Dict mapping phoneme to list of binary labels (1=correct, 0=wrong)
            max_iter: Unused (kept for API compatibility)
        """
        for phoneme in self.phonemes:
            scores = np.array(scores_by_phoneme.get(phoneme, []))
            labels = np.array(labels_by_phoneme.get(phoneme, []))
            
            if len(scores) < 10:
                continue
            
            # Get scores only when this phoneme is the CORRECT answer
            correct_mask = labels == 1
            if correct_mask.sum() < 5:
                # Not enough positive examples
                continue
            
            correct_scores = scores[correct_mask]
            mean = float(np.mean(correct_scores))
            std = float(np.std(correct_scores))
            if std < 0.01:
                std = 0.01  # Prevent division by zero
            
            self.params[phoneme] = np.array([mean, std], dtype=np.float32)
        
        self.is_fitted = True
    
    def calibrate(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply calibration to raw scores for comparable voting.
        
        Normalizes each phoneme's score by its learned distribution, then
        applies sigmoid to convert to probability-like values.
        """
        calibrated = {}
        for phoneme, score in scores.items():
            if phoneme in self.params:
                mean, std = self.params[phoneme]
                z = (score - mean) / std
                z = np.clip(z, -10, 10)
                # Convert z-score to probability-like value via sigmoid
                calibrated[phoneme] = 1.0 / (1.0 + np.exp(-z))
            else:
                calibrated[phoneme] = 1.0 / (1.0 + np.exp(-score))
        return calibrated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'phonemes': self.phonemes,
            'params': {p: v.tolist() for p, v in self.params.items()},
            'is_fitted': self.is_fitted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PlattCalibrator':
        cal = cls(d.get('phonemes', []))
        cal.params = {p: np.array(v, dtype=np.float32) for p, v in d.get('params', {}).items()}
        cal.is_fitted = d.get('is_fitted', False)
        return cal


@dataclass 
class SwarmConfig:
    """Configuration for a fly swarm."""
    specialist_config: SpecialistConfig = field(default_factory=SpecialistConfig)
    phonemes: List[str] = field(default_factory=lambda: PHONEME_LIST.copy())
    seed: int = 42
    vote_strategy: str = DEFAULT_VOTE_STRATEGY  # 'argmax', 'softmax', or 'margin'
    vote_temperature: float = 0.5               # Temperature for softmax voting
    vote_margin: float = 0.1                    # Minimum margin for margin voting
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'specialist_config': self.specialist_config.to_dict(),
            'phonemes': self.phonemes,
            'seed': self.seed,
            'vote_strategy': self.vote_strategy,
            'vote_temperature': self.vote_temperature,
            'vote_margin': self.vote_margin,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SwarmConfig':
        spec_config = SpecialistConfig.from_dict(d.get('specialist_config', {}))
        return cls(
            specialist_config=spec_config,
            phonemes=d.get('phonemes', PHONEME_LIST.copy()),
            seed=d.get('seed', 42),
            vote_strategy=d.get('vote_strategy', DEFAULT_VOTE_STRATEGY),
            vote_temperature=d.get('vote_temperature', 0.5),
            vote_margin=d.get('vote_margin', 0.1),
        )


class FlySwarm:
    """
    Ensemble of specialist flies, one per phoneme.
    
    At inference, all specialists score the letter context. The phoneme is
    selected using a configurable voting strategy (arbitrator):
    
    - argmax: Raw argmax over YES scores (can thrash when specialists overconfident)
    - softmax: Softmax with temperature over YES scores (more robust)
    - margin: Require margin between best and second-best (conservative)
    
    This is closer to biological compartment specialization: each "compartment"
    (specialist fly) responds to its learned association.
    """
    
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Validate vote strategy
        if self.config.vote_strategy not in VALID_VOTE_STRATEGIES:
            print(f"Warning: Unknown vote strategy '{self.config.vote_strategy}', "
                  f"using '{DEFAULT_VOTE_STRATEGY}'")
            self.config.vote_strategy = DEFAULT_VOTE_STRATEGY
        
        # Create shared expansion layer
        self.shared = SharedExpansion(
            self.config.specialist_config,
            self.rng
        )
        
        # Create specialist flies for each phoneme
        self.specialists: Dict[str, SpecialistFly] = {}
        for phoneme in self.config.phonemes:
            p = strip_stress(phoneme)
            if p not in self.specialists:
                self.specialists[p] = SpecialistFly(
                    target_phoneme=p,
                    shared_expansion=self.shared,
                    config=self.config.specialist_config,
                    rng=self.rng,
                )
        
        self.phoneme_list = list(self.specialists.keys())
        
        # Calibrator for normalizing specialist scores (per-MBON bias/gain)
        self.calibrator = PlattCalibrator(self.phoneme_list)
        
        print(f"FlySwarm: {len(self.specialists)} specialist flies for phonemes: "
              f"{', '.join(self.phoneme_list[:5])}{'...' if len(self.phoneme_list) > 5 else ''}")
        print(f"  Vote strategy: {self.config.vote_strategy} "
              f"(temp={self.config.vote_temperature}, margin={self.config.vote_margin})")
    
    def _get_raw_scores(
        self, 
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
    ) -> Dict[str, float]:
        """Get raw YES confidence scores from all specialists."""
        kc_activity = self.shared.encode_to_kc(
            letter_context, phoneme_pos=phoneme_pos, n_phonemes=n_phonemes
        )
        
        scores = {}
        for phoneme, specialist in self.specialists.items():
            is_yes, confidence, mbon_input = specialist.forward(kc_activity)
            # Score = confidence if YES, negative confidence if NO
            if is_yes:
                scores[phoneme] = confidence
            else:
                scores[phoneme] = -confidence
        
        return scores
    
    def _vote_argmax(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """Raw argmax voting: pick highest score."""
        winner = max(scores, key=scores.get)
        return winner, scores[winner]
    
    def _vote_softmax(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Softmax voting: convert scores to probabilities and pick winner.
        
        Note: While softmax+argmax is mathematically equivalent to argmax,
        the probability values matter for downstream uses like beam search
        or confidence thresholding. Lower temperature sharpens the distribution.
        """
        phonemes = list(scores.keys())
        raw_scores = np.array([scores[p] for p in phonemes])
        
        # Apply temperature (lower = sharper peaks)
        temp = max(self.config.vote_temperature, 0.01)
        scaled = raw_scores / temp
        
        # Softmax with numerical stability
        scaled = scaled - scaled.max()
        exp_scores = np.exp(scaled)
        probs = exp_scores / exp_scores.sum()
        
        # Pick highest probability
        winner_idx = np.argmax(probs)
        winner = phonemes[winner_idx]
        
        return winner, float(probs[winner_idx])
    
    def _vote_margin(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Margin voting: require clear margin between best and second-best.
        
        If margin is sufficient, returns best with full confidence.
        If margin is insufficient, does softmax among top-k candidates
        to get a more nuanced decision when specialists disagree.
        """
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        
        if len(sorted_items) < 2:
            return sorted_items[0][0], sorted_items[0][1]
        
        best_phone, best_score = sorted_items[0]
        second_phone, second_score = sorted_items[1]
        
        margin = best_score - second_score
        
        # If margin is sufficient, return best
        if margin >= self.config.vote_margin:
            return best_phone, best_score
        
        # If margin insufficient, apply softmax to top-k candidates only
        # This gives more weight to close alternatives rather than penalizing
        top_k = min(5, len(sorted_items))
        top_phones = [p for p, s in sorted_items[:top_k]]
        top_scores = np.array([s for p, s in sorted_items[:top_k]])
        
        # Softmax with temperature
        temp = max(self.config.vote_temperature, 0.01)
        scaled = top_scores / temp
        scaled = scaled - scaled.max()
        exp_scores = np.exp(scaled)
        probs = exp_scores / exp_scores.sum()
        
        winner_idx = np.argmax(probs)
        return top_phones[winner_idx], float(probs[winner_idx])
    
    def _vote_calibrated(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Calibrated voting: use Platt-scaled probabilities.
        
        Applies per-phoneme calibration to normalize raw scores to comparable
        probabilities, then picks the highest. Requires fit_calibration() first.
        """
        if not self.calibrator.is_fitted:
            # Fallback to softmax if not calibrated
            return self._vote_softmax(scores)
        
        calibrated = self.calibrator.calibrate(scores)
        
        # Normalize to sum to 1 for proper probability distribution
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {p: v / total for p, v in calibrated.items()}
        
        winner = max(calibrated, key=calibrated.get)
        return winner, calibrated[winner]
    
    def predict(
        self, 
        letter_context: str,
        vote_strategy: Optional[str] = None,
        phoneme_pos: float = None,
        n_phonemes: int = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict phoneme by ensemble voting with configurable arbitrator.
        
        Args:
            letter_context: Input context string
            vote_strategy: Override config vote strategy (argmax/softmax/margin/calibrated)
            phoneme_pos: Normalized phoneme position [0, 1] for disambiguation
            n_phonemes: Total phonemes in word
        
        Returns:
            phoneme: Predicted phoneme
            confidence: Winning confidence/probability
            all_scores: Dict mapping phonemes to their raw YES scores
        """
        scores = self._get_raw_scores(letter_context, phoneme_pos=phoneme_pos, n_phonemes=n_phonemes)
        
        strategy = vote_strategy or self.config.vote_strategy
        
        if strategy == VOTE_ARGMAX:
            winner, confidence = self._vote_argmax(scores)
        elif strategy == VOTE_SOFTMAX:
            winner, confidence = self._vote_softmax(scores)
        elif strategy == VOTE_MARGIN:
            winner, confidence = self._vote_margin(scores)
        elif strategy == VOTE_CALIBRATED:
            winner, confidence = self._vote_calibrated(scores)
        else:
            # Fallback to softmax
            winner, confidence = self._vote_softmax(scores)
        
        return winner, confidence, scores
    
    def collect_calibration_data(
        self, 
        pairs: List['AlignedPair'],
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
        """
        Collect raw scores and labels for calibration fitting.
        
        Args:
            pairs: List of AlignedPair training examples
        
        Returns:
            scores_by_phoneme: Raw YES scores grouped by target phoneme
            labels_by_phoneme: Binary labels (1=correct) grouped by target phoneme
        """
        from .alignment import AlignedPair
        
        scores_by_phoneme: Dict[str, List[float]] = {p: [] for p in self.phoneme_list}
        labels_by_phoneme: Dict[str, List[int]] = {p: [] for p in self.phoneme_list}
        
        for pair in pairs:
            target = strip_stress(pair.phoneme)
            if target not in self.phoneme_list:
                continue
            
            # Use phoneme position for better calibration
            scores = self._get_raw_scores(
                pair.letter_context, 
                phoneme_pos=pair.phoneme_pos,
                n_phonemes=pair.n_phonemes,
            )
            
            # For each phoneme, record its score and whether it's the correct target
            for phoneme, score in scores.items():
                scores_by_phoneme[phoneme].append(score)
                labels_by_phoneme[phoneme].append(1 if phoneme == target else 0)
        
        return scores_by_phoneme, labels_by_phoneme
    
    def fit_calibration(self, pairs: List['AlignedPair'], verbose: bool = True):
        """
        Fit per-phoneme calibration on held-out data.
        
        Args:
            pairs: Held-out AlignedPair examples for calibration
            verbose: Print progress
        """
        if verbose:
            print("Collecting calibration data...")
        
        scores_by_phoneme, labels_by_phoneme = self.collect_calibration_data(pairs)
        
        if verbose:
            total_samples = sum(len(v) for v in scores_by_phoneme.values())
            print(f"  Total samples: {total_samples}")
            print("Fitting Platt calibration...")
        
        self.calibrator.fit(scores_by_phoneme, labels_by_phoneme)
        
        if verbose:
            print("  Calibration fitted successfully")
    
    def set_vote_strategy(self, strategy: str, temperature: float = None, margin: float = None):
        """Change voting strategy at runtime."""
        if strategy in VALID_VOTE_STRATEGIES:
            self.config.vote_strategy = strategy
        if temperature is not None:
            self.config.vote_temperature = temperature
        if margin is not None:
            self.config.vote_margin = margin
    
    def predict_word(
        self, 
        word: str, 
        n_phonemes: int, 
        context_size: int = None,
        vote_strategy: Optional[str] = None,
    ) -> List[str]:
        """Predict phoneme sequence for a word."""
        if context_size is None:
            context_size = self.config.specialist_config.context_size
        
        word = word.lower()
        n_letters = len(word)
        phonemes = []
        
        for p_idx in range(n_phonemes):
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
            
            phoneme, _, _ = self.predict(
                letter_context, 
                vote_strategy=vote_strategy,
                phoneme_pos=phoneme_pos,
                n_phonemes=n_phonemes,
            )
            phonemes.append(phoneme)
        
        return phonemes
    
    def predict_word_beam(
        self,
        word: str,
        n_phonemes: int,
        lm: 'PhonemeNGramLM',
        beam_width: int = 5,
        lm_weight: float = 0.3,
        top_k: int = 5,
        vote_strategy: Optional[str] = None,
        context_size: int = None,
    ) -> List[str]:
        """
        Predict phoneme sequence using beam search with LM rescoring.
        
        The swarm proposes top-k candidates per slot; beam search finds
        the best sequence considering both fly scores and LM probability.
        
        Args:
            word: Input word
            n_phonemes: Number of phonemes to predict
            lm: PhonemeNGramLM for rescoring
            beam_width: Number of hypotheses to keep
            lm_weight: Weight for LM score (vs fly score)
            top_k: Candidates to consider per slot
            vote_strategy: Base voting strategy for scores
            context_size: Letter context size
        
        Returns:
            Best phoneme sequence
        """
        from .phoneme_lm import BeamSearchDecoder
        
        decoder = BeamSearchDecoder(
            swarm=self,
            lm=lm,
            beam_width=beam_width,
            lm_weight=lm_weight,
            top_k=top_k,
        )
        
        phonemes, _ = decoder.decode(
            word, n_phonemes, 
            context_size=context_size,
            vote_strategy=vote_strategy,
        )
        return phonemes
    
    def train_step(
        self, 
        letter_context: str, 
        correct_phoneme: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
    ) -> Tuple[bool, str]:
        """
        Train all specialists on one example.
        
        Each specialist learns whether this context is/isn't its target phoneme.
        
        Args:
            letter_context: Input context
            correct_phoneme: Target phoneme
            phoneme_pos: Normalized phoneme position [0, 1]
            n_phonemes: Total phonemes in word
        
        Returns:
            is_correct: Whether ensemble prediction was correct
            predicted: Predicted phoneme
        """
        correct = strip_stress(correct_phoneme)
        
        # Train each specialist with position info
        for phoneme, specialist in self.specialists.items():
            specialist.train_step(
                letter_context, correct,
                phoneme_pos=phoneme_pos, n_phonemes=n_phonemes,
            )
        
        # Get ensemble prediction for accuracy tracking
        predicted, _, _ = self.predict(
            letter_context, phoneme_pos=phoneme_pos, n_phonemes=n_phonemes
        )
        is_correct = (predicted == correct)
        
        return is_correct, predicted
    
    def get_phoneme_accuracies(self) -> Dict[str, float]:
        """Get per-phoneme specialist accuracy."""
        return {p: s.get_accuracy() for p, s in self.specialists.items()}
    
    def reset_stats(self):
        for specialist in self.specialists.values():
            specialist.reset_stats()
    
    def save(self, path: str):
        """Save swarm to directory or npz file, including calibration if fitted."""
        path = Path(path)
        
        # Collect all specialist weights
        specialist_weights = {}
        for phoneme, specialist in self.specialists.items():
            specialist_weights[f"weights_{phoneme}"] = specialist.kc_mbon_weights
        
        np.savez(
            path,
            # Shared expansion
            input_to_pn=self.shared.input_to_pn,
            pn_kc_weights=self.shared.pn_kc_weights,
            # Config
            config=json.dumps(self.config.to_dict()),
            # Calibration parameters
            calibration=json.dumps(self.calibrator.to_dict()),
            # All specialist weights
            **specialist_weights,
        )
        cal_status = "with calibration" if self.calibrator.is_fitted else "uncalibrated"
        print(f"Swarm saved to {path} ({cal_status})")
    
    @classmethod
    def load(cls, path: str) -> 'FlySwarm':
        """Load swarm from file, including calibration if present."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = SwarmConfig.from_dict(json.loads(str(data['config'])))
        swarm = cls(config)
        
        # Load shared expansion
        swarm.shared.input_to_pn = data['input_to_pn']
        swarm.shared.pn_kc_weights = data['pn_kc_weights']
        
        # Load specialist weights
        for phoneme in swarm.specialists:
            key = f"weights_{phoneme}"
            if key in data:
                swarm.specialists[phoneme].kc_mbon_weights = data[key]
        
        # Load calibration if present
        if 'calibration' in data:
            try:
                cal_dict = json.loads(str(data['calibration']))
                swarm.calibrator = PlattCalibrator.from_dict(cal_dict)
            except Exception:
                pass  # Use default uncalibrated
        
        cal_status = "calibrated" if swarm.calibrator.is_fitted else "uncalibrated"
        print(f"Swarm loaded from {path} ({len(swarm.specialists)} specialists, {cal_status})")
        return swarm


def get_demo_phonemes() -> List[str]:
    """
    Get phonemes needed for demo vocabulary.
    
    Uses the actual demo words from get_demo_words() to ensure coverage
    for evaluation. This includes: cat, bat, dog, go, no, hi, bye, yes, me, you
    """
    from .lexicon import get_phonemes, get_demo_words
    
    # Use actual demo words to ensure coverage for evaluation
    demo_words = get_demo_words()
    
    needed = set()
    for word in demo_words:
        try:
            phones = get_phonemes(word, allow_g2p=True)
            for p in phones:
                needed.add(strip_stress(p))
        except Exception:
            pass
    
    return sorted(needed)


if __name__ == "__main__":
    print("Testing FlySwarm...")
    
    # Test with subset of phonemes for speed
    test_phonemes = ['K', 'AE', 'T', 'D', 'AO', 'G']
    config = SwarmConfig(
        phonemes=test_phonemes,
        seed=42,
    )
    
    swarm = FlySwarm(config)
    
    # Test prediction
    test_contexts = ["_ca_t__", "___dog_", "_mush__"]
    for ctx in test_contexts:
        phoneme, conf, scores = swarm.predict(ctx)
        print(f"\nContext '{ctx}':")
        print(f"  Predicted: {phoneme} (confidence: {conf:.3f})")
        top_3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(f"  Top 3: {top_3}")
    
    # Test training step
    print("\nTraining step test:")
    is_correct, pred = swarm.train_step("_ca_t__", "K")
    print(f"  Context '_ca_t__' -> target 'K', predicted '{pred}', correct={is_correct}")
