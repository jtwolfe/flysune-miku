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
        self.input_dim = get_input_dim(config.context_size)
        
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
    
    def encode_to_kc(self, letter_context: str) -> np.ndarray:
        """Transform letter context to sparse KC activity."""
        input_features = encode_letter_context(letter_context)
        
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
    
    def predict(self, letter_context: str) -> Tuple[bool, float]:
        """Predict YES/NO for this context."""
        kc_activity = self.shared.encode_to_kc(letter_context)
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
        self, letter_context: str, actual_phoneme: str
    ) -> Tuple[bool, bool, float]:
        """
        Train on one example.
        
        Args:
            letter_context: Input context
            actual_phoneme: The actual phoneme for this context
        
        Returns:
            is_correct: Whether prediction was correct
            predicted_yes: What the fly predicted
            confidence: Prediction confidence
        """
        actual = strip_stress(actual_phoneme)
        target_yes = (actual == self.target_phoneme)
        
        kc_activity = self.shared.encode_to_kc(letter_context)
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


@dataclass 
class SwarmConfig:
    """Configuration for a fly swarm."""
    specialist_config: SpecialistConfig = field(default_factory=SpecialistConfig)
    phonemes: List[str] = field(default_factory=lambda: PHONEME_LIST.copy())
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'specialist_config': self.specialist_config.to_dict(),
            'phonemes': self.phonemes,
            'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SwarmConfig':
        spec_config = SpecialistConfig.from_dict(d.get('specialist_config', {}))
        return cls(
            specialist_config=spec_config,
            phonemes=d.get('phonemes', PHONEME_LIST.copy()),
            seed=d.get('seed', 42),
        )


class FlySwarm:
    """
    Ensemble of specialist flies, one per phoneme.
    
    At inference, all specialists score the letter context. The phoneme whose
    specialist has the strongest "YES" vote (highest confidence for YES) wins.
    
    This is closer to biological compartment specialization: each "compartment"
    (specialist fly) responds to its learned association.
    """
    
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
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
        print(f"FlySwarm: {len(self.specialists)} specialist flies for phonemes: "
              f"{', '.join(self.phoneme_list[:5])}{'...' if len(self.phoneme_list) > 5 else ''}")
    
    def predict(self, letter_context: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict phoneme by ensemble voting.
        
        All specialists evaluate the context. The phoneme whose specialist
        has the strongest YES vote wins.
        
        Returns:
            phoneme: Predicted phoneme
            confidence: Winning confidence
            all_scores: Dict mapping phonemes to their YES confidence
        """
        kc_activity = self.shared.encode_to_kc(letter_context)
        
        scores = {}
        for phoneme, specialist in self.specialists.items():
            is_yes, confidence, mbon_input = specialist.forward(kc_activity)
            # Score = confidence if YES, negative confidence if NO
            if is_yes:
                scores[phoneme] = confidence
            else:
                # If specialist says NO, give it a lower score
                # Use the inverse of NO confidence as a tie-breaker
                scores[phoneme] = -confidence
        
        # Winner: highest score (strongest YES)
        winner = max(scores, key=scores.get)
        
        return winner, scores[winner], scores
    
    def predict_word(self, word: str, n_phonemes: int, context_size: int = None) -> List[str]:
        """Predict phoneme sequence for a word."""
        if context_size is None:
            context_size = self.config.specialist_config.context_size
        
        word = word.lower()
        n_letters = len(word)
        phonemes = []
        
        for p_idx in range(n_phonemes):
            if n_phonemes == 1:
                letter_pos = n_letters // 2
            else:
                letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
            letter_pos = max(0, min(letter_pos, n_letters - 1))
            
            context_chars = []
            for offset in range(-context_size, context_size + 1):
                idx = letter_pos + offset
                if 0 <= idx < n_letters:
                    context_chars.append(word[idx])
                else:
                    context_chars.append('_')
            letter_context = ''.join(context_chars)
            
            phoneme, _, _ = self.predict(letter_context)
            phonemes.append(phoneme)
        
        return phonemes
    
    def train_step(self, letter_context: str, correct_phoneme: str) -> Tuple[bool, str]:
        """
        Train all specialists on one example.
        
        Each specialist learns whether this context is/isn't its target phoneme.
        
        Returns:
            is_correct: Whether ensemble prediction was correct
            predicted: Predicted phoneme
        """
        correct = strip_stress(correct_phoneme)
        
        # Train each specialist
        for phoneme, specialist in self.specialists.items():
            specialist.train_step(letter_context, correct)
        
        # Get ensemble prediction for accuracy tracking
        predicted, _, _ = self.predict(letter_context)
        is_correct = (predicted == correct)
        
        return is_correct, predicted
    
    def get_phoneme_accuracies(self) -> Dict[str, float]:
        """Get per-phoneme specialist accuracy."""
        return {p: s.get_accuracy() for p, s in self.specialists.items()}
    
    def reset_stats(self):
        for specialist in self.specialists.values():
            specialist.reset_stats()
    
    def save(self, path: str):
        """Save swarm to directory or npz file."""
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
            # All specialist weights
            **specialist_weights,
        )
        print(f"Swarm saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FlySwarm':
        """Load swarm from file."""
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
        
        print(f"Swarm loaded from {path} ({len(swarm.specialists)} specialists)")
        return swarm


def get_demo_phonemes() -> List[str]:
    """
    Get phonemes needed for demo vocabulary.
    
    These are the phonemes that appear in the key demo words:
    cat, dog, mushroom, hatsune, australia, kenyon, chaos, connectome
    """
    from .lexicon import get_phonemes
    
    demo_words = [
        'cat', 'dog', 'mushroom', 'hatsune', 'australia', 
        'kenyon', 'chaos', 'connectome'
    ]
    
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
