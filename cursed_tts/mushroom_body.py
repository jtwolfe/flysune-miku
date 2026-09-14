"""
Mushroom body model for grapheme-to-phoneme classification.

This is a biologically-inspired G2P classifier, like the FlyWire hiragana OCR demo:
- Input: letter context (graphemes around current position)
- Output: phoneme classification (MBON compartments = ARPAbet classes)
- Learning: dopamine-style plasticity when wrong

Architecture:
1. Letter context → PN-like encoding (character features)
2. PN → KC sparse expansion (random connectivity, ~10% active)
3. KC → MBON weights (learned via anti-Hebbian plasticity)
4. Winner-take-all: minimum MBON input wins (inhibitory logic)

Learning rule (Hige 2015 / Handler 2019):
- Only update when wrong (dopamine signal)
- DEPRESS synapses from active KCs to correct MBON (make it win more easily)
- POTENTIATE synapses from active KCs to wrong MBON (make it win less)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import json
from pathlib import Path

from .phonemes import NUM_PHONEMES, index_to_phoneme, phoneme_to_index, PHONEME_LIST


@dataclass
class MushroomBodyConfig:
    """Configuration for the G2P mushroom body model."""
    # Network sizes
    n_pn: int = 500            # Projection neurons
    n_kc: int = 5000           # Kenyon cells
    n_mbon: int = NUM_PHONEMES # MBON compartments = phoneme classes (39)
    
    # Connectivity
    pn_per_kc: int = 10        # PNs randomly connected to each KC
    kc_sparsity: float = 0.10  # Fraction of KCs active (~10%, FlyWire-style)
    
    # Learning
    learning_rate: float = 0.05
    weight_init_mean: float = 1.0
    weight_clip_min: float = 0.01
    weight_clip_max: float = 10.0
    
    # Input encoding
    context_size: int = 3      # Letters on each side (total window = 2*3+1 = 7)
    
    # Random seed
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_pn': self.n_pn, 'n_kc': self.n_kc, 'n_mbon': self.n_mbon,
            'pn_per_kc': self.pn_per_kc, 'kc_sparsity': self.kc_sparsity,
            'learning_rate': self.learning_rate, 'weight_init_mean': self.weight_init_mean,
            'weight_clip_min': self.weight_clip_min, 'weight_clip_max': self.weight_clip_max,
            'context_size': self.context_size, 'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MushroomBodyConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Character vocabulary
CHAR_VOCAB = '_abcdefghijklmnopqrstuvwxyz'  # '_' = padding
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
NUM_CHARS = len(CHAR_VOCAB)  # 27


def encode_letter_context(
    context: str,
    phoneme_pos: float = None,
    n_phonemes: int = None,
) -> np.ndarray:
    """
    Encode letter context as PN-like feature vector.
    
    Features:
    1. Position-specific one-hot for each character (7 positions × 27 chars = 189)
    2. Bigram features (26×26 = 676, only letter pairs)
    3. Center character one-hot (27)
    4. Phone slot position features (8 dims) - helps distinguish slots in short words
    
    Args:
        context: Letter context string (e.g., "_ca_t__")
        phoneme_pos: Normalized phoneme position (0.0 to 1.0), None for legacy mode
        n_phonemes: Total phonemes in word, None for legacy mode
    
    Total: ~900 features (892 base + 8 position)
    """
    context = context.lower()
    context_len = len(context)
    
    features = []
    
    # 1. Position-specific one-hot encoding
    pos_onehot = np.zeros(context_len * NUM_CHARS, dtype=np.float32)
    for i, c in enumerate(context):
        if c in CHAR_TO_IDX:
            pos_onehot[i * NUM_CHARS + CHAR_TO_IDX[c]] = 1.0
        else:
            pos_onehot[i * NUM_CHARS + CHAR_TO_IDX['_']] = 1.0  # Unknown → padding
    features.append(pos_onehot)
    
    # 2. Bigram features (character pairs, ignoring padding)
    bigram_features = np.zeros(26 * 26, dtype=np.float32)
    for i in range(context_len - 1):
        c1, c2 = context[i], context[i + 1]
        if 'a' <= c1 <= 'z' and 'a' <= c2 <= 'z':
            idx = (ord(c1) - ord('a')) * 26 + (ord(c2) - ord('a'))
            bigram_features[idx] = 1.0
    features.append(bigram_features)
    
    # 3. Center character one-hot (emphasized)
    center_idx = context_len // 2
    center_onehot = np.zeros(NUM_CHARS, dtype=np.float32)
    if 0 <= center_idx < context_len:
        c = context[center_idx]
        if c in CHAR_TO_IDX:
            center_onehot[CHAR_TO_IDX[c]] = 2.0  # Stronger weight for center
    features.append(center_onehot)
    
    # 4. Phone slot position features (helps distinguish slots in short words like "me")
    # This tells the network "this is phoneme 0 vs phoneme 1" even when letter contexts overlap
    phone_pos_features = np.zeros(8, dtype=np.float32)
    if phoneme_pos is not None:
        # Continuous position encoding (sinusoidal-like)
        phone_pos_features[0] = phoneme_pos                    # Raw position [0, 1]
        phone_pos_features[1] = 1.0 - phoneme_pos              # Complement
        phone_pos_features[2] = np.sin(np.pi * phoneme_pos)    # Sin encoding
        phone_pos_features[3] = np.cos(np.pi * phoneme_pos)    # Cos encoding
        # Discrete slot indicators (first/middle/last)
        phone_pos_features[4] = 1.0 if phoneme_pos < 0.25 else 0.0   # First slot
        phone_pos_features[5] = 1.0 if phoneme_pos > 0.75 else 0.0   # Last slot
        # Phoneme count hint (short word = more overlap)
        if n_phonemes is not None:
            phone_pos_features[6] = 1.0 / max(n_phonemes, 1)   # Inverse count
            phone_pos_features[7] = 1.0 if n_phonemes <= 2 else 0.0  # Short word flag
    features.append(phone_pos_features)
    
    return np.concatenate(features)


def get_input_dim(context_size: int = 3, include_phone_pos: bool = True) -> int:
    """Get input feature dimension for a context size."""
    context_len = 2 * context_size + 1
    base_dim = context_len * NUM_CHARS + 26 * 26 + NUM_CHARS
    phone_pos_dim = 8 if include_phone_pos else 0
    return base_dim + phone_pos_dim


class MushroomBody:
    """
    G2P mushroom body classifier.
    
    Classifies letter context → phoneme using sparse KC coding and
    dopamine-modulated learning.
    """
    
    def __init__(self, config: Optional[MushroomBodyConfig] = None):
        self.config = config or MushroomBodyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Input dimension (from letter context encoding)
        self.input_dim = get_input_dim(self.config.context_size)
        
        # Initialize network layers
        self._init_input_to_pn()
        self._init_pn_to_kc()
        self._init_kc_to_mbon()
        
        # Training statistics
        self.training_steps = 0
        self.correct_count = 0
        self.total_count = 0
    
    def _init_input_to_pn(self):
        """Initialize projection from input features to PN layer."""
        # Random projection with normalization
        self.input_to_pn = self.rng.standard_normal(
            (self.input_dim, self.config.n_pn)
        ).astype(np.float32)
        # Normalize columns
        norms = np.linalg.norm(self.input_to_pn, axis=0, keepdims=True)
        self.input_to_pn /= (norms + 1e-6)
    
    def _init_pn_to_kc(self):
        """Initialize sparse random PN→KC connectivity (FlyWire-style)."""
        # Use a sparse matrix for efficient PN→KC computation
        # Each KC receives input from a random subset of PNs
        self.pn_kc_weights = np.zeros((self.config.n_pn, self.config.n_kc), dtype=np.float32)
        
        for kc in range(self.config.n_kc):
            connected_pns = self.rng.choice(
                self.config.n_pn,
                size=self.config.pn_per_kc,
                replace=False
            )
            self.pn_kc_weights[connected_pns, kc] = 1.0 / self.config.pn_per_kc
    
    def _init_kc_to_mbon(self):
        """Initialize KC→MBON weights (learnable)."""
        self.kc_mbon_weights = np.full(
            (self.config.n_kc, self.config.n_mbon),
            self.config.weight_init_mean,
            dtype=np.float32
        )
    
    def encode_context(self, letter_context: str) -> np.ndarray:
        """Encode letter context to input features."""
        return encode_letter_context(letter_context)
    
    def input_to_pn_activity(self, input_features: np.ndarray) -> np.ndarray:
        """Transform input features to PN activity."""
        pn_raw = input_features @ self.input_to_pn
        pn_activity = np.maximum(0, pn_raw)  # ReLU activation
        
        # Normalize to reasonable range
        if pn_activity.max() > 0:
            pn_activity = pn_activity / pn_activity.max()
        
        return pn_activity.astype(np.float32)
    
    def pn_to_kc_activity(self, pn_activity: np.ndarray) -> np.ndarray:
        """Transform PN activity to sparse KC activity."""
        # Vectorized: matrix multiply PN activity by PN→KC weights
        kc_input = pn_activity @ self.pn_kc_weights
        
        # Sparse winner-take-all: top ~10% KCs fire
        n_active = max(1, int(self.config.n_kc * self.config.kc_sparsity))
        threshold = np.partition(kc_input, -n_active)[-n_active]
        
        kc_activity = (kc_input >= threshold).astype(np.float32)
        
        return kc_activity
    
    def kc_to_mbon_input(self, kc_activity: np.ndarray) -> np.ndarray:
        """Compute MBON inputs from KC activity."""
        return kc_activity @ self.kc_mbon_weights
    
    def select_winner(self, mbon_input: np.ndarray) -> int:
        """Select winning MBON (minimum input wins - inhibitory logic)."""
        return int(np.argmin(mbon_input))
    
    def forward(self, letter_context: str) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Run forward pass: letter context → phoneme prediction.
        
        Args:
            letter_context: Letter context string (e.g., "_ca_t__" for 'cat' at position 1)
        
        Returns:
            predicted_idx: Index of predicted phoneme
            activations: Dict of intermediate activations
        """
        # Encode input
        input_features = self.encode_context(letter_context)
        
        # PN layer
        pn_activity = self.input_to_pn_activity(input_features)
        
        # KC layer (sparse)
        kc_activity = self.pn_to_kc_activity(pn_activity)
        
        # MBON layer
        mbon_input = self.kc_to_mbon_input(kc_activity)
        
        # Winner selection
        predicted_idx = self.select_winner(mbon_input)
        
        return predicted_idx, {
            'input_features': input_features,
            'pn_activity': pn_activity,
            'kc_activity': kc_activity,
            'mbon_input': mbon_input,
            'n_kc_active': int(kc_activity.sum()),
        }
    
    def learn(self, kc_activity: np.ndarray, predicted_idx: int, correct_idx: int):
        """
        Apply dopamine-modulated anti-Hebbian learning.
        
        Only called when prediction is WRONG:
        - DEPRESS synapses from active KCs to correct MBON (make it win more)
        - POTENTIATE synapses from active KCs to wrong MBON (make it win less)
        """
        if predicted_idx == correct_idx:
            return  # No learning when correct
        
        lr = self.config.learning_rate
        active_kc_mask = kc_activity > 0
        
        # Depression: weaken connections to correct MBON
        self.kc_mbon_weights[active_kc_mask, correct_idx] *= (1 - lr)
        
        # Potentiation: strengthen connections to wrong (predicted) MBON
        self.kc_mbon_weights[active_kc_mask, predicted_idx] *= (1 + lr)
        
        # Clip weights to prevent explosion/collapse
        np.clip(
            self.kc_mbon_weights,
            self.config.weight_clip_min,
            self.config.weight_clip_max,
            out=self.kc_mbon_weights
        )
    
    def train_step(self, letter_context: str, correct_phoneme: str) -> Tuple[bool, str]:
        """
        Perform one training step.
        
        Args:
            letter_context: Letter context string
            correct_phoneme: Target phoneme symbol
        
        Returns:
            is_correct: Whether prediction was correct
            predicted_phoneme: Predicted phoneme symbol
        """
        correct_idx = phoneme_to_index(correct_phoneme)
        predicted_idx, activations = self.forward(letter_context)
        
        is_correct = (predicted_idx == correct_idx)
        
        if not is_correct:
            self.learn(activations['kc_activity'], predicted_idx, correct_idx)
        
        # Update stats
        self.training_steps += 1
        self.total_count += 1
        if is_correct:
            self.correct_count += 1
        
        predicted_phoneme = index_to_phoneme(predicted_idx)
        return is_correct, predicted_phoneme
    
    def predict(self, letter_context: str) -> Tuple[str, float]:
        """
        Predict phoneme for a letter context.
        
        Returns:
            phoneme: Predicted phoneme symbol
            confidence: Confidence score (margin between top two)
        """
        predicted_idx, activations = self.forward(letter_context)
        phoneme = index_to_phoneme(predicted_idx)
        
        # Confidence: margin between winner and runner-up
        mbon_input = activations['mbon_input']
        sorted_inputs = np.sort(mbon_input)
        if len(sorted_inputs) > 1:
            margin = sorted_inputs[1] - sorted_inputs[0]
            confidence = margin / (abs(sorted_inputs[0]) + 1e-6)
        else:
            confidence = 1.0
        
        return phoneme, confidence
    
    def predict_word(self, word: str, n_phonemes: int, context_size: int = None) -> List[str]:
        """
        Predict phoneme sequence for a word.
        
        Args:
            word: The word to predict
            n_phonemes: Number of phonemes to predict
            context_size: Context window size (default: model's config)
        
        Returns:
            List of predicted phoneme symbols
        """
        if context_size is None:
            context_size = self.config.context_size
        
        word = word.lower()
        n_letters = len(word)
        phonemes = []
        
        for p_idx in range(n_phonemes):
            # Interpolate letter position
            if n_phonemes == 1:
                letter_pos = n_letters // 2
            else:
                letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
            letter_pos = max(0, min(letter_pos, n_letters - 1))
            
            # Build letter context
            context_chars = []
            for offset in range(-context_size, context_size + 1):
                idx = letter_pos + offset
                if 0 <= idx < n_letters:
                    context_chars.append(word[idx])
                else:
                    context_chars.append('_')
            letter_context = ''.join(context_chars)
            
            # Predict
            phoneme, _ = self.predict(letter_context)
            phonemes.append(phoneme)
        
        return phonemes
    
    def get_accuracy(self) -> float:
        """Get current training accuracy."""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    def reset_stats(self):
        """Reset training statistics."""
        self.correct_count = 0
        self.total_count = 0
    
    def save(self, path: str):
        """Save model to file."""
        path = Path(path)
        np.savez(
            path,
            kc_mbon_weights=self.kc_mbon_weights,
            pn_kc_weights=self.pn_kc_weights,
            input_to_pn=self.input_to_pn,
            config=json.dumps(self.config.to_dict()),
            training_steps=self.training_steps,
        )
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MushroomBody':
        """Load model from file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = MushroomBodyConfig.from_dict(json.loads(str(data['config'])))
        model = cls(config)
        
        model.kc_mbon_weights = data['kc_mbon_weights']
        model.pn_kc_weights = data['pn_kc_weights']
        model.input_to_pn = data['input_to_pn']
        model.training_steps = int(data['training_steps'])
        
        print(f"Model loaded from {path} (trained for {model.training_steps} steps)")
        return model


if __name__ == "__main__":
    print("Testing G2P MushroomBody...")
    
    mb = MushroomBody()
    print(f"Input dimension: {mb.input_dim}")
    print(f"KC sparsity target: {mb.config.kc_sparsity * 100:.1f}%")
    
    # Test forward pass
    test_contexts = ["_ca_t__", "___dog_", "mushro_"]
    for ctx in test_contexts:
        pred_idx, acts = mb.forward(ctx)
        phoneme = index_to_phoneme(pred_idx)
        print(f"\nContext '{ctx}':")
        print(f"  Predicted: {phoneme}")
        print(f"  Active KCs: {acts['n_kc_active']}/{mb.config.n_kc} "
              f"({100*acts['n_kc_active']/mb.config.n_kc:.1f}%)")
