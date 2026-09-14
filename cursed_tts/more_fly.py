"""
MORE FLY: Enhanced fly-faithful G2P swarm.

This module implements the MORE FLY roadmap:
- Stage A: Richer cues (previous-phone, improved position features)
- Stage B: Fuller data support (full CMUdict, hard-word curriculum)
- Stage C: Real wiring (FlyWire/hemibrain PN→KC connectivity)
- Stage D: DAN teaching (compartment-local dopamine modulation)

Key principles:
1. Stay fly-faithful: no Transformer/LLM G2P, no lexicon-cheat
2. Use real connectome statistics where possible
3. Compartmentalized learning matching real mushroom body

Biology references:
- Scheffer et al. 2020 (hemibrain connectome)
- Zheng et al. 2020 (PN-KC structured sampling)
- Hige et al. 2015 (dopamine plasticity)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
import json
from pathlib import Path
import hashlib

from .phonemes import (
    NUM_PHONEMES, index_to_phoneme, phoneme_to_index, PHONEME_LIST, strip_stress
)
from .mushroom_body import CHAR_VOCAB, CHAR_TO_IDX, NUM_CHARS


# =============================================================================
# Stage A: Enhanced Cue Features
# =============================================================================

@dataclass
class CueConfig:
    """Configuration for cue features used in encoding."""
    use_position_features: bool = True      # Phone slot position (from PR#3)
    use_previous_phone: bool = True         # Previous phoneme cue (teacher/predicted)
    use_focus_features: bool = True         # Focus features for short words
    position_dim: int = 8                   # Position encoding dimension
    previous_phone_dim: int = NUM_PHONEMES  # One-hot for previous phone
    focus_dim: int = 4                      # Focus features for disambiguation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_position_features': self.use_position_features,
            'use_previous_phone': self.use_previous_phone,
            'use_focus_features': self.use_focus_features,
            'position_dim': self.position_dim,
            'previous_phone_dim': self.previous_phone_dim,
            'focus_dim': self.focus_dim,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CueConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def encode_enhanced_context(
    letter_context: str,
    phoneme_pos: float = None,
    n_phonemes: int = None,
    previous_phone: str = None,
    cue_config: CueConfig = None,
) -> np.ndarray:
    """
    Enhanced letter context encoding with richer cues.
    
    Features:
    1. Position-specific one-hot for each character (7 positions × 27 chars = 189)
    2. Bigram features (26×26 = 676, only letter pairs)
    3. Center character one-hot (27)
    4. Phone slot position features (8 dims) - from PR#3
    5. Previous phone cue (39 dims) - NEW: teacher-forced train / predicted decode
    6. Focus features (4 dims) - NEW: helps short words not collapse
    
    Args:
        letter_context: Letter context string (e.g., "_ca_t__")
        phoneme_pos: Normalized phoneme position (0.0 to 1.0)
        n_phonemes: Total phonemes in word
        previous_phone: Previous phoneme (for cue), None for first slot
        cue_config: Configuration for which cues to use
    
    Returns:
        Feature vector
    """
    if cue_config is None:
        cue_config = CueConfig()
    
    context = letter_context.lower()
    context_len = len(context)
    
    features = []
    
    # 1. Position-specific one-hot encoding
    pos_onehot = np.zeros(context_len * NUM_CHARS, dtype=np.float32)
    for i, c in enumerate(context):
        if c in CHAR_TO_IDX:
            pos_onehot[i * NUM_CHARS + CHAR_TO_IDX[c]] = 1.0
        else:
            pos_onehot[i * NUM_CHARS + CHAR_TO_IDX['_']] = 1.0
    features.append(pos_onehot)
    
    # 2. Bigram features
    bigram_features = np.zeros(26 * 26, dtype=np.float32)
    for i in range(context_len - 1):
        c1, c2 = context[i], context[i + 1]
        if 'a' <= c1 <= 'z' and 'a' <= c2 <= 'z':
            idx = (ord(c1) - ord('a')) * 26 + (ord(c2) - ord('a'))
            bigram_features[idx] = 1.0
    features.append(bigram_features)
    
    # 3. Center character one-hot
    center_idx = context_len // 2
    center_onehot = np.zeros(NUM_CHARS, dtype=np.float32)
    if 0 <= center_idx < context_len:
        c = context[center_idx]
        if c in CHAR_TO_IDX:
            center_onehot[CHAR_TO_IDX[c]] = 2.0
    features.append(center_onehot)
    
    # 4. Phone slot position features (from PR#3)
    if cue_config.use_position_features:
        phone_pos_features = np.zeros(cue_config.position_dim, dtype=np.float32)
        if phoneme_pos is not None:
            phone_pos_features[0] = phoneme_pos
            phone_pos_features[1] = 1.0 - phoneme_pos
            phone_pos_features[2] = np.sin(np.pi * phoneme_pos)
            phone_pos_features[3] = np.cos(np.pi * phoneme_pos)
            phone_pos_features[4] = 1.0 if phoneme_pos < 0.25 else 0.0
            phone_pos_features[5] = 1.0 if phoneme_pos > 0.75 else 0.0
            if n_phonemes is not None:
                phone_pos_features[6] = 1.0 / max(n_phonemes, 1)
                phone_pos_features[7] = 1.0 if n_phonemes <= 2 else 0.0
        features.append(phone_pos_features)
    
    # 5. Previous phone cue (NEW - Stage A)
    if cue_config.use_previous_phone:
        prev_phone_features = np.zeros(cue_config.previous_phone_dim, dtype=np.float32)
        if previous_phone is not None:
            try:
                idx = phoneme_to_index(strip_stress(previous_phone))
                prev_phone_features[idx] = 1.0
            except (ValueError, KeyError):
                pass  # Unknown phoneme, leave as zeros
        features.append(prev_phone_features)
    
    # 6. Focus features for short words (NEW - Stage A)
    if cue_config.use_focus_features:
        focus_features = np.zeros(cue_config.focus_dim, dtype=np.float32)
        if n_phonemes is not None and phoneme_pos is not None:
            # Is this a short word? (1-3 phonemes)
            focus_features[0] = 1.0 if n_phonemes <= 3 else 0.0
            # Unique slot indicator (for "me" problem)
            if n_phonemes == 2:
                focus_features[1] = 1.0 if phoneme_pos < 0.5 else -1.0
            # Slot index modulo 3 (repeating pattern)
            if n_phonemes > 0:
                slot_idx = int(phoneme_pos * (n_phonemes - 1)) if n_phonemes > 1 else 0
                focus_features[2] = float(slot_idx % 3) / 2.0 - 0.5
            # Word length signal (normalized)
            n_letters = sum(1 for c in letter_context if c != '_')
            focus_features[3] = min(n_letters / 10.0, 1.0)
        features.append(focus_features)
    
    return np.concatenate(features)


def get_enhanced_input_dim(
    context_size: int = 3,
    cue_config: CueConfig = None,
) -> int:
    """Get input feature dimension for enhanced encoding."""
    if cue_config is None:
        cue_config = CueConfig()
    
    context_len = 2 * context_size + 1
    base_dim = context_len * NUM_CHARS + 26 * 26 + NUM_CHARS
    
    extra_dim = 0
    if cue_config.use_position_features:
        extra_dim += cue_config.position_dim
    if cue_config.use_previous_phone:
        extra_dim += cue_config.previous_phone_dim
    if cue_config.use_focus_features:
        extra_dim += cue_config.focus_dim
    
    return base_dim + extra_dim


# =============================================================================
# Stage C: Real Wiring (FlyWire/Hemibrain)
# =============================================================================

@dataclass
class WiringConfig:
    """Configuration for PN→KC wiring."""
    mode: str = 'random'                    # 'random', 'flywire', 'hemibrain'
    n_pn: int = 180                         # Number of PN types (input channels)
    n_kc: int = 2000                        # Number of Kenyon cells
    pn_per_kc: int = 7                      # Average PNs connected to each KC
    kc_sparsity: float = 0.10               # Fraction of KCs active
    freeze_pn_kc: bool = True               # Freeze PN→KC weights (default)
    pn_kc_learning_rate: float = 0.001      # Very slow PN→KC plasticity if not frozen
    seed: int = 42
    connectome_path: str = None             # Path to connectome data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'n_pn': self.n_pn,
            'n_kc': self.n_kc,
            'pn_per_kc': self.pn_per_kc,
            'kc_sparsity': self.kc_sparsity,
            'freeze_pn_kc': self.freeze_pn_kc,
            'pn_kc_learning_rate': self.pn_kc_learning_rate,
            'seed': self.seed,
            'connectome_path': self.connectome_path,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WiringConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Hemibrain statistics for PN→KC connectivity
# Based on Zheng et al. 2020 and Scheffer et al. 2020
HEMIBRAIN_STATS = {
    'n_pn_types': 180,              # Approximate number of olfactory PN types (stats-matched)
    'n_pn_real': 428,               # Actual PNs in hemibrain v1.2 traced data
    'n_kc_total': 1963,             # KCs on right hemisphere (hemibrain)
    'n_kc_real': 1927,              # Actual KCs in hemibrain v1.2 traced data
    'avg_pn_per_kc': 6.8,           # Average PNs per KC claw (literature)
    'avg_pn_per_kc_real': 5.5,      # Actual from extracted data (binarized ≥3 syn)
    'pn_per_kc_std': 2.1,           # Standard deviation
    'kc_sparsity': 0.05,            # ~5% KC activity in odor response
    'synapse_threshold': 3,         # Min synapses to count as connection
}

# For real hemibrain wiring, use n_pn=400 to preserve connectivity structure
# Subsampling to 180 loses too much real connectivity
REAL_HEMIBRAIN_N_PN = 400


def load_hemibrain_connectivity(path: str = None, prefer_real: bool = True) -> Optional[Tuple[np.ndarray, str]]:
    """
    Load PN→KC connectivity matrix from hemibrain data.
    
    Priority order (if prefer_real=True):
    1. Real hemibrain extracted data (hemibrain_real_pn_kc.npz)
    2. Stats-matched fallback (hemibrain_pn_kc.npz)
    
    Returns:
        Tuple of (sparse binary matrix (n_pn, n_kc), source_label) or None if not available
    """
    # Define paths to try in priority order
    default_paths = []
    
    if path is not None:
        default_paths.append((Path(path), 'provided'))
    
    if prefer_real:
        # Prefer real extracted hemibrain data
        default_paths.append((Path('artifacts/connectome/hemibrain_real_pn_kc.npz'), 'hemibrain_real'))
    
    # Fallback to stats-matched
    default_paths.append((Path('artifacts/connectome/hemibrain_pn_kc.npz'), 'hemibrain_stats'))
    default_paths.append((Path('artifacts/connectome/pn_kc_matrix.npz'), 'legacy'))
    
    for p, source in default_paths:
        if p.exists():
            try:
                data = np.load(str(p), allow_pickle=True)
                # Support different key names
                for key in ['matrix', 'pn_kc_matrix']:
                    if key in data:
                        matrix = data[key]
                        print(f"Loaded {source} connectivity: {matrix.shape}")
                        return matrix, source
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                continue
    
    return None


def create_wiring_matrix(
    config: WiringConfig,
    input_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create PN→KC wiring matrix based on configuration.
    
    Args:
        config: Wiring configuration
        input_dim: Input feature dimension (maps to PN-like channels)
        rng: Random generator
    
    Returns:
        input_to_pn: Input features → PN projection
        pn_kc_weights: PN → KC connectivity
        metadata: Wiring metadata for documentation
    """
    metadata = {
        'mode': config.mode,
        'n_pn': config.n_pn,
        'n_kc': config.n_kc,
        'seed': config.seed,
    }
    
    # Input → PN projection (random, normalized)
    input_to_pn = rng.standard_normal((input_dim, config.n_pn)).astype(np.float32)
    norms = np.linalg.norm(input_to_pn, axis=0, keepdims=True)
    input_to_pn /= (norms + 1e-6)
    
    # PN → KC connectivity based on mode
    if config.mode == 'flywire' or config.mode == 'hemibrain':
        # Try to load real connectivity
        result = load_hemibrain_connectivity(config.connectome_path, prefer_real=True)
        
        if result is not None:
            hemibrain_matrix, source_label = result
            
            # Subsample or pad to match our dimensions
            src_pn, src_kc = hemibrain_matrix.shape
            
            # Subsample KCs if needed
            if src_kc > config.n_kc:
                kc_indices = rng.choice(src_kc, config.n_kc, replace=False)
                hemibrain_matrix = hemibrain_matrix[:, kc_indices]
            elif src_kc < config.n_kc:
                # Tile the matrix
                repeats = (config.n_kc + src_kc - 1) // src_kc
                hemibrain_matrix = np.tile(hemibrain_matrix, (1, repeats))[:, :config.n_kc]
            
            # Subsample PNs if needed
            if src_pn > config.n_pn:
                pn_indices = rng.choice(src_pn, config.n_pn, replace=False)
                hemibrain_matrix = hemibrain_matrix[pn_indices, :]
            elif src_pn < config.n_pn:
                padding = np.zeros((config.n_pn - src_pn, hemibrain_matrix.shape[1]))
                hemibrain_matrix = np.vstack([hemibrain_matrix, padding])
            
            pn_kc_weights = hemibrain_matrix.astype(np.float32)
            
            # Normalize columns
            col_sums = pn_kc_weights.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            pn_kc_weights /= col_sums
            
            metadata['source'] = source_label
            metadata['hash'] = hashlib.md5(pn_kc_weights.tobytes()).hexdigest()[:16]
            print(f"Using {source_label} wiring (hash: {metadata['hash']})")
        else:
            # Fall back to random but with hemibrain statistics
            print(f"Warning: {config.mode} requested but data not found, "
                  f"using deterministic random with hemibrain statistics")
            config.mode = 'random_hemibrain_stats'
            pn_kc_weights = _create_random_wiring_hemibrain_stats(config, rng)
            metadata['source'] = 'random_hemibrain_stats'
            metadata['hash'] = hashlib.md5(pn_kc_weights.tobytes()).hexdigest()[:16]
    else:
        # Random wiring (baseline)
        pn_kc_weights = _create_random_wiring(config, rng)
        metadata['source'] = 'random'
        metadata['hash'] = hashlib.md5(pn_kc_weights.tobytes()).hexdigest()[:16]
    
    # Compute sparsity for verification
    sparsity = np.mean(pn_kc_weights > 0)
    avg_pn_per_kc = np.mean(np.sum(pn_kc_weights > 0, axis=0))
    metadata['sparsity'] = float(sparsity)
    metadata['avg_pn_per_kc'] = float(avg_pn_per_kc)
    
    return input_to_pn, pn_kc_weights, metadata


def _create_random_wiring(config: WiringConfig, rng: np.random.Generator) -> np.ndarray:
    """Create baseline random PN→KC wiring."""
    pn_kc_weights = np.zeros((config.n_pn, config.n_kc), dtype=np.float32)
    
    for kc in range(config.n_kc):
        connected_pns = rng.choice(config.n_pn, size=config.pn_per_kc, replace=False)
        pn_kc_weights[connected_pns, kc] = 1.0 / config.pn_per_kc
    
    return pn_kc_weights


def _create_random_wiring_hemibrain_stats(
    config: WiringConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create random wiring with hemibrain statistics (variable PNs per KC)."""
    pn_kc_weights = np.zeros((config.n_pn, config.n_kc), dtype=np.float32)
    
    # Use hemibrain statistics for number of PNs per KC
    mean_pn = HEMIBRAIN_STATS['avg_pn_per_kc']
    std_pn = HEMIBRAIN_STATS['pn_per_kc_std']
    
    for kc in range(config.n_kc):
        # Sample number of PNs from truncated normal
        n_pns = int(rng.normal(mean_pn, std_pn))
        n_pns = max(3, min(n_pns, 12))  # Clamp to reasonable range
        
        connected_pns = rng.choice(config.n_pn, size=n_pns, replace=False)
        pn_kc_weights[connected_pns, kc] = 1.0 / n_pns
    
    return pn_kc_weights


# =============================================================================
# Stage D: DAN Compartment-Local Teaching
# =============================================================================

@dataclass
class DANConfig:
    """Configuration for DAN (dopamine neuron) teaching signal."""
    enabled: bool = True                    # Enable DAN-style teaching
    compartment_teaching: bool = True       # Per-compartment (specialist) updates
    word_reward_modulation: bool = False    # Optional word-level reward scalar
    learning_rate: float = 0.08             # KC→MBON learning rate
    depression_ratio: float = 1.0           # Depression strength relative to potentiation
    kc_mbon_only: bool = True               # Only update KC→MBON (fly-faithful)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'compartment_teaching': self.compartment_teaching,
            'word_reward_modulation': self.word_reward_modulation,
            'learning_rate': self.learning_rate,
            'depression_ratio': self.depression_ratio,
            'kc_mbon_only': self.kc_mbon_only,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DANConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DANTeacher:
    """
    Dopamine neuron (DAN) teaching signal for compartment-local learning.
    
    Biology analogy (Hige et al. 2015, Handler et al. 2019):
    - Each MBON compartment receives teaching from specific DANs
    - When the ensemble makes an error, only the responsible compartments update
    - True target receives depression signal (YES compartment learns to activate)
    - False winner receives potentiation signal (NO compartment learns to inhibit)
    
    This implements fly-faithful learning where:
    - Only KC→MBON synapses are plastic (default)
    - Teaching is compartmentalized, not whole-brain reward spray
    - Error signal drives learning (dopamine-when-wrong)
    """
    
    def __init__(self, config: DANConfig = None):
        self.config = config or DANConfig()
        self.update_count = 0
        self.compartment_updates: Dict[str, int] = {}
    
    def compute_teaching_signal(
        self,
        predicted: str,
        target: str,
        kc_activity: np.ndarray,
        word_reward: float = 1.0,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute DAN teaching signal for compartment-local updates.
        
        Args:
            predicted: Predicted phoneme
            target: Target (correct) phoneme
            kc_activity: Active KC pattern
            word_reward: Optional word-level reward modulation
        
        Returns:
            Teaching signals per compartment:
            {phoneme: {'depression': delta, 'potentiation': delta}}
        """
        if not self.config.enabled:
            return {}
        
        predicted = strip_stress(predicted)
        target = strip_stress(target)
        
        if predicted == target:
            return {}  # No teaching when correct
        
        signals = {}
        lr = self.config.learning_rate
        
        if self.config.word_reward_modulation:
            lr *= word_reward
        
        active_mask = kc_activity > 0
        
        # Target compartment: DEPRESSION (make YES win more easily)
        # This reduces KC→MBON[YES] weights for the correct phoneme
        signals[target] = {
            'depression': lr * self.config.depression_ratio * active_mask.astype(np.float32),
            'potentiation': np.zeros_like(kc_activity),
        }
        
        # Predicted (wrong) compartment: POTENTIATION (make NO win more easily)
        # This increases KC→MBON[NO] weights for the wrong phoneme
        signals[predicted] = {
            'depression': np.zeros_like(kc_activity),
            'potentiation': lr * active_mask.astype(np.float32),
        }
        
        # Track updates
        self.update_count += 1
        self.compartment_updates[target] = self.compartment_updates.get(target, 0) + 1
        self.compartment_updates[predicted] = self.compartment_updates.get(predicted, 0) + 1
        
        return signals
    
    def apply_teaching(
        self,
        specialist,  # SpecialistFly instance
        kc_activity: np.ndarray,
        predicted_yes: bool,
        target_yes: bool,
        word_reward: float = 1.0,
    ):
        """
        Apply DAN teaching to a single specialist's KC→MBON weights.
        
        Args:
            specialist: The specialist fly to update
            kc_activity: Active KC pattern
            predicted_yes: What the specialist predicted
            target_yes: Whether this is the target phoneme
            word_reward: Optional reward modulation
        """
        if not self.config.enabled:
            return
        
        if predicted_yes == target_yes:
            return  # No teaching when correct
        
        lr = self.config.learning_rate
        if self.config.word_reward_modulation:
            lr *= word_reward
        
        active_mask = kc_activity > 0
        
        # Correct class index (YES=0, NO=1)
        correct_idx = 0 if target_yes else 1
        wrong_idx = 1 - correct_idx
        
        # Depression: weaken connections to correct MBON
        specialist.kc_mbon_weights[active_mask, correct_idx] *= (1 - lr * self.config.depression_ratio)
        
        # Potentiation: strengthen connections to wrong MBON
        specialist.kc_mbon_weights[active_mask, wrong_idx] *= (1 + lr)
        
        # Clip weights
        np.clip(
            specialist.kc_mbon_weights,
            specialist.config.weight_clip_min,
            specialist.config.weight_clip_max,
            out=specialist.kc_mbon_weights,
        )
        
        self.update_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get teaching statistics."""
        return {
            'update_count': self.update_count,
            'compartment_updates': dict(self.compartment_updates),
            'top_updated': sorted(
                self.compartment_updates.items(),
                key=lambda x: -x[1]
            )[:10],
        }


# =============================================================================
# Enhanced Swarm with All MORE FLY Features
# =============================================================================

@dataclass
class MoreFlyConfig:
    """Configuration for MORE FLY enhanced swarm."""
    # Stage A: Cues
    cue_config: CueConfig = field(default_factory=CueConfig)
    
    # Stage B: Data (handled in training, not model)
    
    # Stage C: Wiring
    wiring_config: WiringConfig = field(default_factory=WiringConfig)
    
    # Stage D: DAN Teaching
    dan_config: DANConfig = field(default_factory=DANConfig)
    
    # Specialist settings
    n_mbon: int = 2                         # MBON outputs per specialist (YES/NO)
    learning_rate: float = 0.08
    weight_init_mean: float = 1.0
    weight_clip_min: float = 0.01
    weight_clip_max: float = 10.0
    context_size: int = 3
    seed: int = 42
    
    # Voting
    vote_strategy: str = 'softmax'
    vote_temperature: float = 0.5
    vote_margin: float = 0.1
    
    # Phonemes to train
    phonemes: List[str] = field(default_factory=lambda: PHONEME_LIST.copy())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cue_config': self.cue_config.to_dict(),
            'wiring_config': self.wiring_config.to_dict(),
            'dan_config': self.dan_config.to_dict(),
            'n_mbon': self.n_mbon,
            'learning_rate': self.learning_rate,
            'weight_init_mean': self.weight_init_mean,
            'weight_clip_min': self.weight_clip_min,
            'weight_clip_max': self.weight_clip_max,
            'context_size': self.context_size,
            'seed': self.seed,
            'vote_strategy': self.vote_strategy,
            'vote_temperature': self.vote_temperature,
            'vote_margin': self.vote_margin,
            'phonemes': self.phonemes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MoreFlyConfig':
        cue_config = CueConfig.from_dict(d.get('cue_config', {}))
        wiring_config = WiringConfig.from_dict(d.get('wiring_config', {}))
        dan_config = DANConfig.from_dict(d.get('dan_config', {}))
        
        return cls(
            cue_config=cue_config,
            wiring_config=wiring_config,
            dan_config=dan_config,
            **{k: v for k, v in d.items() 
               if k in cls.__dataclass_fields__ and k not in ['cue_config', 'wiring_config', 'dan_config']}
        )


class MoreFlyExpansion:
    """
    Shared input→PN→KC expansion layer with real wiring support.
    """
    
    def __init__(self, config: MoreFlyConfig, rng: np.random.Generator):
        self.config = config
        self.input_dim = get_enhanced_input_dim(
            config.context_size,
            config.cue_config,
        )
        
        # Create wiring
        self.input_to_pn, self.pn_kc_weights, self.wiring_metadata = create_wiring_matrix(
            config.wiring_config,
            self.input_dim,
            rng,
        )
        
        print(f"MoreFlyExpansion: {self.input_dim} → {config.wiring_config.n_pn} PNs "
              f"→ {config.wiring_config.n_kc} KCs")
        print(f"  Wiring mode: {self.wiring_metadata['source']} "
              f"(hash: {self.wiring_metadata['hash']})")
        print(f"  Avg PN/KC: {self.wiring_metadata['avg_pn_per_kc']:.1f}, "
              f"sparsity: {self.wiring_metadata['sparsity']:.4f}")
    
    def encode_to_kc(
        self,
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
        previous_phone: str = None,
    ) -> np.ndarray:
        """Transform letter context + cues to sparse KC activity."""
        # Enhanced encoding with previous phone cue
        input_features = encode_enhanced_context(
            letter_context,
            phoneme_pos=phoneme_pos,
            n_phonemes=n_phonemes,
            previous_phone=previous_phone,
            cue_config=self.config.cue_config,
        )
        
        # PN activity
        pn_raw = input_features @ self.input_to_pn
        pn_activity = np.maximum(0, pn_raw)
        if pn_activity.max() > 0:
            pn_activity = pn_activity / pn_activity.max()
        
        # KC activity (sparse winner-take-all)
        kc_input = pn_activity @ self.pn_kc_weights
        n_active = max(1, int(self.config.wiring_config.n_kc * self.config.wiring_config.kc_sparsity))
        threshold = np.partition(kc_input, -n_active)[-n_active]
        kc_activity = (kc_input >= threshold).astype(np.float32)
        
        return kc_activity


class MoreFlySpecialist:
    """
    A specialist fly that answers: "Is the phoneme /P/?" (binary YES/NO).
    
    Uses the MORE FLY enhancements:
    - Enhanced cue encoding
    - Real wiring (shared expansion layer)
    - DAN-style teaching
    """
    
    def __init__(
        self,
        target_phoneme: str,
        shared_expansion: MoreFlyExpansion,
        config: MoreFlyConfig,
        rng: np.random.Generator,
    ):
        self.target_phoneme = strip_stress(target_phoneme)
        self.target_idx = phoneme_to_index(self.target_phoneme)
        self.shared = shared_expansion
        self.config = config
        self.rng = rng
        
        # KC → MBON weights (2 outputs: YES=0, NO=1)
        self.kc_mbon_weights = np.full(
            (config.wiring_config.n_kc, config.n_mbon),
            config.weight_init_mean,
            dtype=np.float32,
        )
        
        # Training stats
        self.training_steps = 0
        self.correct_count = 0
        self.total_count = 0
    
    def forward(self, kc_activity: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """Run forward pass with pre-computed KC activity."""
        mbon_input = kc_activity @ self.kc_mbon_weights
        
        # Winner: minimum input wins (inhibitory logic)
        is_yes = mbon_input[0] < mbon_input[1]
        
        # Confidence
        margin = abs(mbon_input[1] - mbon_input[0])
        baseline = abs(mbon_input[0]) + abs(mbon_input[1]) + 1e-6
        confidence = margin / baseline
        
        return is_yes, confidence, mbon_input
    
    def predict(
        self,
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
        previous_phone: str = None,
    ) -> Tuple[bool, float]:
        """Predict YES/NO for this context with enhanced cues."""
        kc_activity = self.shared.encode_to_kc(
            letter_context,
            phoneme_pos=phoneme_pos,
            n_phonemes=n_phonemes,
            previous_phone=previous_phone,
        )
        is_yes, confidence, _ = self.forward(kc_activity)
        return is_yes, confidence
    
    def learn(self, kc_activity: np.ndarray, predicted_yes: bool, target_yes: bool):
        """
        Apply standard anti-Hebbian learning when wrong.
        
        Used when DAN teaching is disabled.
        """
        if predicted_yes == target_yes:
            return  # No learning when correct
        
        lr = self.config.learning_rate
        active_mask = kc_activity > 0
        
        # Correct class index (YES=0, NO=1)
        correct_idx = 0 if target_yes else 1
        wrong_idx = 1 - correct_idx
        
        # Depression: weaken connections to correct MBON
        self.kc_mbon_weights[active_mask, correct_idx] *= (1 - lr)
        
        # Potentiation: strengthen connections to wrong MBON
        self.kc_mbon_weights[active_mask, wrong_idx] *= (1 + lr)
        
        # Clip weights
        np.clip(
            self.kc_mbon_weights,
            self.config.weight_clip_min,
            self.config.weight_clip_max,
            out=self.kc_mbon_weights,
        )
    
    def get_accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    def reset_stats(self):
        self.correct_count = 0
        self.total_count = 0


class MoreFlySwarm:
    """
    Enhanced fly swarm with all MORE FLY features.
    
    Features:
    - Stage A: Previous-phone cue, enhanced position features
    - Stage C: Real wiring (FlyWire/hemibrain or documented fallback)
    - Stage D: DAN compartment-local teaching
    """
    
    def __init__(self, config: Optional[MoreFlyConfig] = None):
        self.config = config or MoreFlyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Create shared expansion layer with real wiring
        self.shared = MoreFlyExpansion(self.config, self.rng)
        
        # Create specialist flies
        self.specialists: Dict[str, MoreFlySpecialist] = {}
        for phoneme in self.config.phonemes:
            p = strip_stress(phoneme)
            if p not in self.specialists:
                self.specialists[p] = MoreFlySpecialist(
                    target_phoneme=p,
                    shared_expansion=self.shared,
                    config=self.config,
                    rng=self.rng,
                )
        
        self.phoneme_list = list(self.specialists.keys())
        
        # DAN teacher for compartment-local learning
        self.dan_teacher = DANTeacher(self.config.dan_config)
        
        print(f"MoreFlySwarm: {len(self.specialists)} specialists")
        print(f"  Cues: pos={self.config.cue_config.use_position_features}, "
              f"prev_phone={self.config.cue_config.use_previous_phone}, "
              f"focus={self.config.cue_config.use_focus_features}")
        print(f"  DAN teaching: {self.config.dan_config.enabled}, "
              f"compartment={self.config.dan_config.compartment_teaching}")
    
    def _get_raw_scores(
        self,
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
        previous_phone: str = None,
    ) -> Dict[str, float]:
        """Get raw YES confidence scores from all specialists."""
        kc_activity = self.shared.encode_to_kc(
            letter_context,
            phoneme_pos=phoneme_pos,
            n_phonemes=n_phonemes,
            previous_phone=previous_phone,
        )
        
        scores = {}
        for phoneme, specialist in self.specialists.items():
            is_yes, confidence, _ = specialist.forward(kc_activity)
            scores[phoneme] = confidence if is_yes else -confidence
        
        return scores
    
    def _vote_softmax(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """Softmax voting."""
        phonemes = list(scores.keys())
        raw_scores = np.array([scores[p] for p in phonemes])
        
        temp = max(self.config.vote_temperature, 0.01)
        scaled = raw_scores / temp
        scaled = scaled - scaled.max()
        exp_scores = np.exp(scaled)
        probs = exp_scores / exp_scores.sum()
        
        winner_idx = np.argmax(probs)
        return phonemes[winner_idx], float(probs[winner_idx])
    
    def predict(
        self,
        letter_context: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
        previous_phone: str = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict phoneme with enhanced cues."""
        scores = self._get_raw_scores(
            letter_context,
            phoneme_pos=phoneme_pos,
            n_phonemes=n_phonemes,
            previous_phone=previous_phone,
        )
        
        winner, confidence = self._vote_softmax(scores)
        return winner, confidence, scores
    
    def predict_word(
        self,
        word: str,
        n_phonemes: int,
        context_size: int = None,
        use_teacher_forcing: bool = False,
        reference_phonemes: List[str] = None,
    ) -> List[str]:
        """
        Predict phoneme sequence for a word.
        
        Args:
            word: Input word
            n_phonemes: Number of phonemes to predict
            context_size: Letter context size
            use_teacher_forcing: Use reference phonemes for previous-phone cue
            reference_phonemes: Reference phonemes (for teacher forcing)
        """
        if context_size is None:
            context_size = self.config.context_size
        
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
            
            # Build letter context
            context_chars = []
            for offset in range(-context_size, context_size + 1):
                idx = letter_pos + offset
                if 0 <= idx < n_letters:
                    context_chars.append(word[idx])
                else:
                    context_chars.append('_')
            letter_context = ''.join(context_chars)
            
            # Get previous phone cue
            if p_idx == 0:
                previous_phone = None
            elif use_teacher_forcing and reference_phonemes:
                previous_phone = reference_phonemes[p_idx - 1]
            else:
                previous_phone = phonemes[p_idx - 1]
            
            phoneme, _, _ = self.predict(
                letter_context,
                phoneme_pos=phoneme_pos,
                n_phonemes=n_phonemes,
                previous_phone=previous_phone,
            )
            phonemes.append(phoneme)
        
        return phonemes
    
    def train_step(
        self,
        letter_context: str,
        correct_phoneme: str,
        phoneme_pos: float = None,
        n_phonemes: int = None,
        previous_phone: str = None,
        word_reward: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Train all specialists using DAN teaching.
        
        Returns:
            is_correct: Whether ensemble prediction was correct
            predicted: Predicted phoneme
        """
        correct = strip_stress(correct_phoneme)
        
        # Get KC activity (shared across all specialists)
        kc_activity = self.shared.encode_to_kc(
            letter_context,
            phoneme_pos=phoneme_pos,
            n_phonemes=n_phonemes,
            previous_phone=previous_phone,
        )
        
        # Train each specialist
        for phoneme, specialist in self.specialists.items():
            target_yes = (phoneme == correct)
            predicted_yes, confidence, _ = specialist.forward(kc_activity)
            
            is_correct_specialist = (predicted_yes == target_yes)
            
            if not is_correct_specialist:
                # Apply learning: DAN teaching if enabled, standard anti-Hebbian otherwise
                if self.config.dan_config.enabled:
                    self.dan_teacher.apply_teaching(
                        specialist,
                        kc_activity,
                        predicted_yes,
                        target_yes,
                        word_reward=word_reward,
                    )
                else:
                    # Standard anti-Hebbian learning
                    specialist.learn(kc_activity, predicted_yes, target_yes)
            
            specialist.training_steps += 1
            specialist.total_count += 1
            if is_correct_specialist:
                specialist.correct_count += 1
        
        # Get ensemble prediction
        scores = {}
        for phoneme, specialist in self.specialists.items():
            is_yes, confidence, _ = specialist.forward(kc_activity)
            scores[phoneme] = confidence if is_yes else -confidence
        
        predicted, _ = self._vote_softmax(scores)
        is_correct = (predicted == correct)
        
        return is_correct, predicted
    
    def get_wiring_metadata(self) -> Dict[str, Any]:
        """Get wiring metadata for verification."""
        return self.shared.wiring_metadata
    
    def is_random_wiring(self) -> bool:
        """Check if wiring is random (not from connectome)."""
        return self.shared.wiring_metadata['source'] in ['random', 'random_hemibrain_stats']
    
    def save(self, path: str):
        """Save swarm to file."""
        path = Path(path)
        
        specialist_weights = {}
        for phoneme, specialist in self.specialists.items():
            specialist_weights[f"weights_{phoneme}"] = specialist.kc_mbon_weights
        
        np.savez(
            path,
            input_to_pn=self.shared.input_to_pn,
            pn_kc_weights=self.shared.pn_kc_weights,
            config=json.dumps(self.config.to_dict()),
            wiring_metadata=json.dumps(self.shared.wiring_metadata),
            dan_stats=json.dumps(self.dan_teacher.get_stats()),
            **specialist_weights,
        )
        print(f"MoreFlySwarm saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MoreFlySwarm':
        """Load swarm from file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        config = MoreFlyConfig.from_dict(json.loads(str(data['config'])))
        swarm = cls(config)
        
        # Load shared expansion
        swarm.shared.input_to_pn = data['input_to_pn']
        swarm.shared.pn_kc_weights = data['pn_kc_weights']
        
        if 'wiring_metadata' in data:
            swarm.shared.wiring_metadata = json.loads(str(data['wiring_metadata']))
        
        # Load specialist weights
        for phoneme in swarm.specialists:
            key = f"weights_{phoneme}"
            if key in data:
                swarm.specialists[phoneme].kc_mbon_weights = data[key]
        
        print(f"MoreFlySwarm loaded from {path} ({len(swarm.specialists)} specialists)")
        return swarm


# =============================================================================
# Utility Functions
# =============================================================================

def verify_wiring_is_not_random(swarm: MoreFlySwarm) -> bool:
    """
    Verify that wiring is not random (uses real connectome data).
    
    Returns True if wiring is from flywire/hemibrain, False otherwise.
    """
    return not swarm.is_random_wiring()


def verify_kc_mbon_only_plasticity(
    swarm: MoreFlySwarm,
    letter_context: str,
    target_phoneme: str,
) -> bool:
    """
    Verify that only KC→MBON weights change during learning.
    
    Returns True if PN→KC weights are frozen.
    """
    # Copy PN→KC weights before
    pn_kc_before = swarm.shared.pn_kc_weights.copy()
    
    # Get a sample KC→MBON weight
    sample_specialist = list(swarm.specialists.values())[0]
    kc_mbon_before = sample_specialist.kc_mbon_weights.copy()
    
    # Run training step
    swarm.train_step(letter_context, target_phoneme)
    
    # Check PN→KC unchanged
    pn_kc_unchanged = np.allclose(pn_kc_before, swarm.shared.pn_kc_weights)
    
    # Check KC→MBON changed (if error occurred)
    # This is a weak check since it depends on whether there was an error
    
    return pn_kc_unchanged


def verify_previous_phone_cue_changes_encoding(
    cue_config: CueConfig,
    letter_context: str = "_ca_t__",
) -> bool:
    """
    Verify that previous-phone cue changes the encoding.
    
    Returns True if encodings differ with different previous phones.
    """
    enc_none = encode_enhanced_context(
        letter_context,
        phoneme_pos=0.5,
        n_phonemes=3,
        previous_phone=None,
        cue_config=cue_config,
    )
    
    enc_k = encode_enhanced_context(
        letter_context,
        phoneme_pos=0.5,
        n_phonemes=3,
        previous_phone='K',
        cue_config=cue_config,
    )
    
    enc_ae = encode_enhanced_context(
        letter_context,
        phoneme_pos=0.5,
        n_phonemes=3,
        previous_phone='AE',
        cue_config=cue_config,
    )
    
    # Check that encodings differ
    return not np.allclose(enc_none, enc_k) and not np.allclose(enc_k, enc_ae)


def verify_short_word_position_differs(
    cue_config: CueConfig,
    word: str = "me",
) -> bool:
    """
    Verify that different slots in short words have different encodings.
    
    Returns True if slot 0 and slot 1 of "me" have different position features.
    """
    context_0 = "___me__"  # Slot 0
    context_1 = "__me___"  # Slot 1
    
    enc_0 = encode_enhanced_context(
        context_0,
        phoneme_pos=0.0,
        n_phonemes=2,
        previous_phone=None,
        cue_config=cue_config,
    )
    
    enc_1 = encode_enhanced_context(
        context_1,
        phoneme_pos=1.0,
        n_phonemes=2,
        previous_phone='M',
        cue_config=cue_config,
    )
    
    return not np.allclose(enc_0, enc_1)


if __name__ == "__main__":
    print("Testing MORE FLY module...")
    
    # Test enhanced encoding
    print("\n1. Testing enhanced encoding:")
    cue_config = CueConfig()
    dim = get_enhanced_input_dim(3, cue_config)
    print(f"   Enhanced input dim: {dim}")
    
    enc = encode_enhanced_context(
        "_ca_t__",
        phoneme_pos=0.33,
        n_phonemes=3,
        previous_phone='K',
        cue_config=cue_config,
    )
    print(f"   Encoding shape: {enc.shape}")
    print(f"   Non-zero: {np.sum(enc != 0)}")
    
    # Test previous phone cue
    print("\n2. Testing previous-phone cue:")
    if verify_previous_phone_cue_changes_encoding(cue_config):
        print("   ✓ Previous phone cue changes encoding")
    else:
        print("   ✗ Previous phone cue has no effect")
    
    # Test short word position
    print("\n3. Testing short word position features:")
    if verify_short_word_position_differs(cue_config):
        print("   ✓ Short word slots have different encodings")
    else:
        print("   ✗ Short word slots have same encoding")
    
    # Test wiring
    print("\n4. Testing wiring creation:")
    wiring_config = WiringConfig(mode='random', seed=42)
    config = MoreFlyConfig(wiring_config=wiring_config)
    swarm = MoreFlySwarm(config)
    
    print(f"   Wiring metadata: {swarm.get_wiring_metadata()}")
    print(f"   Is random: {swarm.is_random_wiring()}")
    
    # Test prediction
    print("\n5. Testing prediction:")
    phoneme, conf, scores = swarm.predict("_ca_t__", phoneme_pos=0.33, n_phonemes=3)
    print(f"   Context '_ca_t__' → {phoneme} (conf={conf:.3f})")
    
    # Test training
    print("\n6. Testing DAN teaching:")
    is_correct, pred = swarm.train_step("_ca_t__", "K", phoneme_pos=0.33, n_phonemes=3)
    print(f"   Training step: pred={pred}, correct={is_correct}")
    print(f"   DAN stats: {swarm.dan_teacher.get_stats()}")
    
    print("\n✓ MORE FLY module tests complete")
