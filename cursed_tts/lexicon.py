"""
CMUdict-based lexicon for open-vocabulary TTS.

Loads pronunciation data from CMUdict and provides:
- Training data generation from thousands of words
- G2P (grapheme-to-phoneme) for OOV words via g2p_en
- The original 10 demo words as a smoke test subset
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from .phonemes import PHONEME_INVENTORY, phoneme_to_index, strip_stress


# Original 10 demo words (must always work)
DEMO_WORDS = ['cat', 'bat', 'dog', 'go', 'no', 'hi', 'bye', 'yes', 'me', 'you']

# Cache for loaded lexicon
_LEXICON_CACHE: Optional[Dict[str, List[str]]] = None
_G2P_MODEL = None


def _load_cmudict() -> Dict[str, List[str]]:
    """
    Load CMUdict pronunciation dictionary.
    
    Returns dict mapping lowercase words to phoneme lists (stress-stripped).
    """
    global _LEXICON_CACHE
    
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    
    try:
        import cmudict
        raw_dict = cmudict.dict()
    except ImportError:
        try:
            import nltk
            nltk.download('cmudict', quiet=True)
            from nltk.corpus import cmudict as nltk_cmudict
            raw_dict = nltk_cmudict.dict()
        except Exception as e:
            raise RuntimeError(
                f"Could not load CMUdict. Install with: pip install cmudict\n"
                f"Or: pip install nltk && python -c \"import nltk; nltk.download('cmudict')\"\n"
                f"Error: {e}"
            )
    
    lexicon = {}
    for word, pronunciations in raw_dict.items():
        word_lower = word.lower()
        
        # Skip non-alphabetic words
        if not word_lower.isalpha():
            continue
        
        # Take first pronunciation, strip stress
        phones = pronunciations[0]
        phones_stripped = [strip_stress(p) for p in phones]
        
        # Verify all phonemes are in our inventory
        if all(p in PHONEME_INVENTORY for p in phones_stripped):
            lexicon[word_lower] = phones_stripped
    
    _LEXICON_CACHE = lexicon
    print(f"Loaded CMUdict: {len(lexicon)} words")
    return lexicon


def _get_g2p():
    """Get or initialize the G2P model for OOV words."""
    global _G2P_MODEL
    
    if _G2P_MODEL is not None:
        return _G2P_MODEL
    
    try:
        from g2p_en import G2p
        _G2P_MODEL = G2p()
        return _G2P_MODEL
    except ImportError:
        return None


def g2p_predict(word: str) -> List[str]:
    """
    Predict phonemes for an OOV word using G2P.
    
    Falls back to simple heuristics if g2p_en not available or fails.
    """
    g2p = _get_g2p()
    
    if g2p is not None:
        try:
            # Use g2p_en
            phones = g2p(word)
            # g2p_en returns list like ['M', 'AH1', 'SH', 'R', 'UW0', 'M']
            # Filter out non-phonemes and strip stress
            result = []
            for p in phones:
                p_clean = strip_stress(p.upper())
                if p_clean in PHONEME_INVENTORY:
                    result.append(p_clean)
            if result:
                return result
        except Exception:
            pass  # Fall through to simple G2P
    
    # Simple fallback G2P (very crude)
    return _simple_g2p(word)


def _simple_g2p(word: str) -> List[str]:
    """
    Very simple rule-based G2P fallback.
    
    This is intentionally crude - the cursed quality is expected.
    """
    word = word.lower()
    phones = []
    
    # Simple letter-to-phoneme mappings
    mapping = {
        'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AA', 'u': 'AH',
        'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
        'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
        'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
        't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'IY',
        'z': 'Z',
    }
    
    i = 0
    while i < len(word):
        # Try digraphs first
        if i + 1 < len(word):
            digraph = word[i:i+2]
            if digraph == 'th':
                phones.append('TH')
                i += 2
                continue
            elif digraph == 'sh':
                phones.append('SH')
                i += 2
                continue
            elif digraph == 'ch':
                phones.append('CH')
                i += 2
                continue
            elif digraph == 'ng':
                phones.append('NG')
                i += 2
                continue
            elif digraph == 'ee':
                phones.append('IY')
                i += 2
                continue
            elif digraph == 'oo':
                phones.append('UW')
                i += 2
                continue
            elif digraph == 'ou':
                phones.append('AW')
                i += 2
                continue
            elif digraph == 'ai' or digraph == 'ay':
                phones.append('EY')
                i += 2
                continue
        
        c = word[i]
        if c in mapping:
            phones.append(mapping[c])
        i += 1
    
    return phones if phones else ['AH']


def get_lexicon() -> Dict[str, List[str]]:
    """Get the full CMUdict lexicon."""
    return _load_cmudict()


def get_phonemes(word: str, allow_g2p: bool = True) -> List[str]:
    """
    Get phoneme sequence for a word.
    
    Uses CMUdict if available, falls back to G2P for OOV words.
    
    Args:
        word: Word to look up
        allow_g2p: Whether to use G2P for OOV words
    
    Returns:
        List of phoneme symbols
    """
    word_lower = word.lower().strip()
    
    # Try CMUdict first
    lexicon = get_lexicon()
    if word_lower in lexicon:
        return lexicon[word_lower]
    
    # OOV - use G2P
    if allow_g2p:
        return g2p_predict(word_lower)
    
    raise ValueError(f"Unknown word: '{word}' (not in CMUdict)")


def is_known_word(word: str) -> bool:
    """Check if a word is in CMUdict."""
    return word.lower().strip() in get_lexicon()


def get_phoneme_indices(word: str) -> List[int]:
    """Get MBON indices for a word's phonemes."""
    return [phoneme_to_index(p) for p in get_phonemes(word)]


def get_demo_words() -> List[str]:
    """Get the original 10 demo words."""
    return DEMO_WORDS.copy()


def create_training_data(
    max_words: int = 10000,
    min_length: int = 2,
    max_length: int = 15,
    max_phonemes: int = 12,
    seed: int = 42,
    include_demo: bool = True,
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Create training and held-out test data from CMUdict.
    
    Args:
        max_words: Maximum number of training words
        min_length: Minimum word length (characters)
        max_length: Maximum word length (characters)
        max_phonemes: Maximum phoneme count
        seed: Random seed for deterministic sampling
        include_demo: Ensure demo words are in training set
    
    Returns:
        training_lexicon: Dict of training words -> phonemes
        training_words: List of training words
        test_words: List of held-out test words
    """
    rng = np.random.default_rng(seed)
    lexicon = get_lexicon()
    
    # Filter words by length constraints
    valid_words = []
    for word, phones in lexicon.items():
        if (min_length <= len(word) <= max_length and 
            len(phones) <= max_phonemes and
            word.isalpha()):
            valid_words.append(word)
    
    # Shuffle deterministically
    rng.shuffle(valid_words)
    
    # Ensure demo words are included
    demo_set = set(DEMO_WORDS)
    if include_demo:
        # Move demo words to front
        demo_in_valid = [w for w in valid_words if w in demo_set]
        other_words = [w for w in valid_words if w not in demo_set]
        valid_words = demo_in_valid + other_words
    
    # Split into train and test
    n_train = min(max_words, int(len(valid_words) * 0.9))
    n_test = min(1000, len(valid_words) - n_train)
    
    train_words = valid_words[:n_train]
    test_words = valid_words[n_train:n_train + n_test]
    
    # Build training lexicon
    training_lexicon = {w: lexicon[w] for w in train_words}
    
    return training_lexicon, train_words, test_words


def create_training_samples(
    lexicon: Optional[Dict[str, List[str]]] = None,
    words: Optional[List[str]] = None,
) -> List[Tuple[str, int, str, int]]:
    """
    Create training samples: (word, position, phoneme, phoneme_idx).
    
    Each sample is one phoneme slot in a word, which the mushroom body
    must learn to classify.
    """
    if lexicon is None:
        lexicon = get_lexicon()
    
    if words is None:
        words = list(lexicon.keys())
    
    samples = []
    for word in words:
        if word not in lexicon:
            continue
        phonemes = lexicon[word]
        for pos, phoneme in enumerate(phonemes):
            samples.append((word, pos, phoneme, phoneme_to_index(phoneme)))
    
    return samples


def get_max_phoneme_length(lexicon: Optional[Dict[str, List[str]]] = None) -> int:
    """Get the maximum phoneme count in the lexicon."""
    if lexicon is None:
        lexicon = get_lexicon()
    return max(len(phones) for phones in lexicon.values()) if lexicon else 12


def print_lexicon_stats():
    """Print statistics about the lexicon."""
    lexicon = get_lexicon()
    
    phone_lengths = [len(phones) for phones in lexicon.values()]
    word_lengths = [len(word) for word in lexicon.keys()]
    
    print(f"\nCMUdict Statistics:")
    print(f"  Total words: {len(lexicon)}")
    print(f"  Word length: {min(word_lengths)}-{max(word_lengths)} chars "
          f"(mean: {np.mean(word_lengths):.1f})")
    print(f"  Phoneme count: {min(phone_lengths)}-{max(phone_lengths)} "
          f"(mean: {np.mean(phone_lengths):.1f})")
    
    # Demo word check
    print(f"\nDemo words:")
    for word in DEMO_WORDS:
        if word in lexicon:
            print(f"  {word}: {' '.join(lexicon[word])}")
        else:
            phones = g2p_predict(word)
            print(f"  {word}: {' '.join(phones)} (G2P)")


if __name__ == "__main__":
    print_lexicon_stats()
    
    # Test G2P for OOV
    oov_words = ['mushroom', 'connectome', 'chaos', 'hatsune', 'australia']
    print("\nOOV G2P predictions:")
    for word in oov_words:
        phones = get_phonemes(word)
        in_dict = "✓" if is_known_word(word) else "(G2P)"
        print(f"  {word}: {' '.join(phones)} {in_dict}")
