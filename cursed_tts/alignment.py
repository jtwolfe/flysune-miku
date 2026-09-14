"""
Grapheme-phoneme alignment for G2P training.

Aligns letters (graphemes) to phonemes using a simple monotonic alignment
with position interpolation. For English, this gives reasonable alignments
without requiring full EM training.

Each training example becomes: (letter_context, phoneme_position, target_phoneme)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class AlignedPair:
    """A single aligned (grapheme_context, phoneme) training example."""
    word: str
    letter_pos: int      # Center letter position in word
    letter_context: str  # Window of letters around position
    phoneme_idx: int     # Which phoneme this maps to
    phoneme: str         # Target phoneme symbol
    
    def __repr__(self):
        return f"'{self.letter_context}' → {self.phoneme}"


def align_word(word: str, phonemes: List[str], context_size: int = 3) -> List[AlignedPair]:
    """
    Align a word's letters to its phonemes using monotonic interpolation.
    
    For a word with L letters and P phonemes, we create P training examples,
    one per phoneme. Each example has a letter context window centered at
    the interpolated position.
    
    Args:
        word: The word (lowercase letters only)
        phonemes: List of phoneme symbols
        context_size: Half-width of letter context window (total = 2*size + 1)
    
    Returns:
        List of AlignedPair training examples
    """
    word = word.lower()
    n_letters = len(word)
    n_phonemes = len(phonemes)
    
    if n_letters == 0 or n_phonemes == 0:
        return []
    
    pairs = []
    
    for p_idx, phoneme in enumerate(phonemes):
        # Interpolate: which letter position corresponds to this phoneme?
        # Map phoneme index [0, P-1] to letter index [0, L-1]
        if n_phonemes == 1:
            letter_pos = n_letters // 2
        else:
            letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
        
        letter_pos = max(0, min(letter_pos, n_letters - 1))
        
        # Extract letter context window
        context_chars = []
        for offset in range(-context_size, context_size + 1):
            idx = letter_pos + offset
            if 0 <= idx < n_letters:
                context_chars.append(word[idx])
            else:
                context_chars.append('_')  # Padding for boundary
        
        letter_context = ''.join(context_chars)
        
        pairs.append(AlignedPair(
            word=word,
            letter_pos=letter_pos,
            letter_context=letter_context,
            phoneme_idx=p_idx,
            phoneme=phoneme,
        ))
    
    return pairs


def align_lexicon(lexicon: Dict[str, List[str]], 
                  context_size: int = 3,
                  verbose: bool = False) -> List[AlignedPair]:
    """
    Align an entire lexicon to create training data.
    
    Args:
        lexicon: Dict mapping words to phoneme lists
        context_size: Letter context window half-width
        verbose: Print progress
    
    Returns:
        List of all AlignedPair training examples
    """
    all_pairs = []
    
    for i, (word, phonemes) in enumerate(lexicon.items()):
        pairs = align_word(word, phonemes, context_size)
        all_pairs.extend(pairs)
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Aligned {i+1}/{len(lexicon)} words...")
    
    if verbose:
        print(f"  Total aligned pairs: {len(all_pairs)}")
    
    return all_pairs


def create_g2p_training_data(
    max_words: int = 10000,
    context_size: int = 3,
    seed: int = 42,
    test_split: float = 0.1,
    include_demo: bool = True,
    verbose: bool = True,
) -> Tuple[List[AlignedPair], List[AlignedPair], List[str], List[str]]:
    """
    Create aligned G2P training and test data from CMUdict.
    
    Args:
        max_words: Maximum training words
        context_size: Letter context window half-width
        seed: Random seed for splitting
        test_split: Fraction of words for test set
        include_demo: Ensure demo words are in training
        verbose: Print progress
    
    Returns:
        train_pairs: Training aligned pairs
        test_pairs: Test aligned pairs
        train_words: Training word list
        test_words: Test word list
    """
    from .lexicon import get_lexicon, get_demo_words
    
    lexicon = get_lexicon()
    demo_words = get_demo_words()
    
    # Filter to alphabetic words with reasonable length
    filtered = {
        w: p for w, p in lexicon.items()
        if w.isalpha() and 2 <= len(w) <= 15 and 1 <= len(p) <= 15
    }
    
    if verbose:
        print(f"Filtered lexicon: {len(filtered)} words")
    
    # Ensure demo words are included
    all_words = list(filtered.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_words)
    
    # Move demo words to front
    demo_in_lexicon = [w for w in demo_words if w in filtered]
    other_words = [w for w in all_words if w not in demo_in_lexicon]
    
    # Select training words
    n_train = min(max_words, len(other_words) + len(demo_in_lexicon))
    n_demo = len(demo_in_lexicon)
    n_other = n_train - n_demo
    
    train_words = demo_in_lexicon + other_words[:n_other]
    
    # Test words from remainder
    n_test = int(len(other_words[n_other:]) * test_split)
    test_words = other_words[n_other:n_other + n_test]
    
    if verbose:
        print(f"Training words: {len(train_words)} (including {n_demo} demo)")
        print(f"Test words: {len(test_words)}")
    
    # Create aligned pairs
    train_lexicon = {w: filtered[w] for w in train_words}
    test_lexicon = {w: filtered[w] for w in test_words}
    
    if verbose:
        print("Aligning training data...")
    train_pairs = align_lexicon(train_lexicon, context_size, verbose)
    
    if verbose:
        print("Aligning test data...")
    test_pairs = align_lexicon(test_lexicon, context_size, verbose=False)
    
    if verbose:
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
    
    return train_pairs, test_pairs, train_words, test_words


def encode_letter_context(context: str, vocab_size: int = 28) -> np.ndarray:
    """
    Encode a letter context string as a feature vector.
    
    Uses one-hot encoding for each position, plus n-gram features.
    
    Args:
        context: Letter context string (e.g., "_ca_t__")
        vocab_size: Number of character types (26 letters + '_' + other)
    
    Returns:
        Feature vector
    """
    # Character to index mapping
    def char_to_idx(c):
        if c == '_':
            return 26  # Padding
        elif 'a' <= c <= 'z':
            return ord(c) - ord('a')
        else:
            return 27  # Other
    
    context_len = len(context)
    
    # One-hot for each position
    one_hot = np.zeros(context_len * vocab_size, dtype=np.float32)
    for i, c in enumerate(context):
        idx = char_to_idx(c)
        one_hot[i * vocab_size + idx] = 1.0
    
    # Bigram features (character pairs)
    n_bigrams = 26 * 26  # Only letter-letter bigrams
    bigrams = np.zeros(n_bigrams, dtype=np.float32)
    for i in range(len(context) - 1):
        c1, c2 = context[i], context[i + 1]
        if 'a' <= c1 <= 'z' and 'a' <= c2 <= 'z':
            idx = (ord(c1) - ord('a')) * 26 + (ord(c2) - ord('a'))
            bigrams[idx] = 1.0
    
    # Trigram hash features (compressed)
    n_trigram_buckets = 256
    trigrams = np.zeros(n_trigram_buckets, dtype=np.float32)
    for i in range(len(context) - 2):
        tri = context[i:i+3]
        if all('a' <= c <= 'z' for c in tri):
            h = hash(tri) % n_trigram_buckets
            trigrams[h] = 1.0
    
    # Combine features
    features = np.concatenate([one_hot, bigrams, trigrams])
    
    return features


def get_feature_dim(context_size: int = 3, vocab_size: int = 28) -> int:
    """Get the feature dimension for a given context size."""
    context_len = 2 * context_size + 1  # e.g., 7 for context_size=3
    one_hot_dim = context_len * vocab_size
    bigram_dim = 26 * 26
    trigram_dim = 256
    return one_hot_dim + bigram_dim + trigram_dim


if __name__ == "__main__":
    # Test alignment
    print("Testing grapheme-phoneme alignment...\n")
    
    test_cases = [
        ("cat", ["K", "AE", "T"]),
        ("dog", ["D", "AO", "G"]),
        ("phone", ["F", "OW", "N"]),
        ("thought", ["TH", "AO", "T"]),
        ("australia", ["AO", "S", "T", "R", "EY", "L", "Y", "AH"]),
    ]
    
    for word, phonemes in test_cases:
        print(f"Word: '{word}' → {' '.join(phonemes)}")
        pairs = align_word(word, phonemes)
        for p in pairs:
            print(f"  {p}")
        print()
    
    # Test feature encoding
    print("Feature encoding test:")
    context = "_cat___"
    features = encode_letter_context(context)
    print(f"Context '{context}' → {len(features)} features, {np.sum(features > 0)} non-zero")
