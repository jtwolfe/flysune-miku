"""
Full CMUdict-compatible ARPAbet phoneme inventory.

This covers all phonemes needed for general English TTS.
Stress markers (0, 1, 2) are stripped - we use stress-free phones.
"""

from typing import NamedTuple
import numpy as np


class PhonemeInfo(NamedTuple):
    """Metadata for a phoneme."""
    symbol: str
    ipa: str
    phoneme_type: str  # 'vowel', 'consonant', 'semivowel'
    example: str
    description: str


# Full CMUdict ARPAbet inventory (39 phonemes, stress-stripped)
# Organized by phoneme type for synthesis parameters
PHONEME_INVENTORY = {
    # === VOWELS (monophthongs) ===
    'AA': PhonemeInfo('AA', 'ɑ', 'vowel', 'odd', 'open back unrounded'),
    'AE': PhonemeInfo('AE', 'æ', 'vowel', 'at', 'near-open front unrounded'),
    'AH': PhonemeInfo('AH', 'ʌ', 'vowel', 'hut', 'open-mid back unrounded'),
    'AO': PhonemeInfo('AO', 'ɔ', 'vowel', 'ought', 'open-mid back rounded'),
    'EH': PhonemeInfo('EH', 'ɛ', 'vowel', 'ed', 'open-mid front unrounded'),
    'ER': PhonemeInfo('ER', 'ɝ', 'vowel', 'hurt', 'r-colored mid central'),
    'IH': PhonemeInfo('IH', 'ɪ', 'vowel', 'it', 'near-close front unrounded'),
    'IY': PhonemeInfo('IY', 'i', 'vowel', 'eat', 'close front unrounded'),
    'UH': PhonemeInfo('UH', 'ʊ', 'vowel', 'hood', 'near-close back rounded'),
    'UW': PhonemeInfo('UW', 'u', 'vowel', 'two', 'close back rounded'),
    
    # === VOWELS (diphthongs) ===
    'AW': PhonemeInfo('AW', 'aʊ', 'vowel', 'cow', 'diphthong'),
    'AY': PhonemeInfo('AY', 'aɪ', 'vowel', 'hide', 'diphthong'),
    'EY': PhonemeInfo('EY', 'eɪ', 'vowel', 'ate', 'diphthong'),
    'OW': PhonemeInfo('OW', 'oʊ', 'vowel', 'oat', 'diphthong'),
    'OY': PhonemeInfo('OY', 'ɔɪ', 'vowel', 'toy', 'diphthong'),
    
    # === CONSONANTS: Stops ===
    'P': PhonemeInfo('P', 'p', 'consonant', 'pea', 'voiceless bilabial stop'),
    'B': PhonemeInfo('B', 'b', 'consonant', 'bee', 'voiced bilabial stop'),
    'T': PhonemeInfo('T', 't', 'consonant', 'tea', 'voiceless alveolar stop'),
    'D': PhonemeInfo('D', 'd', 'consonant', 'dee', 'voiced alveolar stop'),
    'K': PhonemeInfo('K', 'k', 'consonant', 'key', 'voiceless velar stop'),
    'G': PhonemeInfo('G', 'ɡ', 'consonant', 'gee', 'voiced velar stop'),
    
    # === CONSONANTS: Affricates ===
    'CH': PhonemeInfo('CH', 'tʃ', 'consonant', 'cheese', 'voiceless postalveolar affricate'),
    'JH': PhonemeInfo('JH', 'dʒ', 'consonant', 'gee', 'voiced postalveolar affricate'),
    
    # === CONSONANTS: Fricatives ===
    'F': PhonemeInfo('F', 'f', 'consonant', 'fee', 'voiceless labiodental fricative'),
    'V': PhonemeInfo('V', 'v', 'consonant', 'vee', 'voiced labiodental fricative'),
    'TH': PhonemeInfo('TH', 'θ', 'consonant', 'thin', 'voiceless dental fricative'),
    'DH': PhonemeInfo('DH', 'ð', 'consonant', 'then', 'voiced dental fricative'),
    'S': PhonemeInfo('S', 's', 'consonant', 'sea', 'voiceless alveolar fricative'),
    'Z': PhonemeInfo('Z', 'z', 'consonant', 'zee', 'voiced alveolar fricative'),
    'SH': PhonemeInfo('SH', 'ʃ', 'consonant', 'she', 'voiceless postalveolar fricative'),
    'ZH': PhonemeInfo('ZH', 'ʒ', 'consonant', 'measure', 'voiced postalveolar fricative'),
    'HH': PhonemeInfo('HH', 'h', 'consonant', 'he', 'voiceless glottal fricative'),
    
    # === CONSONANTS: Nasals ===
    'M': PhonemeInfo('M', 'm', 'consonant', 'me', 'bilabial nasal'),
    'N': PhonemeInfo('N', 'n', 'consonant', 'knee', 'alveolar nasal'),
    'NG': PhonemeInfo('NG', 'ŋ', 'consonant', 'sing', 'velar nasal'),
    
    # === CONSONANTS: Liquids ===
    'L': PhonemeInfo('L', 'l', 'consonant', 'lee', 'alveolar lateral'),
    'R': PhonemeInfo('R', 'ɹ', 'consonant', 'ray', 'alveolar approximant'),
    
    # === SEMIVOWELS / Glides ===
    'W': PhonemeInfo('W', 'w', 'semivowel', 'way', 'labio-velar approximant'),
    'Y': PhonemeInfo('Y', 'j', 'semivowel', 'yes', 'palatal approximant'),
}

# Ordered list of phonemes (for MBON indexing)
PHONEME_LIST = list(PHONEME_INVENTORY.keys())
NUM_PHONEMES = len(PHONEME_LIST)

# Index mappings
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_LIST)}
IDX_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_LIST)}

# Phoneme categories for synthesis
VOWELS = {p for p, info in PHONEME_INVENTORY.items() if info.phoneme_type == 'vowel'}
CONSONANTS = {p for p, info in PHONEME_INVENTORY.items() if info.phoneme_type == 'consonant'}
SEMIVOWELS = {p for p, info in PHONEME_INVENTORY.items() if info.phoneme_type == 'semivowel'}

# Sub-categories for synthesis
STOPS = {'P', 'B', 'T', 'D', 'K', 'G'}
AFFRICATES = {'CH', 'JH'}
FRICATIVES = {'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'}
NASALS = {'M', 'N', 'NG'}
LIQUIDS = {'L', 'R'}
VOICED_CONSONANTS = {'B', 'D', 'G', 'JH', 'V', 'DH', 'Z', 'ZH', 'M', 'N', 'NG', 'L', 'R'}
DIPHTHONGS = {'AW', 'AY', 'EY', 'OW', 'OY'}


def strip_stress(phoneme: str) -> str:
    """Strip stress marker (0, 1, 2) from a phoneme."""
    return phoneme.rstrip('012')


def phoneme_to_index(phoneme: str) -> int:
    """Convert a phoneme symbol to its MBON compartment index."""
    p = strip_stress(phoneme)
    if p not in PHONEME_TO_IDX:
        raise ValueError(f"Unknown phoneme: {phoneme} (stripped: {p}). Known: {PHONEME_LIST}")
    return PHONEME_TO_IDX[p]


def index_to_phoneme(idx: int) -> str:
    """Convert an MBON compartment index to its phoneme symbol."""
    if idx not in IDX_TO_PHONEME:
        raise ValueError(f"Invalid index: {idx}. Valid range: 0-{NUM_PHONEMES-1}")
    return IDX_TO_PHONEME[idx]


def get_phoneme_info(phoneme: str) -> PhonemeInfo:
    """Get metadata for a phoneme."""
    p = strip_stress(phoneme)
    if p not in PHONEME_INVENTORY:
        raise ValueError(f"Unknown phoneme: {phoneme}")
    return PHONEME_INVENTORY[p]


def is_vowel(phoneme: str) -> bool:
    """Check if a phoneme is a vowel."""
    return strip_stress(phoneme) in VOWELS


def is_consonant(phoneme: str) -> bool:
    """Check if a phoneme is a consonant."""
    return strip_stress(phoneme) in CONSONANTS


def is_voiced(phoneme: str) -> bool:
    """Check if a consonant is voiced."""
    p = strip_stress(phoneme)
    return p in VOICED_CONSONANTS or p in VOWELS or p in SEMIVOWELS


def print_inventory():
    """Print the phoneme inventory for reference."""
    print(f"\nPhoneme Inventory ({NUM_PHONEMES} phonemes):")
    print("-" * 70)
    print(f"{'Symbol':<8} {'IPA':<6} {'Type':<12} {'Example':<10} {'Description'}")
    print("-" * 70)
    
    current_type = None
    for symbol, info in PHONEME_INVENTORY.items():
        if info.phoneme_type != current_type:
            current_type = info.phoneme_type
            print(f"\n  === {current_type.upper()}S ===")
        print(f"{symbol:<8} {info.ipa:<6} {info.phoneme_type:<12} {info.example:<10} {info.description}")
    print()


if __name__ == "__main__":
    print_inventory()
    print(f"\nTotal: {NUM_PHONEMES} phonemes")
    print(f"Vowels: {len(VOWELS)}")
    print(f"Consonants: {len(CONSONANTS)}")
    print(f"Semivowels: {len(SEMIVOWELS)}")
