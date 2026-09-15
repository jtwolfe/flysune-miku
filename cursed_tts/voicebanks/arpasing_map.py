"""
ARPAbet ↔ Arpasing mapping for UTAU/OpenUtau voicebanks.

CMUdict uses uppercase ARPAbet (AA, AE, IY, etc.)
Arpasing voicebanks like MARIAN ILUSTRADO use lowercase (aa, ae, iy, etc.)

This module provides:
- Direct 1:1 mappings between ARPAbet and Arpasing
- Fallback chains for missing phonemes
- Coverage analysis utilities

Reference:
- CMUdict ARPAbet: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- Arpasing spec: https://arpasing.tubs.wtf/
- OpenUtau EN ARPA phonemizer conventions
"""

from typing import Dict, List, Optional, Tuple, Set


# ============================================================================
# Core ARPAbet ↔ Arpasing Mapping (39 CMUdict phonemes)
# ============================================================================

# Direct 1:1 mapping from CMUdict ARPAbet (uppercase) to Arpasing (lowercase)
# This is the standard mapping used by ARPAsing voicebanks
ARPABET_TO_ARPASING: Dict[str, str] = {
    # === VOWELS (monophthongs) ===
    'AA': 'aa',   # ɑ as in "odd"
    'AE': 'ae',   # æ as in "at"
    'AH': 'ah',   # ʌ as in "hut"
    'AO': 'ao',   # ɔ as in "ought"
    'EH': 'eh',   # ɛ as in "ed"
    'ER': 'er',   # ɝ as in "hurt" (r-colored)
    'IH': 'ih',   # ɪ as in "it"
    'IY': 'iy',   # i as in "eat"
    'UH': 'uh',   # ʊ as in "hood"
    'UW': 'uw',   # u as in "two"
    
    # === VOWELS (diphthongs) ===
    'AW': 'aw',   # aʊ as in "cow"
    'AY': 'ay',   # aɪ as in "hide"
    'EY': 'ey',   # eɪ as in "ate"
    'OW': 'ow',   # oʊ as in "oat"
    'OY': 'oy',   # ɔɪ as in "toy"
    
    # === CONSONANTS: Stops ===
    'P': 'p',     # voiceless bilabial
    'B': 'b',     # voiced bilabial
    'T': 't',     # voiceless alveolar
    'D': 'd',     # voiced alveolar
    'K': 'k',     # voiceless velar
    'G': 'g',     # voiced velar
    
    # === CONSONANTS: Affricates ===
    'CH': 'ch',   # voiceless postalveolar (tʃ)
    'JH': 'jh',   # voiced postalveolar (dʒ)
    
    # === CONSONANTS: Fricatives ===
    'F': 'f',     # voiceless labiodental
    'V': 'v',     # voiced labiodental
    'TH': 'th',   # voiceless dental (θ)
    'DH': 'dh',   # voiced dental (ð)
    'S': 's',     # voiceless alveolar
    'Z': 'z',     # voiced alveolar
    'SH': 'sh',   # voiceless postalveolar (ʃ)
    'ZH': 'zh',   # voiced postalveolar (ʒ)
    'HH': 'hh',   # voiceless glottal
    
    # === CONSONANTS: Nasals ===
    'M': 'm',     # bilabial nasal
    'N': 'n',     # alveolar nasal
    'NG': 'ng',   # velar nasal (ŋ)
    
    # === CONSONANTS: Liquids ===
    'L': 'l',     # alveolar lateral
    'R': 'r',     # alveolar approximant (ɹ)
    
    # === SEMIVOWELS / Glides ===
    'W': 'w',     # labio-velar approximant
    'Y': 'y',     # palatal approximant (j)
}

# Reverse mapping: Arpasing → ARPAbet
ARPASING_TO_ARPABET: Dict[str, str] = {v: k for k, v in ARPABET_TO_ARPASING.items()}


# ============================================================================
# Extended Arpasing Aliases (variations found in different voicebanks)
# ============================================================================

# Some voicebanks use alternative aliases for certain sounds
# Map these to the canonical ARPAbet phoneme
ARPASING_ALIASES: Dict[str, str] = {
    # Schwa variations (often mapped to AH)
    'ax': 'AH',      # Reduced schwa
    'axr': 'ER',     # Schwa + r (often merged with ER)
    
    # R-colored vowels (some banks separate these)
    'aar': 'AA',     # AA + r (fallback to AA if not available)
    'aer': 'AE',     # AE + r
    'ihr': 'IH',     # IH + r
    'uhr': 'UH',     # UH + r
    'eyr': 'EY',     # EY + r
    
    # Glottal stop (some banks include this)
    'q': None,       # Glottal stop - handled specially
    'cl': None,      # Closure - handled specially
    
    # Alternative spellings
    'dx': 'D',       # Alveolar tap (flap T/D) - fallback to D
    'nx': 'N',       # Syllabic N - fallback to N
    'el': 'L',       # Syllabic L - fallback to L
    'em': 'M',       # Syllabic M - fallback to M
    'en': 'N',       # Syllabic N - fallback to N
}


# ============================================================================
# Fallback Chains for Missing Phonemes
# ============================================================================

# When a phoneme isn't available, try these fallbacks in order
# Based on phonetic similarity
ARPABET_FALLBACKS: Dict[str, List[str]] = {
    # Vowel fallbacks (based on vowel space proximity)
    'AA': ['AO', 'AH'],           # back open → back mid-open → central
    'AE': ['EH', 'AH'],           # front open → front mid → central
    'AH': ['AA', 'AO'],           # central → back
    'AO': ['AA', 'AH', 'OW'],     # back rounded → back unrounded
    'EH': ['AE', 'IH', 'AH'],     # front mid → front open → front close
    'ER': ['AH', 'R', 'AA'],      # r-colored → central + r approximation
    'IH': ['IY', 'EH'],           # near-close → close → mid
    'IY': ['IH', 'EY'],           # close front → near-close
    'UH': ['UW', 'AH'],           # near-close back → close back
    'UW': ['UH', 'OW'],           # close back → near-close
    
    # Diphthong fallbacks
    'AW': ['AA', 'AO'],           # aʊ → starting vowel
    'AY': ['AA', 'AH'],           # aɪ → starting vowel
    'EY': ['EH', 'IY'],           # eɪ → starting vowel
    'OW': ['AO', 'UW'],           # oʊ → starting vowel
    'OY': ['AO', 'OW'],           # ɔɪ → starting vowel
    
    # Consonant fallbacks (based on manner/place)
    'P': ['B', 'T'],              # voiceless bilabial → voiced → alveolar
    'B': ['P', 'D'],              # voiced bilabial → voiceless → alveolar
    'T': ['D', 'K'],              # voiceless alveolar → voiced → velar
    'D': ['T', 'G'],              # voiced alveolar → voiceless → velar
    'K': ['G', 'T'],              # voiceless velar → voiced → alveolar
    'G': ['K', 'D'],              # voiced velar → voiceless → alveolar
    
    'CH': ['SH', 'T'],            # affricate → fricative → stop
    'JH': ['ZH', 'D'],            # voiced affricate → fricative → stop
    
    'F': ['V', 'TH'],             # labiodental → dental
    'V': ['F', 'DH'],
    'TH': ['F', 'S'],             # dental → labiodental → alveolar
    'DH': ['V', 'Z'],
    'S': ['Z', 'SH'],             # alveolar → voiced → postalveolar
    'Z': ['S', 'ZH'],
    'SH': ['S', 'CH'],            # postalveolar → alveolar → affricate
    'ZH': ['Z', 'JH'],
    'HH': ['AH'],                 # glottal → vowel (aspiration approx)
    
    'M': ['N', 'B'],              # bilabial nasal → alveolar → stop
    'N': ['M', 'NG'],             # alveolar nasal → bilabial → velar
    'NG': ['N', 'G'],             # velar nasal → alveolar → stop
    
    'L': ['R', 'N'],              # lateral → approximant → nasal
    'R': ['L', 'W'],              # approximant → lateral → glide
    
    'W': ['UW', 'R'],             # labio-velar → close back → approximant
    'Y': ['IY', 'L'],             # palatal → close front → lateral
}


# ============================================================================
# Conversion Functions
# ============================================================================

def arpabet_to_arpasing(phoneme: str) -> str:
    """
    Convert an ARPAbet phoneme (CMUdict style) to Arpasing alias.
    
    Strips stress markers (0, 1, 2) and converts to lowercase.
    
    Args:
        phoneme: ARPAbet phoneme (e.g., 'AA', 'AE1', 'IY2')
    
    Returns:
        Arpasing alias (e.g., 'aa', 'ae', 'iy')
    """
    # Strip stress markers
    clean = phoneme.rstrip('012').upper()
    
    if clean in ARPABET_TO_ARPASING:
        return ARPABET_TO_ARPASING[clean]
    
    # Unknown phoneme - return lowercase as-is
    return phoneme.lower().rstrip('012')


def arpasing_to_arpabet(alias: str) -> Optional[str]:
    """
    Convert an Arpasing alias to ARPAbet phoneme.
    
    Args:
        alias: Arpasing alias (e.g., 'aa', 'ae', 'iy')
    
    Returns:
        ARPAbet phoneme (e.g., 'AA', 'AE', 'IY') or None if unknown
    """
    clean = alias.lower().strip()
    
    if clean in ARPASING_TO_ARPABET:
        return ARPASING_TO_ARPABET[clean]
    
    # Check extended aliases
    if clean in ARPASING_ALIASES:
        return ARPASING_ALIASES[clean]
    
    return None


def get_arpasing_fallback(phoneme: str, available: Set[str]) -> Optional[str]:
    """
    Get a fallback Arpasing alias when the primary isn't available.
    
    Args:
        phoneme: ARPAbet phoneme to find fallback for
        available: Set of available Arpasing aliases in the voicebank
    
    Returns:
        Fallback Arpasing alias, or None if no fallback found
    """
    clean = phoneme.rstrip('012').upper()
    
    # First try direct mapping
    primary = arpabet_to_arpasing(clean)
    if primary in available:
        return primary
    
    # Try fallback chain
    if clean in ARPABET_FALLBACKS:
        for fallback_phoneme in ARPABET_FALLBACKS[clean]:
            fallback_alias = arpabet_to_arpasing(fallback_phoneme)
            if fallback_alias in available:
                return fallback_alias
    
    return None


def get_mapping_coverage(available_aliases: Set[str]) -> Dict[str, any]:
    """
    Analyze how well a set of Arpasing aliases covers the ARPAbet inventory.
    
    Args:
        available_aliases: Set of Arpasing aliases in a voicebank
    
    Returns:
        Dictionary with coverage statistics and missing phonemes
    """
    available_lower = {a.lower() for a in available_aliases}
    
    covered = []
    missing_with_fallback = []
    missing_no_fallback = []
    
    for arpabet, arpasing in ARPABET_TO_ARPASING.items():
        if arpasing in available_lower:
            covered.append(arpabet)
        else:
            fallback = get_arpasing_fallback(arpabet, available_lower)
            if fallback:
                missing_with_fallback.append((arpabet, fallback))
            else:
                missing_no_fallback.append(arpabet)
    
    total = len(ARPABET_TO_ARPASING)
    direct_pct = 100 * len(covered) / total
    fallback_pct = 100 * (len(covered) + len(missing_with_fallback)) / total
    
    return {
        'total_phonemes': total,
        'directly_covered': len(covered),
        'covered_with_fallback': len(covered) + len(missing_with_fallback),
        'direct_coverage_pct': direct_pct,
        'total_coverage_pct': fallback_pct,
        'covered': covered,
        'missing_with_fallback': missing_with_fallback,
        'missing_no_fallback': missing_no_fallback,
    }


# ============================================================================
# Arpasing Alias Pattern Matching
# ============================================================================

# Common Arpasing alias patterns for UTAU voicebanks
# These patterns help identify what type of sample a file represents

ARPASING_PATTERNS = {
    # CV (consonant-vowel) patterns: "k aa", "t iy", etc.
    'cv': r'^([a-z]{1,2})\s+([a-z]{2})$',
    
    # VC (vowel-consonant) patterns: "aa k", "iy t", etc.
    'vc': r'^([a-z]{2})\s+([a-z]{1,2})$',
    
    # VV (vowel-vowel) patterns: "aa iy", "ow ah", etc.
    'vv': r'^([a-z]{2})\s+([a-z]{2})$',
    
    # CC (consonant-consonant) patterns: "s t", "n d", etc.
    'cc': r'^([a-z]{1,2})\s+([a-z]{1,2})$',
    
    # Standalone vowel: "aa", "iy", etc.
    'v': r'^([a-z]{2})$',
    
    # Standalone consonant with vowel onset: "- aa", etc.
    'onset': r'^-\s+([a-z]{2})$',
    
    # Vowel with release: "aa -", etc.
    'release': r'^([a-z]{2})\s+-$',
}


def parse_arpasing_alias(alias: str) -> Tuple[str, List[str]]:
    """
    Parse an Arpasing alias into its component phonemes.
    
    Args:
        alias: Arpasing alias (e.g., "k aa", "aa", "aa k")
    
    Returns:
        Tuple of (alias_type, [phonemes])
        alias_type is one of: 'cv', 'vc', 'vv', 'cc', 'v', 'onset', 'release'
    """
    import re
    
    alias = alias.strip().lower()
    
    # Check each pattern
    for pattern_name, pattern in ARPASING_PATTERNS.items():
        match = re.match(pattern, alias)
        if match:
            phonemes = list(match.groups())
            return (pattern_name, phonemes)
    
    # Unknown pattern - treat as single unit
    return ('unknown', [alias])


def get_phonemes_from_alias(alias: str) -> List[str]:
    """
    Extract ARPAbet phonemes from an Arpasing alias.
    
    Args:
        alias: Arpasing alias (e.g., "k aa", "aa k")
    
    Returns:
        List of ARPAbet phonemes (e.g., ['K', 'AA'])
    """
    _, arpasing_phonemes = parse_arpasing_alias(alias)
    
    result = []
    for ap in arpasing_phonemes:
        if ap in ARPASING_TO_ARPABET:
            result.append(ARPASING_TO_ARPABET[ap])
        elif ap in ARPASING_ALIASES and ARPASING_ALIASES[ap]:
            result.append(ARPASING_ALIASES[ap])
        # Skip glottal stops and other special markers
    
    return result


# ============================================================================
# Module Self-Test
# ============================================================================

if __name__ == "__main__":
    print("ARPAbet ↔ Arpasing Mapping")
    print("=" * 60)
    
    print(f"\nTotal phonemes: {len(ARPABET_TO_ARPASING)}")
    
    print("\nMapping table:")
    print(f"{'ARPAbet':<10} {'Arpasing':<10} {'Example'}")
    print("-" * 40)
    
    examples = {
        'AA': 'odd', 'AE': 'at', 'IY': 'eat', 'K': 'key',
        'CH': 'cheese', 'TH': 'thin', 'NG': 'sing', 'W': 'way',
    }
    
    for arpabet, arpasing in ARPABET_TO_ARPASING.items():
        example = examples.get(arpabet, '')
        print(f"{arpabet:<10} {arpasing:<10} {example}")
    
    # Test coverage analysis
    print("\n" + "=" * 60)
    print("Coverage Analysis (hypothetical voicebank)")
    
    # Simulate a voicebank with most but not all phonemes
    test_bank = {
        'aa', 'ae', 'ah', 'ao', 'eh', 'er', 'ih', 'iy', 'uh', 'uw',
        'aw', 'ay', 'ey', 'ow', 'oy',
        'p', 'b', 't', 'd', 'k', 'g',
        'ch', 'jh', 'f', 'v', 's', 'z', 'sh', 'hh',
        'm', 'n', 'ng', 'l', 'r', 'w', 'y',
        # Missing: th, dh, zh
    }
    
    coverage = get_mapping_coverage(test_bank)
    print(f"\nDirect coverage: {coverage['directly_covered']}/{coverage['total_phonemes']} "
          f"({coverage['direct_coverage_pct']:.1f}%)")
    print(f"With fallbacks: {coverage['covered_with_fallback']}/{coverage['total_phonemes']} "
          f"({coverage['total_coverage_pct']:.1f}%)")
    
    if coverage['missing_with_fallback']:
        print("\nMissing (with fallback):")
        for phoneme, fallback in coverage['missing_with_fallback']:
            print(f"  {phoneme} → {fallback}")
    
    if coverage['missing_no_fallback']:
        print("\nMissing (no fallback):")
        for phoneme in coverage['missing_no_fallback']:
            print(f"  {phoneme}")
