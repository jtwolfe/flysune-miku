"""
Voicebank support for real audio samples (UTAU/OpenUtau banks).

This module provides integration with open voicebanks that can replace
or augment the synthetic formant synthesis with real recorded samples.
"""

from .arpasing_map import (
    ARPABET_TO_ARPASING, ARPASING_TO_ARPABET,
    arpabet_to_arpasing, arpasing_to_arpabet,
    get_arpasing_fallback, get_mapping_coverage,
)
from .utau_singer import (
    UTAUSingerFly, UTAUSingerSwarm, MarianSingerSwarm,
    load_utau_sample, find_voicebank_samples,
)

__all__ = [
    'ARPABET_TO_ARPASING', 'ARPASING_TO_ARPABET',
    'arpabet_to_arpasing', 'arpasing_to_arpabet',
    'get_arpasing_fallback', 'get_mapping_coverage',
    'UTAUSingerFly', 'UTAUSingerSwarm', 'MarianSingerSwarm',
    'load_utau_sample', 'find_voicebank_samples',
]
