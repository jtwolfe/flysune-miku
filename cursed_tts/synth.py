"""
Cursed phoneme audio synthesis for full CMUdict ARPAbet.

Generates distinct, audible audio clips for each of the 39 phonemes:
- Vowels: Two-formant synthesis with characteristic F1/F2
- Consonants: Noise bursts, clicks, fricatives, nasals, etc.
- Semivowels: Glide transitions

This is intentionally cursed and robotic - stage 2 would add neural vocoding.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import wave

from .phonemes import (
    PHONEME_LIST, PHONEME_INVENTORY, VOWELS, CONSONANTS, SEMIVOWELS,
    STOPS, AFFRICATES, FRICATIVES, NASALS, LIQUIDS, VOICED_CONSONANTS,
    DIPHTHONGS, strip_stress, is_vowel, is_voiced
)


# Audio parameters
SAMPLE_RATE = 22050
VOWEL_DURATION = 0.15
CONSONANT_DURATION = 0.08
SEMIVOWEL_DURATION = 0.10

# Formant frequencies (F1, F2) for vowels in Hz
# Based on typical American English values
VOWEL_FORMANTS = {
    # Monophthongs
    'AA': (750, 1100),   # ɑ as in "odd"
    'AE': (700, 1800),   # æ as in "at"
    'AH': (600, 1200),   # ʌ as in "hut"
    'AO': (550, 850),    # ɔ as in "ought"
    'EH': (550, 1900),   # ɛ as in "ed"
    'ER': (500, 1400),   # ɝ as in "hurt" (r-colored)
    'IH': (400, 2000),   # ɪ as in "it"
    'IY': (280, 2400),   # i as in "eat"
    'UH': (450, 1050),   # ʊ as in "hood"
    'UW': (310, 870),    # u as in "two"
    # Diphthongs (use starting formants, will glide)
    'AW': (750, 1200),   # aʊ as in "cow" (start)
    'AY': (750, 1200),   # aɪ as in "hide" (start)
    'EY': (450, 2100),   # eɪ as in "ate" (start)
    'OW': (500, 850),    # oʊ as in "oat" (start)
    'OY': (550, 850),    # ɔɪ as in "toy" (start)
}

# Diphthong end formants
DIPHTHONG_END = {
    'AW': (350, 900),    # → ʊ
    'AY': (350, 2200),   # → ɪ
    'EY': (350, 2200),   # → ɪ
    'OW': (350, 900),    # → ʊ
    'OY': (350, 2200),   # → ɪ
}

# Consonant synthesis parameters
CONSONANT_PARAMS = {
    # Stops
    'P': {'type': 'stop', 'freq': 1800, 'noise': 0.7, 'voiced': False, 'burst': 0.5},
    'B': {'type': 'stop', 'freq': 300, 'noise': 0.4, 'voiced': True, 'burst': 0.3},
    'T': {'type': 'stop', 'freq': 3500, 'noise': 0.8, 'voiced': False, 'burst': 0.6},
    'D': {'type': 'stop', 'freq': 400, 'noise': 0.5, 'voiced': True, 'burst': 0.4},
    'K': {'type': 'stop', 'freq': 1500, 'noise': 0.8, 'voiced': False, 'burst': 0.5},
    'G': {'type': 'stop', 'freq': 350, 'noise': 0.5, 'voiced': True, 'burst': 0.3},
    
    # Affricates
    'CH': {'type': 'affricate', 'freq': 4000, 'noise': 0.9, 'voiced': False},
    'JH': {'type': 'affricate', 'freq': 2500, 'noise': 0.7, 'voiced': True},
    
    # Fricatives
    'F': {'type': 'fricative', 'freq': 3500, 'noise': 0.6, 'voiced': False},
    'V': {'type': 'fricative', 'freq': 250, 'noise': 0.4, 'voiced': True},
    'TH': {'type': 'fricative', 'freq': 4500, 'noise': 0.4, 'voiced': False},
    'DH': {'type': 'fricative', 'freq': 300, 'noise': 0.3, 'voiced': True},
    'S': {'type': 'fricative', 'freq': 5500, 'noise': 1.0, 'voiced': False},
    'Z': {'type': 'fricative', 'freq': 4500, 'noise': 0.8, 'voiced': True},
    'SH': {'type': 'fricative', 'freq': 3500, 'noise': 0.9, 'voiced': False},
    'ZH': {'type': 'fricative', 'freq': 2500, 'noise': 0.7, 'voiced': True},
    'HH': {'type': 'fricative', 'freq': 1000, 'noise': 0.5, 'voiced': False, 'aspirate': True},
    
    # Nasals
    'M': {'type': 'nasal', 'freq': 250, 'f2': 1000, 'voiced': True},
    'N': {'type': 'nasal', 'freq': 280, 'f2': 1700, 'voiced': True},
    'NG': {'type': 'nasal', 'freq': 280, 'f2': 2300, 'voiced': True},
    
    # Liquids
    'L': {'type': 'liquid', 'freq': 350, 'f2': 1200, 'f3': 2800, 'voiced': True},
    'R': {'type': 'liquid', 'freq': 350, 'f2': 1000, 'f3': 1600, 'voiced': True},
}

# Semivowel parameters
SEMIVOWEL_PARAMS = {
    'W': {'start': (350, 700), 'end': (400, 1200), 'voiced': True},
    'Y': {'start': (300, 2300), 'end': (400, 1800), 'voiced': True},
}


def _make_envelope(n_samples: int, attack: float = 0.03, release: float = 0.05) -> np.ndarray:
    """Create an amplitude envelope with attack and release."""
    envelope = np.ones(n_samples)
    
    attack_samples = min(int(attack * SAMPLE_RATE), n_samples // 3)
    release_samples = min(int(release * SAMPLE_RATE), n_samples // 3)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    return envelope


def _add_voicing(signal: np.ndarray, freq: float, amount: float = 0.3) -> np.ndarray:
    """Add voicing (low-frequency pitch) to a signal."""
    t = np.linspace(0, len(signal) / SAMPLE_RATE, len(signal))
    voice = np.sin(2 * np.pi * freq * t)
    return signal + voice * amount


def _synthesize_vowel(phoneme: str, duration: float = VOWEL_DURATION) -> np.ndarray:
    """Synthesize a vowel using two-formant additive synthesis."""
    p = strip_stress(phoneme)
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples)
    
    f1_start, f2_start = VOWEL_FORMANTS[p]
    
    # Check if diphthong (formants glide)
    if p in DIPHTHONG_END:
        f1_end, f2_end = DIPHTHONG_END[p]
        f1 = np.linspace(f1_start, f1_end, n_samples)
        f2 = np.linspace(f2_start, f2_end, n_samples)
    else:
        f1 = np.full(n_samples, f1_start)
        f2 = np.full(n_samples, f2_start)
    
    # Fundamental frequency
    f0 = 140  # Hz
    
    # Generate harmonics with formant shaping
    signal = np.zeros(n_samples)
    for harmonic in range(1, 12):
        freq = f0 * harmonic
        
        # Time-varying formant amplitudes
        amp1 = np.exp(-((freq - f1) / 150) ** 2)
        amp2 = np.exp(-((freq - f2) / 200) ** 2)
        amp = (amp1 + amp2 * 0.6) / (harmonic ** 0.7)
        
        # Phase accumulation for smooth frequency changes
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Apply envelope
    signal *= _make_envelope(n_samples)
    
    # Normalize
    if np.abs(signal).max() > 0:
        signal = signal / np.abs(signal).max() * 0.75
    
    return signal.astype(np.float32)


def _synthesize_stop(phoneme: str, params: dict, duration: float) -> np.ndarray:
    """Synthesize a stop consonant (plosive)."""
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    
    # Stop = silence + burst
    burst_start = int(n_samples * 0.35)
    burst_len = int(n_samples * params.get('burst', 0.4))
    
    if burst_start + burst_len > n_samples:
        burst_len = n_samples - burst_start
    
    # Noise burst
    noise = np.random.randn(burst_len) * params['noise']
    
    # Filter around characteristic frequency
    freq = params['freq']
    t_burst = np.linspace(0, burst_len / SAMPLE_RATE, burst_len)
    carrier = np.sin(2 * np.pi * freq * t_burst)
    filtered = noise * 0.6 + carrier * 0.4 * np.abs(noise)
    
    # Burst envelope (sharp attack, quick decay)
    burst_env = np.exp(-np.linspace(0, 6, burst_len))
    filtered *= burst_env
    
    signal[burst_start:burst_start + burst_len] = filtered
    
    # Add voicing for voiced stops
    if params.get('voiced', False):
        voice_freq = params['freq']
        voice = np.sin(2 * np.pi * voice_freq * t) * 0.25
        voice *= _make_envelope(n_samples, attack=0.15, release=0.15)
        signal += voice
    
    return signal


def _synthesize_affricate(phoneme: str, params: dict, duration: float) -> np.ndarray:
    """Synthesize an affricate (stop + fricative)."""
    n_samples = int(duration * SAMPLE_RATE)
    signal = np.zeros(n_samples)
    
    # Stop portion (first 40%)
    stop_len = int(n_samples * 0.4)
    stop_params = {'freq': params['freq'], 'noise': params['noise'] * 0.5, 
                   'voiced': params['voiced'], 'burst': 0.6}
    stop_part = _synthesize_stop(phoneme, stop_params, stop_len / SAMPLE_RATE)
    signal[:stop_len] = stop_part
    
    # Fricative portion (remaining 60%)
    fric_len = n_samples - stop_len
    t_fric = np.linspace(0, fric_len / SAMPLE_RATE, fric_len)
    
    noise = np.random.randn(fric_len) * params['noise']
    carrier = np.sin(2 * np.pi * params['freq'] * t_fric)
    fric_signal = noise * 0.5 + carrier * 0.3 * noise
    
    fric_env = _make_envelope(fric_len, attack=0.05, release=0.1)
    signal[stop_len:] = fric_signal * fric_env
    
    if params.get('voiced', False):
        signal = _add_voicing(signal, 120, 0.2)
    
    return signal


def _synthesize_fricative(phoneme: str, params: dict, duration: float) -> np.ndarray:
    """Synthesize a fricative."""
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples)
    
    noise = np.random.randn(n_samples) * params['noise']
    
    if params.get('aspirate', False):
        # HH is broadband noise
        signal = noise * 0.5
    else:
        # Filtered noise + carrier
        freq = params['freq']
        carrier = np.sin(2 * np.pi * freq * t)
        signal = noise * 0.4 + carrier * 0.2 * np.abs(noise)
    
    signal *= _make_envelope(n_samples, attack=0.05, release=0.08)
    
    if params.get('voiced', False):
        signal = _add_voicing(signal, 130, 0.3)
    
    return signal


def _synthesize_nasal(phoneme: str, params: dict, duration: float) -> np.ndarray:
    """Synthesize a nasal consonant."""
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples)
    
    f0 = 130  # Fundamental
    f1 = params['freq']
    f2 = params['f2']
    
    signal = np.zeros(n_samples)
    
    # Low formants with nasal quality
    for harmonic in range(1, 8):
        freq = f0 * harmonic
        amp1 = np.exp(-((freq - f1) / 100) ** 2)
        amp2 = np.exp(-((freq - f2) / 150) ** 2) * 0.5
        amp = (amp1 + amp2) / (harmonic ** 0.8)
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Add nasal "buzz"
    signal += np.sin(2 * np.pi * f1 * t) * 0.3
    
    signal *= _make_envelope(n_samples, attack=0.08, release=0.08)
    
    return signal


def _synthesize_liquid(phoneme: str, params: dict, duration: float) -> np.ndarray:
    """Synthesize a liquid (L or R)."""
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples)
    
    f0 = 130
    f1 = params['freq']
    f2 = params['f2']
    f3 = params['f3']
    
    signal = np.zeros(n_samples)
    
    for harmonic in range(1, 10):
        freq = f0 * harmonic
        amp1 = np.exp(-((freq - f1) / 100) ** 2)
        amp2 = np.exp(-((freq - f2) / 150) ** 2) * 0.6
        amp3 = np.exp(-((freq - f3) / 200) ** 2) * 0.4
        amp = (amp1 + amp2 + amp3) / (harmonic ** 0.6)
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    signal *= _make_envelope(n_samples, attack=0.08, release=0.1)
    
    return signal


def _synthesize_semivowel(phoneme: str, duration: float = SEMIVOWEL_DURATION) -> np.ndarray:
    """Synthesize a semivowel (glide)."""
    params = SEMIVOWEL_PARAMS[phoneme]
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples)
    
    f1_start, f2_start = params['start']
    f1_end, f2_end = params['end']
    
    # Gliding formants
    f1 = np.linspace(f1_start, f1_end, n_samples)
    f2 = np.linspace(f2_start, f2_end, n_samples)
    
    f0 = 140
    signal = np.zeros(n_samples)
    
    for harmonic in range(1, 10):
        freq = f0 * harmonic
        amp1 = np.exp(-((freq - f1) / 120) ** 2)
        amp2 = np.exp(-((freq - f2) / 180) ** 2) * 0.7
        amp = (amp1 + amp2) / (harmonic ** 0.7)
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    signal *= _make_envelope(n_samples, attack=0.06, release=0.08)
    
    if np.abs(signal).max() > 0:
        signal = signal / np.abs(signal).max() * 0.7
    
    return signal.astype(np.float32)


def _synthesize_consonant(phoneme: str, duration: float = CONSONANT_DURATION) -> np.ndarray:
    """Synthesize a consonant."""
    p = strip_stress(phoneme)
    params = CONSONANT_PARAMS[p]
    ctype = params['type']
    
    if ctype == 'stop':
        signal = _synthesize_stop(p, params, duration)
    elif ctype == 'affricate':
        signal = _synthesize_affricate(p, params, duration)
    elif ctype == 'fricative':
        signal = _synthesize_fricative(p, params, duration)
    elif ctype == 'nasal':
        signal = _synthesize_nasal(p, params, duration)
    elif ctype == 'liquid':
        signal = _synthesize_liquid(p, params, duration)
    else:
        signal = np.random.randn(int(duration * SAMPLE_RATE)) * 0.3
    
    # Normalize
    if np.abs(signal).max() > 0:
        signal = signal / np.abs(signal).max() * 0.65
    
    return signal.astype(np.float32)


def synthesize_phoneme(phoneme: str) -> np.ndarray:
    """
    Synthesize audio for a single phoneme.
    
    Args:
        phoneme: ARPAbet phoneme symbol (stress markers stripped automatically)
    
    Returns:
        Audio signal as float32 numpy array
    """
    p = strip_stress(phoneme)
    
    if p in VOWELS:
        return _synthesize_vowel(p)
    elif p in SEMIVOWELS:
        return _synthesize_semivowel(p)
    elif p in CONSONANTS:
        return _synthesize_consonant(p)
    else:
        raise ValueError(f"Unknown phoneme: {phoneme}")


def synthesize_all_phonemes() -> Dict[str, np.ndarray]:
    """
    Synthesize audio for all phonemes in the inventory.
    
    Returns:
        Dictionary mapping phoneme symbols to audio arrays
    """
    phoneme_audio = {}
    for phoneme in PHONEME_LIST:
        phoneme_audio[phoneme] = synthesize_phoneme(phoneme)
    return phoneme_audio


def concatenate_phonemes(phoneme_sequence: list, phoneme_audio: Dict[str, np.ndarray],
                         crossfade_ms: float = 15) -> np.ndarray:
    """
    Concatenate phoneme audio clips with crossfading.
    """
    if not phoneme_sequence:
        return np.array([], dtype=np.float32)
    
    crossfade_samples = int(crossfade_ms / 1000 * SAMPLE_RATE)
    
    # Get first phoneme (strip stress)
    first_p = strip_stress(phoneme_sequence[0])
    result = phoneme_audio[first_p].copy()
    
    for phoneme in phoneme_sequence[1:]:
        p = strip_stress(phoneme)
        audio = phoneme_audio[p]
        
        if crossfade_samples > 0 and len(result) >= crossfade_samples and len(audio) >= crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            result[-crossfade_samples:] *= fade_out
            audio_copy = audio.copy()
            audio_copy[:crossfade_samples] *= fade_in
            
            result[-crossfade_samples:] += audio_copy[:crossfade_samples]
            result = np.concatenate([result, audio_copy[crossfade_samples:]])
        else:
            result = np.concatenate([result, audio])
    
    return result


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save audio to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    
    with wave.open(str(path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    duration = len(audio) / sample_rate
    print(f"Saved WAV: {path} ({duration:.2f}s)")


if __name__ == "__main__":
    print("Testing phoneme synthesis for all 39 phonemes...")
    
    phoneme_audio = synthesize_all_phonemes()
    
    for phoneme, audio in phoneme_audio.items():
        info = PHONEME_INVENTORY[phoneme]
        print(f"  {phoneme:<4} ({info.ipa}): {len(audio)} samples, "
              f"{len(audio)/SAMPLE_RATE:.3f}s")
    
    print(f"\nTotal: {len(phoneme_audio)} phonemes synthesized")
