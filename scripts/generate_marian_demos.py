#!/usr/bin/env python3
"""
Generate demo WAVs comparing formant synthesis vs MARIAN voice.

Creates side-by-side audio demos in artifacts/eval/marian/ for:
1. Demo words (cat, dog, mushroom, etc.)
2. Short sentences (if available)

Usage:
    python scripts/generate_marian_demos.py
    python scripts/generate_marian_demos.py --voicebank /path/to/marian
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cursed_tts.speak import SwarmSpeaker, MarianSwarmSpeaker
from cursed_tts.singer_fly import SingerSwarm
from cursed_tts.synth import SAMPLE_RATE, save_wav
from cursed_tts.phonemes import strip_stress
from cursed_tts.lexicon import get_phonemes
import numpy as np


DEMO_WORDS = [
    'cat', 'dog', 'bat', 'yes', 'no', 'me', 'you', 'hi', 'bye', 'go',
]

EXTENDED_WORDS = [
    'mushroom', 'connectome', 'chaos', 'hatsune', 'australia',
    'kenyon', 'flywire', 'neural', 'miku', 'voice',
]

SHORT_PHRASES = [
    ('hello world', ['HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D']),
    ('marian speaks', ['M', 'EH', 'R', 'IY', 'AE', 'N', 'S', 'P', 'IY', 'K', 'S']),
]


def generate_comparison(
    word: str,
    swarm_speaker: SwarmSpeaker,
    marian_speaker: MarianSwarmSpeaker,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    """Generate comparison WAVs for a single word."""
    word_clean = ''.join(c for c in word.lower() if c.isalnum())
    
    result = {'word': word}
    
    # Get phonemes
    phonemes = get_phonemes(word, allow_g2p=True)
    phonemes_stripped = [strip_stress(p) for p in phonemes]
    result['phonemes'] = phonemes_stripped
    
    # Formant synthesis (using swarm speaker's singer)
    formant_audio = swarm_speaker.singer_swarm.synthesize_sequence(phonemes_stripped)
    formant_path = output_dir / f"{word_clean}_formant.wav"
    save_wav(formant_audio, str(formant_path), SAMPLE_RATE)
    result['formant_path'] = str(formant_path)
    
    # MARIAN synthesis
    marian_audio = marian_speaker.singer_swarm.synthesize_sequence(phonemes_stripped)
    marian_path = output_dir / f"{word_clean}_marian.wav"
    save_wav(marian_audio, str(marian_path), SAMPLE_RATE)
    result['marian_path'] = str(marian_path)
    
    # Check MARIAN sample coverage
    has_samples = sum(
        1 for p in phonemes_stripped
        if p in marian_speaker.singer_swarm.singers
        and marian_speaker.singer_swarm.singers[p].has_sample
    )
    result['marian_samples'] = has_samples
    result['total_phonemes'] = len(phonemes_stripped)
    
    if verbose:
        print(f"  {word:15} ({' '.join(phonemes_stripped):30}) "
              f"MARIAN: {has_samples}/{len(phonemes_stripped)} samples")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate MARIAN comparison demos')
    parser.add_argument('--voicebank', type=str, default=None,
                       help='Path to MARIAN voicebank')
    parser.add_argument('--swarm-model', type=str, default='model_swarm.npz',
                       help='Picker swarm model path')
    parser.add_argument('--output-dir', type=str, default='artifacts/eval/marian',
                       help='Output directory for demo WAVs')
    parser.add_argument('--extended', action='store_true',
                       help='Include extended word list')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MARIAN vs Formant Demo Generation")
    print("=" * 70)
    
    # Check if swarm model exists
    if not Path(args.swarm_model).exists():
        print(f"\nWarning: Swarm model not found: {args.swarm_model}")
        print("Using direct phoneme synthesis (no picker swarm)")
        
        # Create singer swarms directly
        from cursed_tts.singer_fly import SingerSwarm
        from cursed_tts.voicebanks import MarianSingerSwarm
        
        formant_swarm = SingerSwarm(seed=42)
        marian_swarm = MarianSingerSwarm(
            voicebank_path=Path(args.voicebank) if args.voicebank else None,
            fallback_synth=True,
        )
        
        use_direct = True
    else:
        # Load speakers
        print(f"\nLoading swarm model: {args.swarm_model}")
        swarm_speaker = SwarmSpeaker.from_model_file(args.swarm_model)
        marian_speaker = MarianSwarmSpeaker.from_model_file(
            args.swarm_model,
            voicebank_path=args.voicebank,
        )
        formant_swarm = swarm_speaker.singer_swarm
        marian_swarm = marian_speaker.singer_swarm
        use_direct = False
    
    # Print MARIAN coverage
    print(f"\n{marian_swarm.get_coverage_summary()}")
    
    # Generate demos
    print("\n" + "-" * 70)
    print("Generating Demo Comparisons")
    print("-" * 70)
    
    words = DEMO_WORDS.copy()
    if args.extended:
        words.extend(EXTENDED_WORDS)
    
    results = []
    
    for word in words:
        try:
            phonemes = get_phonemes(word, allow_g2p=True)
            phonemes_stripped = [strip_stress(p) for p in phonemes]
            word_clean = ''.join(c for c in word.lower() if c.isalnum())
            
            # Formant
            formant_audio = formant_swarm.synthesize_sequence(phonemes_stripped)
            formant_path = output_dir / f"{word_clean}_formant.wav"
            save_wav(formant_audio, str(formant_path), SAMPLE_RATE)
            
            # MARIAN
            marian_audio = marian_swarm.synthesize_sequence(phonemes_stripped)
            marian_path = output_dir / f"{word_clean}_marian.wav"
            save_wav(marian_audio, str(marian_path), SAMPLE_RATE)
            
            # Check coverage
            has_samples = sum(
                1 for p in phonemes_stripped
                if p in marian_swarm.singers
                and marian_swarm.singers[p].has_sample
            )
            
            print(f"  {word:15} ({' '.join(phonemes_stripped):30}) "
                  f"MARIAN: {has_samples}/{len(phonemes_stripped)} samples")
            
            results.append({
                'word': word,
                'phonemes': phonemes_stripped,
                'formant_path': str(formant_path),
                'marian_path': str(marian_path),
                'marian_samples': has_samples,
                'total_phonemes': len(phonemes_stripped),
            })
            
        except Exception as e:
            print(f"  {word:15} ERROR: {e}")
    
    # Generate manifest
    import json
    manifest = {
        'voicebank': 'MARIAN ILUSTRADO',
        'author': 'Kanabun',
        'voicebank_path': str(args.voicebank) if args.voicebank else None,
        'marian_available': MarianSwarmSpeaker.is_available(),
        'words': results,
        'total_demos': len(results),
    }
    
    manifest_path = output_dir / 'demo_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Total demos: {len(results)} words × 2 versions = {len(results) * 2} WAVs")
    
    total_samples = sum(r['marian_samples'] for r in results)
    total_phonemes = sum(r['total_phonemes'] for r in results)
    coverage = 100 * total_samples / total_phonemes if total_phonemes > 0 else 0
    
    print(f"MARIAN sample coverage: {total_samples}/{total_phonemes} ({coverage:.1f}%)")
    
    if coverage < 100:
        print("\nNote: Without MARIAN voicebank installed, all demos use formant synthesis.")
        print("Download MARIAN from: https://downloadmarian.carrd.co/")
    
    print(f"\nManifest: {manifest_path}")


if __name__ == '__main__':
    main()
