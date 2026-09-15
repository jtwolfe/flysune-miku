#!/usr/bin/env python3
"""
Generate comparison demos: formant baseline vs trained acoustic flies.

Creates side-by-side WAVs for evaluation:
1. Formant baseline (MORE FLY picker + formant synth)
2. Acoustic flies (MORE FLY picker + trained acoustic synth)

Output to: artifacts/eval/acoustic_flies/
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cursed_tts.speak import MoreFlySpeaker, AcousticSpeaker
from cursed_tts.lexicon import get_demo_words


def main():
    output_dir = Path("artifacts/eval/acoustic_flies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    picker_path = "artifacts/eval/more_fly/swarm__A_B_data.npz"
    acoustic_path = "model_acoustic.npz"
    
    print("=" * 60)
    print("ACOUSTIC FLIES COMPARISON DEMO")
    print("=" * 60)
    
    # Demo words + sentences
    demo_words = ['cat', 'dog', 'hi', 'bye', 'yes', 'me', 'you', 'go', 'no', 'bat']
    test_words = ['mushroom', 'connectome', 'chaos', 'hatsune', 'australia',
                  'neural', 'phoneme', 'cursed', 'flywire', 'kenyon']
    
    all_words = demo_words + test_words
    
    print(f"\n1. Loading formant baseline speaker...")
    formant_speaker = MoreFlySpeaker.from_model_file(picker_path)
    
    print(f"\n2. Loading acoustic flies speaker...")
    acoustic_speaker = AcousticSpeaker.from_model_files(picker_path, acoustic_path)
    
    results = {
        'formant': {},
        'acoustic': {},
        'comparison': {},
    }
    
    # Generate formant baseline
    print(f"\n3. Generating formant baseline WAVs...")
    formant_dir = output_dir / "formant_baseline"
    formant_dir.mkdir(exist_ok=True)
    
    for word in all_words:
        word_clean = ''.join(c for c in word.lower() if c.isalnum())
        output_path = formant_dir / f"{word_clean}.wav"
        
        try:
            audio, audio_ph, ref_ph, known = formant_speaker.speak(
                word, str(output_path), verbose=False
            )
            results['formant'][word] = {
                'phonemes': audio_ph,
                'reference': ref_ph,
                'match': sum(1 for r, a in zip(ref_ph, audio_ph) if r == a) / len(ref_ph),
            }
            print(f"  {word}: {' '.join(audio_ph)} ({results['formant'][word]['match']*100:.0f}%)")
        except Exception as e:
            print(f"  {word}: ERROR - {e}")
            results['formant'][word] = {'error': str(e)}
    
    # Generate acoustic flies
    print(f"\n4. Generating acoustic flies WAVs...")
    acoustic_dir = output_dir / "acoustic_flies"
    acoustic_dir.mkdir(exist_ok=True)
    
    for word in all_words:
        word_clean = ''.join(c for c in word.lower() if c.isalnum())
        output_path = acoustic_dir / f"{word_clean}.wav"
        
        try:
            audio, audio_ph, ref_ph, known = acoustic_speaker.speak(
                word, str(output_path), verbose=False
            )
            results['acoustic'][word] = {
                'phonemes': audio_ph,
                'reference': ref_ph,
                'match': sum(1 for r, a in zip(ref_ph, audio_ph) if r == a) / len(ref_ph),
            }
            print(f"  {word}: {' '.join(audio_ph)} ({results['acoustic'][word]['match']*100:.0f}%)")
        except Exception as e:
            print(f"  {word}: ERROR - {e}")
            results['acoustic'][word] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Word':<15} {'Formant':<25} {'Acoustic':<25} {'Match'}")
    print("-" * 75)
    
    for word in all_words:
        formant_ph = results['formant'].get(word, {}).get('phonemes', ['ERR'])
        acoustic_ph = results['acoustic'].get(word, {}).get('phonemes', ['ERR'])
        ref_ph = results['formant'].get(word, {}).get('reference', ['?'])
        
        formant_str = ' '.join(formant_ph)[:22]
        acoustic_str = ' '.join(acoustic_ph)[:22]
        
        # Compare
        same = formant_ph == acoustic_ph
        marker = "✓ same" if same else "✗ diff"
        
        results['comparison'][word] = {
            'formant': formant_ph,
            'acoustic': acoustic_ph,
            'reference': ref_ph,
            'same': same,
        }
        
        print(f"{word:<15} {formant_str:<25} {acoustic_str:<25} {marker}")
    
    # Save results
    results_path = output_dir / "demo_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print(f"\nFormant WAVs: {formant_dir}")
    print(f"Acoustic WAVs: {acoustic_dir}")
    
    # Count matches
    n_same = sum(1 for r in results['comparison'].values() if r.get('same'))
    print(f"\nPicker agreement: {n_same}/{len(all_words)} words have same phonemes")


if __name__ == "__main__":
    main()
