"""
CLI entry point for Cursed TTS: G2P mushroom body classifier.

The correct architecture:
- MB = grapheme-to-phoneme CLASSIFIER (like hiragana OCR demo)
- Synth = clean concatenative renderer (formant crumbs)
- Default speak uses MB predictions; --lexicon uses dictionary phonemes

Swarm mode (experimental):
- Train specialists: python -m cursed_tts train-swarm [--phones SUBSET]
- Speak with swarm: python -m cursed_tts speak <word> --swarm

Usage:
    python -m cursed_tts train [--epochs N] [--words N]
    python -m cursed_tts train-swarm [--epochs N] [--phones demo]
    python -m cursed_tts speak <word> [--lexicon] [--swarm]
    python -m cursed_tts speak-all [--lexicon] [--swarm]
    python -m cursed_tts eval [--swarm]
    python -m cursed_tts info

Legacy (experimental, not recommended):
    python -m cursed_tts train-stage2 / train-stage2b
    python -m cursed_tts speak <word> --stage2 / --stage2b
"""

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    """Train G2P mushroom body classifier on CMUdict."""
    from .train import train_and_save
    
    model = train_and_save(
        output_path=args.output,
        n_epochs=args.epochs,
        max_words=args.words,
        context_size=args.context,
        seed=args.seed,
        verbose=True,
    )
    print(f"\nModel saved to: {args.output}")


def cmd_speak(args):
    """Speak a word using G2P-MB + clean synth (default) or lexicon."""
    word_clean = ''.join(c for c in args.word.lower() if c.isalnum())
    
    # Check for legacy stage2/2b flags
    if getattr(args, 'stage2b', False):
        # Legacy Stage 2b
        from .stage2b import speak_stage2b
        
        if args.output:
            output_path = args.output
        else:
            Path("artifacts/stage2b").mkdir(parents=True, exist_ok=True)
            output_path = f"artifacts/stage2b/{word_clean}.wav"
        
        print("\n[LEGACY] Using Stage 2b (formant track regression) - not recommended")
        audio, phonemes = speak_stage2b(
            args.word,
            mb_path=args.model,
            stage2b_path=args.stage2b_model,
            output_path=output_path,
            verbose=True,
        )
        print(f"\nSaved to: {output_path}")
        return
    
    if getattr(args, 'stage2', False):
        # Legacy Stage 2
        from .stage2 import speak_stage2
        
        if args.output:
            output_path = args.output
        else:
            Path("artifacts/stage2").mkdir(parents=True, exist_ok=True)
            output_path = f"artifacts/stage2/{word_clean}.wav"
        
        print("\n[LEGACY] Using Stage 2 (mel + Griffin-Lim) - not recommended")
        audio, phonemes = speak_stage2(
            args.word,
            mb_path=args.model,
            stage2_path=args.stage2_model,
            output_path=output_path,
            verbose=True,
        )
        print(f"\nSaved to: {output_path}")
        return
    
    # Check for swarm mode
    if getattr(args, 'swarm', False):
        from .speak import SwarmSpeaker
        
        if args.output:
            output_path = args.output
        else:
            Path("artifacts/swarm").mkdir(parents=True, exist_ok=True)
            output_path = f"artifacts/swarm/{word_clean}.wav"
        
        swarm_model = getattr(args, 'swarm_model', 'model_swarm.npz')
        speaker = SwarmSpeaker.from_model_file(swarm_model)
        audio, audio_ph, ref_ph, known = speaker.speak(
            args.word, output_path, verbose=True
        )
        print(f"\nSaved to: {output_path}")
        return
    
    # Default: G2P-MB + clean synth
    from .speak import Speaker
    
    if args.output:
        output_path = args.output
    else:
        Path("artifacts/g2p").mkdir(parents=True, exist_ok=True)
        output_path = f"artifacts/g2p/{word_clean}.wav"
    
    use_lexicon = getattr(args, 'lexicon', False)
    
    speaker = Speaker.from_model_file(args.model)
    audio, audio_ph, ref_ph, known = speaker.speak(
        args.word, output_path, use_lexicon=use_lexicon, verbose=True
    )
    print(f"\nSaved to: {output_path}")


def cmd_speak_all(args):
    """Speak demo words and OOV examples."""
    
    # Check for legacy flags
    if getattr(args, 'stage2b', False):
        from .stage2b import speak_stage2b
        
        output_dir = Path(args.output_dir) / "stage2b"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        words = ['cat', 'bat', 'dog', 'go', 'no', 'hi', 'bye', 'yes', 'me', 'you',
                 'mushroom', 'connectome', 'chaos', 'hatsune', 'australia',
                 'neural', 'phoneme', 'cursed', 'flywire', 'kenyon']
        
        print("\n[LEGACY] Using Stage 2b - not recommended")
        for word in words:
            output_path = output_dir / f"{word}.wav"
            try:
                speak_stage2b(word, args.model, args.stage2b_model, str(output_path), verbose=True)
            except Exception as e:
                print(f"Error on '{word}': {e}")
        
        print(f"\nStage 2b WAVs saved to: {output_dir}")
        return
    
    if getattr(args, 'stage2', False):
        from .stage2 import speak_stage2
        
        output_dir = Path(args.output_dir) / "stage2"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        words = ['cat', 'bat', 'dog', 'go', 'no', 'hi', 'bye', 'yes', 'me', 'you',
                 'mushroom', 'connectome', 'chaos', 'hatsune', 'australia',
                 'neural', 'phoneme', 'cursed', 'flywire', 'kenyon']
        
        print("\n[LEGACY] Using Stage 2 - not recommended")
        for word in words:
            output_path = output_dir / f"{word}.wav"
            try:
                speak_stage2(word, args.model, args.stage2_model, str(output_path), verbose=True)
            except Exception as e:
                print(f"Error on '{word}': {e}")
        
        print(f"\nStage 2 WAVs saved to: {output_dir}")
        return
    
    # Check for swarm mode
    if getattr(args, 'swarm', False):
        from .speak import SwarmSpeaker
        
        output_dir = Path(args.output_dir) / "swarm"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        swarm_model = getattr(args, 'swarm_model', 'model_swarm.npz')
        speaker = SwarmSpeaker.from_model_file(swarm_model)
        results = speaker.speak_all(str(output_dir), verbose=True)
        return
    
    # Default: G2P-MB + clean synth
    from .speak import Speaker
    
    use_lexicon = getattr(args, 'lexicon', False)
    output_dir = args.output_dir + "/g2p" if not use_lexicon else args.output_dir + "/lexicon"
    
    speaker = Speaker.from_model_file(args.model)
    results = speaker.speak_all(output_dir, use_lexicon=use_lexicon, verbose=True)


def cmd_eval(args):
    """Evaluate G2P-MB accuracy on demo and test words."""
    # Check for swarm mode
    if getattr(args, 'swarm', False):
        from .specialist_fly import FlySwarm
        from .train_swarm import evaluate_swarm, compare_to_baseline
        
        swarm_model = getattr(args, 'swarm_model', 'model_swarm.npz')
        swarm = FlySwarm.load(swarm_model)
        evaluate_swarm(swarm, verbose=True)
        compare_to_baseline(swarm, args.model, verbose=True)
        return
    
    from .mushroom_body import MushroomBody
    from .train import evaluate
    
    model = MushroomBody.load(args.model)
    evaluate(model, verbose=True)


def cmd_info(args):
    """Print phoneme inventory and system info."""
    from .phonemes import print_inventory, NUM_PHONEMES
    from .lexicon import print_lexicon_stats, get_demo_words
    
    print_inventory()
    
    print("\nDemo words (should achieve high accuracy):")
    for word in get_demo_words():
        print(f"  {word}")
    
    try:
        print_lexicon_stats()
    except Exception as e:
        print(f"\n(CMUdict not loaded: {e})")
        print("Install with: pip install cmudict g2p_en")
    
    print("\n" + "="*60)
    print("ARCHITECTURE (correct)")
    print("="*60)
    print("  MB = G2P classifier (letter context → phoneme)")
    print("  Synth = clean concatenative renderer (formant crumbs)")
    print("  Default: MB predictions + formant synth")
    print("  --lexicon: Dictionary phonemes + formant synth (baseline)")
    print()
    print("LEGACY (experimental, not recommended):")
    print("  --stage2: MB trajectory → mel → Griffin-Lim (scratchy)")
    print("  --stage2b: MB trajectory → formant tracks → synth (better but wrong)")
    print()
    print("Usage:")
    print("  python -m cursed_tts train         # Train G2P-MB on CMUdict")
    print("  python -m cursed_tts speak cat     # MB predictions + clean synth")
    print("  python -m cursed_tts speak cat --lexicon  # Dictionary baseline")
    print("  python -m cursed_tts speak-all     # All demo + OOV words")
    print("  python -m cursed_tts eval          # Evaluate accuracy")


# Swarm training command
def cmd_train_swarm(args):
    """Train one-phoneme-per-fly specialist ensemble."""
    from .train_swarm import train_and_save_swarm, train_swarm_demo
    from .specialist_fly import get_demo_phonemes
    from .phonemes import PHONEME_LIST
    
    # Determine phoneme subset
    phones_arg = getattr(args, 'phones', None)
    if phones_arg == 'demo':
        phonemes = get_demo_phonemes()
        print(f"Training demo subset: {len(phonemes)} phonemes")
    elif phones_arg == 'all' or phones_arg is None:
        phonemes = None  # All 39
        print("Training full swarm: 39 phonemes")
    else:
        # Parse comma-separated phoneme list
        phonemes = [p.strip().upper() for p in phones_arg.split(',')]
        print(f"Training custom subset: {len(phonemes)} phonemes")
    
    swarm = train_and_save_swarm(
        output_path=args.output,
        phonemes=phonemes,
        n_epochs=args.epochs,
        max_words=args.words,
        seed=args.seed,
        verbose=True,
    )
    print(f"\nSwarm saved to: {args.output}")


# Legacy training commands
def cmd_train_stage2(args):
    """[LEGACY] Train Stage 2 (mel spectrogram regression)."""
    print("\n[LEGACY] Stage 2 is not recommended. Use default 'train' instead.")
    from .stage2 import train_stage2
    
    model = train_stage2(
        mb_path=args.model,
        output_path=args.output,
        max_words=args.words,
        seed=args.seed,
        verbose=True,
    )


def cmd_train_stage2b(args):
    """[LEGACY] Train Stage 2b (formant track regression)."""
    print("\n[LEGACY] Stage 2b is not recommended. Use default 'train' instead.")
    from .stage2b import train_stage2b
    
    model = train_stage2b(
        mb_path=args.model,
        output_path=args.output,
        max_words=args.words,
        seed=args.seed,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cursed TTS: G2P mushroom body classifier + clean synth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The mushroom body is a CLASSIFIER (like hiragana OCR), not an acoustic generator.
Audio comes from clean formant synthesis, not neural trajectory regression.

Examples:
  python -m cursed_tts train                 # Train G2P-MB
  python -m cursed_tts speak mushroom        # MB predictions (cursed G2P)
  python -m cursed_tts speak mushroom --lexicon  # Dictionary baseline
  python -m cursed_tts speak-all             # Generate all demo WAVs
  python -m cursed_tts eval                  # Evaluate accuracy
  python -m cursed_tts info                  # Show phoneme inventory
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command (G2P-MB)
    train_parser = subparsers.add_parser('train', help='Train G2P mushroom body')
    train_parser.add_argument('--epochs', type=int, default=15,
                              help='Training epochs (default: 15)')
    train_parser.add_argument('--words', type=int, default=10000,
                              help='Max training words from CMUdict (default: 10000)')
    train_parser.add_argument('--context', type=int, default=3,
                              help='Letter context size (default: 3)')
    train_parser.add_argument('--output', '-o', type=str, default='model.npz',
                              help='Output model path (default: model.npz)')
    train_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed (default: 42)')
    
    # Train-swarm command
    train_swarm_parser = subparsers.add_parser('train-swarm',
                                                help='Train one-phoneme-per-fly ensemble')
    train_swarm_parser.add_argument('--epochs', type=int, default=10,
                                     help='Training epochs (default: 10)')
    train_swarm_parser.add_argument('--words', type=int, default=10000,
                                     help='Max training words (default: 10000)')
    train_swarm_parser.add_argument('--phones', type=str, default=None,
                                     help='Phoneme subset: "demo", "all", or comma-separated list')
    train_swarm_parser.add_argument('--output', '-o', type=str, default='model_swarm.npz',
                                     help='Output model path (default: model_swarm.npz)')
    train_swarm_parser.add_argument('--seed', type=int, default=42,
                                     help='Random seed (default: 42)')
    
    # Speak command
    speak_parser = subparsers.add_parser('speak', help='Speak a word')
    speak_parser.add_argument('word', type=str, help='Word to speak')
    speak_parser.add_argument('--output', '-o', type=str, default=None,
                              help='Output WAV path')
    speak_parser.add_argument('--model', '-m', type=str, default='model.npz',
                              help='Model path (default: model.npz)')
    speak_parser.add_argument('--lexicon', action='store_true',
                              help='Use dictionary phonemes (baseline, not cursed)')
    # Swarm flag
    speak_parser.add_argument('--swarm', action='store_true',
                              help='Use one-phoneme-per-fly ensemble')
    speak_parser.add_argument('--swarm-model', type=str, default='model_swarm.npz',
                              help='Swarm model path (default: model_swarm.npz)')
    # Legacy flags
    speak_parser.add_argument('--stage2', action='store_true',
                              help='[LEGACY] Use Stage 2 mel + Griffin-Lim')
    speak_parser.add_argument('--stage2-model', type=str, default='model_stage2.npz',
                              help='[LEGACY] Stage 2 model path')
    speak_parser.add_argument('--stage2b', action='store_true',
                              help='[LEGACY] Use Stage 2b formant tracks')
    speak_parser.add_argument('--stage2b-model', type=str, default='model_stage2b.npz',
                              help='[LEGACY] Stage 2b model path')
    
    # Speak-all command
    speak_all_parser = subparsers.add_parser('speak-all', help='Speak demo + OOV words')
    speak_all_parser.add_argument('--output-dir', '-o', type=str, default='artifacts',
                                   help='Output directory (default: artifacts)')
    speak_all_parser.add_argument('--model', '-m', type=str, default='model.npz',
                                   help='Model path (default: model.npz)')
    speak_all_parser.add_argument('--lexicon', action='store_true',
                                   help='Use dictionary phonemes (baseline)')
    # Swarm flag
    speak_all_parser.add_argument('--swarm', action='store_true',
                                   help='Use one-phoneme-per-fly ensemble')
    speak_all_parser.add_argument('--swarm-model', type=str, default='model_swarm.npz',
                                   help='Swarm model path')
    # Legacy flags
    speak_all_parser.add_argument('--stage2', action='store_true',
                                   help='[LEGACY] Use Stage 2')
    speak_all_parser.add_argument('--stage2-model', type=str, default='model_stage2.npz')
    speak_all_parser.add_argument('--stage2b', action='store_true',
                                   help='[LEGACY] Use Stage 2b')
    speak_all_parser.add_argument('--stage2b-model', type=str, default='model_stage2b.npz')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate G2P accuracy')
    eval_parser.add_argument('--model', '-m', type=str, default='model.npz',
                             help='Model path (default: model.npz)')
    eval_parser.add_argument('--swarm', action='store_true',
                             help='Evaluate swarm model instead of single MB')
    eval_parser.add_argument('--swarm-model', type=str, default='model_swarm.npz',
                             help='Swarm model path (default: model_swarm.npz)')
    
    # Info command
    subparsers.add_parser('info', help='Show phoneme/system info')
    
    # Legacy training commands
    train_s2_parser = subparsers.add_parser('train-stage2', 
                                             help='[LEGACY] Train Stage 2')
    train_s2_parser.add_argument('--words', type=int, default=2000)
    train_s2_parser.add_argument('--model', '-m', type=str, default='model.npz')
    train_s2_parser.add_argument('--output', '-o', type=str, default='model_stage2.npz')
    train_s2_parser.add_argument('--seed', type=int, default=42)
    
    train_s2b_parser = subparsers.add_parser('train-stage2b',
                                              help='[LEGACY] Train Stage 2b')
    train_s2b_parser.add_argument('--words', type=int, default=2000)
    train_s2b_parser.add_argument('--model', '-m', type=str, default='model.npz')
    train_s2b_parser.add_argument('--output', '-o', type=str, default='model_stage2b.npz')
    train_s2b_parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        'train': cmd_train,
        'train-swarm': cmd_train_swarm,
        'speak': cmd_speak,
        'speak-all': cmd_speak_all,
        'eval': cmd_eval,
        'info': cmd_info,
        'train-stage2': cmd_train_stage2,
        'train-stage2b': cmd_train_stage2b,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
