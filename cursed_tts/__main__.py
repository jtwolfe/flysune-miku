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
    
    # Two-swarm: picker G2P → speaker flies (NO KC in the speaker path)
    if getattr(args, 'swarm', False) or getattr(args, 'speakers', False):
        from .two_swarm import TwoSwarmSpeaker
        
        if args.output:
            output_path = args.output
        else:
            Path("artifacts/eval/two_swarm").mkdir(parents=True, exist_ok=True)
            output_path = f"artifacts/eval/two_swarm/{word_clean}.wav"
        
        picker_model = getattr(args, 'swarm_model', 'model_swarm.npz')
        if Path('model_more_fly_best.npz').exists() and picker_model == 'model_swarm.npz':
            picker_model = 'model_more_fly_best.npz'
        picker_type = 'more_fly' if 'more_fly' in picker_model else 'swarm'
        speaker_path = getattr(args, 'speaker_model', 'model_speaker.npz')
        if not Path(speaker_path).exists():
            speaker_path = None
        use_formant = getattr(args, 'formant_baseline', False)
        
        speaker = TwoSwarmSpeaker.from_model_files(
            picker_path=picker_model,
            speaker_path=speaker_path,
            use_formant_baseline=use_formant,
            picker_type=picker_type,
        )
        speaker.speak(args.word, output_path, verbose=True)
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
        vote_strategy = getattr(args, 'vote', None)
        speaker = SwarmSpeaker.from_model_file(swarm_model, vote_strategy=vote_strategy)
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
        from .train_swarm import evaluate_swarm_words
        from .lexicon import get_lexicon, get_demo_words
        import numpy as np
        
        swarm_model = getattr(args, 'swarm_model', 'model_swarm.npz')
        swarm = FlySwarm.load(swarm_model)
        
        lexicon = get_lexicon()
        demo_words = get_demo_words()
        demo_in_lex = [w for w in demo_words if w in lexicon]
        
        # Build held-out test sample
        all_words = list(lexicon.keys())
        rng = np.random.default_rng(123)
        test_sample = [
            w for w in rng.choice(all_words, size=500, replace=False)
            if w not in demo_words and w.isalpha() and 2 <= len(w) <= 12
        ][:200]
        
        # Check for beam search / LM
        beam_width = getattr(args, 'beam', 0)
        lm_weight = getattr(args, 'lm_weight', 0.3)
        vote_strategy = getattr(args, 'vote', None)
        
        lm = None
        if beam_width > 0:
            from .phoneme_lm import build_phoneme_lm
            print(f"\nBuilding phoneme LM for beam search (beam={beam_width}, lm_weight={lm_weight})...")
            lm = build_phoneme_lm(n=3, max_words=20000, verbose=True)
        
        print("\n" + "="*60)
        print("SWARM EVALUATION")
        if lm:
            print(f"  Mode: beam search (beam={beam_width}, lm_weight={lm_weight})")
        else:
            print(f"  Mode: greedy ({vote_strategy or swarm.config.vote_strategy} voting)")
        print("="*60)
        
        # Demo evaluation
        print("\n--- Demo Words ---\n")
        print(f"{'Word':15} {'Expected':25} {'Predicted':25}")
        print("-" * 70)
        
        demo_phoneme, demo_word, demo_details = evaluate_swarm_words(
            swarm, demo_in_lex, lexicon, verbose=True,
            lm=lm, beam_width=beam_width, lm_weight=lm_weight,
            vote_strategy=vote_strategy,
        )
        
        print(f"\nDemo: phoneme={100*demo_phoneme:.1f}%, word={100*demo_word:.1f}%")
        
        # Held-out evaluation
        test_phoneme, test_word, _ = evaluate_swarm_words(
            swarm, test_sample, lexicon, verbose=False,
            lm=lm, beam_width=beam_width, lm_weight=lm_weight,
            vote_strategy=vote_strategy,
        )
        
        print(f"Held-out ({len(test_sample)} words): phoneme={100*test_phoneme:.1f}%, word={100*test_word:.1f}%")
        print("="*60)
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
    from .specialist_fly import get_demo_phonemes, VALID_VOTE_STRATEGIES
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
    
    # Get voting parameters
    vote_strategy = getattr(args, 'vote', 'softmax')
    if vote_strategy not in VALID_VOTE_STRATEGIES:
        print(f"Warning: Unknown vote strategy '{vote_strategy}', using 'softmax'")
        vote_strategy = 'softmax'
    
    vote_temp = getattr(args, 'vote_temp', 0.5)
    vote_margin = getattr(args, 'vote_margin', 0.1)
    patience = getattr(args, 'patience', 0)
    save_best = not getattr(args, 'no_save_best', False)
    
    # Confusion mining parameters
    confusion_mine = getattr(args, 'confusion_mine', False)
    confusion_mine_epoch = getattr(args, 'confusion_mine_epoch', 2)
    hard_negative_weight = getattr(args, 'hard_negative_weight', 2.0)
    oversample_factor = getattr(args, 'oversample_factor', 1.5)
    confusion_report_path = getattr(args, 'confusion_report', None)
    use_known_hard_pairs = getattr(args, 'use_known_hard_pairs', False)
    
    swarm = train_and_save_swarm(
        output_path=args.output,
        phonemes=phonemes,
        n_epochs=args.epochs,
        max_words=args.words,
        seed=args.seed,
        vote_strategy=vote_strategy,
        vote_temperature=vote_temp,
        vote_margin=vote_margin,
        early_stop_patience=patience,
        save_best=save_best,
        confusion_mine=confusion_mine,
        confusion_mine_epoch=confusion_mine_epoch,
        hard_negative_weight=hard_negative_weight,
        oversample_factor=oversample_factor,
        confusion_report_path=confusion_report_path,
        use_known_hard_pairs=use_known_hard_pairs,
        verbose=True,
    )
    print(f"\nSwarm saved to: {args.output}")


# MORE FLY training commands
def cmd_train_more_fly(args):
    """Train MORE FLY enhanced swarm."""
    from .train_more_fly import (
        train_more_fly, ABLATION_CONFIGS,
        get_baseline_config, get_stage_a_config, get_stage_ab_config,
        get_stage_abc_config, get_full_config,
    )
    
    config_name = getattr(args, 'config', '+A+B+C+D_full')
    wiring_mode = getattr(args, 'wiring', 'flywire')
    
    # Get config based on selection
    if config_name == 'baseline':
        config = get_baseline_config(args.seed)
    elif config_name == '+A_cues':
        config = get_stage_a_config(args.seed)
    elif config_name == '+A+B_data':
        config = get_stage_ab_config(args.seed)
    elif config_name == '+A+B+C_wiring':
        config = get_stage_abc_config(args.seed, wiring_mode)
    else:  # Full
        config = get_full_config(args.seed, wiring_mode)
    
    # Determine if hard curriculum should be used
    use_hard = config_name in ['+A+B_data', '+A+B+C_wiring', '+A+B+C+D_full']
    
    swarm, history = train_more_fly(
        config=config,
        max_words=args.words,
        n_epochs=args.epochs,
        seed=args.seed,
        use_hard_curriculum=use_hard,
        early_stop_patience=getattr(args, 'patience', 3),
        save_best=True,
        best_path=args.output.replace('.npz', '_best.npz'),
        verbose=True,
    )
    
    swarm.save(args.output)
    print(f"\nMORE FLY model saved to: {args.output}")
    print(f"Best checkpoint: {args.output.replace('.npz', '_best.npz')}")


def cmd_eval_more_fly(args):
    """Run MORE FLY ablation evaluation."""
    from .train_more_fly import run_all_ablations, format_ablation_table
    
    results = run_all_ablations(
        max_words=args.words,
        n_epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=True,
    )
    
    # Print markdown table
    print("\n" + "="*60)
    print("MARKDOWN TABLE (for PR body)")
    print("="*60)
    print(format_ablation_table(results))


# =============================================================================
# TWO-SWARM COMMANDS (Clean Architecture)
# =============================================================================

def cmd_train_speaker(args):
    """Train speaker flies (synthesis, no KC dependency)."""
    from .train_speaker import (
        train_speaker_swarm, SpeakerTrainingConfig
    )
    from .phonemes import PHONEME_LIST
    
    # Determine phonemes
    phones_arg = getattr(args, 'phones', None)
    if phones_arg == 'demo':
        from .specialist_fly import get_demo_phonemes
        phonemes = get_demo_phonemes()
        print(f"Training demo subset: {len(phonemes)} phonemes")
    elif phones_arg == 'all' or phones_arg is None:
        phonemes = None  # All 39
        print("Training full speaker swarm: 39 phonemes")
    else:
        # Parse comma-separated phoneme list
        phonemes = [p.strip().upper() for p in phones_arg.split(',')]
        print(f"Training custom subset: {len(phonemes)} phonemes")
    
    # Create config
    marian_dir = getattr(args, 'marian_dir', 'data/marian_crumbs')
    voicebank = getattr(args, 'voicebank', None)
    if voicebank:
        from .train_speaker import extract_marian_crumbs
        extract_marian_crumbs(
            voicebank,
            output_dir=marian_dir,
            duration_cap_ms=getattr(args, 'duration_cap', 250.0),
            verbose=True,
        )

    config = SpeakerTrainingConfig(
        mode=getattr(args, 'mode', 'formant-bootstrap'),
        n_iterations=getattr(args, 'iterations', 50),
        learning_rate=getattr(args, 'lr', 0.01),
        seed=args.seed,
        marian_dir=marian_dir,
        duration_cap_ms=getattr(args, 'duration_cap', 250.0),
    )
    
    swarm = train_speaker_swarm(
        output_path=args.output,
        config=config,
        phonemes=phonemes,
        verbose=True,
    )
    
    print(f"\nSpeaker swarm saved to: {args.output}")
    print("\nTwo-swarm architecture:")
    print("  - Recognition flies pick (G2P via KC/MB)")
    print("  - Speaking flies speak (NO KC dependency)")


def cmd_speak_two_swarm(args):
    """Speak using two-swarm pipeline (picker → speakers)."""
    from .two_swarm import TwoSwarmSpeaker
    from pathlib import Path
    
    word_clean = ''.join(c for c in args.word.lower() if c.isalnum())
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        Path("artifacts/eval/two_swarm").mkdir(parents=True, exist_ok=True)
        output_path = f"artifacts/eval/two_swarm/{word_clean}.wav"
    
    # Determine picker type
    picker_type = getattr(args, 'picker_type', 'swarm')
    
    # Load speaker
    use_formant = getattr(args, 'formant_baseline', False)
    speaker_path = getattr(args, 'speaker_model', 'model_speaker.npz')
    if not Path(speaker_path).exists():
        speaker_path = None  # Will use formant baseline
    
    speaker = TwoSwarmSpeaker.from_model_files(
        picker_path=args.picker_model,
        speaker_path=speaker_path,
        use_formant_baseline=use_formant,
        picker_type=picker_type,
    )
    
    audio, picker_ph, ref_ph, known = speaker.speak(
        args.word, output_path, verbose=True
    )
    
    print(f"\nSaved to: {output_path}")


def cmd_speak_two_swarm_sentence(args):
    """Speak a sentence using two-swarm pipeline."""
    from .two_swarm import TwoSwarmSpeaker
    from pathlib import Path
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        Path("artifacts/eval/two_swarm").mkdir(parents=True, exist_ok=True)
        sentence_clean = '_'.join(args.text.lower().split()[:3])
        output_path = f"artifacts/eval/two_swarm/sentence_{sentence_clean}.wav"
    
    # Load speaker
    use_formant = getattr(args, 'formant_baseline', False)
    speaker_path = getattr(args, 'speaker_model', 'model_speaker.npz')
    if not Path(speaker_path).exists():
        speaker_path = None
    
    picker_type = getattr(args, 'picker_type', 'swarm')
    
    speaker = TwoSwarmSpeaker.from_model_files(
        picker_path=args.picker_model,
        speaker_path=speaker_path,
        use_formant_baseline=use_formant,
        picker_type=picker_type,
    )
    
    audio, phonemes = speaker.speak_sequence(
        args.text, output_path, verbose=True
    )


def cmd_speak_two_swarm_demo(args):
    """Generate demo WAVs using two-swarm pipeline."""
    from .two_swarm import TwoSwarmSpeaker, compare_two_swarm_vs_formant
    from pathlib import Path
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load speaker
    use_formant = getattr(args, 'formant_baseline', False)
    speaker_path = getattr(args, 'speaker_model', 'model_speaker.npz')
    if not Path(speaker_path).exists():
        speaker_path = None
    
    picker_type = getattr(args, 'picker_type', 'swarm')
    
    speaker = TwoSwarmSpeaker.from_model_files(
        picker_path=args.picker_model,
        speaker_path=speaker_path,
        use_formant_baseline=use_formant,
        picker_type=picker_type,
    )
    
    # Generate demos (words + sentences + paragraph)
    results = speaker.speak_eval_suite(str(output_dir), verbose=True)
    
    # Generate comparison if speaker model available
    if speaker_path and Path(speaker_path).exists() and not use_formant:
        print("\n" + "="*60)
        print("Generating speaker vs formant comparison...")
        demo_words = ['cat', 'dog', 'mushroom', 'hello']
        comparison = compare_two_swarm_vs_formant(
            demo_words,
            args.picker_model,
            speaker_path,
            str(output_dir / "comparison"),
            verbose=True,
            picker_type=picker_type,
        )
    
    print(f"\nTwo-swarm demos saved to: {output_dir}")


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
    # Voting/arbitration options
    train_swarm_parser.add_argument('--vote', type=str, default='softmax',
                                     choices=['argmax', 'softmax', 'margin', 'calibrated'],
                                     help='Voting strategy: argmax, softmax, margin, calibrated (default: softmax)')
    train_swarm_parser.add_argument('--vote-temp', type=float, default=0.5,
                                     help='Softmax temperature (default: 0.5, lower=sharper)')
    train_swarm_parser.add_argument('--vote-margin', type=float, default=0.1,
                                     help='Margin threshold for margin voting (default: 0.1)')
    # Early stopping / checkpointing
    train_swarm_parser.add_argument('--patience', type=int, default=0,
                                     help='Early stop if demo_phoneme stagnates for N epochs (0=disabled)')
    train_swarm_parser.add_argument('--no-save-best', action='store_true',
                                     help='Disable saving best checkpoint by demo_phoneme')
    # Confusion mining / hard negatives
    train_swarm_parser.add_argument('--confusion-mine', action='store_true',
                                     help='Enable confusion mining (mine confusions and oversample hard pairs)')
    train_swarm_parser.add_argument('--confusion-mine-epoch', type=int, default=2,
                                     help='Epoch after which to mine confusions (default: 2)')
    train_swarm_parser.add_argument('--hard-negative-weight', type=float, default=2.0,
                                     help='Weight multiplier for hard negative pairs (default: 2.0)')
    train_swarm_parser.add_argument('--oversample-factor', type=float, default=1.5,
                                     help='Oversample factor for hard pairs (default: 1.5 = 50%% more)')
    train_swarm_parser.add_argument('--confusion-report', type=str, default=None,
                                     help='Path to save confusion report (default: <output>_confusion.txt)')
    train_swarm_parser.add_argument('--use-known-hard-pairs', action='store_true',
                                     help='Use pre-defined known hard pairs (IY/EH, AE/AA, etc.) without mining')
    
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
                              help='Picker swarm then speaker flies (two-swarm, no KC in speakers)')
    speak_parser.add_argument('--speakers', action='store_true',
                              help='Same as --swarm: picker → speaker flies')
    speak_parser.add_argument('--swarm-model', type=str, default='model_swarm.npz',
                              help='Picker swarm model path (default: model_swarm.npz)')
    speak_parser.add_argument('--speaker-model', type=str, default='model_speaker.npz',
                              help='Speaker swarm model (default: model_speaker.npz)')
    speak_parser.add_argument('--formant-baseline', action='store_true',
                              help='A/B: formant crumbs instead of trained speakers')
    speak_parser.add_argument('--vote', type=str, default=None,
                              choices=['argmax', 'softmax', 'margin', 'calibrated'],
                              help='Override swarm voting strategy')
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
    speak_all_parser.add_argument('--vote', type=str, default=None,
                                   choices=['argmax', 'softmax', 'margin', 'calibrated'],
                                   help='Override swarm voting strategy')
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
    eval_parser.add_argument('--vote', type=str, default=None,
                             choices=['argmax', 'softmax', 'margin', 'calibrated'],
                             help='Voting strategy for swarm evaluation')
    # Beam search / LM options
    eval_parser.add_argument('--beam', type=int, default=0,
                             help='Beam width for LM-rescored search (0=greedy, no LM)')
    eval_parser.add_argument('--lm-weight', type=float, default=0.3,
                             help='Weight for phoneme LM score (default: 0.3)')
    
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
    
    # MORE FLY training command
    train_mf_parser = subparsers.add_parser('train-more-fly',
                                             help='Train MORE FLY enhanced swarm')
    train_mf_parser.add_argument('--epochs', type=int, default=8,
                                  help='Training epochs (default: 8)')
    train_mf_parser.add_argument('--words', type=int, default=10000,
                                  help='Max training words (0 = all CMUdict)')
    train_mf_parser.add_argument('--output', '-o', type=str, default='model_more_fly.npz',
                                  help='Output model path')
    train_mf_parser.add_argument('--seed', type=int, default=42,
                                  help='Random seed (default: 42)')
    # Stage selection
    # Default: +A+B_data (random wiring + cues + curriculum) — best balance of accuracy and simplicity
    # hemibrain wiring is experimental: requires init_seed tuning and ≤ random+data on held-out in ablations
    train_mf_parser.add_argument('--config', type=str, default='+A+B_data',
                                  choices=['baseline', '+A_cues', '+A+B_data', '+A+B+C_wiring', '+A+B+C+D_full'],
                                  help='Ablation config (default: +A+B_data = cues + curriculum)')
    train_mf_parser.add_argument('--wiring', type=str, default='random',
                                  choices=['random', 'flywire', 'hemibrain'],
                                  help='Wiring mode (default: random; flywire/hemibrain are experimental)')
    train_mf_parser.add_argument('--patience', type=int, default=3,
                                  help='Early stop patience (default: 3)')
    
    # Ablation evaluation command
    eval_mf_parser = subparsers.add_parser('eval-more-fly',
                                            help='Run MORE FLY ablation evaluation')
    eval_mf_parser.add_argument('--words', type=int, default=5000,
                                 help='Max training words per config')
    eval_mf_parser.add_argument('--epochs', type=int, default=6,
                                 help='Epochs per config')
    eval_mf_parser.add_argument('--seed', type=int, default=42)
    eval_mf_parser.add_argument('--output-dir', type=str, default='artifacts/eval/more_fly',
                                 help='Output directory for results')
    
    # ==========================================================================
    # TWO-SWARM COMMANDS (Clean Architecture)
    # ==========================================================================
    
    # Train speaker flies
    train_speaker_parser = subparsers.add_parser('train-speaker-flies',
                                                   help='Train speaker flies (synthesis, no KC)')
    train_speaker_parser.add_argument('--output', '-o', type=str, default='model_speaker.npz',
                                       help='Output model path (default: model_speaker.npz)')
    train_speaker_parser.add_argument('--phones', type=str, default=None,
                                       help='Phoneme subset: "demo", "all", or comma-separated')
    train_speaker_parser.add_argument('--mode', type=str, default='formant-bootstrap',
                                       choices=['formant-bootstrap', 'marian-fit'],
                                       help='Training mode (default: formant-bootstrap)')
    train_speaker_parser.add_argument('--iterations', type=int, default=50,
                                       help='Training iterations per phoneme (default: 50)')
    train_speaker_parser.add_argument('--lr', type=float, default=0.01,
                                       help='Learning rate (default: 0.01)')
    train_speaker_parser.add_argument('--seed', type=int, default=42,
                                       help='Random seed (default: 42)')
    train_speaker_parser.add_argument('--marian-dir', type=str, default='data/marian_crumbs',
                                       help='Marian crumbs directory (default: data/marian_crumbs)')
    train_speaker_parser.add_argument('--voicebank', type=str, default=None,
                                       help='Path to unpacked MARIAN ILUSTRADO folder (extracts capped crumbs first)')
    train_speaker_parser.add_argument('--duration-cap', type=float, default=250.0,
                                       help='Max crumb duration in ms (default: 250)')
    
    # Speak with two-swarm pipeline
    speak_ts_parser = subparsers.add_parser('speak-two-swarm',
                                             help='Speak using picker→speaker two-swarm pipeline')
    speak_ts_parser.add_argument('word', type=str, help='Word to speak')
    speak_ts_parser.add_argument('--output', '-o', type=str, default=None,
                                  help='Output WAV path')
    speak_ts_parser.add_argument('--picker-model', type=str, default='model_swarm.npz',
                                  help='Picker swarm model (default: model_swarm.npz)')
    speak_ts_parser.add_argument('--speaker-model', type=str, default='model_speaker.npz',
                                  help='Speaker swarm model (default: model_speaker.npz)')
    speak_ts_parser.add_argument('--picker-type', type=str, default='swarm',
                                  choices=['swarm', 'more_fly'],
                                  help='Picker type (default: swarm)')
    speak_ts_parser.add_argument('--formant-baseline', action='store_true',
                                  help='Use formant synthesis instead of trained speakers')
    
    # Speak sentence with two-swarm
    speak_ts_sent_parser = subparsers.add_parser('speak-two-swarm-sentence',
                                                   help='Speak sentence using two-swarm pipeline')
    speak_ts_sent_parser.add_argument('text', type=str, help='Sentence to speak')
    speak_ts_sent_parser.add_argument('--output', '-o', type=str, default=None,
                                       help='Output WAV path')
    speak_ts_sent_parser.add_argument('--picker-model', type=str, default='model_swarm.npz',
                                       help='Picker swarm model')
    speak_ts_sent_parser.add_argument('--speaker-model', type=str, default='model_speaker.npz',
                                       help='Speaker swarm model')
    speak_ts_sent_parser.add_argument('--picker-type', type=str, default='swarm',
                                       choices=['swarm', 'more_fly'])
    speak_ts_sent_parser.add_argument('--formant-baseline', action='store_true')
    
    # Demo generation with two-swarm
    speak_ts_demo_parser = subparsers.add_parser('speak-two-swarm-demo',
                                                   help='Generate two-swarm demo WAVs')
    speak_ts_demo_parser.add_argument('--output-dir', '-o', type=str, 
                                       default='artifacts/eval/two_swarm',
                                       help='Output directory')
    speak_ts_demo_parser.add_argument('--picker-model', type=str, default='model_swarm.npz')
    speak_ts_demo_parser.add_argument('--speaker-model', type=str, default='model_speaker.npz')
    speak_ts_demo_parser.add_argument('--picker-type', type=str, default='swarm',
                                       choices=['swarm', 'more_fly'])
    speak_ts_demo_parser.add_argument('--formant-baseline', action='store_true')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        'train': cmd_train,
        'train-swarm': cmd_train_swarm,
        'train-more-fly': cmd_train_more_fly,
        'eval-more-fly': cmd_eval_more_fly,
        'speak': cmd_speak,
        'speak-all': cmd_speak_all,
        'eval': cmd_eval,
        'info': cmd_info,
        'train-stage2': cmd_train_stage2,
        'train-stage2b': cmd_train_stage2b,
        # Two-swarm commands
        'train-speaker-flies': cmd_train_speaker,
        'speak-two-swarm': cmd_speak_two_swarm,
        'speak-two-swarm-sentence': cmd_speak_two_swarm_sentence,
        'speak-two-swarm-demo': cmd_speak_two_swarm_demo,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
