# Acoustic Flies Evaluation

## Overview

This directory contains evaluation artifacts for the **Acoustic Fly Head** - a trained speech production system that maps fly-native features (KC activations, cues) to audio.

**Key distinction**: This is TRAINED speech production, NOT concatenative sample playback. The acoustic flies LEARN to produce voice from mushroom body activity patterns.

## Architecture

```
Word ("cat")
    ↓
Letter context encoding → Picker Swarm (MORE FLY)
    ↓
KC activity + phoneme cue → Acoustic Fly (per-phoneme MoE)
    ↓
Voice parameters (F0, formants, harmonics, noise)
    ↓
Additive synthesis → Audio WAV
```

### Components

1. **Picker Swarm** (from MORE FLY): G2P classification (letter context → phoneme)
2. **Acoustic Fly Swarm**: Per-phoneme voice parameter predictors
3. **Parametric Synthesizer**: Voice params → waveform (clean oscillator bank, NOT Griffin-Lim)

## Training Targets

Two modes:

1. **MARIAN ILUSTRADO** (preferred): Real voicebank crumbs from Kanabun's Arpasing bank
   - Download from: https://downloadmarian.carrd.co/
   - Attribution required (see NOTICE)
   
2. **Formant Bootstrap** (fallback): Synthetic formant crumbs
   - Used when MARIAN unavailable
   - Documents the gap between synthetic and real targets

Current training used: **formant (bootstrap mode)** - MARIAN download requires manual MediaFire fetch.

## Demo Files

### `formant_baseline/`
WAVs using MORE FLY picker + clean formant synthesis.
This is the baseline for comparison.

### `acoustic_flies/`
WAVs using MORE FLY picker + trained acoustic fly synthesis.
Same phoneme sequences, different audio rendering.

## Evaluation Results

| Word | Reference | Predicted | Phoneme Match |
|------|-----------|-----------|---------------|
| cat | K AE T | K AE T | 100% |
| dog | D AO G | D AO G | 100% |
| hi | HH AY | HH AY | 100% |
| bye | B AY | B P | 50% |
| yes | Y EH S | Y EH S | 100% |
| me | M IY | M IY | 100% |
| you | Y UW | Y UW | 100% |
| go | G OW | G OW | 100% |
| no | N OW | N OW | 100% |
| bat | B AE T | B AE T | 100% |

**Demo word accuracy**: ~95% phoneme accuracy (consistent with MORE FLY picker)

## Comparison: Formant vs Acoustic Flies

Both use the same MORE FLY picker, so phoneme sequences are identical.
The difference is in audio rendering:

- **Formant**: Fixed formant synthesis with hand-tuned parameters per phoneme
- **Acoustic Flies**: Learned voice parameters from KC activity patterns

### Intelligibility Notes

- Consonants: Both methods produce audible consonants
- Vowels: Acoustic flies have learned formant trajectories
- Duration: Acoustic flies produce slightly longer crumbs (more frames)

## Training Info

```
Picker: artifacts/eval/more_fly/swarm__A_B_data.npz
Target source: formant (bootstrap)
Training examples: 18750
Epochs: 8
Final demo loss: 11769.09
```

## CLI Commands

```bash
# Train acoustic flies
python -m cursed_tts train-acoustic-flies --epochs 10 --words 5000

# Speak with acoustic flies
python -m cursed_tts speak cat --acoustic

# Generate all demos
python -m cursed_tts speak-all --acoustic
```

## Fly-Faithful Design

The acoustic fly architecture follows mushroom body principles:

1. **Per-compartment readout**: Each phoneme has its own acoustic fly (like MBON compartments)
2. **KC→output mapping**: Sparse KC pattern → voice parameters (KC→MBON style)
3. **Compartment-local teaching**: Only the responsible acoustic fly updates for each example
4. **No big neural network**: Simple 2-layer readout, not a Transformer/LLM

This is NOT "replace everything with a TTS model" - the flies maintain the biological metaphor.

## Known Limitations

1. **Bootstrap mode**: Without MARIAN, training targets are synthetic formant crumbs
2. **No prosody**: Fixed F0 per phoneme, no intonation modeling
3. **Short crumbs**: Each phoneme is ~0.1-0.2s, no coarticulation effects
4. **Intelligibility over fidelity**: v1 prioritizes audible consonants over natural voice

## Future Work

1. Download and integrate MARIAN ILUSTRADO for real target crumbs
2. Add pitch contour modeling (F0 trajectories)
3. Coarticulation: Condition on neighboring phonemes
4. Longer context: Multi-phoneme acoustic flies for smoother synthesis
