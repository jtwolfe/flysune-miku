# Testing: Two-Swarm Architecture

This document explains the clean two-swarm architecture and how to test it.

## Architecture Summary

**Recognition flies pick; speaking flies speak.**

```
word → PICKER SWARM → phoneme list → SPEAKER SWARM → WAV
        (G2P via KC)    [K AE T]      (synthesis)
```

### Picker Swarm (Recognition)
- Uses KC/MB for G2P classification (letter context → phoneme)
- Each picker fly is a specialist that votes YES/NO for its target phoneme
- Output: phoneme sequence

### Speaker Swarm (Synthesis)  
- **NO KC/MB dependency** - receives only phoneme IDs
- Each speaker fly produces audio for ONE phoneme
- Trainable parameters: F0, formants, noise, duration
- Output: concatenated audio crumbs

## Why This Architecture?

PR #6 ("Acoustic Fly Head") stacked audio generation on picker KC features:
```
word → picker → KC activity → acoustic fly → audio  ← WRONG
```

This caused all words to sound similar because:
- KC activity patterns are sparse and similar for different words
- Audio output was coupled to picker state, not phoneme identity

The correct architecture decouples recognition from synthesis:
```
word → picker → phonemes → speakers → audio  ← CORRECT
```

Now different words produce different audio because:
- Different words → different phoneme sequences (from picker)
- Different phonemes → different audio crumbs (from speakers)

## Testing

### Run All Two-Swarm Tests

```bash
python3 -m pytest tests/test_two_swarm.py -v
```

### Key Test Classes

#### TestSpeakerNoKCDependency
Verifies speakers have NO KC/MB dependency:
- `test_speaker_fly_synthesize_signature`: No KC params in synthesize()
- `test_speaker_swarm_synthesize_signature`: No KC params in swarm methods
- `test_speaker_params_no_kc_fields`: No KC fields in SpeakerParams
- `test_verify_no_kc_dependency_function`: Built-in verification passes

#### TestSpeakerDifferentPhonemes
Verifies different phonemes produce different audio:
- `test_different_phonemes_different_audio`: K ≠ AE ≠ T ≠ M ≠ IY ≠ S
- `test_different_words_different_audio`: "cat" ≠ "bat" ≠ "dog"

#### TestSpeakerDurationCaps
Verifies crumb duration constraints:
- Each crumb within 60-250ms bounds

#### TestTrainSpeaker
Verifies speaker training:
- Formant-bootstrap mode produces trained speakers
- Trained parameters are within reasonable ranges

#### TestTwoSwarmPipeline
End-to-end pipeline tests (requires trained picker):
- Full pipeline produces audio
- Different words → different audio through pipeline

## Manual Testing

### Train Speakers

```bash
# Train speaker flies (formant bootstrap)
python3 -m cursed_tts train-speaker-flies --iterations 50

# Train with Marian voicebank targets (if available)
python3 -m cursed_tts train-speaker-flies --mode marian-fit --marian-dir data/marian_crumbs
```

### Speak with Two-Swarm Pipeline

```bash
# Single word
python3 -m cursed_tts speak-two-swarm cat

# Sentence  
python3 -m cursed_tts speak-two-swarm-sentence "hello world this is a test"

# Generate demo suite
python3 -m cursed_tts speak-two-swarm-demo --output-dir artifacts/eval/two_swarm
```

### Compare Speaker vs Formant Baseline

```bash
# Speak with formant baseline (for A/B comparison)
python3 -m cursed_tts speak-two-swarm cat --formant-baseline

# The demo command generates both versions for comparison
python3 -m cursed_tts speak-two-swarm-demo
```

## Verifying Architecture Correctness

### Programmatic Verification

```python
from cursed_tts.speaker_fly import verify_no_kc_dependency

# This should return True
assert verify_no_kc_dependency()
```

### Signature Inspection

```python
import inspect
from cursed_tts.speaker_fly import SpeakerFly, SpeakerSwarm

# These should NOT contain 'kc', 'mb', 'mushroom', etc.
print(inspect.signature(SpeakerFly.synthesize))
print(inspect.signature(SpeakerSwarm.synthesize_sequence))
```

## File Structure

```
cursed_tts/
├── speaker_fly.py      # SpeakerFly/SpeakerSwarm (NO KC dependency)
├── train_speaker.py    # Speaker training (formant-bootstrap, marian-fit)
├── two_swarm.py        # Two-swarm pipeline integration
├── specialist_fly.py   # Picker swarm (uses KC)
└── ...

tests/
└── test_two_swarm.py   # Two-swarm architecture tests

artifacts/eval/two_swarm/
├── *.wav               # Demo outputs
└── comparison/         # Speaker vs formant comparisons
```

## Deprecation Notice

The PR #6 "acoustic flies" approach that passed KC activity to speakers is deprecated.
Do not use KC/MB features for audio synthesis - they should only be used for G2P classification.

**Correct**: Speakers receive phoneme ID + optional context (prev-phone, position)
**Wrong**: Speakers receive KC activity or picker state

---

*One architecture sentence: Recognition flies pick; speaking flies speak.*
