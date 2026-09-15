# Testing: Two-Swarm Architecture

**Recognition flies pick; speaking flies speak.**

```
word → PICKER SWARM → phoneme list → SPEAKER SWARM → WAV
        (G2P via KC)    [K AE T]      (no KC)
```

Picker flies classify letter context into a phoneme sequence. Speaker flies
then render each phoneme as a short audio crumb. Speakers are conditioned
**only on phoneme id** (plus optional prev-phone / position). They never see
picker Kenyon-cell activity.

## How speakers were trained (this PR)

Speakers were trained with **`train-speaker-flies --mode marian-fit`**.

1. Downloaded **MARIAN (ILUSTRADO)** from https://downloadmarian.carrd.co/
   (MediaFire: `MARIAN_ILUSTRADO_Series.zip`).
2. Parsed `ARPAsing/oto.ini` and resolved aliases the same way as PR #5/#6:
   standalone (`aa`, `k`) first, then `- C` / `C -`, then diphone `C V` / `V C`.
   Trailing digits (`iy1`) are stripped for matching.
3. Sliced **duration-capped** crumbs (80–250 ms; vowels ~200 ms, consonants
   ~140 ms). Full diphone oto windows are **not** concatenated — that is the
   path that failed in PR #5.
4. Fit a small per-phone parametric voice (F0 / formant shifts / noise /
   duration) to those crumbs. Phones with no usable alias fall back to
   formant self-targets.
5. Saved `model_speaker.npz`. Attribution: `data/NOTICE`.

If Marian cannot be downloaded, the same command with
`--mode formant-bootstrap` trains against the built-in formant synth instead.
This PR **did** obtain Marian and used `marian-fit`.

```bash
# Re-extract crumbs + fit (needs an unpacked ILUSTRADO folder)
python3 -m cursed_tts train-speaker-flies --mode marian-fit \
  --voicebank /path/to/MARIAN\ ILUSTRADO\ Series \
  --iterations 50 --output model_speaker.npz

# Formant-only fallback (no voicebank)
python3 -m cursed_tts train-speaker-flies --mode formant-bootstrap --iterations 50
```

## Tests

```bash
python3 -m pytest tests/test_two_swarm.py -v
```

| Class | What it proves |
|-------|----------------|
| `TestSpeakerNoKCDependency` | `SpeakerFly` / `SpeakerSwarm` have no KC/MB parameters |
| `TestSpeakerDifferentPhonemes` | `cat` ≠ `bat` ≠ `dog`; consonants do not collapse |
| `TestSpeakerDurationCaps` | crumbs stay in the 60–250 ms band |
| `TestMarianAliasAndCaps` | standalone/`- C` alias order; oto windows hard-capped |
| `TestTrainSpeaker` | formant-bootstrap smoke + sane param ranges |
| `TestTwoSwarmPipeline` | picker → phone list → speakers → WAV |

## Demo suite

```bash
python3 -m cursed_tts speak-two-swarm-demo \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz \
  --output-dir artifacts/eval/two_swarm
```

| Path | Contents |
|------|----------|
| `artifacts/eval/two_swarm/words/` | cat/bat/dog/me/yes/… plus longer demo words |
| `artifacts/eval/two_swarm/sentences/` | four short sentences |
| `artifacts/eval/two_swarm/paragraph_four_sentences.wav` | 4-sentence paragraph |
| `artifacts/eval/two_swarm/*_speakers.wav` / `*_formant.wav` | A/B vs formant-only speakers |

Speak one word or a sentence:

```bash
python3 -m cursed_tts speak-two-swarm cat --picker-type more_fly \
  --picker-model model_more_fly_best.npz
python3 -m cursed_tts speak-two-swarm-sentence "hello world" \
  --picker-type more_fly --picker-model model_more_fly_best.npz
```

`--formant-baseline` keeps the same picker phones but renders with the
legacy formant crumbs (A/B).

## Architecture check

```python
from cursed_tts.speaker_fly import verify_no_kc_dependency
assert verify_no_kc_dependency()
```

Speaker input: phoneme id + optional prev-phone / position.
Speaker output: WAV crumb.
**Wrong fork:** [PR #6](https://github.com/jtwolfe/flysune-miku/pull/6)
passed picker KC into an acoustic head; do not ship that as default.
