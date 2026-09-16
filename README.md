# Cursed TTS: Flysune Miku

<p align="center">
  <img src="assets/flysune-miku.jpg" alt="Flysune Miku - Project Mascot" width="300">
  <br>
  <em>Flysune Miku: Hatsune Miku with a picker swarm in her skull</em>
  <br>
  <small>(Project mascot art. Not affiliated with Crypton Future Media.)</small>
</p>

A cursed text-to-speech toy: **recognition flies pick phonemes; speaking flies speak them.** Inspired by the [FlyWire hiragana OCR demo](https://hae.satoru.net/).

## Current architecture

Two swarms. The picker never sings; the speakers never see Kenyon cells.

```
word / sentence
    ↓
PICKER SWARM  (MORE FLY G2P, KC/MB classifier)
    ↓  phoneme ids only   e.g. [K AE T]
SPEAKER SWARM  (Marian-fit flies, NO KC)
    ↓
WAV  (+ silence after punctuation)
```

| Swarm | Job | Sees |
|-------|-----|------|
| **Picker** | letter context → phoneme list | graphemes, KC/MB |
| **Speakers** | phoneme id → audio crumb | phone id (+ optional prev-phone / position). **Never KC.** |

**This freeze (`v0.3.1-punct-pauses`):** fly-only speak path — picker G2P → Marian speaker flies, with sentence pauses. Listen to [`artifacts/eval/two_swarm/avocado/`](artifacts/eval/two_swarm/avocado/). Lexicon phones → same speakers is still available (the `v0.3.0` hybrid); it is not this freeze's listen path.

### Punctuation pauses

Concat (`speak-two-swarm-sentence` / `speak_sequence`) does **not** send `,` or `.` to either swarm. Silence is inserted at join time:

| Mark | Pause |
|------|------:|
| word gap (no punct) | 150 ms |
| `,` | 300 ms |
| `.` | 550 ms |

### Rejected: KC → voice

[PR #6](https://github.com/jtwolfe/flysune-miku/pull/6) stacked an acoustic head on picker KC activity (`word → picker KC → voice`). Words collapsed toward the same buzz. **Do not revive that fork.** Speakers take phoneme ids only.

Older wrong objectives (Stage 2 mel + Griffin-Lim, Stage 2b formant-track regression from MB trajectories) are the same mistake: treating the mushroom body as an acoustic generator. They stay in the tree as `stage2.py` / `stage2b.py` only.

## Listen

**Reference render (this freeze):** picker G2P + Marian speakers + pauses.

> Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside.

- WAV: [`artifacts/eval/two_swarm/avocado/avocados_grow_speakers.wav`](artifacts/eval/two_swarm/avocado/avocados_grow_speakers.wav) (~15.5 s)
- Notes: [`artifacts/eval/two_swarm/avocado/README.md`](artifacts/eval/two_swarm/avocado/README.md)

Word / sentence suites: [`artifacts/eval/two_swarm/`](artifacts/eval/two_swarm/).

## Quick start

```bash
pip install -e .
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# Fly-only speak (needs checked-in models)
python -m cursed_tts speak-two-swarm mushroom \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz

python -m cursed_tts speak-two-swarm-sentence \
  "Avocados grow on trees. The trees are tall, and the fruit is green." \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz

python -m cursed_tts speak-two-swarm-demo \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz
```

`--formant-baseline` keeps the same picker phones but renders with the old formant crumbs (A/B only).

**Lexicon hybrid** (phones from CMUdict/G2P, same Marian speakers — `v0.3.0` path): `generate_speaker_demo()` without a picker, or the old `speak --lexicon` formant baseline. Not the avocado listen reference.

Do **not** retrain picker or speakers for this freeze. Tests:

```bash
python -m pytest tests/test_two_swarm.py -v
```

Full commands and pause checks: [TESTING.md](TESTING.md).

## Freeze tags

Do not rewrite these tips.

| Tag | Branch | What it freezes |
|-----|--------|-----------------|
| `v0.3.1-punct-pauses` | `freeze/punct-pauses-acceptable` | **This freeze.** Picker G2P → Marian speakers; `,` 300 ms / `.` 550 ms; avocado listen reference |
| `v0.3.0-speakers-lexicon` | `freeze/speakers-lexicon-acceptable` | Lexicon phones → Marian-fit speakers (hybrid baseline) |
| `v0.2.0-more-fly` | `freeze/more-fly-acceptable` | MORE FLY picker |
| `v0.1.0-g2p-acceptable` | `freeze/g2p-acceptable` | Single-MB G2P + formant crumbs |

## Earlier experiments (don't start here)

Kept for archaeology; none of these are the default speak path.

- **Single MB G2P** (`train` / `speak`): one mushroom body, 39 MBON classes, formant crumbs. Fine as a classifier demo.
- **Specialist picker swarm** (`train-swarm` / `speak --swarm`): one fly per phoneme. Educational, usually worse than a single MB.
- **MORE FLY** (`train-more-fly`): cues + curriculum + optional hemibrain wiring / DAN teaching. **This freeze's picker.** Random `+A+B_data` is the recommended train config; hemibrain wiring is experimental. Ablations: `artifacts/eval/more_fly/`.
- **Singer-fly MoE voices** (`singer_fly.py`): per-phone F0/formant choir on formant crumbs. Superseded by Marian-fit speaker flies.

## Phonemes

39-class CMUdict ARPAbet (stress stripped):

| Type | Phonemes |
|------|----------|
| Vowels (mono) | AA, AE, AH, AO, EH, ER, IH, IY, UH, UW |
| Vowels (diphthong) | AW, AY, EY, OW, OY |
| Stops | P, B, T, D, K, G |
| Affricates | CH, JH |
| Fricatives | F, V, TH, DH, S, Z, SH, ZH, HH |
| Nasals | M, N, NG |
| Liquids | L, R |
| Semivowels | W, Y |

## Speakers (already fitted)

Marian-fit speaker flies live in `model_speaker.npz`. Crumbs were duration-capped slices of **MARIAN (ILUSTRADO)** ([download](https://downloadmarian.carrd.co/), attribution in `data/NOTICE`). `ZH` had no oto alias and used a formant target. Do not re-fit for this freeze.

```bash
# Only if you intentionally re-fit (not needed for v0.3.1)
python -m cursed_tts train-speaker-flies --mode marian-fit \
  --voicebank /path/to/MARIAN\ ILUSTRADO\ Series --iterations 50
python -m cursed_tts train-speaker-flies --mode formant-bootstrap   # no voicebank
```

## References

- [FlyWire Hiragana OCR Demo](https://hae.satoru.net/)
- [FlyWire](https://flywire.ai/)
- [Hemibrain](https://neuprint.janelia.org/) — Scheffer et al. 2020
- [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- [Hige et al. 2015](https://doi.org/10.1016/j.neuron.2015.04.027) — dopamine plasticity in MB
- [Handler et al. 2019](https://doi.org/10.1038/s41593-019-0435-7)

## License

MIT. Marian crumbs: see `data/NOTICE` (Kanabun / MARIAN ILUSTRADO). Hemibrain extracts: CC-BY, Scheffer et al. 2020.
