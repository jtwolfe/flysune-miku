# Testing

**Recognition flies pick; speaking flies speak.** Speakers never see picker KC.

```
word → PICKER (MORE FLY G2P) → phone ids → SPEAKER FLIES (Marian-fit) → WAV
```

**This freeze (`v0.3.1-punct-pauses`, `freeze/punct-pauses-acceptable`):** fly-only speak path. Listen to `artifacts/eval/two_swarm/avocado/`. Punctuation is silence at concat, not phones.

**Prior freeze (`v0.3.0-speakers-lexicon`):** lexicon phones → the same Marian speakers. Still available; not the avocado reference.

Rejected: [PR #6](https://github.com/jtwolfe/flysune-miku/pull/6) KC→voice. Stage 2 / 2b trajectory→audio. Do not ship those.

## Tests

```bash
python3 -m pytest tests/test_two_swarm.py -v
```

| Class | What it proves |
|-------|----------------|
| `TestSpeakerNoKCDependency` | `SpeakerFly` / `SpeakerSwarm` have no KC/MB parameters |
| `TestSpeakerDifferentPhonemes` | `cat` ≠ `bat` ≠ `dog`; consonants do not collapse |
| `TestSpeakerDurationCaps` | crumbs stay in the 60–250 ms band |
| `TestMarianAliasAndCaps` | standalone / `- C` alias order; oto windows hard-capped |
| `TestTrainSpeaker` | formant-bootstrap smoke + sane param ranges |
| `TestTwoSwarmPipeline` | picker → phone list → speakers → WAV |
| `TestPunctuationPauses` | tokenize + duration vs glued concat (`,` 300 ms, `.` 550 ms) |

Architecture assert:

```python
from cursed_tts.speaker_fly import verify_no_kc_dependency
assert verify_no_kc_dependency()
```

Speaker input: phoneme id + optional prev-phone / position. Speaker output: WAV crumb.

## Demos (no retraining)

Checked-in models: `model_more_fly_best.npz` (picker), `model_speaker.npz` (speakers).

```bash
# Word
python3 -m cursed_tts speak-two-swarm cat --picker-type more_fly \
  --picker-model model_more_fly_best.npz --speaker-model model_speaker.npz

# Sentence / paragraph (pauses: 150 ms word, 300 ms comma, 550 ms period)
python3 -m cursed_tts speak-two-swarm-sentence \
  "Avocados grow on trees. The trees are tall, and the fruit is green." \
  --picker-type more_fly --picker-model model_more_fly_best.npz \
  --speaker-model model_speaker.npz

# Suite
python3 -m cursed_tts speak-two-swarm-demo \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz \
  --output-dir artifacts/eval/two_swarm
```

`--formant-baseline` = same picker phones, old formant crumbs (A/B only).

| Path | Contents |
|------|----------|
| `artifacts/eval/two_swarm/avocado/` | **Listen reference** — fly-only avocado paragraph with pauses |
| `artifacts/eval/two_swarm/words/` | cat/bat/dog/me/yes/… |
| `artifacts/eval/two_swarm/sentences/` | short sentences |
| `artifacts/eval/two_swarm/paragraph_four_sentences.wav` | 4-sentence paragraph |
| `artifacts/eval/two_swarm/*_speakers.wav` / `*_formant.wav` | A/B vs formant-only |

Lexicon hybrid (CMUdict/G2P phones → Marian speakers): `cursed_tts.train_speaker.generate_speaker_demo()` with `picker_swarm=None`. Old formant lexicon: `python -m cursed_tts speak <word> --lexicon`.

## Speakers (already fitted)

`train-speaker-flies --mode marian-fit` against duration-capped **MARIAN (ILUSTRADO)** crumbs (`data/NOTICE`). `ZH` formant-fallback. Do not re-fit for this freeze.

```bash
# Re-fit only if you mean to (needs unpacked ILUSTRADO)
python3 -m cursed_tts train-speaker-flies --mode marian-fit \
  --voicebank /path/to/MARIAN\ ILUSTRADO\ Series \
  --iterations 50 --output model_speaker.npz

python3 -m cursed_tts train-speaker-flies --mode formant-bootstrap --iterations 50
```

## Freeze tags

| Tag | Branch |
|-----|--------|
| `v0.3.1-punct-pauses` | `freeze/punct-pauses-acceptable` |
| `v0.3.0-speakers-lexicon` | `freeze/speakers-lexicon-acceptable` |
| `v0.2.0-more-fly` | `freeze/more-fly-acceptable` |
| `v0.1.0-g2p-acceptable` | `freeze/g2p-acceptable` |

MORE FLY picker ablations (historical): `artifacts/eval/more_fly/`.

## Security / PII scan (v0.3.1 tip)

Working tree + `git grep` over `HEAD` history blobs for high-signal secrets (`AKIA…`, `sk_live_`, `ghp_`, `github_pat_`, `xox[baprs]-`, PEM/SSH private keys, `.env`, `aws_secret`). No hits. No `.env`, `*.pem`, or credential files. `.gitignore` now ignores `.env` / `*.pem` / `id_rsa*`.

**PII:** no emails, phone numbers, home paths, or street addresses in tracked files. Public attributions left as-is (MARIAN / Kanabun, FlyWire, hemibrain authors, Hatsune Miku mascot disclaimer). Commit metadata still has the repo owner's git author email (public Git history — not rewritten).

**Binaries:** checked-in `*.npz` models (largest `model.npz` ~12 MB) and demo WAVs, all expected. No surprise disk images or voicebank dumps.

**Dependencies:** no npm tree. Declared Python deps (`numpy`, `soundfile`, `cmudict`, `g2p_en`) had no OSV hits on current/latest versions checked. `g2p_en` pulls **NLTK** for the optional tagger download; NLTK's corpus/downloader APIs have many 2026 advisories. This repo is a local CLI toy and does not expose those APIs. VM `pip-audit` also flagged image-level `pip`/`setuptools`/`urllib3`/etc. — not project requirements. No dependency bumps in this freeze.
