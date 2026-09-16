# Two-swarm eval demos

**This freeze (`v0.3.1-punct-pauses`):** picker G2P → Marian speaker flies. Listen to `avocado/`.

```
word → picker (MORE FLY) → phone ids → speakers → WAV
```

Speakers: **marian-fit** (38/39 phones from MARIAN ILUSTRADO crumbs; `ZH` formant-bootstrap). Attribution: `data/NOTICE`. No KC→voice.

Sentence/paragraph concat inserts silence after punctuation: **150 ms** between words, **300 ms** after `,`, **550 ms** after `.`. Punctuation is never sent to picker or speakers.

| Path | What |
|------|------|
| `avocado/` | **Listen reference** — fly-only avocado paragraph with pauses |
| `words/` | 22 words including cat/bat/dog/me so consonants can be A/B'd |
| `sentences/` | 4 short sentences |
| `paragraph_four_sentences.wav` | 4-sentence paragraph |
| `comparison/` | same phones rendered by trained speakers vs formant-only |

Lexicon phones → the same speakers is the older `v0.3.0-speakers-lexicon` hybrid, not this folder's avocado render.

```bash
python3 -m cursed_tts speak-two-swarm-demo \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz \
  --output-dir artifacts/eval/two_swarm
```
