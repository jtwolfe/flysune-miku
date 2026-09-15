# Two-swarm eval demos

Pipeline: `word → picker (MORE FLY G2P) → phone list → speaker flies → WAV`

Speakers were trained with **marian-fit** (38/39 phones from MARIAN ILUSTRADO crumbs;
`ZH` formant-bootstrap). See `data/NOTICE` and repo `TESTING.md`.

| Path | What |
|------|------|
| `words/` | 22 words including cat/bat/dog/me so consonants can be A/B'd |
| `sentences/` | 4 short sentences |
| `paragraph_four_sentences.wav` | 4-sentence paragraph (~9s) |
| `comparison/` | same phones rendered by trained speakers vs formant-only |

Stale single-file WAVs at this directory root (if any) are leftovers from the first pass.
