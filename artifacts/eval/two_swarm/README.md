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
| `avocado/` | fly-only avocado re-render (picker G2P → Marian speakers) with `,` / `.` pauses |

Sentence/paragraph concat (`speak_sequence` / `speak-two-swarm-sentence`) inserts
silence after punctuation: **150ms** between words, **300ms** after `,`,
**550ms** after `.`. Speakers still receive phone ids only.

Stale single-file WAVs at this directory root (if any) are leftovers from the first pass.
