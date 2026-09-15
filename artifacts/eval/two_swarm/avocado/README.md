# Fly-only avocado paragraph

Picker G2P (`model_more_fly_best.npz`) → Marian-fit speaker flies (`model_speaker.npz`).
Not lexicon phones, not formant baseline.

**Text:**

> Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside.

Prior avocado A/B / `avocados_grow_speakers` paths were not in `main` after the
`v0.3.0-speakers-lexicon` freeze, so this is the standard multi-sentence
avocado-growth paragraph used for this re-render.

**Pauses** (see `cursed_tts.two_swarm`): 150ms inter-word, 300ms after `,`,
550ms after `.`. Punctuation is stripped before picker/speakers.

| File | What |
|------|------|
| `avocados_grow_speakers.wav` | fly-only render with punctuation pauses |
| `avocados_grow_speakers.zip` | same WAV, zipped for download |

```bash
python3 -m cursed_tts speak-two-swarm-sentence \
  "Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside." \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz \
  --output artifacts/eval/two_swarm/avocado/avocados_grow_speakers.wav
```
