# Fly-only avocado (listen reference)

**Freeze:** `v0.3.1-punct-pauses` / `freeze/punct-pauses-acceptable`

Picker G2P (`model_more_fly_best.npz`) → Marian-fit speaker flies (`model_speaker.npz`). Not lexicon phones, not formant baseline, not KC→voice.

**Text:**

> Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside.

**Pauses** (`cursed_tts.two_swarm`): 150 ms inter-word, 300 ms after `,`, 550 ms after `.`. Punctuation is stripped before picker/speakers.

| File | What |
|------|------|
| `avocados_grow_speakers.wav` | fly-only render with punctuation pauses (~15.5 s) |
| `avocados_grow_speakers.zip` | same WAV, zipped |

```bash
python3 -m cursed_tts speak-two-swarm-sentence \
  "Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside." \
  --picker-model model_more_fly_best.npz --picker-type more_fly \
  --speaker-model model_speaker.npz \
  --output artifacts/eval/two_swarm/avocado/avocados_grow_speakers.wav
```
