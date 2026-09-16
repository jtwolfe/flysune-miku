# Flysune Miku — in-browser two-swarm lab

Browser port of the **v0.3.1-punct-pauses** freeze:

```
text
  → MORE FLY picker (Kenyon cells, 39 YES/NO specialists, softmax)
  → phoneme ids only
  → Marian-fit speaker flies (formant crumbs, 15 ms crossfade)
  → WAV  (+ 150 / 300 / 550 ms silence at word / comma / period)
```

Speakers never see Kenyon cells. Slot count is a lexicon cheat (CMUdict, else simple G2P) — the flies do not choose how many phones a word has. Do not revive [PR #6](https://github.com/jtwolfe/flysune-miku/pull/6) (KC → voice).

## What this directory is

The interactive lab (slow-mo theatre, choir, KC grid, Listen / WAV) lives in the Grok App Builder preview. This folder is the **portable JS inference** that preview runs:

| Path | Role |
|------|------|
| `src/engine/` | Picker, speakers, lexicon, concat, pauses |
| `src/lab/` | Timeline, letter window, KC grid, choir, speaker inspector, waveform |
| `public/models/picker.bin` + `picker.json` | MORE FLY weights (hemibrain_real PN→KC) |
| `public/models/speaker.json` | Marian-fit speaker params |
| `public/models/cmudict.bin` | Packed CMUdict for slot counts |
| `public/models/golden.json` | Traces for `cat` / `bat` / `me` / `yes` / `dog` |

The picker here is the checked-in hemibrain wiring (`nPn=400`, `nKc=2000`). Freeze docs still recommend random `+A+B_data` for *training*; this lab does not retrain.

## Honesty

- **Slot count** comes from CMUdict / simple G2P, not from the flies.
- **Picker vs reference** is shown per slot (green match / amber miss). `dog` is `D OW G` vs CMUdict `D AO G` on this hemibrain picker — that miss is real, not a UI lie.
- **Speakers see** phoneme id (+ prev-phone / position). Never KC.
- Punctuation is silence, not a phoneme.

## Local engine check

Golden traces (`cat`, `bat`, `me`, `yes` match; `dog` miss as above) can be replayed by loading `picker.bin` through `src/engine/picker.ts`. The live lab does that in the browser after fetching `/models/*`.

## Attribution

Project mascot only. Not affiliated with Crypton Future Media. Marian crumbs: Kanabun / MARIAN ILUSTRADO. Hemibrain: Scheffer et al. 2020, CC-BY.
