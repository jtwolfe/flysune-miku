# MORE FLY Demo Proofs

Picker-only G2P audio proofs (historical). **v0.3.1 listen reference:** `artifacts/eval/two_swarm/avocado/`.

Audio proof files demonstrating the MORE FLY G2P system.

## Directory Structure

```
proofs/
├── random/       # +A+B_data (recommended default)
│   ├── cat.wav
│   ├── bat.wav
│   ├── ... (20 words total)
├── hemibrain/    # +A+B+C_wiring (experimental)
│   ├── cat.wav
│   ├── bat.wav
│   ├── ... (20 words total)
└── sentences/    # Multi-word phrases (using random wiring)
    ├── hi_me.wav
    ├── go_fly_miku.wav
    ├── hello_world.wav
    ├── yes_I_love_you.wav
    ├── open_the_brain.wav
    ├── see_you_bye.wav
    └── avocados_grow.wav
```

## Demo Words

| Word | Random Wiring | Hemibrain Wiring | Reference |
|------|---------------|------------------|-----------|
| cat | K AE T ✓ | K AE T ✓ | K AE T |
| bat | B AE T ✓ | B AE T ✓ | B AE T |
| dog | D AO G ✓ | D OW G ✗ | D AO G |
| go | G OW ✓ | G OW ✓ | G OW |
| no | N OW ✓ | N OW ✓ | N OW |
| hi | HH AY ✓ | HH AY ✓ | HH AY |
| bye | B P ✗ | B AY ✓ | B AY |
| yes | Y EH S ✓ | Y EH S ✓ | Y EH S |
| me | M IY ✓ | M IY ✓ | M IY |
| you | Y UW ✓ | Y UW ✓ | Y UW |

**Key observation**: Both wiring modes correctly predict `me = M IY` after the hemibrain `me`/IY fixes. These proofs are picker-only; the v0.3.1 listen reference is `artifacts/eval/two_swarm/avocado/`.

## OOV (Out-of-Vocabulary) Words

These words test generalization beyond the training lexicon:

| Word | Random | Hemibrain |
|------|--------|-----------|
| mushroom | M M UW R UW M | M SH AA M UW M |
| connectome | K EH N ER K T EH M | K AA N EH K T AY AY |
| chaos | K EY OW Z | CH R OW Z |
| hatsune | HH AE T AH N IY | HH AE T UW N N |
| australia | AO AO S T AA L D AH | Z N S T AA L AH AH |
| neural | N EH UH AA L | N N ER AH L |
| phoneme | F OW IH IH M | P AH M N AY |
| cursed | K ER S D | K R S T |
| flywire | F L K AH AY ER | F L AY AY AY DH |
| kenyon | K EH N AH UW N | K EH N IY G N |

OOV accuracy is lower as expected—the system relies on learned letter-to-phoneme patterns.

## Sentences

Multi-word phrases synthesized with the recommended random wiring model:

| Filename | Words |
|----------|-------|
| hi_me.wav | "hi me" |
| go_fly_miku.wav | "go fly miku" |
| hello_world.wav | "hello world" |
| yes_I_love_you.wav | "yes I love you" |
| open_the_brain.wav | "open the brain" |
| see_you_bye.wav | "see you bye" |
| avocados_grow.wav | "avocados grow" |

## How to Play

```bash
# Play with any audio player
play artifacts/eval/more_fly/proofs/random/me.wav
play artifacts/eval/more_fly/proofs/sentences/hi_me.wav

# Or use Python
import scipy.io.wavfile as wav
rate, audio = wav.read("artifacts/eval/more_fly/proofs/random/me.wav")
```

## Regeneration

To regenerate these proofs from the trained models:

```python
from cursed_tts.more_fly import MoreFlySwarm
from cursed_tts.synth import synthesize_phoneme
from cursed_tts.phonemes import strip_stress
import numpy as np
import scipy.io.wavfile as wav

# Load model
swarm = MoreFlySwarm.load("artifacts/eval/more_fly/swarm__A_B_data.npz")

# Predict and synthesize
word = "me"
predicted = swarm.predict_word(word, 2)  # 2 phonemes
audio = np.concatenate([synthesize_phoneme(strip_stress(p)) for p in predicted])
wav.write("me.wav", 22050, (audio * 32767).astype(np.int16))
```

## Models Used

- **Random wiring** (`swarm__A_B_data.npz`): Recommended default, +A+B_data config
- **Hemibrain wiring** (`swarm__A_B_C_wiring.npz`): Experimental, +A+B+C_wiring config

See `TESTING.md` for full ablation results and analysis.
