# Cursed TTS: G2P Mushroom Body Classifier

A cursed text-to-speech toy that uses a **simplified mushroom body neural circuit as a grapheme-to-phoneme (G2P) classifier**. Inspired by the [FlyWire hiragana OCR demo](https://hae.satoru.net/) — a real fruit fly brain learning to read Japanese characters.

## The Correct Architecture

The mushroom body is a **CLASSIFIER**, not an acoustic signal generator:

```
Word ("cat")
    ↓
Letter context encoding (character n-grams, position)
    ↓
Projection Neurons (PNs) — input layer
    ↓
Kenyon Cells (KCs) — sparse expansion (~10% active, FlyWire-style)
    ↓
MBONs — output compartments (39 phoneme classes)
    ↓
Winner-take-all → Predicted phoneme
    ↓
Clean formant synthesis → Audio
```

The MB classifies **letter context → phoneme**, just like the OCR fly classifies images → hiragana. Audio is rendered by a **separate clean synthesizer** (concatenative formant crumbs), not by regressing acoustic features from neural activity.

## Why Stage 2/2b Were Wrong

Previous attempts (Stage 2: mel + Griffin-Lim, Stage 2b: formant track regression) tried to generate audio directly from MB activity trajectories. This was the wrong objective:

- **Stage 2**: MB trajectory → mel spectrogram → Griffin-Lim = scratchy audio
- **Stage 2b**: MB trajectory → formant tracks → synthesis = better but still wrong

The problem: we were treating the MB as an **acoustic generator** instead of a **classifier**. The real fly MB is a classifier (odor → mushroom body → behavior), not a signal generator.

**The fix**: MB = G2P classifier (letter context → phoneme), Synth = clean renderer (phoneme → audio crumbs).

## Quick Start

```bash
# Install
pip install -e .

# Download NLTK data for G2P fallback
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# Train G2P mushroom body on CMUdict (~6 minutes)
python -m cursed_tts train --epochs 15 --words 10000

# Speak using MB predictions (default - cursed G2P)
python -m cursed_tts speak mushroom

# Speak using dictionary phonemes (baseline)
python -m cursed_tts speak mushroom --lexicon

# Generate all demo WAVs
python -m cursed_tts speak-all

# Evaluate accuracy
python -m cursed_tts eval
```

## The Biology

### Mushroom Body as Classifier

Like the hiragana OCR demo, our MB learns to classify inputs:

1. **Input encoding**: Letter context → PN-like activity (character n-grams, position features)
2. **Sparse expansion**: PNs → KCs via random connectivity (~10% KC sparsity, FlyWire-style)
3. **Classification**: KC activity → MBON compartments (39 phoneme classes)
4. **Winner-take-all**: Minimum MBON input wins (inhibitory logic, matching real fly)

### Learning Rule (Hige 2015 / Handler 2019)

Dopamine-modulated anti-Hebbian learning — **only update when wrong**:

- **DEPRESS** KC→correct_MBON synapses (less input = wins more easily)
- **POTENTIATE** KC→wrong_MBON synapses (more input = wins less)

This matches the real mushroom body plasticity observed in flies.

## Performance

After training on 10,000 CMUdict words (15 epochs, ~6 minutes):

| Metric | Value |
|--------|-------|
| Demo word phoneme accuracy | 100% |
| Demo word accuracy | 100% |
| Held-out test phoneme accuracy | ~71% |
| Held-out word accuracy | ~24% |

The test accuracy reflects the difficulty of English G2P — spelling is highly irregular. Demo words achieve 100% because they're in the training set.

## Phoneme Inventory (39 phonemes)

Full CMUdict-compatible ARPAbet:

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

## CLI Reference

```bash
# Training (G2P classifier)
python -m cursed_tts train                  # Default: 15 epochs, 10k words
python -m cursed_tts train --epochs 25      # More training
python -m cursed_tts train --words 5000     # Smaller vocabulary

# Speaking (MB predictions = cursed G2P)
python -m cursed_tts speak cat              # MB predicts phonemes
python -m cursed_tts speak mushroom         # Works on any word

# Speaking (dictionary baseline)
python -m cursed_tts speak cat --lexicon    # CMUdict/G2P phonemes

# Batch generation
python -m cursed_tts speak-all              # Demo + OOV words
python -m cursed_tts speak-all --lexicon    # Dictionary baseline

# Evaluation
python -m cursed_tts eval                   # Accuracy on demo/test

# Info
python -m cursed_tts info                   # Phoneme inventory, stats
```

## Project Structure

```
cursed_tts/
├── __init__.py          # Package metadata
├── __main__.py          # CLI entry point
├── phonemes.py          # 39-phoneme ARPAbet inventory
├── lexicon.py           # CMUdict loader + G2P fallback
├── alignment.py         # Grapheme-phoneme alignment for training
├── mushroom_body.py     # G2P classifier (the brain)
├── synth.py             # Clean formant synthesis (the voice)
├── train.py             # G2P training loop
├── speak.py             # Inference + audio generation
├── stage2.py            # [LEGACY] Mel trajectory regression
└── stage2b.py           # [LEGACY] Formant track regression
```

## Example Outputs

### artifacts/g2p/ (default, recommended)

WAVs generated using MB phoneme predictions + clean formant synthesis:

- Demo words: cat, bat, dog, go, no, hi, bye, yes, me, you (100% accuracy)
- OOV words: mushroom, connectome, chaos, hatsune, australia, neural, phoneme, cursed, flywire, kenyon (varying accuracy)

### artifacts/lexicon/ (baseline)

Same words using dictionary phonemes — shows what "perfect G2P" sounds like.

### artifacts/stage2/, artifacts/stage2b/ (legacy)

Scratchy/experimental outputs from the trajectory-regression approach. Not recommended.

## Limitations

1. **English spelling is hard**: ~71% test accuracy (irregular orthography)
2. **No prosody**: No pitch, stress, or duration modeling
3. **Not real FlyWire**: Simplified MB, not actual connectome weights
4. **Cursed quality**: This is a toy, not production TTS

---

## One Phoneme Per Fly (Experimental)

*"What if each phoneme had its own dedicated fly brain?"*

### The Joke (and the Biology)

In the real fruit fly mushroom body, different **MBON compartments** respond to different learned associations. The standard multi-class MB uses one MBON per phoneme class (39 compartments), with winner-take-all selecting the prediction.

The **one-phoneme-per-fly** experiment takes this to the extreme: instead of one brain with 39 output compartments, we train an **ensemble of 39 specialist flies**. Each specialist answers a simple binary question: *"Is the next phoneme /K/?"* At inference, all 39 flies evaluate the letter context, and we pick the phoneme whose specialist is most confident it's seeing its target.

This is closer to the biological concept of **compartment specialization** — each MBON compartment in the real fly responds preferentially to specific odor-reward associations. Here, each "compartment" (specialist fly) specializes in detecting one phoneme.

### Architecture

```
    Letter context ("_ca_t__")
            ↓
    ┌───────────────────────────────────────┐
    │   Shared PN→KC Expansion              │
    │   (Random projection + sparse coding) │
    └───────────────────────────────────────┘
            ↓ KC activity (shared)
    ┌─────┬─────┬─────┬─────┬─────┐
    │ /K/ │ /AE/│ /T/ │ /D/ │ ... │  ← 39 Specialist Flies
    │ Fly │ Fly │ Fly │ Fly │     │
    │YES/NO│YES/NO│YES/NO│YES/NO│   │
    └──┬──┴──┬──┴──┬──┴──┬──┴─────┘
       │     │     │     │
       ↓     ↓     ↓     ↓
    Pick phoneme with strongest YES vote
            ↓
    Clean formant synthesis → Audio
```

Each specialist has:
- **Shared input layer**: Same PN→KC expansion as others (efficiency)
- **Private output weights**: KC→MBON with 2 outputs (YES/NO)
- **Same learning rule**: Dopamine-when-wrong (anti-Hebbian)

### Quick Start (Swarm)

```bash
# Train a subset of specialists (phonemes needed for demo words)
python -m cursed_tts train-swarm --phones demo --epochs 8

# Train all 39 specialists (full ARPAbet)
python -m cursed_tts train-swarm --epochs 10

# Speak using the fly swarm
python -m cursed_tts speak mushroom --swarm

# Generate all swarm demo WAVs
python -m cursed_tts speak-all --swarm

# Compare swarm vs baseline accuracy
python -m cursed_tts eval --swarm
```

### Performance

The swarm is an experiment, not an improvement. Expected results:

| Model | Demo Phoneme Acc | Demo Word Acc | Test Phoneme Acc | Test Word Acc |
|-------|------------------|---------------|------------------|---------------|
| Single MB (baseline) | ~100% | ~100% | ~71% | ~24% |
| Fly Swarm (39 specialists) | ~70-80% | ~40-60% | ~50-60% | ~5-15% |

**Why worse?** The specialists vote independently — they don't see each other's outputs. The single MB has one unified decision boundary across all classes; the swarm has 39 independent binary classifiers that can disagree. This is biologically interesting but mathematically suboptimal.

### Output Examples

Swarm predictions for demo words (demo-subset swarm):

| Word | Reference | Swarm Prediction | Match |
|------|-----------|------------------|-------|
| cat | K AE T | K AE T | 100% |
| dog | D AO G | D AO G | 100% |
| mushroom | M AH SH R UW M | M AH SH R AH M | 83% |
| australia | AO S T R EY L Y AH | AO AH S S AH L AH AH | 38% |
| chaos | K EY AA S | K EH R S | 50% |

### artifacts/swarm/

WAV files generated using swarm predictions for:
- cat, dog, mushroom, hatsune, australia, kenyon, chaos, connectome

### Why Do This?

1. **Meme science**: It's a cursed TTS project, so why not?
2. **Biology exploration**: Tests whether compartment specialization can work for classification
3. **Failure is data**: Understanding why the swarm underperforms reveals the value of unified multi-class decision boundaries
4. **Educational**: Demonstrates ensemble methods vs single classifiers

### Freeze Tags

- `v0.1.0-g2p-acceptable`: The baseline single-MB model (don't modify)
- `freeze/g2p-acceptable`: Frozen branch with baseline (don't modify)

### Future Ideas (Not Implemented)

- **Hierarchical swarms**: Specialists for phoneme *categories* (vowels, stops, fricatives) that then dispatch to sub-specialists
- **Shared attention**: Let specialists see top-K votes from other specialists before final decision
- **CMU Arctic integration**: Replace formant crumbs with real phoneme audio units (would require downloading corpora — not done to keep the project lightweight)

---

## References

- [FlyWire Hiragana OCR Demo](https://hae.satoru.net/) — The inspiration
- [FlyWire](https://flywire.ai/) — Complete fruit fly brain connectome
- [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) — Pronunciation dictionary
- [Hige et al. 2015](https://doi.org/10.1016/j.neuron.2015.04.027) — Dopamine plasticity in MB
- [Handler et al. 2019](https://doi.org/10.1038/s41593-019-0435-7) — Timing-dependent plasticity

## License

MIT
