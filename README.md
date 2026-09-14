# Cursed TTS: G2P Mushroom Body Classifier

<p align="center">
  <img src="assets/flysune-miku.jpg" alt="Flysune Miku - Project Mascot" width="300">
  <br>
  <em>Flysune Miku: Hatsune Miku with a picker swarm in her skull</em>
  <br>
  <small>(Project mascot art. Not affiliated with Crypton Future Media.)</small>
</p>

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

### Architecture: Picker Flies + Singer Flies (MoE)

The swarm has two stages: **picker flies** choose the phoneme sequence, then **singer flies** render each phoneme with personalized voices.

```
    Letter context ("_ca_t__")
            ↓
    ┌───────────────────────────────────────┐
    │   PICKER FLIES (G2P Classification)   │
    │   Shared PN→KC + per-phoneme MBON     │
    └───────────────────────────────────────┘
            ↓ Predicted phoneme sequence
    ┌─────┬─────┬─────┬─────┬─────┐
    │ /K/ │ /AE/│ /T/ │ /D/ │ ... │  ← SINGER FLIES
    │ ♪   │ ♪   │ ♪   │ ♪   │     │  (personalized voices)
    │F0=119│F0=194│F0=112│F0=113│   │
    └──┬──┴──┬──┴──┬──┴──┬──┴─────┘
       │     │     │     │
       ↓     ↓     ↓     ↓
    Concatenate audio crumbs → Final WAV
```

**Picker flies** (specialist classification):
- Shared input layer: Same PN→KC expansion (efficiency)
- Private output weights: KC→MBON with 2 outputs (YES/NO)
- Learning rule: Dopamine-when-wrong (anti-Hebbian)

**Singer flies** (personalized synthesis):
- Each phoneme has its own "voice" with distinct characteristics
- F0 (pitch): 100-200 Hz range, spread across phonemes
- Formant shift: 0.9-1.2x (brighter/darker timbres)
- Vibrato: Some vowels get subtle pitch modulation
- Breathiness: Fricatives get more noise, vowels less
- Duration scaling: Diphthongs longer, stops shorter

### Quick Start (Swarm)

```bash
# Train a subset of specialists (phonemes needed for demo words)
python -m cursed_tts train-swarm --phones demo --epochs 8

# Train all 39 specialists (full ARPAbet) with softmax voting
python -m cursed_tts train-swarm --epochs 10 --vote softmax

# Speak using the fly swarm
python -m cursed_tts speak mushroom --swarm

# Speak with a specific voting strategy
python -m cursed_tts speak mushroom --swarm --vote margin

# Generate all swarm demo WAVs
python -m cursed_tts speak-all --swarm

# Compare swarm vs baseline accuracy
python -m cursed_tts eval --swarm
```

### Voting Strategies (Arbitrator)

The swarm uses a configurable **arbitrator** to combine specialist votes:

| Strategy | Flag | Description |
|----------|------|-------------|
| **softmax** | `--vote softmax` | Softmax over YES scores (default, most robust) |
| **margin** | `--vote margin` | Require margin between best/second; fallback to softmax |
| **argmax** | `--vote argmax` | Raw argmax (original, can thrash when specialists overconfident) |

```bash
# Train with softmax voting (recommended)
python -m cursed_tts train-swarm --vote softmax --vote-temp 0.5

# Train with margin voting
python -m cursed_tts train-swarm --vote margin --vote-margin 0.1

# Override at inference
python -m cursed_tts speak mushroom --swarm --vote margin
```

**Why this matters**: Raw argmax voting can degrade as training continues — specialists get overconfident on wrong phonemes. Softmax/margin voting provides more robust ensemble decisions.

### Best Checkpoint & Early Stopping

Training automatically tracks demo_phoneme accuracy and saves the best weights:

```bash
# Train with early stopping (stop if no improvement for 5 epochs)
python -m cursed_tts train-swarm --patience 5

# Disable best checkpoint saving
python -m cursed_tts train-swarm --no-save-best
```

Output files:
- `model_swarm.npz` — Final weights (may be degraded on long runs)
- `model_swarm_best.npz` — Best weights by demo_phoneme accuracy

**Recommendation**: Use `model_swarm_best.npz` for inference after long training runs.

### Confusion Mining / Hard Negatives (Step 4)

Train with attention to confusable phoneme pairs — learn more when wrong on hard pairs:

```bash
# Train with confusion mining (mine confusions after epoch 2, oversample hard pairs)
python -m cursed_tts train-swarm --confusion-mine --epochs 6 --patience 3

# Custom parameters
python -m cursed_tts train-swarm \
    --confusion-mine \
    --confusion-mine-epoch 2 \
    --hard-negative-weight 2.0 \
    --oversample-factor 1.5 \
    --confusion-report confusion_report.txt

# Use pre-defined known hard pairs without mining (IY↔EH, AE↔AA, etc.)
python -m cursed_tts train-swarm --use-known-hard-pairs
```

**How it works**:

1. **Mining phase** (after epoch N): Build confusion matrix on held-out data to identify frequently confused phoneme pairs (e.g., AA↔AH, IY↔IH, Z↔S)
2. **Oversampling**: Augment training data by duplicating samples involving confusable pairs
3. **Re-training**: Continue training with the augmented data — the dopamine-when-wrong learning rule naturally focuses more on the now-overrepresented hard cases

**Biology analogy**: Selective attention to errors. When a fly brain repeatedly confuses two similar odors, more learning happens on those specific cases.

**Results** (250-word held-out evaluation):

| Metric | Baseline | +Confusion Mining | Delta |
|--------|----------|-------------------|-------|
| Phoneme accuracy | 64.9% | 66.9% | **+2.1%** |
| Word accuracy | 8.4% | 10.8% | **+2.4%** |

**Known confusable pairs** (from CMUdict):
- AA ↔ AH (17.1% confusion rate)
- AE ↔ AH (20.0% confusion rate)
- IY ↔ IH (10.4% confusion rate)
- IY ↔ EH (5.3% - the "me" problem)
- Z ↔ S (14.3% confusion rate)

### Performance

The swarm is an experiment, not an improvement. Expected results:

| Model | Demo Phoneme Acc | Demo Word Acc | Test Phoneme Acc | Test Word Acc |
|-------|------------------|---------------|------------------|---------------|
| Single MB (baseline) | ~100% | ~100% | ~71% | ~24% |
| Fly Swarm (softmax) | ~80-90% | ~50-70% | ~55-65% | ~10-20% |
| Fly Swarm (argmax) | ~70-80% | ~40-60% | ~50-60% | ~5-15% |

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

### Singer Fly Voice Examples

Each phoneme's singer fly has unique voice characteristics:

| Phoneme | F0 (Hz) | Formant Shift | Vibrato | Notes |
|---------|---------|---------------|---------|-------|
| K | 119 | 0.97 | none | Lower, darker stop |
| AE | 194 | 1.12 | 6 Hz | High, bright with vibrato |
| T | 112 | 0.97 | none | Low, crisp stop |
| AO | 120 | 1.01 | 5 Hz | Warm with subtle vibrato |
| M | ~130 | ~1.0 | none | Resonant nasal |
| S | ~140 | ~1.05 | none | Breathy fricative |

The result is a "choir of flies" effect where different phonemes are literally sung by different voices — musically cursed but biologically poetic.

### Future Ideas (Not Implemented)

- **Hierarchical swarms**: Specialists for phoneme *categories* (vowels, stops, fricatives) that then dispatch to sub-specialists
- **Shared attention**: Let specialists see top-K votes from other specialists before final decision
- **UTAU/OpenUTAU voicebank slices**: Replace synthetic formant crumbs with real phoneme audio samples from open voicebanks (see hook in `singer_fly.py`). This would require downloading voicebank files but could provide much higher quality phoneme rendering while keeping the MoE architecture.
- **CMU Arctic integration**: Use diphone/triphone units from CMU Arctic for more natural concatenative synthesis

---

## References

- [FlyWire Hiragana OCR Demo](https://hae.satoru.net/) — The inspiration
- [FlyWire](https://flywire.ai/) — Complete fruit fly brain connectome
- [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) — Pronunciation dictionary
- [Hige et al. 2015](https://doi.org/10.1016/j.neuron.2015.04.027) — Dopamine plasticity in MB
- [Handler et al. 2019](https://doi.org/10.1038/s41593-019-0435-7) — Timing-dependent plasticity

## License

MIT
