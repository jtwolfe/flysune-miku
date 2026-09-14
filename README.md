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
# Train with confusion mining (mine confusions after epoch 2)
python -m cursed_tts train-swarm --confusion-mine --epochs 6 --patience 3

# With confusion report
python -m cursed_tts train-swarm --confusion-mine --confusion-report confusion.txt

# Use pre-defined known hard pairs without mining (IY↔EH, AE↔AA, etc.)
python -m cursed_tts train-swarm --use-known-hard-pairs
```

**How it works**:

1. **Mining phase** (after epoch N): Build confusion matrix on held-out data to identify frequently confused phoneme pairs (e.g., AA↔AH, IY↔IH, Z↔S)
2. **Target-only oversampling**: Augment training data by duplicating samples where `target=T` for a confusion `(T→P)` — NOT when target is P (v2 fix to avoid flooding with common phonemes)
3. **Capped at 15%**: Duplicates don't exceed 15% of corpus to avoid overfitting
4. **Position encoding**: 8-dim phoneme slot position features help distinguish slots in short words like "me"

**Biology analogy**: Selective attention to errors. When a fly brain repeatedly confuses two similar odors, more learning happens on those specific cases.

**Results** (250-word held-out evaluation):

| Metric | Baseline | +Mining | Delta |
|--------|----------|---------|-------|
| Test pair accuracy | 64.9% | 65.8% | **+0.9%** |
| Held-out phoneme | 65.8% | 66.3% | **+0.5%** |
| `me` correct? | NO (M EH) | **YES (M IY)** | ✓ |

**Confusion rate improvements**:

| Pair | Baseline | +Mining | Note |
|------|----------|---------|------|
| IY→EH | 6.6% | 5.1% | "me" problem fixed |
| AE→AA | 9.3% | **2.5%** | -6.7% big win! |
| IY→IH | 10.8% | 7.9% | -2.9% |
| AE→AH | 22.3% | 19.9% | -2.4% |

**Hard fraction**: 45.1% of samples marked hard (capped at 15% oversample)

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
- **CMU Arctic integration**: Use diphone/triphone units from CMU Arctic for more natural concatenative synthesis

---

## MARIAN ILUSTRADO Voice Integration

*"Real voice samples instead of synthetic formants"*

The project now supports **MARIAN ILUSTRADO**, an English ARPAsing UTAU voicebank by Kanabun, as an optional audio source for singer flies. This replaces (or supplements) the synthetic formant synthesis with real recorded phoneme samples.

**Voice characteristics** (per [ARPAsing directory](https://arpasing.tubs.wtf/en/directories/voicebanks)):
- **Type**: Masculine, soft/shy
- **Range**: Approximately C#5–A2
- **Monopitch** voicebank

### Quick Start (MARIAN)

```bash
# Speak with MARIAN voice (requires voicebank download)
python -m cursed_tts speak mushroom --voice marian

# Speak all demo words with MARIAN
python -m cursed_tts speak-all --voice marian

# Specify custom voicebank path
python -m cursed_tts speak cat --voice marian --voicebank /path/to/marian
```

### Getting the MARIAN Voicebank

MARIAN ILUSTRADO is free to use with attribution. To download:

1. Visit: https://downloadmarian.carrd.co/
2. Scroll to "Ilustrado SERIES"
3. Download from MediaFire
4. Extract to `data/marian_crumbs/` or `~/.cursed_tts/voicebanks/marian/`

Or use the fetch script:

```bash
# Show download instructions
python scripts/fetch_marian.py download

# After manual download, extract
python scripts/fetch_marian.py extract /path/to/downloaded.zip

# Create minimal crumb pack (essential phonemes only)
python scripts/fetch_marian.py crumb-pack --input /path/to/full/voicebank
```

### ARPAbet ↔ Arpasing Mapping

The integration includes a complete mapping between CMUdict ARPAbet (uppercase, e.g., `AA`, `AE`, `IY`) and Arpasing aliases (lowercase, e.g., `aa`, `ae`, `iy`). Fallback chains handle missing phonemes:

| ARPAbet | Arpasing | If Missing, Fallback To |
|---------|----------|------------------------|
| AA | aa | AO, AH |
| AE | ae | EH, AH |
| TH | th | F, S |
| ZH | zh | Z, JH |
| ... | ... | (see arpasing_map.py) |

### Coverage

With full MARIAN ILUSTRADO voicebank: **100% direct coverage** (all 39 ARPAbet phonemes).

Without voicebank: Falls back to formant synthesis for missing samples.

### Attribution

When using MARIAN voice samples, credit is required:

> Voice: MARIAN ILUSTRADO by Kanabun
> https://downloadmarian.carrd.co/

See `NOTICE` file for full license terms.

### Technical Details

The MARIAN integration maintains the picker/singer fly architecture:

1. **Picker flies** (G2P classification): Choose phoneme sequence (unchanged)
2. **MARIAN singer flies**: Render each phoneme using real WAV samples
3. **Fallback**: Formant synthesis for unavailable phonemes

```python
from cursed_tts.voicebanks import MarianSingerSwarm

# Create MARIAN swarm
swarm = MarianSingerSwarm(voicebank_path="data/marian_crumbs")

# Synthesize a word
audio = swarm.synthesize_sequence(['K', 'AE', 'T'])

# Check coverage
print(swarm.get_coverage_summary())
```

---

## More Fly: Real Wiring & DAN Teaching

*"Upgrade from mushroom-shaped to mushroom-wired"*

The MORE FLY module (`cursed_tts/more_fly.py`) implements biologically-faithful enhancements.

### Recommended Default: `+A+B_data` (Random Wiring + Cues + Curriculum)

```bash
# Train with recommended config (now the default)
python -m cursed_tts train-more-fly --epochs 8
```

This uses:
- **Stage A cues**: Previous-phone, focus features, enhanced position encoding
- **Stage B curriculum**: Hard-word oversampling (IY↔EH, AE↔AA, IY↔UW)
- **Random wiring**: Simpler and matches or exceeds hemibrain on held-out accuracy

### Stage A: Richer Cues

- **Previous-phone cue**: Teacher-forced during training, autoregressive at decode
- **Focus features**: Help short words like "me" maintain distinct slot encodings
- **Position encoding**: Enhanced sinusoidal + discrete slot indicators

### Stage B: Fuller Data

- **Full CMUdict support**: `--words 0` uses all ~117k words
- **Hard-word curriculum**: Fixed 15% oversample of known hard pairs (IY↔EH, AE↔AA, IY↔UW)
- **Fixed seeds**: Reproducible splits documented in training

### Stage C: Real Wiring (Experimental)

⚠️ **Experimental**: Real hemibrain wiring does not outperform random wiring on our task.
Use `+A+B_data` (random) as the recommended default.

**Now with real hemibrain synapse data** extracted from v1.2 compact adjacencies:

- **428 PNs → 1927 KCs** (real traced neurons)
- **~5.5 PN inputs per KC** (binarized with ≥3 synapse threshold)
- **Frozen PN→KC**: Only KC→MBON weights are plastic (fly-faithful)
- Uses n_pn=400 to preserve connectivity structure

```bash
# Train with real hemibrain wiring (experimental)
python -m cursed_tts train-more-fly --config +A+B+C_wiring --wiring flywire --epochs 8
```

**Why random wiring is still better**: The real hemibrain PN→KC connectivity is specialized 
for olfactory processing, not letter-to-phoneme classification. Random wiring provides more 
flexibility for our task (95.8% vs 87.5% demo accuracy).

### Stage D: DAN Teaching

**Compartment-local dopamine** following Hige et al. 2015:

- **Only update responsible specialists**: When wrong, depress KC→MBON for correct class, potentiate for wrong class
- **Per-compartment teaching**: Not whole-brain reward spray
- **KC→MBON only**: PN→KC weights frozen by default

```bash
# Train full MORE FLY (all stages)
python -m cursed_tts train-more-fly --config +A+B+C+D_full --epochs 8
```

### Ablation Results

| Config | Demo Ph | Demo W | Test Ph | Test W | me | yes | hi | IY→EH | AE→AA |
|--------|---------|--------|---------|--------|----|----|-----|-------|-------|
| baseline | 95.8% | 90.0% | 65.0% | 13.2% | M IY | Y EH Z | HH AY | 5.8% | 13.9% |
| +A_cues | 95.8% | 90.0% | 61.7% | 10.8% | M IY | Y EH Z | HH AY | 5.6% | 4.8% |
| +A+B_data | 95.8% | 90.0% | 62.8% | 11.2% | M IY | Y EH S | HH AY | 9.3% | 9.4% |
| +A+B+C_wiring | 95.8% | 90.0% | 61.5% | 9.2% | M IY | Y EH S | HH AY | 6.3% | 16.1% |
| +A+B+C+D_full | 95.8% | 90.0% | 61.5% | 9.2% | M IY | Y EH S | HH AY | 6.3% | 16.1% |

*Trained on 5000 words, 6-8 epochs, seed=42. Full logs in `artifacts/eval/more_fly/`.*

**Note on Stage B**: In this 5k-word ablation, "Stage B" refers to the **hard-word curriculum** 
(IY↔EH, AE↔AA, IY↔UW oversampling), not larger vocabulary. Use `--words 0` for full CMUdict.

**Note on hemibrain wiring (experimental)**: The hemibrain expansion geometry requires a 
different KC→MBON initialization (init_seed=1000) to correctly discriminate IY in short words 
like "me". This is a temporary workaround until real synapse data is available. Hemibrain 
wiring does not outperform random+data on held-out accuracy in current ablations, so 
`+A+B_data` with random wiring is the recommended default.

### CLI Reference

```bash
# Train MORE FLY with recommended default (+A+B_data, random wiring)
python -m cursed_tts train-more-fly --epochs 8

# Train with specific configuration
python -m cursed_tts train-more-fly --config <CONFIG> --wiring <MODE> --epochs N

# Configs: baseline, +A_cues, +A+B_data (default), +A+B+C_wiring, +A+B+C+D_full
# Wiring modes: random (default), flywire, hemibrain (experimental)

# Example: hemibrain wiring experiment
python -m cursed_tts train-more-fly --config +A+B+C_wiring --wiring flywire --epochs 8

# Run all ablations
python -m cursed_tts eval-more-fly --words 5000 --epochs 6
```

### Unit Tests

```bash
# Run MORE FLY tests
python -m pytest tests/test_more_fly.py -v

# Tests verify:
# - Previous-phone cue changes encoding
# - Short word slots differ by position features
# - PN→KC frozen during DAN training
# - KC→MBON changes on errors
# - Wiring hash differs when seed changes
```

### Biology Analogy

```
Input Features (letter context)
         ↓
    PN Layer (~180 "glomeruli")
         ↓ [sparse random, from connectome]
    KC Layer (~2000 Kenyon cells, ~10% active)
         ↓ [plastic, anti-Hebbian]
    MBON Compartments (YES/NO per phoneme)
         ↓
    Winner-take-all → Prediction
         
    DAN Teaching (when wrong):
    - Target compartment: DEPRESS KC→MBON[YES]
    - Wrong compartment: POTENTIATE KC→MBON[NO]
```

---

## References

- [FlyWire Hiragana OCR Demo](https://hae.satoru.net/) — The inspiration
- [FlyWire](https://flywire.ai/) — Complete fruit fly brain connectome
- [Hemibrain Connectome](https://neuprint.janelia.org/) — Janelia FlyEM dataset
- [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) — Pronunciation dictionary
- [Scheffer et al. 2020](https://doi.org/10.7554/eLife.57443) — Hemibrain connectome paper
- [Zheng et al. 2020](https://doi.org/10.1016/j.cub.2022.06.012) — PN-KC structured sampling
- [Hige et al. 2015](https://doi.org/10.1016/j.neuron.2015.04.027) — Dopamine plasticity in MB
- [Handler et al. 2019](https://doi.org/10.1038/s41593-019-0435-7) — Timing-dependent plasticity

## License

MIT
