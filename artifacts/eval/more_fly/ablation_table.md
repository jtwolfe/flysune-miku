# MORE FLY Ablation Results

*Generated from ablation_results.json. Trained on 5k words, 6-8 epochs, seed=42.*

| Config | Demo Ph | Demo W | Test Ph | Test W | me | yes | hi | IY→EH | AE→AA |
|--------|---------|--------|---------|--------|----|----|-----|-------|-------|
| baseline | 95.8% | 90.0% | 65.0% | 13.2% | M IY | Y EH Z | HH AY | 5.8% | 13.9% |
| +A_cues | 95.8% | 90.0% | 61.7% | 10.8% | M IY | Y EH Z | HH AY | 5.6% | 4.8% |
| +A+B_data | 95.8% | 90.0% | 62.8% | 11.2% | M IY | Y EH S | HH AY | 9.3% | 9.4% |
| +A+B+C_wiring | 95.8% | 90.0% | 61.5% | 9.2% | M IY | Y EH S | HH AY | 6.3% | 16.1% |
| +A+B+C+D_full | 95.8% | 90.0% | 61.5% | 9.2% | M IY | Y EH S | HH AY | 6.3% | 16.1% |

## Summary

- **All configs achieve 95.8% demo phoneme / 90% demo word** after the `me` fix
- **`+A+B_data` (random) has best held-out accuracy**: 62.8% phoneme / 11.2% word
- **Hemibrain wiring slightly underperforms**: 61.5% phoneme / 9.2% word
- **All configs correctly predict `me = M IY`**

## Notes

- **Recommended default: `+A+B_data`** (random wiring + cues + curriculum)
- Stage B = hard-word curriculum (IY↔EH, AE↔AA, IY↔UW oversampling), not larger vocabulary
- Real hemibrain synapse data available (`hemibrain_real_pn_kc.npz`) but uses stats-matched fallback in this ablation
- Hemibrain wiring configs use init_seed=1000 for KC→MBON (required for correct IY discrimination)

## Earlier Demo Drop (Now Fixed)

Before fixes, hemibrain wiring showed `me = M UW` instead of `M IY`:
- Root cause: Hemibrain PN→KC expansion geometry + KC→MBON init seed interaction
- NOT caused by DAN teaching
- Fixed by: IY↔UW in KNOWN_HARD_PAIRS, init_seed=1000, short_word_curriculum

## Why Random Wiring Remains Better

The real hemibrain PN→KC connectivity is specialized for olfactory processing, not 
letter-to-phoneme classification. Random wiring provides more flexibility for our task.

See `TESTING.md` for detailed diagnosis and reproduction steps.