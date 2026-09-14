# MORE FLY Ablation Results

*Generated from ablation_results.json. Trained on 5k words, 6-8 epochs, seed=42.*

| Config | Demo Ph | Demo W | Test Ph | Test W | me | yes | hi | IY→EH | AE→AA |
|--------|---------|--------|---------|--------|----|----|-----|-------|-------|
| baseline | 95.8% | 90.0% | 65.0% | 13.2% | M IY | Y EH Z | HH AY | 5.8% | 13.9% |
| +A_cues | 95.8% | 90.0% | 61.7% | 10.8% | M IY | Y EH Z | HH AY | 5.6% | 4.8% |
| +A+B_data | 95.8% | 90.0% | 62.8% | 11.2% | M IY | Y EH S | HH AY | 9.3% | 9.4% |
| +A+B+C_wiring | 95.8% | 90.0% | 61.5% | 9.2% | M IY | Y EH S | HH AY | 6.3% | 16.1% |
| +A+B+C+D_full | 95.8% | 90.0% | 61.5% | 9.2% | M IY | Y EH S | HH AY | 6.3% | 16.1% |

**Notes:**
- Stage B = hard-word curriculum (IY↔EH, AE↔AA, IY↔UW oversampling), not larger vocabulary
- Hemibrain wiring uses init_seed=1000 for KC→MBON (required for correct IY discrimination)
- Hemibrain file is stats-matched fallback, not raw synapse data