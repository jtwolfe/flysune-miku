# Connectome Data for MORE FLY

This directory contains PN→KC connectivity data from published fly connectome datasets.

## Files

### hemibrain_real_pn_kc.npz (REAL SYNAPSE DATA)

**Real PN→KC connectivity matrix extracted from hemibrain v1.2:**
- Source: Janelia hemibrain compact adjacencies (traced-total-connections.csv)
- Shape: (428 PNs, 1927 KCs)
- Binarized with ≥3 synapse threshold (per Zheng et al. 2020)
- Mean 5.5 PN inputs per KC (binarized)
- **REAL synapse data** from traced neurons

### hemibrain_pn_kc.npz (stats-matched fallback)

Deterministic fallback matrix for when real data isn't available:
- Shape: (180 PNs, 2000 KCs)
- Matches published statistics but not real synapses
- Generated with seed=42 for reproducibility

### hemibrain_pn_metadata.csv / hemibrain_kc_metadata.csv

Neuron metadata from hemibrain v1.2:
- bodyId, type, instance for PNs and KCs

## Data Source

### hemibrain v1.2 Compact Adjacencies
- **Reference**: Scheffer et al. 2020 (eLife 9:e57443)
- **License**: CC-BY
- **Downloaded via**: fruitloops Python package
- **Archive**: exported-traced-adjacencies-v1.2.tar.gz (44MB)
- **Files**: traced-neurons.csv, traced-total-connections.csv

### PN→KC Statistics (Zheng et al. 2020)
- **Reference**: Zheng et al. 2020 (Current Biology)
- **GitHub**: https://github.com/bocklab/pn_kc
- Key statistics: ~180 PN types, ~2000 KCs, 6.8 ± 2.1 PNs per KC claw

## Usage Note

**Random wiring remains the recommended default** for train-more-fly because:
1. Real hemibrain is specialized for olfaction, not letter-to-phoneme
2. Random wiring achieves better accuracy on our task (95.8% vs 87.5% demo)
3. Real hemibrain requires n_pn=400 to preserve connectivity structure

To experiment with real hemibrain wiring:
```bash
python -m cursed_tts train-more-fly --config +A+B+C_wiring --wiring flywire --epochs 8
```

### generate_fallback.py
Script to regenerate the fallback matrix with documented seed.

## Usage

```python
from cursed_tts.more_fly import load_hemibrain_connectivity

# Try to load real data, falls back to deterministic random
matrix = load_hemibrain_connectivity()
```

## Data Extraction Notes

The full hemibrain v1.0.1 dataset is ~47GB uncompressed and requires significant 
processing to extract just the PN→KC connections. Key files in the dataset:
- `Neuprint_Neurons_*.csv` (8GB) - neuron metadata including cell types
- `Neuprint_Neuron_Connections_*.csv` (4.8GB) - connection weights

For full extraction, use:
- neuPrint API (https://neuprint.janelia.org)
- hemibrainr R package (https://github.com/flyconnectome/hemibrainr)
- Python CAVE/fafbseg tools

## License

Hemibrain data is licensed under CC-BY. Please cite:
Scheffer et al., "A connectome and analysis of the adult Drosophila central brain", 
eLife 2020;9:e57443
