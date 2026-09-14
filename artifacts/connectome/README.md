# Connectome Data for MORE FLY

This directory contains PN→KC connectivity data derived from published fly connectome datasets.

## Sources

### hemibrain v1.0.1
- **Reference**: Scheffer et al. 2020 (eLife 9:e57443)
- **License**: CC-BY
- **URL**: https://neuprint.janelia.org
- **Download**: https://storage.cloud.google.com/hemibrain-release/neuprint/hemibrain_v1.0.1_neo4j_inputs.zip

### PN→KC Statistics (Zheng et al. 2020)
- **Reference**: Zheng et al. 2020 (Current Biology)
- **GitHub**: https://github.com/bocklab/pn_kc
- **Figshare**: https://doi.org/10.6084/m9.figshare.19092242.v1

Key statistics from the literature:
- ~180 olfactory PN types
- ~1963 KCs in hemibrain (one hemisphere)
- Average 6.8 ± 2.1 PNs per KC claw
- ~5% KC activity during odor response
- Binarization threshold: ≥3 synapses per PN-KC pair

## Files

### hemibrain_pn_kc.npz (fallback)
A deterministic connectivity matrix generated using hemibrain statistics:
- Shape: (180 PNs, 2000 KCs)
- Connectivity pattern matches published statistics
- Hash documented for reproducibility
- NOT raw hemibrain data (extraction was impractical in cloud env)

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
