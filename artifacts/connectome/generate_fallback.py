#!/usr/bin/env python3
"""
Generate deterministic fallback PN→KC connectivity matrix.

This creates a connectivity matrix that matches published hemibrain statistics
but uses deterministic random generation when actual connectome extraction
is not practical.

Statistics from Zheng et al. 2020 / Scheffer et al. 2020:
- ~180 PN types
- ~2000 KCs
- 6.8 ± 2.1 PNs per KC on average
- Binary connectivity (≥3 synapses = connected)

Usage:
    python generate_fallback.py
"""

import numpy as np
import hashlib
from pathlib import Path

# Parameters matching hemibrain literature
N_PN = 180          # Approximate number of olfactory PN types
N_KC = 2000         # Kenyon cells (rounded from 1963)
AVG_PN_PER_KC = 6.8 # Mean from Zheng et al.
STD_PN_PER_KC = 2.1 # Std from Zheng et al.
SEED = 20200417     # Paper submission date as seed

def generate_pn_kc_matrix(
    n_pn: int = N_PN,
    n_kc: int = N_KC,
    avg_pn: float = AVG_PN_PER_KC,
    std_pn: float = STD_PN_PER_KC,
    seed: int = SEED,
) -> np.ndarray:
    """
    Generate PN→KC connectivity matrix with hemibrain-like statistics.
    
    Each KC receives input from a variable number of PNs (drawn from
    truncated normal distribution matching published statistics).
    
    Returns:
        Binary matrix (n_pn, n_kc) where 1 = connected
    """
    rng = np.random.default_rng(seed)
    
    matrix = np.zeros((n_pn, n_kc), dtype=np.float32)
    
    for kc in range(n_kc):
        # Sample number of PNs from truncated normal
        n_pns = int(rng.normal(avg_pn, std_pn))
        n_pns = max(3, min(n_pns, 12))  # Clamp to biological range
        
        # Select random PNs
        connected_pns = rng.choice(n_pn, size=n_pns, replace=False)
        matrix[connected_pns, kc] = 1.0
    
    return matrix


def main():
    print("Generating hemibrain-derived PN→KC fallback matrix...")
    print(f"Parameters: {N_PN} PNs × {N_KC} KCs")
    print(f"Avg PN/KC: {AVG_PN_PER_KC} ± {STD_PN_PER_KC}")
    print(f"Seed: {SEED}")
    
    matrix = generate_pn_kc_matrix()
    
    # Compute statistics
    sparsity = np.mean(matrix > 0)
    avg_pn_per_kc = np.mean(np.sum(matrix > 0, axis=0))
    std_pn_per_kc = np.std(np.sum(matrix > 0, axis=0))
    
    # Compute hash for verification
    matrix_hash = hashlib.md5(matrix.tobytes()).hexdigest()[:16]
    
    print(f"\nGenerated matrix:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Sparsity: {sparsity:.4f}")
    print(f"  Avg PN/KC: {avg_pn_per_kc:.2f} ± {std_pn_per_kc:.2f}")
    print(f"  Hash: {matrix_hash}")
    
    # Save
    output_path = Path(__file__).parent / "hemibrain_pn_kc.npz"
    np.savez(
        output_path,
        pn_kc_matrix=matrix,
        metadata={
            'source': 'deterministic_hemibrain_stats',
            'seed': SEED,
            'n_pn': N_PN,
            'n_kc': N_KC,
            'avg_pn_per_kc': float(avg_pn_per_kc),
            'std_pn_per_kc': float(std_pn_per_kc),
            'sparsity': float(sparsity),
            'hash': matrix_hash,
            'reference': 'Zheng et al. 2020, Scheffer et al. 2020',
        }
    )
    
    print(f"\nSaved to: {output_path}")
    return matrix


if __name__ == "__main__":
    main()
