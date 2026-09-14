"""
Optional visualization of mushroom body activity.

Requires matplotlib: pip install matplotlib

Usage:
    from cursed_tts.viz import plot_activity
    plot_activity(model, "cat", output_path="activity.png")
"""

import numpy as np
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .mushroom_body import MushroomBody
from .lexicon import get_phonemes
from .phonemes import index_to_phoneme, NUM_PHONEMES, PHONEME_LIST


def plot_activity(
    model: MushroomBody,
    word: str,
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[object]:
    """
    Plot KC and MBON activity for a word.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None
    
    expected = get_phonemes(word.lower())
    n_positions = len(expected)
    
    # Collect activations
    kc_activities = []
    mbon_inputs = []
    predictions = []
    
    for pos in range(n_positions):
        pred_idx, activations = model.forward(word.lower(), pos)
        kc_activities.append(activations['kc_activity'])
        mbon_inputs.append(activations['mbon_input'])
        predictions.append(index_to_phoneme(pred_idx))
    
    # Create figure
    fig, axes = plt.subplots(2, n_positions, figsize=(4 * n_positions, 8))
    
    if n_positions == 1:
        axes = axes.reshape(-1, 1)
    
    for pos in range(n_positions):
        # KC activity
        ax_kc = axes[0, pos]
        kc = kc_activities[pos]
        
        side = int(np.ceil(np.sqrt(len(kc))))
        kc_padded = np.zeros(side * side)
        kc_padded[:len(kc)] = kc
        kc_grid = kc_padded.reshape(side, side)
        
        ax_kc.imshow(kc_grid, cmap='Greys', aspect='auto')
        n_active = int(kc.sum())
        ax_kc.set_title(f"KC (pos {pos})\n{n_active}/{len(kc)} active")
        ax_kc.set_xticks([])
        ax_kc.set_yticks([])
        
        # MBON inputs
        ax_mbon = axes[1, pos]
        mbon = mbon_inputs[pos]
        
        colors = ['green' if PHONEME_LIST[i] == expected[pos] else 
                  ('red' if PHONEME_LIST[i] == predictions[pos] and 
                   predictions[pos] != expected[pos] else 'steelblue')
                  for i in range(NUM_PHONEMES)]
        
        bars = ax_mbon.barh(range(NUM_PHONEMES), mbon, color=colors)
        ax_mbon.set_yticks(range(0, NUM_PHONEMES, 3))
        ax_mbon.set_yticklabels([PHONEME_LIST[i] for i in range(0, NUM_PHONEMES, 3)], fontsize=7)
        ax_mbon.invert_yaxis()
        ax_mbon.set_xlabel('MBON Input')
        
        winner_idx = np.argmin(mbon)
        match_str = "✓" if predictions[pos] == expected[pos] else "✗"
        ax_mbon.set_title(f"MBON: {predictions[pos]} {match_str}\n(expect {expected[pos]})")
        
        bars[winner_idx].set_edgecolor('black')
        bars[winner_idx].set_linewidth(2)
    
    fig.suptitle(f"Mushroom Body Activity: '{word}'", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    from .mushroom_body import MushroomBody
    
    model = MushroomBody.load("model.npz")
    plot_activity(model, "cat", output_path="artifacts/activity_cat.png", show=False)
