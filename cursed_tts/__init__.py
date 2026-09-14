"""
Cursed TTS: A mushroom-body phoneme picker text-to-speech toy.

Inspired by the FlyWire hiragana OCR demo at https://hae.satoru.net/
This implements a simplified mushroom body model where:
- Kenyon cells provide sparse encoding of word/phoneme context
- MBON compartments compete to identify phonemes (weakest input wins)
- Dopamine-style plasticity learns the mapping when wrong

Stage 1: Phoneme picker with canned audio concatenation.
"""

__version__ = "0.1.0"
