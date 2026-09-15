#!/usr/bin/env python3
"""
Fetch and prepare MARIAN ILUSTRADO voicebank samples.

MARIAN ILUSTRADO is an English ARPAsing UTAU voicebank by Kanabun.
Download: https://downloadmarian.carrd.co/

This script provides utilities for:
1. Downloading the voicebank from the official source
2. Extracting and organizing samples
3. Creating a minimal crumb pack with essential phonemes

Usage:
    python scripts/fetch_marian.py --help
    python scripts/fetch_marian.py download
    python scripts/fetch_marian.py extract /path/to/downloaded.zip
    python scripts/fetch_marian.py crumb-pack --output data/marian_crumbs

License:
    MARIAN voicebank terms permit redistribution with attribution.
    See README/NOTICE for full terms.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Set, Dict, Optional
import urllib.request
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cursed_tts.voicebanks.arpasing_map import (
        ARPABET_TO_ARPASING, arpabet_to_arpasing,
    )
    from cursed_tts.voicebanks.utau_singer import (
        load_oto_ini, find_voicebank_samples,
    )
except ImportError:
    print("Warning: Could not import voicebank modules. Running in standalone mode.")
    ARPABET_TO_ARPASING = {}


# ============================================================================
# Constants
# ============================================================================

MARIAN_INFO = {
    'name': 'MARIAN ILUSTRADO',
    'author': 'Kanabun',
    'download_page': 'https://downloadmarian.carrd.co/',
    'arpasing_directory': 'https://arpasing.tubs.wtf/en/directories/voicebanks',
    'license': 'Free to use with attribution',
}

# Essential phonemes for basic English coverage
# These are the minimum needed for demo words
ESSENTIAL_PHONEMES = [
    # Vowels
    'AA', 'AE', 'AH', 'AO', 'EH', 'ER', 'IH', 'IY', 'UH', 'UW',
    # Diphthongs
    'AW', 'AY', 'EY', 'OW', 'OY',
    # Stops
    'P', 'B', 'T', 'D', 'K', 'G',
    # Fricatives
    'F', 'V', 'S', 'Z', 'SH', 'HH',
    # Nasals
    'M', 'N', 'NG',
    # Liquids
    'L', 'R',
    # Semivowels
    'W', 'Y',
    # Affricates
    'CH', 'JH',
]

# Default output directory
DEFAULT_OUTPUT = Path(__file__).parent.parent / 'data' / 'marian_crumbs'


# ============================================================================
# Download Utilities
# ============================================================================

def print_download_instructions():
    """Print manual download instructions."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           MARIAN ILUSTRADO Download Instructions                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  MARIAN ILUSTRADO is an English ARPAsing voicebank by Kanabun.  ║
║                                                                  ║
║  To download:                                                    ║
║                                                                  ║
║  1. Visit: {MARIAN_INFO['download_page']:<42} ║
║                                                                  ║
║  2. Scroll down to "Ilustrado SERIES"                           ║
║                                                                  ║
║  3. Click the download button (MediaFire link)                  ║
║                                                                  ║
║  4. Extract the ZIP file                                        ║
║                                                                  ║
║  5. Run this script again with the extracted path:              ║
║     python scripts/fetch_marian.py extract /path/to/marian      ║
║                                                                  ║
║  Or create a minimal crumb pack:                                ║
║     python scripts/fetch_marian.py crumb-pack --input /path     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Alternatively, you can find the voicebank on the ARPAsing directory:
  {MARIAN_INFO['arpasing_directory']}

License: {MARIAN_INFO['license']}
""")


def download_marian(output_dir: Path) -> bool:
    """
    Attempt to download MARIAN ILUSTRADO.
    
    Note: MediaFire direct links are not easily scripted.
    This function provides instructions for manual download.
    """
    print_download_instructions()
    
    print("\nAutomatic download is not supported for MediaFire links.")
    print("Please download manually and use the 'extract' or 'crumb-pack' command.")
    return False


# ============================================================================
# Extraction and Organization
# ============================================================================

def extract_voicebank(source: Path, output_dir: Path) -> bool:
    """
    Extract and organize a downloaded voicebank.
    
    Args:
        source: Path to ZIP file or extracted directory
        output_dir: Where to place organized samples
    
    Returns:
        True if successful
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If source is a ZIP, extract it first
    if source.suffix.lower() == '.zip':
        print(f"Extracting {source}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(source, 'r') as zf:
                zf.extractall(tmpdir)
            
            # Find the voicebank root (contains oto.ini or WAV files)
            tmppath = Path(tmpdir)
            vb_root = find_voicebank_root(tmppath)
            
            if vb_root:
                return copy_voicebank(vb_root, output_dir)
            else:
                print("Error: Could not find voicebank files in ZIP")
                return False
    
    elif source.is_dir():
        vb_root = find_voicebank_root(source)
        if vb_root:
            return copy_voicebank(vb_root, output_dir)
        else:
            print("Error: Could not find voicebank files in directory")
            return False
    
    else:
        print(f"Error: Unknown source type: {source}")
        return False


def find_voicebank_root(path: Path) -> Optional[Path]:
    """Find the root directory of a voicebank."""
    # Look for oto.ini first
    oto_files = list(path.rglob('oto.ini'))
    if oto_files:
        return oto_files[0].parent
    
    # Look for WAV files
    wav_files = list(path.rglob('*.wav'))
    if wav_files:
        # Find common parent
        parents = set(f.parent for f in wav_files)
        if len(parents) == 1:
            return parents.pop()
        # Return the path with most WAV files
        return max(parents, key=lambda p: len(list(p.glob('*.wav'))))
    
    return None


def copy_voicebank(source: Path, output_dir: Path) -> bool:
    """Copy voicebank files to output directory."""
    print(f"Copying voicebank from {source} to {output_dir}")
    
    # Copy all WAV files and oto.ini
    count = 0
    for f in source.iterdir():
        if f.suffix.lower() in ['.wav', '.ini', '.txt', '.md']:
            shutil.copy2(f, output_dir / f.name)
            count += 1
    
    print(f"Copied {count} files")
    
    # Create attribution file
    create_attribution_file(output_dir)
    
    return count > 0


def create_attribution_file(output_dir: Path):
    """Create NOTICE file with attribution."""
    notice_content = f"""MARIAN ILUSTRADO Voicebank Attribution
=====================================

Voicebank: {MARIAN_INFO['name']}
Author: {MARIAN_INFO['author']}
Download: {MARIAN_INFO['download_page']}

Terms of Use (Summary):
- Attribution required: State the name "MARIAN" when publishing
- Commercial use allowed with proper credit
- Editing and redistribution allowed
- No NFT/cryptocurrency usage
- No illegal, bigoted, or explicit underage content

Full terms available at: {MARIAN_INFO['download_page']}

This crumb pack contains a subset of phoneme samples for integration
with the flysune-miku cursed TTS project.
"""
    
    (output_dir / 'NOTICE.txt').write_text(notice_content)
    print(f"Created attribution file: {output_dir / 'NOTICE.txt'}")


# ============================================================================
# Crumb Pack Creation
# ============================================================================

def create_crumb_pack(
    input_dir: Path,
    output_dir: Path,
    phonemes: Optional[List[str]] = None,
) -> bool:
    """
    Create a minimal crumb pack with essential phoneme samples.
    
    This extracts only the WAV files needed for the specified phonemes,
    creating a compact redistributable package.
    
    Args:
        input_dir: Full voicebank directory
        output_dir: Where to create crumb pack
        phonemes: List of ARPAbet phonemes to include (None = essentials)
    
    Returns:
        True if successful
    """
    if phonemes is None:
        phonemes = ESSENTIAL_PHONEMES
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating crumb pack with {len(phonemes)} phonemes...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Find available samples in input
    samples = find_voicebank_samples(input_dir)
    if not samples:
        print("Error: No samples found in input directory")
        return False
    
    print(f"Found {len(samples)} samples in voicebank")
    
    # Copy samples for each phoneme
    copied = []
    missing = []
    
    for phoneme in phonemes:
        arpasing = arpabet_to_arpasing(phoneme)
        
        if arpasing in samples:
            wav_path, oto_entry = samples[arpasing]
            dest_path = output_dir / wav_path.name
            
            # Copy the WAV file
            if not dest_path.exists():
                shutil.copy2(wav_path, dest_path)
            
            copied.append((phoneme, arpasing, wav_path.name))
        else:
            missing.append((phoneme, arpasing))
    
    print(f"\nCopied {len(copied)} samples:")
    for phoneme, arpasing, filename in copied:
        print(f"  {phoneme:4} ({arpasing:4}): {filename}")
    
    if missing:
        print(f"\nMissing {len(missing)} samples:")
        for phoneme, arpasing in missing:
            print(f"  {phoneme:4} ({arpasing:4})")
    
    # Copy oto.ini if present
    oto_path = input_dir / 'oto.ini'
    if oto_path.exists():
        # Filter oto.ini to only include our samples
        create_filtered_oto(input_dir, output_dir, {c[1] for c in copied})
    
    # Create attribution and manifest
    create_attribution_file(output_dir)
    create_manifest(output_dir, copied, missing)
    
    coverage = 100 * len(copied) / len(phonemes)
    print(f"\nCrumb pack created with {coverage:.1f}% coverage")
    
    return len(copied) > 0


def create_filtered_oto(input_dir: Path, output_dir: Path, aliases: Set[str]):
    """Create filtered oto.ini with only specified aliases."""
    oto_path = input_dir / 'oto.ini'
    if not oto_path.exists():
        return
    
    output_lines = []
    
    # Read and filter oto.ini
    for encoding in ['utf-8', 'shift_jis', 'cp932', 'latin-1']:
        try:
            content = oto_path.read_text(encoding=encoding)
            
            for line in content.splitlines():
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith(';'):
                    output_lines.append(line)
                    continue
                
                # Check if this entry's alias is in our set
                parts = line.split('=')
                if len(parts) >= 2:
                    alias_part = parts[1].split(',')[0].strip().lower()
                    if alias_part in aliases:
                        output_lines.append(line)
            
            break
        except UnicodeDecodeError:
            continue
    
    if output_lines:
        (output_dir / 'oto.ini').write_text('\n'.join(output_lines), encoding='utf-8')
        print(f"Created filtered oto.ini with {len(output_lines)} entries")


def create_manifest(output_dir: Path, copied: List, missing: List):
    """Create a manifest file documenting the crumb pack."""
    manifest = {
        'voicebank': MARIAN_INFO['name'],
        'author': MARIAN_INFO['author'],
        'download_url': MARIAN_INFO['download_page'],
        'included_phonemes': [c[0] for c in copied],
        'missing_phonemes': [m[0] for m in missing],
        'coverage_pct': 100 * len(copied) / (len(copied) + len(missing)),
    }
    
    import json
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch and prepare MARIAN ILUSTRADO voicebank samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s download
      Print download instructions and open browser
  
  %(prog)s extract /path/to/marian.zip -o data/marian_crumbs
      Extract voicebank from ZIP file
  
  %(prog)s crumb-pack --input /path/to/full/voicebank
      Create minimal crumb pack with essential phonemes

{MARIAN_INFO['name']} by {MARIAN_INFO['author']}
Download: {MARIAN_INFO['download_page']}
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Download command
    dl_parser = subparsers.add_parser('download', help='Show download instructions')
    
    # Extract command
    ext_parser = subparsers.add_parser('extract', help='Extract voicebank from ZIP')
    ext_parser.add_argument('source', type=Path, help='Path to ZIP or directory')
    ext_parser.add_argument('-o', '--output', type=Path, default=DEFAULT_OUTPUT,
                          help='Output directory')
    
    # Crumb pack command
    crumb_parser = subparsers.add_parser('crumb-pack', help='Create minimal crumb pack')
    crumb_parser.add_argument('--input', type=Path, required=True,
                             help='Full voicebank directory')
    crumb_parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                             help='Output directory for crumb pack')
    crumb_parser.add_argument('--all', action='store_true',
                             help='Include all phonemes (not just essentials)')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        print_download_instructions()
    
    elif args.command == 'extract':
        if not args.source.exists():
            print(f"Error: Source not found: {args.source}")
            sys.exit(1)
        
        success = extract_voicebank(args.source, args.output)
        sys.exit(0 if success else 1)
    
    elif args.command == 'crumb-pack':
        if not args.input.exists():
            print(f"Error: Input directory not found: {args.input}")
            sys.exit(1)
        
        phonemes = None if args.all else ESSENTIAL_PHONEMES
        success = create_crumb_pack(args.input, args.output, phonemes)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        print_download_instructions()


if __name__ == '__main__':
    main()
