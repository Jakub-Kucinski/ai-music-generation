#!/usr/bin/env python3
"""
Batch-convert MIDI files to text token sequences using
``ai_music_generation``'s ``MidiQuantizedConverter``.

**How to use**
--------------
1. Edit the *USER-CONFIG* section below - set the paths to your MIDI files and
   where you want the corresponding ``.txt`` outputs written.
2. Toggle ``INCLUDE_OFFSETS`` to choose whether note-offset tokens are emitted.
3. Optionally specify ``TRANSPOSITIONS`` - a list of semitone shifts.  For each
   MIDI file the script will emit *one* text file per transposition.  Use
   ``None`` to disable transposition entirely (equivalent to passing
   ``transpose_pitches_by_n=None``)  or include ``0`` in the list to keep an
   un-transposed copy alongside shifted variants.
4. Run the script directly: ``python convert_midi_texts.py``.
"""

from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.encodings.quantized_converter import (
    MidiQuantizedConverter,
)

# ──────────────────── USER-CONFIG ────────────────────
# Input directory containing .mid/.midi files
MIDI_DIR = Path("data/03_converted/music21_bach/train/midi")

# Output directory for .txt token files
RESULT_DIR = Path("data/03_converted/music21_bach/train/midi_texts_augmented")

# Whether to include note-offset tokens
INCLUDE_OFFSETS = True

# A list of semitone transpositions to apply *per MIDI file*.
# Example: [-2, 0, 2] will create three outputs for each file: shifted down a
# tone, the original (0), and shifted up a tone.  If set to ``None`` the script
# performs **no** transposition (passes ``transpose_pitches_by_n=None``).
TRANSPOSITIONS: list[int] | None = None  # e.g. [-2, 0, 2]  or  None

# Number of parallel worker processes (default: all CPU cores)
NUM_WORKERS = cpu_count()
# ─────────────────────────────────────────────────────

# Globals that each worker process can access (populated in ``main``)
_GLOBAL_MIDI_DIR: Path
_GLOBAL_RESULT_DIR: Path
converter: Optional[MidiQuantizedConverter] = None  # Will be initialized per worker


def _init_worker(include_offsets: bool) -> None:  # noqa: D401
    """Initialise a ``MidiQuantizedConverter`` once per worker process."""
    global converter
    settings = EncodingSetting(include_offset_in_notes=include_offsets)
    converter = MidiQuantizedConverter(settings)


def _process_file(filename: str) -> None:
    """Convert a single MIDI file to one or more text files (with transpositions)."""
    if not filename.lower().endswith((".mid", ".midi")):
        return

    midi_path = _GLOBAL_MIDI_DIR / filename

    transpose_values: list[None] | list[int]
    # Decide which transposition values to apply for this run.
    if not TRANSPOSITIONS:
        transpose_values = [None]
    else:
        # Preserve order; also ensure at least one element present
        transpose_values = TRANSPOSITIONS

    for n in transpose_values:
        try:
            name_to_text: dict[str, str] = converter.filepath_to_texts(  # type: ignore
                midi_path, transpose_pitches_by_n=n
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {midi_path} (transpose={n}): {exc}")
            continue

        suffix = "" if (n == 0 or n is None) else f"_tr{n:+d}"

        for relative_name, text in name_to_text.items():
            stem = Path(relative_name).stem
            out_path = _GLOBAL_RESULT_DIR / f"{stem}{suffix}.txt"
            with open(out_path, "w", encoding="utf-8") as out_file:
                out_file.write(text)


def main() -> None:  # noqa: D401
    """Launch a pool of workers to convert MIDI files."""
    global _GLOBAL_MIDI_DIR, _GLOBAL_RESULT_DIR  # noqa: PLW0603
    _GLOBAL_MIDI_DIR = MIDI_DIR.expanduser()
    _GLOBAL_RESULT_DIR = RESULT_DIR.expanduser()
    os.makedirs(_GLOBAL_RESULT_DIR, exist_ok=True)

    # Gather MIDI filenames once in the master process.
    all_files: List[str] = [f for f in os.listdir(_GLOBAL_MIDI_DIR) if f.lower().endswith((".mid", ".midi"))]

    if not all_files:
        print(f"No MIDI files found in {_GLOBAL_MIDI_DIR!s} - nothing to do.")
        return

    # Spawn worker pool.
    with Pool(processes=NUM_WORKERS, initializer=_init_worker, initargs=(INCLUDE_OFFSETS,)) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_file, all_files),
            total=len(all_files),
            desc="Converting MIDI → text",
            unit="file",
        ):
            pass

    print("All MIDI files have been successfully converted to text.")


if __name__ == "__main__":
    main()
