import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.encodings.quantized_converter import (
    MidiQuantizedConverter,
)

# Define paths
midi_dir = "data/03_converted/irishman/train_leadsheet/midi/abc2midi"
result_dir = "data/03_converted/irishman/train_leadsheet/midi_texts_no_offsets"
# midi_dir = "data/03_converted/irishman/validation_leadsheet/midi/abc2midi"
# result_dir = "data/03_converted/irishman/validation_leadsheet/midi_texts_no_offsets"
# midi_dir = "data/03_converted/music21_bach/train/midi"
# result_dir = "data/03_converted/music21_bach/train/midi_texts_no_offsets"
# midi_dir = "data/03_converted/music21_bach/validation/midi"
# result_dir = "data/03_converted/music21_bach/validation/midi_texts_no_offsets"

# Create result directory if it doesn't exist
os.makedirs(result_dir, exist_ok=True)

# Globals for worker processes
_GLOBAL_MIDI_DIR: Path = Path(midi_dir)
_GLOBAL_RESULT_DIR: Path = Path(result_dir)
converter: Optional[MidiQuantizedConverter] = None  # will be initialized in each worker


def init_worker() -> None:
    """
    Initializer for each Pool worker: create one converter instance per process.
    """
    global converter
    settings: EncodingSetting = EncodingSetting(include_offset_in_notes=False)
    converter = MidiQuantizedConverter(settings)


def process_file(filename: str) -> None:
    """
    Worker function: convert a single MIDI file to text and write out .txt.
    """
    if not filename.lower().endswith((".mid", ".midi")):
        return

    midi_path: Path = _GLOBAL_MIDI_DIR / filename
    try:
        name_to_text_dict: dict[str, str] = converter.filepath_to_texts(midi_path)  # type: ignore
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return

    for relative_name, text in name_to_text_dict.items():
        stem: str = Path(relative_name).stem
        out_path: Path = _GLOBAL_RESULT_DIR / f"{stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    # List and filter MIDI filenames once
    all_files: List[str] = [f for f in os.listdir(midi_dir) if f.lower().endswith((".mid", ".midi"))]

    # Spawn a pool of workers
    num_workers: int = cpu_count()
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        # imap_unordered gives us finished-as-they-come; total for tqdm
        for _ in tqdm(
            pool.imap_unordered(process_file, all_files),
            total=len(all_files),
            desc="Converting MIDI → text",
            unit="file",
        ):
            pass

    print("All MIDI files have been successfully converted to text.")
