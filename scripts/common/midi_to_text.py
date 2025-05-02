import os
from pathlib import Path

from tqdm import tqdm

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.encodings.quantized_converter import (
    MidiQuantizedConverter,
)

# Define paths
midi_dir = "data/03_converted/irishman/train_leadsheet/midi/abc2midi"
result_dir = "data/03_converted/irishman/train_leadsheet/midi_texts"
# midi_dir = "data/03_converted/irishman/validation_leadsheet/midi/abc2midi"
# result_dir = "data/03_converted/irishman/validation_leadsheet/midi_texts"
# midi_dir = "data/03_converted/music21_bach/train/midi"
# result_dir = "data/03_converted/music21_bach/train/midi_texts"
# midi_dir = "data/03_converted/music21_bach/validation/midi"
# result_dir = "data/03_converted/music21_bach/validation/midi_texts"

# Create result directory if it doesn't exist
os.makedirs(result_dir, exist_ok=True)

# Set up the converter
settings = EncodingSetting()
converter = MidiQuantizedConverter(settings)

# Loop through each file in the directory
for filename in tqdm(os.listdir(midi_dir)):
    # Check for MIDI file extensions
    if filename.lower().endswith((".mid", ".midi")):
        midi_path = os.path.join(midi_dir, filename)
        # Convert the single MIDI file to text.
        # Assuming this function returns a dictionary with one key (the file path) and its text.
        try:
            name_to_text_dict = converter.filepath_to_texts(Path(midi_path))
        except Exception as e:
            print(f"Got exception when processing midi file {midi_path}:\n{e}")
            continue

        # Save the text for each file (typically one per conversion)
        for file_path, text in name_to_text_dict.items():
            base_name = os.path.basename(file_path)
            name_without_ext, _ = os.path.splitext(base_name)
            output_filepath = os.path.join(result_dir, name_without_ext + ".txt")
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(text)

print("All MIDI files have been successfully converted to text.")
