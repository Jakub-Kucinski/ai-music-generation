import os

from tqdm import tqdm

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.encodings.quantized_converter import (
    MidiQuantizedConverter,
)

# Define the input directory for text files and the output directory for MIDI files.
text_dir = "data/04_generated/music21_bach_no_offsets_512_context/conditioned_4_bars/midi_texts"
midi_output_dir = "data/04_generated/music21_bach_no_offsets_512_context/conditioned_4_bars/midi"

# Create the output directory if it doesn't exist.
os.makedirs(midi_output_dir, exist_ok=True)

# Set up the converter with appropriate settings.
settings = EncodingSetting(include_offset_in_notes=False)  # IMPORTANT
converter = MidiQuantizedConverter(settings)

# Loop through each file in the text directory.
for filename in tqdm(os.listdir(text_dir)):
    # Process only text files.
    if filename.lower().endswith(".txt"):
        text_path = os.path.join(text_dir, filename)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Convert the text representation back into a Music21 score.
        score = converter.text_to_score(text)

        # Determine the output MIDI file path.
        base_name = os.path.splitext(filename)[0]
        output_filepath = os.path.join(midi_output_dir, base_name + ".mid")

        # Write the score to a MIDI file.
        score.write("midi", fp=output_filepath)

print("All text files have been successfully converted to MIDI files.")
