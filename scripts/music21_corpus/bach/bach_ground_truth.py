import os

from music21 import corpus
from tqdm import tqdm

# Search for Bach scores in the corpus.
metadata_bundle = corpus.search(composer="bach")

# Create the output directory if it does not exist.
output_dir = "data/03_converted/music21_bach/full_dataset/midi/"
os.makedirs(output_dir, exist_ok=True)

defective_chorals = ["bwv299", "bwv315"]
multiple_soprano = [
    "bwv8.6",
    "bwv27.6",
]
multiple_instruments = [
    "bwv19.7",
    "bwv70.11",
    "bwv91.6",
    "bwv112.5-sc",
    "bwv250",
    "bwv251",
    "bwv252",
]
non_standard_rhythm_and_multiple_instruments = [
    "bwv29.8",
    "bwv41.6",
    "bwv248.9-1",
    "bwv248.23-2",
    "bwv248.42-4",
]
chorals_to_omit = (
    defective_chorals + multiple_soprano + multiple_instruments + non_standard_rhythm_and_multiple_instruments
)


for metadata in tqdm(metadata_bundle):
    if not metadata.sourcePath.stem[:3].startswith("bwv") or metadata.sourcePath.stem in chorals_to_omit:
        continue
    score = metadata.parse()
    # Define the output file path.
    output_path = os.path.join(output_dir, f"{metadata.sourcePath.stem}.mid")
    # Write the new score as a MIDI file.
    score.write("midi", fp=output_path)
    # print(f"Saved {output_path}")
