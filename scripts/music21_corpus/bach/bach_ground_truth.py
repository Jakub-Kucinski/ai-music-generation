import os

from music21 import corpus
from tqdm import tqdm

# Search for Bach scores in the corpus.
metadata_bundle = corpus.search(composer="bach")

# Create the output directory if it does not exist.
output_dir = "data/03_converted/music21_bach/midi/"
os.makedirs(output_dir, exist_ok=True)

for metadata in tqdm(metadata_bundle):
    if (
        not metadata.sourcePath.stem[:3].startswith("bwv")
        or metadata.sourcePath.stem == "bwv299"
        or metadata.sourcePath.stem == "bwv315"
    ):
        continue
    score = metadata.parse()
    # Define the output file path.
    output_path = os.path.join(output_dir, f"{metadata.sourcePath.stem}.mid")
    # Write the new score as a MIDI file.
    score.write("midi", fp=output_path)
    # print(f"Saved {output_path}")
