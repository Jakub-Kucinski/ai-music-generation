import copy
import os
import random

from music21 import corpus, stream
from tqdm import tqdm

# Search for Bach scores in the corpus.
metadata_bundle = corpus.search(composer="bach")

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

# Build a pool of parts from all Bach scores.
parts_pool = []
for metadata in tqdm(metadata_bundle):
    if not metadata.sourcePath.stem[:3].startswith("bwv") or metadata.sourcePath.stem in chorals_to_omit:
        continue
    try:
        score = metadata.parse()
        # Extend the pool with all parts from the score.
        parts_pool.extend(score.parts)
    except Exception as e:
        print(f"Error parsing score {metadata}: {e}")

# Parameters for sampling and file generation.
n_parts_to_sample = 4
num_files_to_create = 100

# Create the output directory if it does not exist.
output_dir = "data/04_generated/music21_bach/sampled_bach_4_parts/midi/"
os.makedirs(output_dir, exist_ok=True)

# Generate new scores by sampling parts and writing them as MIDI files.
for i in tqdm(range(num_files_to_create)):
    n_tries = 10
    for _ in range(n_tries):
        try:
            # Randomly sample n_parts_to_sample parts from the parts pool.
            sampled_parts = random.sample(parts_pool, n_parts_to_sample)
            min_part_length = min(len(part) for part in sampled_parts)

            # Create a new score and add each sampled part.
            new_score = stream.Score()
            for part in sampled_parts:
                new_score.append(stream.Part(copy.deepcopy(part[:min_part_length])))

            # Define the output file path.
            output_path = os.path.join(output_dir, f"file_{i+1}.mid")
            # Write the new score as a MIDI file.
            new_score.write("midi", fp=output_path)
            # print(f"Saved {output_path}")
            break
        except Exception:
            pass
