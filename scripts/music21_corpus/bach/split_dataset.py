import os
import random
import shutil


def split_midi_files(source_dir: str, train_dir: str, validation_dir: str, train_ratio: float = 0.9) -> None:
    # Get a list of all MIDI files (.mid or .midi) in the source directory
    midi_files: list[str] = [
        f
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and (f.lower().endswith(".mid") or f.lower().endswith(".midi"))
    ]

    # Shuffle the list to randomize the split
    random.shuffle(midi_files)

    # Calculate the index for the split (90% for training)
    split_index: int = int(len(midi_files) * train_ratio)
    train_files: list[str] = midi_files[:split_index]
    validation_files: list[str] = midi_files[split_index:]

    # Create target directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Copy each file to the appropriate directory
    for file in train_files:
        source_file: str = os.path.join(source_dir, file)
        target_file: str = os.path.join(train_dir, file)
        shutil.copy(source_file, target_file)

    for file in validation_files:
        source_file: str = os.path.join(source_dir, file)
        target_file: str = os.path.join(validation_dir, file)
        shutil.copy(source_file, target_file)

    print(f"Total files: {len(midi_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(validation_files)}")


if __name__ == "__main__":
    # Define paths with type annotations for clarity
    source_directory: str = "data/03_converted/music21_bach/full_dataset/midi"
    train_directory: str = "data/03_converted/music21_bach/train/midi"
    validation_directory: str = "data/03_converted/music21_bach/validation/midi"

    # (Optional) Set a seed for reproducibility
    random.seed(42)

    # Split the MIDI files
    split_midi_files(source_directory, train_directory, validation_directory)
