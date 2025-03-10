import os

from midi2audio import FluidSynth
from music21 import converter
from music21.stream import Opus
from tqdm import tqdm


def convert_midi_to_wav(input_folder: str, output_folder: str, soundfont: str, sample_rate: int = 16_000) -> None:
    """
    Walk through all subdirectories in input_folder looking for MIDI files.
    For each MIDI file, parse it with music21.
    If the parsed object is an Opus (i.e. contains multiple scores), iterate over each score;
    otherwise, treat it as a single score.
    For each score, write a temporary MIDI file and then convert it to a WAV file using timidity.

    Note: Ensure timidity (or another MIDI-to-WAV converter) is installed and available.
    """
    for root, _, files in tqdm(os.walk(input_folder)):
        for file in files:
            if file.lower().endswith((".mid", ".midi")):
                midi_path = os.path.join(root, file)
                print(f"Processing: {midi_path}")

                try:
                    score = converter.parse(midi_path)
                except Exception as e:
                    print(f"Error parsing {midi_path}: {e}")
                    continue

                # Prepare output subdirectory (mirroring the input folder structure)
                relative_path = os.path.relpath(root, input_folder)
                output_subdir = os.path.join(output_folder, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                base_name = os.path.splitext(file)[0]

                # Check if the parsed object is an Opus (multiple scores) or a single score.
                if type(score) is Opus:
                    # Iterate through each contained score.
                    for idx, s in enumerate(score.scores):
                        temp_midi = os.path.join(output_subdir, "midi", f"{base_name}_piece{idx}.mid")
                        wav_output = os.path.join(output_subdir, "wav", f"{base_name}_piece{idx}.wav")
                        os.makedirs(os.path.join(output_subdir, "midi"), exist_ok=True)
                        os.makedirs(os.path.join(output_subdir, "wav"), exist_ok=True)
                        try:
                            s.write("midi", fp=temp_midi)
                            # Convert MIDI to WAV using timidity.
                            fs = FluidSynth(sound_font=soundfont, sample_rate=sample_rate)
                            fs.midi_to_audio(temp_midi, wav_output)
                            print(f"Created {wav_output}")
                        except Exception as e:
                            print(f"Error processing piece {idx} from {midi_path}: {e}")
                else:
                    # Single score MIDI file.
                    temp_midi = os.path.join(output_subdir, "midi", f"{base_name}.mid")
                    wav_output = os.path.join(output_subdir, "wav", f"{base_name}.wav")
                    os.makedirs(os.path.join(output_subdir, "midi"), exist_ok=True)
                    os.makedirs(os.path.join(output_subdir, "wav"), exist_ok=True)
                    try:
                        score.write("midi", fp=temp_midi)
                        fs = FluidSynth(sound_font=soundfont, sample_rate=sample_rate)
                        fs.midi_to_audio(temp_midi, wav_output)
                        print(f"Created {wav_output}")
                    except Exception as e:
                        print(f"Error processing {midi_path}: {e}")


if __name__ == "__main__":
    # Set the paths to your folders:
    input_folder = "data/01_raw/Classical Music MIDI"  # Change to your MIDI files folder
    output_folder = "data/03_converted/Classical Music MIDI"  # Change to your desired output folder
    soundfont = os.path.join(
        "sound_fonts", "Essential Keys-sforzando-v9.6.sf2"
    )  # Path to your FluidSynth-compatible soundfont file

    convert_midi_to_wav(input_folder, output_folder, soundfont)
