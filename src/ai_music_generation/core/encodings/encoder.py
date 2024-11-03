from collections import defaultdict
from pathlib import Path
from typing import Union, cast

import music21
import music21.meter.base
import numpy as np

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.pydantic_models.instrument_types import InstrumentTypes
from ai_music_generation.core.pydantic_models.musical_notation import (
    BarModel,
    ClefModel,
    KeySignatureModel,
    NoteModel,
    RestModel,
    TimeSignatureModel,
)
from ai_music_generation.core.vocab.vocab import Vocab


# TODO: ADD HANDLING MEASURES IN PARTS
# TODO: CONSIDER ALTERNATIVE REPRESENTATION - without time shift, but with offsets
# in quarterLength from last bar
class MidiEncoder:
    """Class containing the implementations of methods for transforming data
    from different forms e.g. midi -> text"""

    def __init__(
        self,
        vocab: Vocab,
        encoding_settings: EncodingSetting = EncodingSetting(),
    ) -> None:
        self.vocab = vocab

        self.include_bars = encoding_settings.include_bars
        self.include_rests = encoding_settings.include_rests
        self.include_clef = encoding_settings.include_clef
        self.include_key_signature = encoding_settings.include_key_signature
        self.include_time_signature = encoding_settings.include_time_signature
        self.join_parts = encoding_settings.join_parts

        self.shortest_note_duration = encoding_settings.shortest_note_duration
        self.longest_note_duration = encoding_settings.longest_note_duration
        self.allowed_instruments = encoding_settings.allowed_instruments

    def midi_stream_to_texts(
        self,
        stream: music21.stream.Score,
    ) -> np.ndarray:
        raise NotImplementedError()

    def midi_file_to_texts(
        self,
        midi_file: music21.midi.MidiFile,
    ) -> np.ndarray:
        stream = music21.midi.translate.midiFileToStream(midi_file)
        return self.midi_stream_to_texts(stream=stream)

    def filepath_to_texts(  # TODO: Change this function to return text(s)
        self,
        midi_path: Path,
    ) -> list[str]:
        mf = music21.midi.MidiFile()
        mf.open(midi_path)
        mf.read()
        mf.close()
        mf.tracks = self.filter_allowed_tracks(mf.tracks)
        score = cast(music21.stream.Score, music21.midi.translate.midiFileToStream(mf))

        result_texts: list[str] = []
        offset_to_notes_list: list[dict[int, list[Union[NoteModel, RestModel, BarModel]]]] = []
        offset_to_notes: dict[int, list[Union[NoteModel, RestModel, BarModel]]] = defaultdict(list)
        time_signature = music21.meter.base.TimeSignature(value="4/4")
        time_signature_model = TimeSignatureModel(numerator=4, denominator=4)
        clef: music21.clef.Clef = music21.clef.NoClef()
        clef_model = ClefModel(sign="G", line="1", octaveChange=0)
        key_signature = music21.key.KeySignature(sharps=0)
        key_signature_model = KeySignatureModel(sharps=0)

        for idx, part in enumerate(score.parts):
            offset_to_notes = defaultdict(list)
            flattened_part = part.flatten()
            if flattened_part.timeSignature is None:
                flattened_part.timeSignature = music21.meter.base.TimeSignature(value="4/4")
            for elem in part.flatten():
                if isinstance(elem, music21.note.Note):
                    duration = self.get_note_chord_rest_duration_as_int(elem)
                    offset = self.duration_or_offset_to_int_enc(elem.offset)
                    note_model = NoteModel(pitch=elem.pitch.midi, duration=duration)
                    offset_to_notes[offset].append(note_model)
                elif isinstance(elem, music21.chord.Chord):
                    offset = self.duration_or_offset_to_int_enc(elem.offset)
                    duration = self.get_note_chord_rest_duration_as_int(elem)
                    for pitch in elem.pitches:
                        note_model = NoteModel(pitch=pitch.midi, duration=duration)
                        offset_to_notes[offset].append(note_model)
                elif isinstance(elem, music21.note.Rest) and self.include_rests:
                    duration = self.get_note_chord_rest_duration_as_int(elem)
                    offset = self.duration_or_offset_to_int_enc(elem.offset)
                    res_model = RestModel(duration=duration)
                    offset_to_notes[offset].append(res_model)

            if flattened_part.timeSignature is not None:
                time_signature = flattened_part.timeSignature
            time_signature_model = TimeSignatureModel(
                numerator=(time_signature.numerator if time_signature.numerator is not None else 4),
                denominator=(time_signature.denominator if time_signature.denominator is not None else 4),
            )
            clef = (
                flattened_part.clef
                if flattened_part.clef is not None and not isinstance(flattened_part.clef, music21.clef.NoClef)
                else music21.clef.bestClef(flattened_part)
            )
            clef_model = ClefModel(
                sign=clef.sign if clef.sign is not None else "G",
                line=clef.line if clef.line is not None else "1",
                octaveChange=(clef.octaveChange if clef.octaveChange is not None else 0),
            )
            key_signature = (
                flattened_part.keySignature
                if flattened_part.keySignature is not None
                else music21.key.KeySignature(sharps=0)
            )
            key_signature_model = KeySignatureModel(sharps=key_signature.sharps if key_signature.sharps else 0)

            highest_time = self.duration_or_offset_to_int_enc(flattened_part.highestTime)
            sorted_offsets = sorted(offset_to_notes)

            # Add bars to offsets
            if self.include_bars:
                i = 0
                current_bar_offset = self.duration_or_offset_to_int_enc(i * time_signature.barDuration.quarterLength)
                while current_bar_offset < highest_time:
                    i += 1
                    current_bar_offset = self.duration_or_offset_to_int_enc(
                        i * time_signature.barDuration.quarterLength
                    )
                    offset_to_notes[current_bar_offset] = BarModel()
            if self.join_parts:
                offset_to_notes_list.append(sorted_offsets)
            else:
                # TODO: add saving somewhere
                result_texts.append(
                    self.vocab.offset_mapping_to_text(
                        sorted_offsets=sorted_offsets,
                        time_signature=(time_signature_model if self.include_time_signature else None),
                        clef=clef_model if self.include_clef else None,
                        key_signature=(key_signature_model if self.include_key_signature else None),
                    )
                )
        if self.join_parts and len(score.parts) > 0:
            offset_to_notes = offset_to_notes_list[0]
            for offset_to_notes_mapping in offset_to_notes_list[1:]:
                for offset, note in offset_to_notes_mapping.items():
                    offset_to_notes[offset].append(note)
            sorted_offsets = sorted(offset_to_notes)
            # TODO: add saving somewhere
            result_texts.append(
                self.vocab.offset_mapping_to_text(
                    sorted_offsets=sorted_offsets,
                    time_signature=(time_signature_model if self.include_time_signature else None),
                    clef=clef_model if self.include_clef else None,
                    key_signature=(key_signature_model if self.include_key_signature else None),
                )
            )
        return result_texts
        # score = music21.converter.parse(midi_path)
        # return self.midi_stream_to_texts(score)

    def filter_allowed_tracks(self, tracks: list[music21.midi.MidiTrack]) -> list[music21.midi.MidiTrack]:
        accepted_tracks = []
        for track in tracks:
            if cast(music21.midi.MidiTrack, track).hasNotes():
                if self.is_allowed_instrument(track):
                    accepted_tracks.append(track)
        return accepted_tracks

    def is_allowed_instrument(self, track: music21.midi.MidiTrack) -> bool:
        if InstrumentTypes.PERCUSSIVE in self.allowed_instruments and 10 in track.getChannels():
            return True
        instruments = cast(list[int], track.getProgramChanges())
        for instrument in instruments:
            is_instrument_allowed = False
            for allowed_instrument in self.allowed_instruments:
                if instrument in allowed_instrument.value:
                    is_instrument_allowed = True
            if not is_instrument_allowed:
                return False
        return True

    def convert_to_texts(
        self,
        obj: Union[music21.stream.Score, music21.midi.MidiFile, Path, str],
    ) -> np.ndarray:
        if isinstance(obj, music21.stream.Score):
            return self.midi_stream_to_texts(obj)
        if isinstance(obj, music21.midi.MidiFile):
            return self.midi_file_to_texts(obj)
        if isinstance(obj, Path):
            return self.filepath_to_texts(obj)
        if isinstance(obj, str):
            path = Path(obj)
            return self.filepath_to_texts(path)
        raise TypeError(
            "convert_to_texts support input objects of types "
            "Union[music21.stream.Score, music21.midi.MidiFile, Path, str], "
            f"but provided object {obj} was of type {type(obj)}"
        )

    def duration_or_offset_to_int_enc(self, quarterLength: float) -> int:
        duration_as_int = quarterLength * (self.shortest_note_duration / 4)
        if not duration_as_int.is_integer():
            raise ValueError(
                f"Encountered note whose duration {quarterLength / 4} couldn't be "
                "represented as integer multiple of "
                f"self.shortest_note_duration {self.shortest_note_duration}"
            )
        if quarterLength > self.longest_note_duration * 4:
            raise ValueError(
                f"Encountered note whose duration ({quarterLength / 4} in whole notes, "
                f"{quarterLength} in quarterLength) is bigger than "
                f"self.longest_note_duration {self.longest_note_duration}"
            )
        return int(duration_as_int)

    def get_note_chord_rest_duration_as_int(
        self,
        note_chord_rest: Union[music21.note.Note, music21.chord.Chord, music21.note.Rest],
    ) -> int:
        duration_as_int = self.duration_or_offset_to_int_enc(note_chord_rest.quarterLength)
        return duration_as_int
