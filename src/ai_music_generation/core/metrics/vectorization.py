from fractions import Fraction
from pathlib import Path
from typing import cast

import music21
from music21.chord import Chord
from music21.common.numberTools import opFrac
from music21.note import Note
from music21.stream import Measure, Score

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.encodings.quantized_converter import (
    MidiQuantizedConverter,
)


class MidiVectorizer:
    def __init__(self, settings: EncodingSetting | None = None):
        if settings is None:
            settings = EncodingSetting()
        self.settings = settings
        self.converter = MidiQuantizedConverter(self.settings)

    def midi_or_score_to_notes_and_offsets_feature_vectors(
        self,
        midi_path_or_score: str | Score,
        normalize_by_n_parts: bool = False,
    ) -> tuple[list[list[float]], list[list[float | Fraction]]]:
        if isinstance(midi_path_or_score, Score):
            score = midi_path_or_score
        else:
            stream = music21.converter.parseFile(Path(midi_path_or_score))
            if type(stream) is not Score:
                raise ValueError(
                    f"Midi saved under midi_path {midi_path_or_score} is expected to be parsable Score, "
                    f"but got {type(stream)}"
                )
            score = stream
        score = cast(Score, self.converter._quantize_stream(score))
        # score.show()
        parts_measures: list[list[Measure]] = []
        for part in score.parts:
            measures: list[Measure] = []
            for potential_measure in part:
                if type(potential_measure) is Measure:
                    measures.append(potential_measure)
            parts_measures.append(measures)

        # Pad shorter parts with empty Measure() to match the longest part
        max_measures = max(len(pm) for pm in parts_measures)
        for pm in parts_measures:
            if len(pm) < max_measures:
                pm.extend([Measure() for _ in range(max_measures - len(pm))])
        n = max_measures

        pitches_distributions: list[list[float]] = []
        offsets: list[list[float | Fraction]] = []
        for i in range(n):
            measure_stack: list[Measure] = [part_measure[i] for part_measure in parts_measures]

            measure_stack_pitches_with_durations: list[tuple[int, float | Fraction]] = []
            measure_stack_offsets: list[float | Fraction] = []
            measure_duration: float | None = None
            for measure in measure_stack:
                if measure_duration is None:
                    measure_duration = float(measure.duration.quarterLength)
                for element in measure.getElementsByClass([Note, Chord]):
                    if isinstance(element, Note):
                        measure_stack_offsets.append(element.offset)
                        measure_stack_pitches_with_durations.append(
                            (element.pitch.midi, element.duration.quarterLength)
                        )
                    elif isinstance(element, Chord):
                        for pitch in element.pitches:
                            measure_stack_offsets.append(element.offset)
                            measure_stack_pitches_with_durations.append((pitch.midi, element.duration.quarterLength))
            if measure_duration is None or measure_duration == 0:
                measure_duration = 4.0

            measure_pitches_distribution: list[float | Fraction] = [0.0] * 12
            for midi_pitch, duration in measure_stack_pitches_with_durations:
                measure_pitches_distribution[midi_pitch % 12] = cast(
                    float | Fraction, opFrac(measure_pitches_distribution[midi_pitch % 12] + duration)
                )

            measure_pitches_distribution_normalized: list[float] = [
                float(val) / measure_duration / (len(measure_stack) if normalize_by_n_parts else 1)
                for val in measure_pitches_distribution
            ]
            pitches_distributions.append(measure_pitches_distribution_normalized)
            offsets.append(measure_stack_offsets)
        return pitches_distributions, offsets
