import re
from collections import defaultdict
from enum import StrEnum
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
from typing import Any, Tuple, cast

import music21
import music21.meter
from devtools import pprint
from loguru import logger
from music21 import Music21Object
from music21.bar import Barline, Repeat
from music21.chord import Chord
from music21.clef import Clef
from music21.common.numberTools import opFrac
from music21.common.types import OffsetQL, OffsetQLIn
from music21.duration import Duration, Tuplet
from music21.key import KeySignature
from music21.meter import TimeSignature
from music21.note import GeneralNote, Note, NotRest, Rest
from music21.stream import Measure, Opus, Part, Score, Stream
from music21.stream.iterator import OffsetIterator
from pydantic import BaseModel, ConfigDict

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.pydantic_models.instrument_types import InstrumentTypes


class TokenType(StrEnum):
    PITCH = "PITCH"
    DURATION = "DURATION"
    REST = "REST"
    BAR = "BAR"
    TIME_SHIFT = "TIME_SHIFT"
    TIME_SIGNATURE = "TIME_SIGNATURE"
    CLEF = "CLEF"
    KEY_SIGNATURE = "KEY_SIGNATURE"


class BarModel(BaseModel):
    bar_duration_quarterLength: OffsetQL
    real_duration_quarterLength: OffsetQL
    is_repeat: bool = False
    is_end: bool = True
    times: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MidiQuantizedConverter:
    def __init__(
        self,
        settings: EncodingSetting = EncodingSetting(),
    ) -> None:
        self.settings = settings
        self.durations_range: Tuple[int, int] = (
            1,
            settings.longest_note_duration
            * settings.shortest_note_duration
            * (3 if self.settings.allow_triplet_quarterLength else 1),
        )  # TODO: Ensure this is correct
        self.durations: list[str] = [  # TODO: Ensure this is correct
            f"d{i}" for i in range(self.durations_range[0], self.durations_range[1] + 1)
        ]
        self.pitches_range = settings.notes_range
        self.pitches: list[str] = [f"p{i}" for i in range(self.pitches_range[0], self.pitches_range[1] + 1)]
        self.rest: str = "rest"
        self.time_shift: str = "shift"
        self.bar: str = "|"
        self.tuplet_start: str = "tuplet_start"
        self.tuplet_end: str = "tuplet_end"
        self.parts_separator: str = "/"
        self.repeat_start: str = "repeat_start"
        self.repeat_end: str = "repeat_end"

        self.time_signatures: list[str] = (  # Most common time signatures
            []
            if not self.settings.include_time_signature
            else [
                f"time_signature_{i}"
                for i in [
                    "1/2,",
                    "2/2",
                    "3/2",
                    "4/2",
                    "1/4",
                    "2/4",
                    "3/4",
                    "4/4",
                    "5/4",
                    "6/4",
                    "7/4",
                    "8/4",
                    "1/8",
                    "2/8",
                    "3/8",
                    "4/8",
                    "5/8",
                    "6/8",
                    "7/8",
                    "8/8",
                    "9/8",
                    "10/8",
                    "11/8",
                    "12/8",
                ]
            ]
        )
        # [f"time_signature_{i}" for i in ["2/4", "3/4", "4/4", "2/2", "3/8", "6/8", "9/8", "12/8"]]
        self._clef_params = [  # Most common clefs
            ("G", 1, 0),
            ("G", 2, 0),
            ("G", 2, -1),
            ("G", 2, 1),
            ("G", 3, 0),
            ("C", 1, 0),
            ("C", 2, 0),
            ("C", 3, 0),
            ("C", 4, 0),
            ("C", 5, 0),
            ("F", 3, 0),
            ("F", 4, 0),
            ("F", 4, 1),
            ("F", 4, -1),
            ("F", 5, 0),
            ("TAB", 5, 0),
        ]
        self.clefs: list[str] = (
            []
            if not self.settings.include_clef
            else [f"clef_{sign}_{line}_{octaveChange}" for sign, line, octaveChange in self._clef_params]
        )
        self.key_signatures: list[str] = (
            [] if not self.settings.include_key_signature else [f"key_signature_{i}" for i in range(-7, 8)]
        )
        # TODO: Add all possible tokens for creating tokenizer
        self.all_possible_tokens, self.tokens_types = self._create_all_possible_tokens_list()

    def _create_all_possible_tokens_list(self) -> Tuple[list[str], list[TokenType]]:
        all_possible_tokens: list[str] = []
        tokens_types: list[TokenType] = []

        all_possible_tokens.append(self.time_shift)
        tokens_types.append(TokenType.TIME_SHIFT)

        if self.settings.include_bars:
            all_possible_tokens.append(self.bar)
            tokens_types.append(TokenType.BAR)
        if self.settings.include_rests:
            all_possible_tokens.append(self.rest)
            tokens_types.append(TokenType.REST)

        if self.settings.include_clef:
            all_possible_tokens.extend(self.clefs)
            tokens_types.extend([TokenType.CLEF] * len(self.clefs))
        if self.settings.include_key_signature:
            all_possible_tokens.extend(self.key_signatures)
            tokens_types.extend([TokenType.KEY_SIGNATURE] * len(self.key_signatures))
        if self.settings.include_time_signature:
            all_possible_tokens.extend(self.time_signatures)
            tokens_types.extend([TokenType.TIME_SIGNATURE] * len(self.time_signatures))

        all_possible_tokens.extend(self.durations)
        tokens_types.extend([TokenType.DURATION] * len(self.durations))
        all_possible_tokens.extend(self.pitches)
        tokens_types.extend([TokenType.PITCH] * len(self.pitches))

        if len(all_possible_tokens) != len(tokens_types):
            raise RuntimeError(
                f"Created all_possible_tokens and tokens_types of different lengths {len(all_possible_tokens)} != {len(tokens_types)}"  # noqa: E501
            )
        return all_possible_tokens, tokens_types

    def filepath_to_texts(
        self,
        midi_path: Path,
    ) -> dict[str, str]:
        stream = music21.converter.parseFile(midi_path)
        stream = self._quantize_stream(stream)
        score_name_to_text = self.stream_to_texts(stream, midi_path.name)
        return score_name_to_text

    def _quantize_stream(self, stream: Score | Part | Opus) -> Score | Part | Opus:
        quantized_stream = cast(
            Score | Part | Opus, stream.quantize(quarterLengthDivisors=self._get_quarterLengthDivisors(), recurse=True)
        )
        if quantized_stream is None:
            raise ValueError("Stream became None after quantization")
        return quantized_stream

    def stream_to_texts(
        self,
        stream: Opus | Score | Part,
        file_name: str,
    ) -> dict[str, str]:
        score_name_to_text: dict[str, str] = {}
        # TODO:
        # Maybe add saving info about instrument
        if type(stream) is Opus:
            for i, score in enumerate(stream.scores):
                single_score_name_to_text = self.stream_to_texts(score, f"{file_name}_{i}")
                for score_name, text in single_score_name_to_text.items():
                    score_name_to_text[f"{score_name}"] = text
            return score_name_to_text
        elif type(stream) is Score:
            score = stream
        elif type(stream) is Part:
            score = Score(stream)
        else:
            raise ValueError(f"Got stream of type {type(stream)}, but expected Opus | Score | Part")

        score.quantize(quarterLengthDivisors=self._get_quarterLengthDivisors(), inPlace=True, recurse=True)
        if self.settings.repeats_handling == "Expand":
            score = score.expandRepeats()

        parts = self.filter_allowed_parts(score)
        if len(parts) == 0:
            return {}
        parts_measures_dicts: list[
            list[
                dict[
                    float | Fraction,
                    list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
                ]
            ]
        ] = []
        for part in parts:
            # updated_part = cast(Part, cast(Part, part.makeNotation()).makeRests(fillGaps=True))
            updated_part = cast(Part, part.makeNotation())
            updated_part.makeTies(inPlace=True)
            # updated_part.show("text", addEndTimes=True)

            last_clef: Clef | None = None
            last_time_signature: TimeSignature | None = None
            last_key_signature: KeySignature | None = None

            measures_dicts: list[
                dict[
                    float | Fraction,
                    list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
                ]
            ] = []

            partOffsetIterator: OffsetIterator = OffsetIterator(updated_part)
            for elementGroup in partOffsetIterator:
                measure = None
                for possible_measure in elementGroup:
                    if type(possible_measure) is Measure:
                        measure = possible_measure
                        break
                if measure is None:
                    continue

                if measure.hasVoices():
                    measure.flattenUnnecessaryVoices(force=False, inPlace=True)
                    if measure.hasVoices():
                        measure = cast(Measure, measure.chordify())

                if isinstance(measure.offset, Fraction):
                    continue

                measure_offset_dict: dict[
                    float | Fraction,
                    list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
                ] = defaultdict(list)
                measureOffsetIterator: OffsetIterator = OffsetIterator(measure)

                for elements in measureOffsetIterator.getElementsByClass(
                    [Clef, KeySignature, TimeSignature, Note, Chord, Rest]
                ):
                    for element in elements:
                        element = cast(Clef | KeySignature | TimeSignature | Note | Chord | Rest, element)
                        if (
                            isinstance(element, Clef)
                            or isinstance(element, KeySignature)
                            or isinstance(element, TimeSignature)
                        ):
                            self._add_clef_key_or_time_signature_to_dict_if_changed(
                                last_clef,
                                last_key_signature,
                                last_time_signature,
                                measure_offset_dict,
                                element,
                            )
                            if isinstance(element, Clef):
                                last_clef = element
                            elif isinstance(element, KeySignature):
                                last_key_signature = element
                            elif isinstance(element, TimeSignature):
                                last_time_signature = element
                            continue
                        measure_offset_dict[element.offset].append(element)
                bar_model = BarModel(
                    bar_duration_quarterLength=measure.barDuration.quarterLength,
                    real_duration_quarterLength=measure.duration.quarterLength,
                )
                if measure.leftBarline:
                    if self.settings.repeats_handling == "Special tokens" and isinstance(measure.leftBarline, Repeat):
                        bar_model.is_repeat = True
                        bar_model.is_end = False
                if measure.rightBarline:
                    if self.settings.repeats_handling == "Special tokens" and isinstance(measure.rightBarline, Repeat):
                        bar_model.is_repeat = True
                        bar_model.is_end = True
                measure_offset_dict[0].append(bar_model)
                measures_dicts.append(measure_offset_dict)
            parts_measures_dicts.append(measures_dicts)
            # pprint(measures_dicts)
        text = self._convert_offset_dicts_to_text(parts_measures_dicts=parts_measures_dicts)
        score_name_to_text[file_name] = text
        return score_name_to_text

    def _convert_offset_dicts_to_text(
        self,
        parts_measures_dicts: list[
            list[
                dict[
                    float | Fraction,
                    list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
                ]
            ]
        ],
    ) -> str:
        if self.settings.joining_parts_strategy == "Join parallel measures":
            return self._convert_offset_dicts_to_text_by_joining_parallel_measures(parts_measures_dicts)
        elif self.settings.joining_parts_strategy == "Queue parallel measures":
            return self._convert_offset_dicts_to_text_by_queuing_parallel_measures(parts_measures_dicts)
        else:
            raise ValueError(f"Got unexpected joining_parts_strategy {self.settings.joining_parts_strategy}")

    def _convert_offset_dicts_to_text_by_queuing_parallel_measures(
        self,
        parts_measures_dicts: list[
            list[
                dict[
                    float | Fraction,
                    list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
                ]
            ]
        ],
    ) -> str:
        n_measures = max(len(part_measures) for part_measures in parts_measures_dicts)
        tokens: list[str] = []
        for measure_number in range(n_measures):
            # is_any_non_empty_measure: bool = False
            time_signature: TimeSignature | None = None
            bar_model: BarModel | None = None
            is_first_part: bool = True
            for part_number, measures_dicts in enumerate(parts_measures_dicts):
                if len(measures_dicts) <= measure_number:
                    continue

                measure_offset_dict = measures_dicts[measure_number]
                if self.settings.skip_measures_without_notes and not any(
                    isinstance(element, (Note, Chord))
                    for list_of_elements in measure_offset_dict.values()
                    for element in list_of_elements
                ):
                    continue

                offsets = sorted(measure_offset_dict.keys())

                # Add repetition bar start token
                if is_first_part:
                    is_first_part = False
                    if len(offsets) > 0 and offsets[0] == 0:
                        elements = measure_offset_dict[0]
                        # print(elements)
                        bar_model = next((element for element in elements if isinstance(element, BarModel)), None)
                        if bar_model is not None:
                            if bar_model.is_repeat and not bar_model.is_end:
                                tokens.append(self.repeat_start)

                tokens.append(f"{self.parts_separator}{part_number}")
                # is_any_non_empty_measure = True
                for offset in offsets:
                    elements = measure_offset_dict[offset]

                    clef = next((element for element in elements if isinstance(element, Clef)), None)
                    if clef is not None and self.settings.include_clef:
                        tokens.append(f"clef_{clef.sign}_{clef.line}_{clef.octaveChange}")
                    key_signature = next((element for element in elements if isinstance(element, KeySignature)), None)
                    if key_signature is not None and self.settings.include_key_signature:
                        tokens.append(f"key_signature_{key_signature.sharps}")
                    time_signature = next((element for element in elements if isinstance(element, TimeSignature)), None)
                    if time_signature is not None and self.settings.include_time_signature:
                        tokens.append(f"time_signature_{time_signature.numerator}/{time_signature.denominator}")
                    # bar_model = next((element for element in elements if isinstance(element, BarModel)), None)

                    if self.settings.include_offset_in_notes and (
                        any(isinstance(element, (Note, Chord)) for element in elements)
                        or (self.settings.include_rests and any(isinstance(element, Rest) for element in elements))
                    ):
                        tokens.append(f"o{self.duration_or_offset_to_int_enc(offset)}")

                    for element in elements:
                        if isinstance(element, Note):
                            tokens.append(f"p{element.pitch.midi}")
                            tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                        elif isinstance(element, Rest) and self.settings.include_rests:
                            tokens.append(self.rest)
                            tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                        elif isinstance(element, Chord):
                            for pitch in element.pitches:
                                tokens.append(f"p{pitch.midi}")
                            tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")

            # if not is_any_non_empty_measure:
            #     if self.settings.include_rests:
            #         if self.settings.include_offset_in_notes:
            #             tokens.append("o0")
            #         tokens.append(self.rest)
            #         if time_signature is not None:
            #             tokens.append(
            #                 f"d{self.duration_or_offset_to_int_enc(time_signature.barDuration.quarterLength)}"
            #             )
            #         else:
            #             tokens.append(f"d{self.duration_or_offset_to_int_enc(4)}")

            # for pickup/anacrusis bars
            tokens.append(self.parts_separator)
            if bar_model is not None:
                tokens.append(f"o{self.duration_or_offset_to_int_enc(bar_model.real_duration_quarterLength)}")
                if bar_model.is_repeat and bar_model.is_end:
                    tokens.append(self.repeat_end)
            else:
                tokens.append(f"o{self.duration_or_offset_to_int_enc(4)}")
            tokens.append(self.bar)
        return " ".join(tokens)

    def _convert_offset_dicts_to_text_by_joining_parallel_measures(
        self,
        parts_measures_dicts: list[
            list[
                dict[
                    float | Fraction,
                    list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
                ]
            ]
        ],
    ) -> str:
        joined_measures_dicts: list[
            dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ]
        ] = []
        for measures_dicts_with_nones in zip_longest(*parts_measures_dicts, fillvalue=None):
            measures_dicts = tuple(item for item in measures_dicts_with_nones if item is not None)
            joined_measure: dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ] = {}

            for measure_offset_dict in measures_dicts:
                if self.settings.skip_measures_without_notes and not any(
                    isinstance(element, (Note, Chord))
                    for list_of_elements in measure_offset_dict.values()
                    for element in list_of_elements
                ):
                    continue
                for offset, value in measure_offset_dict.items():
                    if offset not in joined_measure:
                        joined_measure[offset] = value
                    else:
                        joined_measure[offset] = joined_measure[offset] + value
            joined_measures_dicts.append(joined_measure)

        tokens: list[str] = []
        for joined_measure in joined_measures_dicts:
            is_nonempty_measure: bool = False
            time_signature: TimeSignature | None = None
            bar_model: BarModel | None = None
            offsets = sorted(joined_measure.keys())
            for offset in offsets:
                elements = joined_measure[offset]

                clef = next((element for element in elements if isinstance(element, Clef)), None)
                if clef is not None and self.settings.include_clef:
                    tokens.append(f"clef_{clef.sign}_{clef.line}_{clef.octaveChange}")
                key_signature = next((element for element in elements if isinstance(element, KeySignature)), None)
                if key_signature is not None and self.settings.include_key_signature:
                    tokens.append(f"key_signature_{key_signature.sharps}")
                time_signature = next((element for element in elements if isinstance(element, TimeSignature)), None)
                if time_signature is not None and self.settings.include_time_signature:
                    tokens.append(f"time_signature_{time_signature.numerator}/{time_signature.denominator}")
                bar_model = next((element for element in elements if isinstance(element, BarModel)), None)

                if any(isinstance(element, (Note, Chord)) for element in elements) or (
                    self.settings.include_rests and any(isinstance(element, Rest) for element in elements)
                ):
                    is_nonempty_measure = True
                    if self.settings.include_offset_in_notes:
                        tokens.append(f"o{self.duration_or_offset_to_int_enc(offset)}")

                for element in elements:
                    if isinstance(element, Note):
                        tokens.append(f"p{element.pitch.midi}")
                        tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                    elif isinstance(element, Rest) and self.settings.include_rests:
                        tokens.append(self.rest)
                        tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                    elif isinstance(element, Chord):
                        for pitch in element.pitches:
                            tokens.append(f"p{pitch.midi}")
                            tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")

            if not is_nonempty_measure:
                if self.settings.include_rests:
                    if self.settings.include_offset_in_notes:
                        tokens.append("o0")
                    tokens.append(self.rest)
                    if time_signature is not None:
                        tokens.append(
                            f"d{self.duration_or_offset_to_int_enc(time_signature.barDuration.quarterLength)}"
                        )
                    else:
                        tokens.append(f"d{self.duration_or_offset_to_int_enc(4)}")
            # for pickup/anacrusis bars
            if self.settings.include_offset_in_notes:
                if bar_model is not None:
                    tokens.append(f"o{self.duration_or_offset_to_int_enc(bar_model.real_duration_quarterLength)}")
                else:
                    tokens.append(f"o{self.duration_or_offset_to_int_enc(4)}")
            tokens.append(self.bar)
        return " ".join(tokens)

    def text_to_score(self, text: str) -> Score:
        parts_numbers = re.compile(rf"\s*(?<!\d){re.escape("/")}(\d?)(?!\d)\s*").findall(text)
        n_parts = 0
        for part_number in parts_numbers:
            if part_number:
                if int(part_number) + 1 > n_parts:
                    n_parts = int(part_number) + 1

        measure_regex = re.compile(rf"\s*{re.escape(self.bar)}\s*")
        measures: list[str] = measure_regex.split(text)
        measures = [measure for measure in measures if measure]

        # n_measures = len(measures)
        part_regex = re.compile(rf"(\s*(?<!\d){re.escape(self.parts_separator)}\d?(?!\d)\s*)")
        measures_parts: list[list[str]] = [part_regex.split(measure) for measure in measures]
        pre_measures_tokens: list[str] = [measures_part[0] for measures_part in measures_parts]
        measures_parts = [measures_part[1:] for measures_part in measures_parts]
        measures_parts = [
            [(measure[i] + measure[i + 1]).strip() for i in range(0, len(measure), 2)] for measure in measures_parts
        ]

        measures_padding_parts = [
            measure_parts[-1] if len(measure_parts) > 0 and measure_parts[-1].startswith("/ ") else None
            for measure_parts in measures_parts
        ]
        # print(measures_padding_parts)

        corrected_measures_parts: list[list[str]] = []
        for measure_parts in measures_parts:
            if len(measure_parts) == 0:
                corrected_measures_parts.append(measure_parts)
            else:
                corrected_measures_parts.append(measure_parts[:-1])
        measures_parts = corrected_measures_parts

        # measures_parts = [[measure for measure in measure_parts if measure] for measure_parts in measures_parts]
        parts = [Part() for _ in range(n_parts)]
        # part_indexes = list(range(n_parts))
        # last_key_signatures: list[KeySignature | None] = [None for _ in range(n_parts)]

        n_invalid_tokens = 0
        for measure_parts, padding_part, pre_measure_tokens in zip(
            measures_parts, measures_padding_parts, pre_measures_tokens, strict=True
        ):
            was_measure_added_in_part = [False for _ in range(n_parts)]
            # for part_index, part, measure_part in zip_longest(part_indexes, parts, measure_parts, fillvalue=None):
            for measure_part in measure_parts:
                # if part is None:
                #     raise ValueError("Got None part in zip_longest in text_to_score")
                # if part_index is None:
                #     raise ValueError("Got None part_index in zip_longest in text_to_score")
                try:
                    part_index = int(measure_part.split()[0][1:])
                except Exception as e:
                    logger.warning(f"Got measure_part that with invalid part_index {measure_part}, error:: {e}")
                    continue

                measure, n_new_invalid_tokens = self.parse_single_measure_part(measure_part=measure_part)
                n_invalid_tokens += n_new_invalid_tokens
                # last_key_signature = last_key_signatures[part_index]
                # if last_key_signature is not None:
                #     for n in measure.getElementsByClass(Note):
                #         nStep = n.pitch.step
                #         rightAccidental = last_key_signature.accidentalByStep(nStep)
                #         n.pitch.accidental = rightAccidental
                parts[part_index].append(measure)
                was_measure_added_in_part[part_index] = True
                if padding_part is not None:
                    tokens = padding_part.split()
                    bar_offset: int | None = None
                    for token in tokens:
                        if token.startswith("o"):
                            bar_offset = int(token[1:])
                        elif token == self.repeat_end:
                            measure.rightBarline = Repeat(direction="end", times=None)
                    if bar_offset is not None:
                        measure.paddingLeft = measure.barDuration.quarterLength - self.int_enc_to_quarterLength(
                            bar_offset
                        )
                tokens = pre_measure_tokens.split()
                if self.repeat_start in tokens:
                    measure.leftBarline = Repeat(direction="start", times=None)
            for part_index, part in enumerate(parts):
                if not was_measure_added_in_part[part_index]:
                    measure = Measure()
                    part.append(measure)
                    if padding_part is not None:
                        tokens = padding_part.split()
                        bar_offset = None
                        for token in tokens:
                            if token.startswith("o"):
                                bar_offset = int(token[1:])
                            elif token == self.repeat_end:
                                measure.rightBarline = Repeat(direction="end", times=None)
                        if bar_offset is not None:
                            measure.paddingLeft = measure.barDuration.quarterLength - self.int_enc_to_quarterLength(
                                bar_offset
                            )
                    tokens = pre_measure_tokens.split()
                    if self.repeat_start in tokens:
                        measure.leftBarline = Repeat(direction="start", times=None)
        if n_invalid_tokens > 0:
            logger.warning(f"Got total of {n_invalid_tokens} invalid tokens")
        return Score(parts)

    def parse_single_measure_part(self, measure_part: str) -> tuple[Measure, int]:
        if self.settings.include_offset_in_notes:
            return self.parse_single_measure_part_with_notes_offsets(measure_part)
        else:
            return self.parse_single_measure_part_without_notes_offsets(measure_part)

    def parse_single_measure_part_with_notes_offsets(self, measure_part: str) -> tuple[Measure, int]:
        n_invalid_tokens = 0
        measure = Measure()

        offset: int | None = None
        pitches: list[int] | None = None
        duration: int | None = None
        tokens = measure_part.split()[1:]
        for token in tokens:
            # print(token)
            if token.startswith("clef"):
                _, sign, line, octave_change = token.split("_")
                clef = music21.clef.clefFromString(
                    f"{sign}{line}",
                    octaveShift=int(octave_change),
                )
                measure.append(clef)
            elif token.startswith("key_signature"):
                n_sharps = token.split("_")[-1]
                key_signature = music21.key.KeySignature(sharps=int(n_sharps))
                measure.append(key_signature)
            elif token.startswith("time_signature"):
                fraction = token.split("_")[-1]
                numerator, denominator = fraction.split("/", maxsplit=1)
                time_signature = TimeSignature(value=f"{int(numerator)}/{int(denominator)}")
                measure.append(time_signature)
            elif token.startswith("o"):
                if pitches is not None:
                    logger.warning(f"Got invalid offset token {token} in measure {measure_part}")
                offset = int(token[1:])
                pitches = None
                duration = None
            elif token.startswith("p"):
                if pitches is None:
                    pitches = []
                pitches.append(int(token[1:]))
                if offset is None:
                    n_invalid_tokens += 1
                    logger.warning(f"Got invalid pitch token {token} in measure {measure_part}")
            elif token.startswith("d"):
                duration = int(token[1:])
                if offset is None or pitches is None:
                    n_invalid_tokens += 1
                    logger.warning(f"Got invalid duration token {token} in measure {measure_part}")
                else:
                    pitches = [pitch for pitch in pitches if pitch > 0]
                    quarterLength_offset = self.int_enc_to_quarterLength(offset)
                    if len(pitches) == 0:
                        rest = music21.note.Rest(length=self.int_enc_to_quarterLength(duration))
                        rest.offset = quarterLength_offset
                        measure.insert(quarterLength_offset, rest)
                    elif len(pitches) == 1:
                        note = music21.note.Note(pitch=pitches[0])
                        note.duration = music21.duration.Duration(self.int_enc_to_quarterLength(duration))
                        note.offset = quarterLength_offset
                        measure.insert(quarterLength_offset, note)
                    else:
                        chord = music21.chord.Chord(pitches)
                        chord.duration = music21.duration.Duration(self.int_enc_to_quarterLength(duration))
                        chord.offset = quarterLength_offset
                        measure.insert(quarterLength_offset, chord)
                    pitches = None
                    duration = None
            elif token == self.rest:
                pitches = [0]
                if offset is None:
                    n_invalid_tokens += 1
                    logger.warning(f"Got invalid rest token {token} in measure {measure_part}")
            else:
                logger.warning(f"Got unexpected token {token}")
        return measure, n_invalid_tokens

    def parse_single_measure_part_without_notes_offsets(self, measure_part: str) -> tuple[Measure, int]:
        n_invalid_tokens = 0
        measure = Measure()

        quarterLength_offset: OffsetQL = 0.0
        pitches: list[int] | None = None
        duration: int | None = None
        tokens = measure_part.split()[1:]
        for token in tokens:
            if token.startswith("clef"):
                _, sign, line, octave_change = token.split("_")
                clef = music21.clef.clefFromString(
                    f"{sign}{line}",
                    octaveShift=int(octave_change),
                )
                measure.append(clef)
            elif token.startswith("key_signature"):
                n_sharps = token.split("_")[-1]
                key_signature = music21.key.KeySignature(sharps=int(n_sharps))
                measure.append(key_signature)
            elif token.startswith("time_signature"):
                fraction = token.split("_")[-1]
                numerator, denominator = fraction.split("/", maxsplit=1)
                time_signature = TimeSignature(value=f"{int(numerator)}/{int(denominator)}")
                measure.append(time_signature)
            elif token.startswith("p"):
                if pitches is None:
                    pitches = []
                pitches.append(int(token[1:]))
            elif token.startswith("d"):
                duration = int(token[1:])
                if pitches is None:
                    n_invalid_tokens += 1
                    logger.warning(f"Got invalid duration token {token} in measure {measure_part}")
                else:
                    quarterLength_duration = self.int_enc_to_quarterLength(duration)
                    pitches = [pitch for pitch in pitches if pitch > 0]
                    if len(pitches) == 0:
                        rest = music21.note.Rest(length=quarterLength_duration)
                        rest.offset = quarterLength_offset
                        measure.insert(quarterLength_offset, rest)
                    elif len(pitches) == 1:
                        note = music21.note.Note(pitch=pitches[0])
                        note.duration = music21.duration.Duration(quarterLength_duration)
                        note.offset = quarterLength_offset
                        measure.insert(quarterLength_offset, note)
                    else:
                        chord = music21.chord.Chord(pitches)
                        chord.duration = music21.duration.Duration(quarterLength_duration)
                        chord.offset = quarterLength_offset
                        measure.insert(quarterLength_offset, chord)
                    pitches = None
                    duration = None
                    quarterLength_offset = cast(OffsetQL, opFrac(quarterLength_offset + quarterLength_duration))
            elif token == self.rest:
                pitches = [0]
            else:
                logger.warning(f"Got unexpected token {token}")
        return measure, n_invalid_tokens

    def _add_clef_key_or_time_signature_to_dict_if_changed(
        self,
        last_clef: Clef | None,
        last_key_signature: KeySignature | None,
        last_time_signature: TimeSignature | None,
        measures_dicts: dict[
            float | Fraction, list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel]
        ],
        element: Clef | KeySignature | TimeSignature,
    ) -> None:
        if_element_changed = False
        if isinstance(element, Clef):
            if_element_changed = (
                last_clef is None
                or element.sign != last_clef.sign
                or element.line != last_clef.line
                or element.octaveChange != last_clef.octaveChange
            )
        elif isinstance(element, KeySignature):
            if_element_changed = last_key_signature is None or (
                isinstance(element.sharps, int)
                and isinstance(last_key_signature.sharps, int)
                and element.sharps != last_key_signature.sharps
            )
        elif isinstance(element, TimeSignature):
            if_element_changed = (
                last_time_signature is None
                or element.numerator != last_time_signature.numerator
                or element.denominator != last_time_signature.denominator
            )
        if if_element_changed:
            measures_dicts[element.offset].append(element)

    def _get_quarterLengthDivisors(self) -> list[int]:
        shortest_note_quarterLength = self.settings.shortest_note_duration / 4
        if not shortest_note_quarterLength.is_integer():
            raise RuntimeError(
                f"shortest_note_quarterLength is expected to be an integer but got {shortest_note_quarterLength} "
                f"for shortest_note_duration {self.settings.shortest_note_duration}"
            )
        quarterLengthDivisors: list[int] = [int(shortest_note_quarterLength)]
        if self.settings.allow_triplet_quarterLength:
            triplet_length = int(shortest_note_quarterLength) / 2 * 3
            if not triplet_length.is_integer():
                raise RuntimeError(
                    f"triplet_length is expected to be an integer but got {triplet_length} "
                    f"for shortest_note_duration {self.settings.shortest_note_duration}"
                )
            quarterLengthDivisors.append(int(triplet_length))
        return quarterLengthDivisors

    def filter_allowed_parts(self, score: Score) -> list[Part]:
        accepted_parts = []
        for part in score.parts:
            if bool(part.recurse().getElementsByClass(Note)) or bool(part.recurse().getElementsByClass(Chord)):
                if self.settings.only_SATB_parts:
                    if self.is_SATB_part(part):
                        accepted_parts.append(part)
                elif not self.settings.allowed_instruments:
                    accepted_parts.append(part)
                elif self.is_allowed_part_instrument(part):
                    accepted_parts.append(part)
        return accepted_parts

    def is_SATB_part(self, part: Part) -> bool:
        n_instruments = 0
        for instrument in part.getInstruments():
            n_instruments += 1
            if instrument.partName in ["Soprano", "Alto", "Tenor", "Bass"]:
                return True
        return False

    def is_allowed_part_instrument(self, part: Part) -> bool:
        n_instruments = 0
        for instrument in part.getInstruments():
            n_instruments += 1
            if instrument.midiProgram is not None:
                is_allowed = False
                for allowed_instrument in self.settings.allowed_instruments:
                    if instrument.midiProgram in allowed_instrument.value:
                        is_allowed = True
                if not is_allowed:
                    return False
            elif instrument.midiChannel is not None:
                if instrument.midiChannel == 9 and InstrumentTypes.PERCUSSIVE in self.settings.allowed_instruments:
                    pass
                else:
                    return False
            else:
                return False
        # If loop ended and there was at least 1 instrument, we mark this part as allowed
        return n_instruments > 0

    def duration_or_offset_to_int_enc(self, quarterLength: float | Fraction | None) -> int:
        if quarterLength is None:
            raise ValueError(f"Got quarterLength which is None {quarterLength}")
        duration_as_int = quarterLength * (self.settings.shortest_note_duration / 4)
        if self.settings.allow_triplet_quarterLength:
            duration_as_int *= 3
        if not duration_as_int.is_integer():
            error_message = (
                f"Encountered note whose duration {quarterLength / 4} couldn't be "
                "represented as integer multiple of "
                f"self.settings.shortest_note_duration {self.settings.shortest_note_duration}"
            )
            logger.warning(error_message)
            if self.settings.raise_duration_errors:
                raise ValueError(error_message)
            return max(1, int(duration_as_int))
        if quarterLength > self.settings.longest_note_duration * 4:
            error_message = (
                f"Encountered note whose duration ({quarterLength / 4} in whole notes, "
                f"{quarterLength} in quarterLength) is bigger than "
                f"self.settings.longest_note_duration {self.settings.longest_note_duration}"
            )
            logger.warning(error_message)
            if self.settings.raise_duration_errors:
                raise ValueError(error_message)
            return self.settings.longest_note_duration * 4
        return int(duration_as_int)

    def int_enc_to_quarterLength(self, int_enc: int) -> OffsetQL:
        quarterLength: OffsetQL | None = opFrac(int_enc * 4 / self.settings.shortest_note_duration)
        if quarterLength is None:
            raise ValueError(f"Got quarterLength which is None {quarterLength} for int_encoding {int_enc}")
        if self.settings.allow_triplet_quarterLength:
            quarterLength = opFrac(quarterLength / 3)
        if quarterLength is None:
            raise ValueError(f"Got quarterLength which is None {quarterLength} for int_encoding {int_enc}")
        return quarterLength
