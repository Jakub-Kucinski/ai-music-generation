from collections import defaultdict
from enum import StrEnum
from fractions import Fraction
from math import gcd, prod
from pathlib import Path
from typing import Any, Tuple, cast

import music21
import music21.meter
from devtools import pprint
from loguru import logger
from music21 import Music21Object
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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MidiConverter:
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
        self.parts_separator = "/"

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

    def stream_to_texts(
        self,
        stream: Opus | Score | Part,
        file_name: str,
    ) -> dict[str, list[str]]:
        score_name_to_texts: dict[str, list[str]] = {}
        # TODO:
        # Maybe add saving info about instrument
        if type(stream) is Opus:
            for i, score in enumerate(stream.scores):
                single_score_name_to_texts = self.stream_to_texts(score, f"{file_name}_{i}")
                for score_name, texts in single_score_name_to_texts.items():
                    score_name_to_texts[f"{score_name}"] = texts
            return score_name_to_texts
        elif type(stream) is Score:
            score = stream
        elif type(stream) is Part:
            score = Score(stream)
        else:
            raise ValueError(f"Got stream of type {type(stream)}, but expected Opus | Score | Part")

        score.quantize(quarterLengthDivisors=self._get_quarterLengthDivisors(), inPlace=True, recurse=True)
        parts = self.filter_allowed_parts(score)
        if len(parts) == 0:
            return {}
        parts_offset_to_result_elements: list[
            dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ]
        ] = []
        for part in parts:
            updated_part = cast(Part, cast(Part, part.makeNotation()).makeRests(fillGaps=True))
            updated_part.makeTies(inPlace=True)

            last_clef: Clef | None = None
            last_time_signature: TimeSignature | None = None
            last_key_signature: KeySignature | None = None

            offset_to_result_elements: dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ] = defaultdict(list)

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

                offset_to_result_elements[measure.offset].append(
                    BarModel(
                        bar_duration_quarterLength=measure.barDuration.quarterLength,
                        real_duration_quarterLength=measure.duration.quarterLength,
                    )
                )
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
                                offset_to_result_elements,
                                element,
                                measure.offset,
                            )
                            if isinstance(element, Clef):
                                last_clef = element
                            elif isinstance(element, KeySignature):
                                last_key_signature = element
                            elif isinstance(element, TimeSignature):
                                last_time_signature = element
                            continue

                        offset_to_result_elements[cast(OffsetQL, opFrac(measure.offset + element.offset))].append(
                            element
                        )
            parts_offset_to_result_elements.append(offset_to_result_elements)
        text = self._convert_offset_dicts_to_text(parts_offset_to_result_elements=parts_offset_to_result_elements)
        score_name_to_texts[file_name] = [text]
        return score_name_to_texts

    def _convert_offset_dicts_to_text(
        self,
        parts_offset_to_result_elements: list[
            dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ],
        ],
    ) -> str:
        if self.settings.joining_parts_strategy == "Join measures":
            return self._convert_offset_dicts_to_text_by_joining_measures(parts_offset_to_result_elements)
        elif self.settings.joining_parts_strategy == "Queue parallel measures":
            return self._convert_offset_dicts_to_text_by_queuing_parallel_measures(parts_offset_to_result_elements)
        else:
            raise ValueError(f"Got unexpected joining_parts_strategy {self.settings.joining_parts_strategy}")

    def _convert_offset_dicts_to_text_by_queuing_parallel_measures(
        self,
        parts_offset_to_result_elements: list[
            dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ],
        ],
    ) -> str:
        # Sort dicts
        sorted_offsets: list[list[float | Fraction]] = [
            sorted(offset_dict.keys()) for offset_dict in parts_offset_to_result_elements
        ]

        # Get offsets of bars in all parts
        all_bars_offsets: list[list[float | Fraction]] = [[] for _ in parts_offset_to_result_elements]
        for offsets, bars_offsets, offset_to_result_elements in zip(
            sorted_offsets, all_bars_offsets, parts_offset_to_result_elements, strict=True
        ):
            for offset in offsets:
                elements = offset_to_result_elements[offset]
                bar = next((element for element in elements if isinstance(element, BarModel)), None)
                if bar is not None:
                    bars_offsets.append(offset)

        tokens: list[str] = []
        is_first_bar: bool = True
        for offset_to_result_elements in parts_offset_to_result_elements:
            pass
        raise NotImplementedError()

    def _convert_offset_dicts_to_text_by_joining_measures(
        self,
        parts_offset_to_result_elements: list[
            dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
            ],
        ],
    ) -> str:
        offset_to_result_elements: dict[
            float | Fraction,
            list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel],
        ] = {}
        for d in parts_offset_to_result_elements:
            for k, v in d.items():
                if k not in offset_to_result_elements:
                    offset_to_result_elements[k] = v
                else:
                    offset_to_result_elements[k].extend(v)
        tokens: list[str] = []
        offsets = sorted(offset_to_result_elements.keys())
        is_first_bar: bool = True
        element = None
        measure_offset: float | Fraction = 0.0
        for offset in offsets:
            elements = offset_to_result_elements[offset]

            clef = next((element for element in elements if isinstance(element, Clef)), None)
            if clef is not None and self.settings.include_clef:
                tokens.append(f"clef_{clef.sign}_{clef.line}_{clef.octaveChange}")
            key_signature = next((element for element in elements if isinstance(element, KeySignature)), None)
            if key_signature is not None and self.settings.include_key_signature:
                tokens.append(f"key_signature_{key_signature.sharps}")
            time_signature = next((element for element in elements if isinstance(element, TimeSignature)), None)
            if time_signature is not None and self.settings.include_time_signature:
                tokens.append(f"time_signature_{time_signature.numerator}/{time_signature.denominator}")

            bar = next((element for element in elements if isinstance(element, BarModel)), None)
            if bar is not None:
                if self.settings.include_bars:
                    if is_first_bar:
                        is_first_bar = False
                    else:
                        offset_int = self.duration_or_offset_to_int_enc(opFrac(offset - measure_offset))
                        tokens.append(f"o{offset_int}")
                        tokens.append(self.bar)
                measure_offset = offset

            in_bar_offset = cast(OffsetQL, opFrac(offset - measure_offset))
            if (
                self.settings.include_offset
                and any(isinstance(element, (Note, Chord)) for element in elements)
                or (self.settings.include_rests and any(isinstance(element, Rest) for element in elements))
            ):
                offset_int = self.duration_or_offset_to_int_enc(in_bar_offset)
                tokens.append(f"o{offset_int}")

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
        # Add last bar
        if self.settings.include_bars and element is not None:
            if isinstance(element, BarModel):
                pass
            else:
                if element.duration.quarterLength > 0 and self.settings.include_offset:
                    offset_int = self.duration_or_offset_to_int_enc(
                        opFrac(element.offset + element.duration.quarterLength)
                    )
                    tokens.append(f"o{offset_int}")
                tokens.append(self.bar)
        return " ".join(tokens)

    def _add_clef_key_or_time_signature_to_dict_if_changed(
        self,
        last_clef: Clef | None,
        last_key_signature: KeySignature | None,
        last_time_signature: TimeSignature | None,
        offset_to_result_elements: dict[
            float | Fraction, list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | BarModel]
        ],
        element: Clef | KeySignature | TimeSignature,
        measure_offset: float,
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
            if isinstance(element.offset, Fraction):
                offset_to_result_elements[measure_offset].append(element)
            else:
                offset_to_result_elements[cast(OffsetQL, opFrac(measure_offset + element.offset))].append(element)

    def filepath_to_texts(
        self,
        midi_path: Path,
    ) -> dict[str, list[str]]:
        stream = music21.converter.parseFile(
            midi_path, quantizePost=True, quarterLengthDivisors=self._get_quarterLengthDivisors()
        )
        score_name_to_texts = self.stream_to_texts(stream, midi_path.name)
        return score_name_to_texts

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
                if not self.settings.allowed_instruments:
                    accepted_parts.append(part)
                elif self.is_allowed_part_instrument(part):
                    accepted_parts.append(part)
        return accepted_parts

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
