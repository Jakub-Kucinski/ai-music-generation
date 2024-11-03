from typing import Optional, Tuple, Union

from devtools import pformat

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.pydantic_models.musical_notation import (
    BarModel,
    ClefModel,
    KeySignatureModel,
    NoteModel,
    RestModel,
    TimeSignatureModel,
)


class Vocab:
    def __init__(
        self,
        encoding_settings: EncodingSetting = EncodingSetting(),
    ) -> None:
        self.encoding_settings = encoding_settings
        self.durations_range: Tuple[int, int] = (
            1,
            encoding_settings.longest_note_duration * encoding_settings.shortest_note_duration,
        )  # TODO: Ensure this is correct
        self.durations: list[str] = [  # TODO: Ensure this is correct
            f"d{i}" for i in range(self.durations_range[0], self.durations_range[0] + 1)
        ]
        self.notes_range = encoding_settings.notes_range
        self.notes: list[str] = [f"n{i}" for i in range(self.notes_range[0], self.notes_range[0] + 1)]
        self.rest: str = "rest"
        self.time_shift: str = "shift"
        self.bar: str = "bar"

        self.time_signatures: list[str] = (  # Most common time signatures
            []
            if not self.encoding_settings.include_time_signature
            else [f"time_signature_{i}" for i in ["2/4", "3/4", "4/4", "2/2", "6/8", "9/8", "12/8"]]
        )
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
            if not self.encoding_settings.include_clef
            else [f"clef_{sign}_{line}_{octaveChange}" for sign, line, octaveChange in self._clef_params]
        )
        self.key_signatures: list[str] = (
            [] if not self.encoding_settings.include_key_signature else [f"key_signature_{i}" for i in range(-7, 8)]
        )
        # TODO: Add all possible tokens for creating tokenizer
        self.all_possible_tokens: list[str] = self._create_all_possible_tokens_list()

    def _create_all_possible_tokens_list(self) -> list[str]:
        all_possible_tokens: list[str] = []

        all_possible_tokens.append(self.time_shift)
        if self.encoding_settings.include_bars:
            all_possible_tokens.append(self.bar)
        if self.encoding_settings.include_rests:
            all_possible_tokens.append(self.rest)

        if self.encoding_settings.include_clef:
            all_possible_tokens.extend(self.clefs)
        if self.encoding_settings.include_key_signature:
            all_possible_tokens.extend(self.key_signatures)
        if self.encoding_settings.include_time_signature:
            all_possible_tokens.extend(self.time_signatures)

        all_possible_tokens.extend(self.durations)
        all_possible_tokens.extend(self.notes)
        return all_possible_tokens

    def offset_mapping_to_text(
        self,
        offset_to_notes: dict[int, list[Union[NoteModel, RestModel, BarModel]]],
        clef: Optional[ClefModel],
        key_signature: Optional[KeySignatureModel],
        time_signature: Optional[TimeSignatureModel],
    ) -> str:
        if self.encoding_settings.include_clef and clef is None:
            raise ValueError(f"include_clef was {self.encoding_settings.include_clef}, but got clef {clef}")
        if self.encoding_settings.include_key_signature and key_signature is None:
            raise ValueError(
                f"include_key_signature was {self.encoding_settings.include_key_signature}, but got key_signature {key_signature}"  # noqa: E501
            )
        if self.encoding_settings.include_time_signature and time_signature is None:
            raise ValueError(
                f"include_time_signature was {self.encoding_settings.include_time_signature}, but got time_signature {time_signature}"  # noqa: E501
            )

        tokens: list[str] = []
        if clef is not None:
            clef_str = f"clef_{clef.sign}_{clef.line}_{clef.octaveChange}"
            if clef_str not in self.clefs:
                raise ValueError(f"Clef {clef_str} was not found in self.clefs {self.clefs}")
            tokens.append(clef_str)
        if key_signature is not None:
            key_signature_str = f"key_signature_{key_signature.sharps}"
            if key_signature_str not in self.key_signatures:
                raise ValueError(
                    f"key_signature {key_signature_str} was not found in self.key_signatures {self.key_signatures}"
                )
            tokens.append(key_signature_str)
        if time_signature is not None:
            time_signature_str = f"time_signature_{time_signature.numerator}/{time_signature.denominator}"
            if time_signature_str not in self.time_signatures:
                raise ValueError(
                    f"time_signature {time_signature_str} was not found in self.time_signatures {self.time_signatures}"
                )
            tokens.append(time_signature_str)

        offset_to_notes = sorted(offset_to_notes)
        last_offset: int = 0
        for offset, notes in offset_to_notes.items():
            if offset != 0:
                tokens.append(self.time_shift)
                tokens.append(f"d{offset-last_offset}")
            bar_to_save_at_current_offset = False
            for note in notes:
                if isinstance(note, NoteModel):
                    tokens.append(f"n_{note.pitch}")
                    tokens.append(f"d_{note.duration}")
                elif isinstance(note, RestModel):
                    tokens.append(self.rest)
                    tokens.append(f"d_{note.duration}")
                elif isinstance(note, BarModel):
                    bar_to_save_at_current_offset = True
            if bar_to_save_at_current_offset:
                tokens.append(self.bar)
        self._check_if_tokens_in_vocab(tokens)
        return " ".join(tokens)

    def _check_if_tokens_in_vocab(self, tokens: list[str]) -> bool:
        disallowed_tokens: list[str] = []
        for token in tokens:
            if token not in self.all_possible_tokens:
                disallowed_tokens.append(token)
        if len(disallowed_tokens) > 0:
            raise ValueError(
                f"Provided tokens contained values that were not allowed by encoding settings {pformat(self.encoding_settings)}\n"  # noqa: E501
                f"disallowed_tokens: {disallowed_tokens}"
            )
        return True

    def get_tokenizer(self) -> None:
        raise NotImplementedError()

    def text_to_offset_mapping(self, text: str) -> Tuple[
        dict[int, list[Union[NoteModel, RestModel, BarModel]]],
        Optional[ClefModel],
        Optional[KeySignatureModel],
        Optional[TimeSignatureModel],
    ]:
        # tokens = text.split(" ")
        raise NotImplementedError()
