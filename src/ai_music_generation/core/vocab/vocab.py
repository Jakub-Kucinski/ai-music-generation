from typing import Optional, Tuple, Union

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
        self.durations_range: Tuple[int, int]  # TODO: Add durations
        self.durations: list[str] = [  # TODO: Ensure this is correct
            f"d{i}" for i in range(self.durations_range[0], self.durations_range[0] + 1)
        ]
        self.notes_range = encoding_settings.notes_range
        self.notes: list[str] = [
            f"n{i}" for i in range(self.notes_range[0], self.notes_range[0] + 1)
        ]
        self.rest: str = "rest"
        self.time_shift: str = "shift"
        self.time_signatures: list[str] = (  # Most common time signatures
            []
            if not self.encoding_settings.include_time_signature
            else [
                f"time_signature_{i}"
                for i in ["2/4", "3/4", "4/4", "2/2", "6/8", "9/8", "12/8"]
            ]
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
            else [
                f"clef_{sign}_{line}_{octaveChange}"
                for sign, line, octaveChange in self._clef_params
            ]
        )
        self.key_signatures: list[str] = (
            []
            if not self.encoding_settings.include_key_signature
            else [f"key_signature_{i}" for i in range(-7, 8)]
        )

    def offset_mapping_to_text(
        self,
        offset_to_notes: dict[int, list[Union[NoteModel, RestModel, BarModel]]],
        clef: Optional[ClefModel],
        key_signature: Optional[KeySignatureModel],
        time_signature: Optional[TimeSignatureModel],
    ) -> str:
        raise NotImplementedError()

    def get_tokenizer(self) -> None:
        raise NotImplementedError()
