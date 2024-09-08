import music21
from pydantic import BaseModel

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting


class Vocab:
    def __init__(
        self,
        encoding_settings: EncodingSetting = EncodingSetting(),
    ) -> None:
        pass

    def construct_mappings(self) -> None:
        raise NotImplementedError()

    def offset_to_notes_to_text() -> None:
        raise NotImplementedError()
