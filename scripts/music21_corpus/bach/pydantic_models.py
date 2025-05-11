from pydantic import BaseModel


class BachChord(BaseModel):
    offset: float
    midi: tuple[int, int, int, int]
    is_start: tuple[bool, bool, bool, bool]

    def midi_mod12(self) -> tuple[int, int, int, int]:
        return self.midi[0] % 12, self.midi[1] % 12, self.midi[2] % 12, self.midi[3] % 12

    def negated_is_start(self) -> tuple[bool, bool, bool, bool]:
        return not self.is_start[0], not self.is_start[1], not self.is_start[2], not self.is_start[3]


class BachMeasure(BaseModel):
    measure_duration: float
    time_signature: str
    bach_chords: list[BachChord]


class BachProgression(BaseModel):
    bach_chords: list[BachChord]
