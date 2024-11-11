from pydantic import BaseModel


class NoteModel(BaseModel):
    pitch: int
    duration: int
    # offset_from_bar: int
    # nth_bar: int


class RestModel(BaseModel):
    duration: int
    # offset_from_bar: int
    # nth_bar: int


class TimeShiftModel(BaseModel):
    duration: int


class TimeSignatureModel(BaseModel):
    numerator: int
    denominator: int


class ClefModel(BaseModel):
    sign: str
    line: int
    octaveChange: int
    is_percussion: bool = False


class KeySignatureModel(BaseModel):
    sharps: int = 0


class BarModel(BaseModel):
    # nth_bar: int
    pass
