from dataclasses import dataclass
from core.enums import GemColor

@dataclass(frozen=True)
class Noble:
    id: int
    Name: str
    points: int
    requirement: dict[GemColor, int]