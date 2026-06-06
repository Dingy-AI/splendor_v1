from dataclasses import dataclass
from enums import GemColor

@dataclass(frozen=True)
class Noble:
    id: int
    points: int
    requirement: dict[GemColor,int,]