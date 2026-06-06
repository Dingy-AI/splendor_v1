from dataclasses import dataclass
from enums import GemColor

@dataclass
class Noble:
    id: int
    points: int
    requirement: dict[GemColor,int,]