from dataclasses import dataclass
from env.core.enums import CardColor

@dataclass(frozen=True)
class Noble:
    id: int
    Name: str
    points: int
    requirement: dict[CardColor, int]