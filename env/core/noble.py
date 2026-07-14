from dataclasses import dataclass
from splendor_v1.env.core.enums import GemColor

@dataclass(frozen=True)
class Noble:
    id: int
    Name: str
    points: int
    requirement: dict[GemColor, int]