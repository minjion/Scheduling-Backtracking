from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Task:
    """Represents a single job in the schedule."""

    name: str
    duration: int
    deadline: int
    release_time: int = 0


@dataclass
class ScheduleResult:
    """Algorithm output container."""

    algo: str
    schedule: List[Tuple[str, int, int]]
    total_tardiness: float
    runtime_ms: float
    feasible: bool
    iterations: Optional[int] = None
