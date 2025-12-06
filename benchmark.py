"""
Batch runner to compare Backtracking vs GWO on multiple scheduling scenarios.

Usage:
    python benchmark.py

The output is a text table with per-scenario tardiness and runtime so you can
see which algorithm performs better under different conditions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from algorithms import backtracking_schedule, gwo_schedule
from models import Task


@dataclass
class Scenario:
    name: str
    tasks: List[Task]
    horizon: int
    step: int = 1
    pack: int = 30
    iterations: int = 80
    seed: Optional[int] = None


def _random_tasks(seed: int, n: int, max_duration: int, horizon: int) -> List[Task]:
    random.seed(seed)
    tasks: List[Task] = []
    for i in range(n):
        duration = random.randint(1, max_duration)
        release = random.randint(0, horizon // 2)
        deadline = random.randint(release + duration, horizon)
        tasks.append(Task(name=f"J{i+1}", duration=duration, deadline=deadline, release_time=release))
    return tasks


def build_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="tight_deadlines",
            horizon=20,
            step=1,
            pack=28,
            iterations=40,
            seed=10,
            tasks=[
                Task("A", 3, 5, 0),
                Task("B", 2, 6, 0),
                Task("C", 4, 7, 1),
                Task("D", 1, 4, 0),
            ],
        ),
        Scenario(
            name="staggered_releases",
            horizon=30,
            step=1,
            pack=22,
            iterations=60,
            seed=0,
            tasks=[
                Task("A", 5, 12, 0),
                Task("B", 3, 18, 5),
                Task("C", 4, 16, 8),
                Task("D", 2, 15, 10),
                Task("E", 6, 28, 12),
            ],
        ),
        Scenario(
            name="precedence_chain",
            horizon=30,
            step=1,
            pack=38,
            iterations=60,
            seed=6,
            tasks=[
                Task("A", 3, 8, 0),
                Task("B", 2, 12, 1, predecessors=["A"]),
                Task("C", 4, 16, 2, predecessors=["A"]),
                Task("D", 3, 20, 5, predecessors=["B", "C"]),
                Task("E", 2, 18, 0),
                Task("F", 1, 15, 4, predecessors=["E"]),
            ],
        ),
        Scenario(
            name="overload_near_deadline",
            horizon=25,
            step=1,
            pack=20,
            iterations=40,
            seed=11,
            tasks=[
                Task("A", 4, 10, 0),
                Task("B", 4, 11, 0),
                Task("C", 4, 12, 0),
                Task("D", 4, 13, 0),
                Task("E", 4, 14, 0),
            ],
        ),
        Scenario(
            name="late_releases",
            horizon=40,
            step=1,
            pack=45,
            iterations=90,
            seed=3,
            tasks=[
                Task("A", 4, 28, 10),
                Task("B", 3, 26, 12),
                Task("C", 5, 35, 18, predecessors=["A"]),
                Task("D", 2, 24, 15),
                Task("E", 3, 32, 20, predecessors=["B", "D"]),
                Task("F", 2, 38, 25),
            ],
        ),
        Scenario(
            name="step2_balanced",
            horizon=26,
            step=2,
            pack=40,
            iterations=60,
            seed=7,
            tasks=[
                Task("A", 4, 7, 0),
                Task("B", 4, 7, 0),
                Task("C", 5, 22, 4, predecessors=["A"]),
                Task("D", 2, 18, 8, predecessors=["B"]),
                Task("E", 4, 30, 12, predecessors=["B", "C"]),
            ],
        ),
        Scenario(
            name="random_seeded_small",
            horizon=30,
            step=1,
            pack=28,
            iterations=55,
            seed=0,
            tasks=_random_tasks(seed=42, n=7, max_duration=5, horizon=30),
        ),
    ]


def run_comparison(scenario: Scenario) -> Dict[str, Tuple[float, float, bool]]:
    if scenario.seed is not None:
        random.seed(scenario.seed)
    backtracking = backtracking_schedule(
        scenario.tasks, horizon=scenario.horizon, step=scenario.step
    )
    gwo = gwo_schedule(
        scenario.tasks,
        horizon=scenario.horizon,
        pack_size=scenario.pack,
        iterations=scenario.iterations,
    )
    return {
        "backtracking": (backtracking.total_tardiness, backtracking.runtime_ms, backtracking.feasible),
        "gwo": (gwo.total_tardiness, gwo.runtime_ms, gwo.feasible),
    }


def format_table(results: Dict[str, Dict[str, Tuple[float, float, bool]]]) -> str:
    lines = []
    header = f"{'Scenario':20} | {'Algo':12} | {'Tardiness':10} | {'Runtime(ms)':11} | Feasible"
    lines.append(header)
    lines.append("-" * len(header))
    for scenario, row in results.items():
        for algo in ("backtracking", "gwo"):
            tardiness, runtime_ms, feasible = row[algo]
            lines.append(
                f"{scenario:20} | {algo:12} | {tardiness:10.2f} | {runtime_ms:11.2f} | {'yes' if feasible else 'no'}"
            )
    return "\n".join(lines)


def main(run: Callable[[Scenario], Dict[str, Tuple[float, float, bool]]] = run_comparison) -> None:
    scenarios = build_scenarios()
    results: Dict[str, Dict[str, Tuple[float, float, bool]]] = {}
    for scenario in scenarios:
        results[scenario.name] = run(scenario)
    print(format_table(results))


if __name__ == "__main__":
    main()
