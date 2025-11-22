from __future__ import annotations

import math
import random
import time
from typing import Iterable, List, Optional, Tuple

from models import ScheduleResult, Task


def compute_metrics(
    tasks: Iterable[Task], schedule: List[Tuple[str, int, int]], horizon: int
) -> Tuple[float, bool]:
    """Return (total_tardiness, feasible_flag) for a concrete schedule."""

    task_map = {t.name: t for t in tasks}
    tardiness = 0.0
    feasible = True
    ordered = sorted(schedule, key=lambda entry: entry[1])

    for idx, (name, start, end) in enumerate(ordered):
        task = task_map.get(name)
        if task is None:
            feasible = False
            continue

        if start < task.release_time or end > horizon or (end - start) != task.duration:
            feasible = False

        if idx > 0 and start < ordered[idx - 1][2]:
            feasible = False

        tardiness += max(0, end - task.deadline)

    # Ensure we scheduled every task exactly once.
    if len(task_map) != len(ordered):
        feasible = False

    return tardiness, feasible


def backtracking_schedule(
    tasks: List[Task], horizon: int, step: int = 1
) -> ScheduleResult:
    """Depth-first backtracking search on discrete time slots."""

    start_time = time.perf_counter()
    if not tasks:
        runtime_ms = (time.perf_counter() - start_time) * 1000
        return ScheduleResult(
            algo="Backtracking",
            schedule=[],
            total_tardiness=0.0,
            runtime_ms=runtime_ms,
            feasible=True,
        )

    tasks_sorted = sorted(tasks, key=lambda t: (t.deadline, -t.duration))
    best_schedule: Optional[List[Tuple[str, int, int]]] = None
    best_score = math.inf
    placed: List[Tuple[str, int, int]] = []

    def overlaps(start: int, duration: int) -> bool:
        end = start + duration
        if end > horizon:
            return True

        for _, s, e in placed:
            if max(s, start) < min(e, end):
                return True
        return False

    def search(idx: int, current_cost: float) -> None:
        nonlocal best_schedule, best_score

        if idx == len(tasks_sorted):
            snapshot = sorted(placed.copy(), key=lambda entry: entry[1])
            best_schedule = snapshot
            best_score = current_cost
            return

        task = tasks_sorted[idx]
        latest_start = max(task.release_time, horizon - task.duration)

        for start in range(task.release_time, latest_start + 1, step):
            if overlaps(start, task.duration):
                continue

            end = start + task.duration
            tardiness = max(0, end - task.deadline)
            next_cost = current_cost + tardiness

            # Branch and bound: prune if already worse than best known.
            if next_cost >= best_score:
                continue

            placed.append((task.name, start, end))
            search(idx + 1, next_cost)
            placed.pop()

    search(0, 0.0)
    runtime_ms = (time.perf_counter() - start_time) * 1000

    if best_schedule is None:
        return ScheduleResult(
            algo="Backtracking",
            schedule=[],
            total_tardiness=math.inf,
            runtime_ms=runtime_ms,
            feasible=False,
        )

    total_tardiness, feasible = compute_metrics(tasks, best_schedule, horizon)

    return ScheduleResult(
        algo="Backtracking",
        schedule=best_schedule,
        total_tardiness=total_tardiness,
        runtime_ms=runtime_ms,
        feasible=feasible,
    )


def _score_position(
    tasks: List[Task], position: List[float], bounds: List[Tuple[int, int]], horizon: int
) -> float:
    """Objective for PSO: tardiness + overlap/out-of-window penalties."""

    tardiness = 0.0
    penalty = 0.0
    entries: List[Tuple[float, float]] = []

    for idx, value in enumerate(position):
        lo, hi = bounds[idx]
        start = max(lo, min(value, hi))
        task = tasks[idx]
        end = start + task.duration
        tardiness += max(0.0, end - task.deadline)

        # Penalize if we cannot fit the job.
        if end > horizon or start < task.release_time:
            penalty += 50.0

        entries.append((start, end))

    entries.sort(key=lambda item: item[0])

    # Penalize pairwise overlaps so the swarm learns to separate tasks.
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            overlap = max(0.0, entries[i][1] - entries[j][0])
            if overlap > 0:
                penalty += overlap * 10.0

    return tardiness + penalty


def _positions_to_schedule(
    tasks: List[Task], position: List[float], bounds: List[Tuple[int, int]], horizon: int
) -> List[Tuple[str, int, int]]:
    """Convert swarm position to a feasible, non-overlapping schedule."""

    blocks: List[Tuple[float, float, Task]] = []
    for idx, task in enumerate(tasks):
        lo, hi = bounds[idx]
        start = max(lo, min(position[idx], hi))
        end = start + task.duration
        blocks.append((start, end, task))

    blocks.sort(key=lambda entry: entry[0])
    schedule: List[Tuple[str, int, int]] = []
    current_time = 0.0

    for start, end, task in blocks:
        start = max(start, current_time, task.release_time)
        end = start + task.duration

        if end > horizon:
            start = max(task.release_time, horizon - task.duration)
            end = start + task.duration

        schedule.append((task.name, int(round(start)), int(round(end))))
        current_time = end

    return schedule


def pso_schedule(
    tasks: List[Task],
    horizon: int,
    swarm_size: int = 30,
    iterations: int = 80,
    inertia: float = 0.6,
    c1: float = 1.5,
    c2: float = 1.5,
) -> ScheduleResult:
    """Particle Swarm Optimization for the same scheduling objective."""

    start_time = time.perf_counter()
    if not tasks:
        runtime_ms = (time.perf_counter() - start_time) * 1000
        return ScheduleResult(
            algo="PSO",
            schedule=[],
            total_tardiness=0.0,
            runtime_ms=runtime_ms,
            feasible=True,
            iterations=iterations,
        )

    bounds: List[Tuple[int, int]] = []
    for task in tasks:
        latest_start = max(task.release_time, horizon - task.duration)
        bounds.append((task.release_time, latest_start))

    dim = len(tasks)
    positions: List[List[float]] = []
    velocities: List[List[float]] = []
    personal_best: List[List[float]] = []
    personal_scores: List[float] = []

    for _ in range(swarm_size):
        pos = [random.uniform(lo, hi) for lo, hi in bounds]
        vel = [0.0 for _ in range(dim)]
        positions.append(pos)
        velocities.append(vel)
        personal_best.append(pos.copy())
        personal_scores.append(_score_position(tasks, pos, bounds, horizon))

    global_best_idx = min(range(swarm_size), key=lambda i: personal_scores[i])
    global_best = personal_best[global_best_idx].copy()
    global_best_score = personal_scores[global_best_idx]

    for _ in range(iterations):
        for i in range(swarm_size):
            current_score = _score_position(tasks, positions[i], bounds, horizon)
            if current_score < personal_scores[i]:
                personal_scores[i] = current_score
                personal_best[i] = positions[i].copy()

        global_best_idx = min(range(swarm_size), key=lambda i: personal_scores[i])
        if personal_scores[global_best_idx] < global_best_score:
            global_best_score = personal_scores[global_best_idx]
            global_best = personal_best[global_best_idx].copy()

        for i in range(swarm_size):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                velocities[i][d] = (
                    inertia * velocities[i][d]
                    + c1 * r1 * (personal_best[i][d] - positions[i][d])
                    + c2 * r2 * (global_best[d] - positions[i][d])
                )
                positions[i][d] += velocities[i][d]

                lo, hi = bounds[d]
                positions[i][d] = max(lo, min(positions[i][d], hi))

    runtime_ms = (time.perf_counter() - start_time) * 1000
    schedule = _positions_to_schedule(tasks, global_best, bounds, horizon)
    total_tardiness, feasible = compute_metrics(tasks, schedule, horizon)

    return ScheduleResult(
        algo="PSO",
        schedule=schedule,
        total_tardiness=total_tardiness,
        runtime_ms=runtime_ms,
        feasible=feasible,
        iterations=iterations,
    )


def describe_schedule(result: ScheduleResult) -> str:
    """Human-friendly multi-line summary for the GUI."""

    lines = [
        f"Thuat toan: {result.algo}",
        f"Thoi gian chay: {result.runtime_ms:.2f} ms",
        f"Tong do tre: {result.total_tardiness:.2f} (don vi thoi gian)",
        f"Tinh kha thi: {'Co' if result.feasible else 'Khong'}",
    ]

    if result.iterations is not None:
        lines.append(f"So vong lap: {result.iterations}")

    if result.schedule:
        lines.append("Lich chi tiet (ten, bat dau, ket thuc) - don vi thoi gian:")
        for name, start, end in result.schedule:
            lines.append(f"  - {name}: {start} -> {end} (dv)")
    else:
        lines.append("Chua co lich kha thi.")

    return "\n".join(lines)
