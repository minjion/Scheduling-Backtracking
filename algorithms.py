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
    """Objective: tardiness + overlap/out-of-window penalties."""

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
    """Convert a continuous position vector to a feasible, non-overlapping schedule."""

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


def gwo_schedule(
    tasks: List[Task],
    horizon: int,
    pack_size: int = 30,
    iterations: int = 80,
) -> ScheduleResult:
    """Grey Wolf Optimizer cho cùng hàm mục tiêu xếp lịch."""

    start_time = time.perf_counter()
    if not tasks:
        runtime_ms = (time.perf_counter() - start_time) * 1000
        return ScheduleResult(
            algo="GWO",
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
    wolves: List[List[float]] = []

    alpha_pos: Optional[List[float]] = None
    beta_pos: Optional[List[float]] = None
    delta_pos: Optional[List[float]] = None
    alpha_score = math.inf
    beta_score = math.inf
    delta_score = math.inf

    for _ in range(pack_size):
        pos = [random.uniform(lo, hi) for lo, hi in bounds]
        wolves.append(pos)
        score = _score_position(tasks, pos, bounds, horizon)
        if score < alpha_score:
            delta_score, delta_pos = beta_score, beta_pos
            beta_score, beta_pos = alpha_score, alpha_pos
            alpha_score, alpha_pos = score, pos.copy()
        elif score < beta_score:
            delta_score, delta_pos = beta_score, beta_pos
            beta_score, beta_pos = score, pos.copy()
        elif score < delta_score:
            delta_score, delta_pos = score, pos.copy()

    for t in range(iterations):
        a = 2 - 2 * (t / max(1, iterations))
        for i in range(pack_size):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                r3, r4 = random.random(), random.random()
                r5, r6 = random.random(), random.random()

                alpha_d = alpha_pos[d] if alpha_pos else wolves[i][d]
                beta_d = beta_pos[d] if beta_pos else wolves[i][d]
                delta_d = delta_pos[d] if delta_pos else wolves[i][d]

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_d - wolves[i][d])
                X1 = alpha_d - A1 * D_alpha

                A2 = 2 * a * r3 - a
                C2 = 2 * r4
                D_beta = abs(C2 * beta_d - wolves[i][d])
                X2 = beta_d - A2 * D_beta

                A3 = 2 * a * r5 - a
                C3 = 2 * r6
                D_delta = abs(C3 * delta_d - wolves[i][d])
                X3 = delta_d - A3 * D_delta

                new_pos = (X1 + X2 + X3) / 3.0
                lo, hi = bounds[d]
                wolves[i][d] = max(lo, min(new_pos, hi))

        for i in range(pack_size):
            score = _score_position(tasks, wolves[i], bounds, horizon)
            if score < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = alpha_score, alpha_pos
                alpha_score, alpha_pos = score, wolves[i].copy()
            elif score < beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = score, wolves[i].copy()
            elif score < delta_score:
                delta_score, delta_pos = score, wolves[i].copy()

    best_position = alpha_pos if alpha_pos is not None else wolves[0]
    runtime_ms = (time.perf_counter() - start_time) * 1000
    schedule = _positions_to_schedule(tasks, best_position, bounds, horizon)
    total_tardiness, feasible = compute_metrics(tasks, schedule, horizon)

    return ScheduleResult(
        algo="GWO",
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
