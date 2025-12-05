import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List

from algorithms import backtracking_schedule, describe_schedule, gwo_schedule
from models import Task


class SchedulerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Scheduling - Backtracking vs GWO")
        self.tasks: List[Task] = []

        self._build_task_form()
        self._build_params_section()
        self._build_actions()
        self._build_output()
        self._load_demo_tasks()

    def _build_task_form(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Nhập công việc")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.name_var = tk.StringVar()
        self.duration_var = tk.StringVar(value="2")
        self.deadline_var = tk.StringVar(value="6")
        self.release_var = tk.StringVar(value="0")

        ttk.Label(frame, text="Tên").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.name_var, width=20).grid(
            row=0, column=1, sticky="ew", pady=2
        )

        ttk.Label(frame, text="Thời lượng").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.duration_var, width=10).grid(
            row=1, column=1, sticky="ew", pady=2
        )

        ttk.Label(frame, text="Deadline").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.deadline_var, width=10).grid(
            row=2, column=1, sticky="ew", pady=2
        )

        ttk.Label(frame, text="Release").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.release_var, width=10).grid(
            row=3, column=1, sticky="ew", pady=2
        )

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=4, sticky="ew")
        ttk.Button(btn_frame, text="Thêm", command=self._add_task).grid(
            row=0, column=0, padx=4
        )
        ttk.Button(btn_frame, text="Xóa", command=self._remove_task).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(btn_frame, text="Demo", command=self._load_demo_tasks).grid(
            row=0, column=2, padx=4
        )

        columns = ("name", "duration", "deadline", "release")
        self.tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            height=8,
            selectmode="browse",
        )
        self.tree.heading("name", text="Tên")
        self.tree.heading("duration", text="Thời lượng")
        self.tree.heading("deadline", text="Deadline")
        self.tree.heading("release", text="Release")
        self.tree.column("name", width=90, anchor="center")
        self.tree.column("duration", width=90, anchor="center")
        self.tree.column("deadline", width=80, anchor="center")
        self.tree.column("release", width=80, anchor="center")
        self.tree.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=4)

        frame.columnconfigure(1, weight=1)

    def _build_params_section(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Tham số chung")
        frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.horizon_var = tk.StringVar(value="24")
        self.step_var = tk.StringVar(value="1")
        self.swarm_var = tk.StringVar(value="30")
        self.iter_var = tk.StringVar(value="80")

        ttk.Label(frame, text="Horizon (tổng thời gian, dv)").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.horizon_var, width=10).grid(
            row=0, column=1, sticky="ew", pady=2
        )

        ttk.Label(frame, text="Bước thời gian (dv, backtracking)").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.step_var, width=10).grid(
            row=1, column=1, sticky="ew", pady=2
        )

        ttk.Label(frame, text="Pack size (GWO)").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.swarm_var, width=10).grid(
            row=2, column=1, sticky="ew", pady=2
        )

        ttk.Label(frame, text="Số vòng lặp (GWO)").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.iter_var, width=10).grid(
            row=3, column=1, sticky="ew", pady=2
        )

        frame.columnconfigure(1, weight=1)

    def _build_actions(self) -> None:
        frame = ttk.Frame(self.root)
        frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Button(frame, text="Chạy Backtracking", command=self.run_backtracking).grid(
            row=0, column=0, padx=4
        )
        ttk.Button(frame, text="Chạy GWO", command=self.run_gwo).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(frame, text="So sánh", command=self.compare_algorithms).grid(
            row=0, column=2, padx=4
        )

    def _build_output(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Kết quả")
        frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.output = tk.Text(frame, height=16, wrap="word")
        self.output.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.output.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scroll.set)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)

    def _add_task(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Lỗi", "Tên không được để trống")
            return

        try:
            duration = int(self.duration_var.get())
            deadline = int(self.deadline_var.get())
            release = int(self.release_var.get())
        except ValueError:
            messagebox.showerror("Lỗi", "Thời lượng/deadline/release phải là số nguyên")
            return

        if duration <= 0:
            messagebox.showerror("Lỗi", "Thời lượng phải lớn hơn 0")
            return

        if any(task.name == name for task in self.tasks):
            messagebox.showerror("Lỗi", "Tên công việc phải duy nhất")
            return

        task = Task(name=name, duration=duration, deadline=deadline, release_time=release)
        self.tasks.append(task)
        self.tree.insert("", "end", iid=name, values=(name, duration, deadline, release))

    def _remove_task(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return

        item_id = selection[0]
        self.tree.delete(item_id)
        self.tasks = [t for t in self.tasks if t.name != item_id]

    def _load_demo_tasks(self) -> None:
        demo = [
            Task("A", duration=3, deadline=8, release_time=0),
            Task("B", duration=2, deadline=5, release_time=1),
            Task("C", duration=4, deadline=12, release_time=0),
            Task("D", duration=1, deadline=4, release_time=0),
        ]
        self.tasks = demo
        for item in self.tree.get_children():
            self.tree.delete(item)

        for task in demo:
            self.tree.insert(
                "",
                "end",
                iid=task.name,
                values=(task.name, task.duration, task.deadline, task.release_time),
            )

    def _read_common_params(self) -> tuple[int, int]:
        try:
            horizon = int(self.horizon_var.get())
            step = int(self.step_var.get())
        except ValueError:
            raise ValueError("Horizon/step phải là số nguyên")

        if horizon <= 0 or step <= 0:
            raise ValueError("Horizon và step phải lớn hơn 0")

        return horizon, step

    def _read_gwo_params(self) -> tuple[int, int]:
        try:
            pack = int(self.swarm_var.get())
            iterations = int(self.iter_var.get())
        except ValueError:
            raise ValueError("Pack size và số vòng lặp phải là số nguyên")

        if pack <= 0 or iterations <= 0:
            raise ValueError("Pack size và số vòng lặp phải lớn hơn 0")

        return pack, iterations

    def run_backtracking(self) -> None:
        if not self.tasks:
            messagebox.showwarning("Lỗi", "Chưa có công việc nào")
            return

        try:
            horizon, step = self._read_common_params()
        except ValueError as exc:
            messagebox.showerror("Lỗi", str(exc))
            return

        result = backtracking_schedule(self.tasks, horizon=horizon, step=step)
        self._show_result(result)

    def run_gwo(self) -> None:
        if not self.tasks:
            messagebox.showwarning("Lỗi", "Chưa có công việc nào")
            return

        try:
            horizon, _ = self._read_common_params()
            pack, iterations = self._read_gwo_params()
        except ValueError as exc:
            messagebox.showerror("Lỗi", str(exc))
            return

        result = gwo_schedule(
            self.tasks,
            horizon=horizon,
            pack_size=pack,
            iterations=iterations,
        )
        self._show_result(result)

    def compare_algorithms(self) -> None:
        if not self.tasks:
            messagebox.showwarning("Lỗi", "Chưa có công việc nào")
            return

        try:
            horizon, step = self._read_common_params()
            pack, iterations = self._read_gwo_params()
        except ValueError as exc:
            messagebox.showerror("Lỗi", str(exc))
            return

        backtracking = backtracking_schedule(self.tasks, horizon=horizon, step=step)
        gwo = gwo_schedule(self.tasks, horizon=horizon, pack_size=pack, iterations=iterations)

        summary = [
            "=== Backtracking ===",
            describe_schedule(backtracking),
            "",
            "=== GWO ===",
            describe_schedule(gwo),
        ]

        if backtracking.total_tardiness != math.inf and gwo.total_tardiness != math.inf:
            better = "Backtracking" if backtracking.total_tardiness <= gwo.total_tardiness else "GWO"
            summary.append(f"\nGiải pháp tốt hơn: {better}")

        self._show_text("\n".join(summary))

    def _show_result(self, result) -> None:
        self._show_text(describe_schedule(result))

    def _show_text(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    SchedulerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
