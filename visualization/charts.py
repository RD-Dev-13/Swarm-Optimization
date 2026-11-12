"""
Visualization Module for Cloud Resource Scheduling Simulation.

Implements all 10 academic benchmark plots specified in Component 12:
1. Task CPU demand bar chart.
2. Task arrival timeline.
3. VM utilization chart.
4. VM workload comparison.
5. Algorithm convergence curves.
6. Makespan comparison.
7. Load imbalance comparison.
8. Adaptive algorithm selection timeline.
9. Population diversity over time.
10. Fitness improvement over time.

Generates publication-ready Matplotlib figures with clear axis labels, units, and legends.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated script/test execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator
from algorithms.base import OptimizationResult


class CloudSchedulerVisualizer:
    """
    Renders and exports graphical visualizations for cloud scheduling analysis.
    """

    @staticmethod
    def _save_or_return(fig: plt.Figure, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Helper to save figure to disk if path is provided, and close it cleanly."""
        if save_path:
            p = Path(save_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
        return fig

    @classmethod
    def plot_task_cpu_demands(
        cls,
        tasks: List[Task],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """1. Task CPU Demand Bar Chart."""
        fig, ax = plt.subplots(figsize=(10, 4.5))
        task_ids = [t.task_id for t in tasks]
        demands = [t.cpu_demand_mi for t in tasks]
        priorities = [t.priority for t in tasks]

        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(tasks)))
        bars = ax.bar(task_ids, demands, color=colors, edgecolor="black", alpha=0.85)

        ax.set_title("Computational Workload Demand per Cloud Task", fontsize=12, fontweight="bold")
        ax.set_xlabel("Task ID", fontsize=10)
        ax.set_ylabel("CPU Demand (Million Instructions - MI)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45, ha="right", fontsize=8)

        # Annotate max value
        max_idx = int(np.argmax(demands))
        ax.annotate(
            f"Peak: {demands[max_idx]:.0f} MI",
            xy=(max_idx, demands[max_idx]),
            xytext=(max_idx, demands[max_idx] * 1.08),
            arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=5),
            ha="center",
            fontsize=8,
            fontweight="bold"
        )
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_task_arrivals(
        cls,
        tasks: List[Task],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """2. Task Arrival Timeline."""
        fig, ax = plt.subplots(figsize=(10, 4))
        arrivals = [t.arrival_time for t in tasks]
        indices = list(range(1, len(tasks) + 1))

        ax.step(arrivals, indices, where="post", color="#1f77b4", linewidth=2, label="Cumulative Tasks")
        ax.scatter(arrivals, indices, color="#d62728", s=30, zorder=3, label="Arrival Event")

        ax.set_title("Chronological Task Arrival Progression", fontsize=12, fontweight="bold")
        ax.set_xlabel("Simulation Time (seconds)", fontsize=10)
        ax.set_ylabel("Cumulative Tasks Submitted", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_vm_utilization(
        cls,
        vm_utilizations: Dict[str, float],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """3. VM Utilization Percentage Bar Chart."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vm_names = list(vm_utilizations.keys())
        utils = list(vm_utilizations.values())

        bars = ax.bar(vm_names, utils, color="#2ca02c", edgecolor="black", alpha=0.85, width=0.5)
        ax.set_ylim(0, 105)
        ax.axhline(100, color="gray", linestyle=":", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax.set_title("CPU Utilization per Virtual Machine", fontsize=12, fontweight="bold")
        ax.set_xlabel("Virtual Machine ID", fontsize=10)
        ax.set_ylabel("CPU Utilization (%)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_vm_workload_distribution(
        cls,
        simulator: VMSimulator,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """4. VM Workload and Busy Time Comparison."""
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        vms = simulator.vm_list
        vm_ids = [vm.vm_id for vm in vms]
        workloads = [vm.current_workload_mi for vm in vms]
        busy_times = [vm.busy_time for vm in vms]

        x = np.arange(len(vm_ids))
        width = 0.35

        rects1 = ax1.bar(x - width/2, workloads, width, label="Workload (MI)", color="#3b528b", edgecolor="black")
        ax1.set_ylabel("Total Task Workload (MI)", color="#3b528b", fontsize=10)
        ax1.tick_params(axis="y", labelcolor="#3b528b")

        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, busy_times, width, label="Busy Time (s)", color="#5dc863", edgecolor="black")
        ax2.set_ylabel("Active Busy Time (seconds)", color="#21918c", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#21918c")

        ax1.set_xticks(x)
        ax1.set_xticklabels(vm_ids)
        ax1.set_title("VM Workload and Active Execution Time Distribution", fontsize=12, fontweight="bold")
        ax1.grid(axis="x", linestyle="--", alpha=0.5)
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_convergence_curves(
        cls,
        algorithm_results: Dict[str, OptimizationResult],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """5. Algorithm Convergence Curves."""
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"PSO": "#1f77b4", "GWO": "#ff7f0e", "ABC": "#2ca02c", "Adaptive": "#d62728"}

        for name, res in algorithm_results.items():
            if res.convergence_history:
                iters = list(range(1, len(res.convergence_history) + 1))
                color = colors.get(name, "#7f7f7f")
                linewidth = 2.5 if name == "Adaptive" else 1.5
                linestyle = "-" if name == "Adaptive" else "--"
                ax.plot(iters, res.convergence_history, label=f"{name} (Final: {res.best_fitness:.2f})",
                        color=color, linewidth=linewidth, linestyle=linestyle)

        ax.set_title("Optimization Convergence Velocity Comparison", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generation / Iteration", fontsize=10)
        ax.set_ylabel("Best Objective Fitness (Lower is Better)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_makespan_comparison(
        cls,
        comparison_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """6. Makespan Comparison Bar Chart."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        algos = comparison_df["algorithm"].tolist()
        makespans = comparison_df["makespan_sec"].tolist()

        colors = ["#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bar_colors = colors[:len(algos)]

        bars = ax.bar(algos, makespans, color=bar_colors, edgecolor="black", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=8)

        ax.set_title("Makespan Benchmark Comparison Across Schedulers", fontsize=12, fontweight="bold")
        ax.set_xlabel("Scheduling Algorithm", fontsize=10)
        ax.set_ylabel("Total Makespan (seconds)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=25, ha="right")
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_load_imbalance_comparison(
        cls,
        comparison_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """7. Load Imbalance Comparison Bar Chart."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        algos = comparison_df["algorithm"].tolist()
        imbalance = comparison_df["load_imbalance"].tolist()

        bars = ax.bar(algos, imbalance, color="#9467bd", edgecolor="black", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=8)

        ax.set_title("Degree of Load Imbalance (DI) Comparison", fontsize=12, fontweight="bold")
        ax.set_xlabel("Scheduling Algorithm", fontsize=10)
        ax.set_ylabel("Degree of Imbalance (Lower is Better)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=25, ha="right")
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_adaptive_strategy_timeline(
        cls,
        decision_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """8. Adaptive Controller Strategy Selection Timeline."""
        fig, ax = plt.subplots(figsize=(10, 3.5))
        if decision_df.empty or "selected_algorithm" not in decision_df:
            return fig

        algo_map = {"PSO": 0, "GWO": 1, "ABC": 2}
        colors = {"PSO": "#1f77b4", "GWO": "#ff7f0e", "ABC": "#2ca02c"}

        iters = decision_df["iteration"].tolist()
        algos = decision_df["selected_algorithm"].tolist()
        y_vals = [algo_map.get(a, 0) for a in algos]
        pt_colors = [colors.get(a, "#000000") for a in algos]

        ax.step(iters, y_vals, where="mid", color="gray", linestyle=":", alpha=0.7)
        ax.scatter(iters, y_vals, c=pt_colors, s=50, edgecolor="black", zorder=3)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["PSO (Momentum)", "GWO (Exploitation)", "ABC (Exploration)"])
        ax.set_title("Adaptive Strategy Selection Trail Over Generations", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generation / Iteration", fontsize=10)
        ax.set_ylabel("Selected Heuristic Strategy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_diversity_over_time(
        cls,
        algorithm_results: Dict[str, OptimizationResult],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """9. Population Diversity Trajectory Over Time."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = {"PSO": "#1f77b4", "GWO": "#ff7f0e", "ABC": "#2ca02c", "Adaptive": "#d62728"}

        for name, res in algorithm_results.items():
            if res.diversity_history:
                iters = list(range(1, len(res.diversity_history) + 1))
                color = colors.get(name, "#7f7f7f")
                lw = 2.5 if name == "Adaptive" else 1.5
                ax.plot(iters, res.diversity_history, label=name, color=color, linewidth=lw)

        ax.axhline(0.15, color="red", linestyle=":", label="Critical Low Diversity Threshold (0.15)")
        ax.set_ylim(-0.02, 1.0)
        ax.set_title("Population Diversity Over Optimization Generations", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generation / Iteration", fontsize=10)
        ax.set_ylabel("Normalized Pairwise Hamming Diversity in [0, 1]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        return cls._save_or_return(fig, save_path)

    @classmethod
    def plot_fitness_improvement_over_time(
        cls,
        decision_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """10. Fitness Improvement Rate Over Time."""
        fig, ax = plt.subplots(figsize=(10, 4))
        if decision_df.empty or "fitness_improvement" not in decision_df:
            return fig

        iters = decision_df["iteration"].tolist()
        improvements = (decision_df["fitness_improvement"] * 100.0).tolist()  # Percentage

        ax.bar(iters, improvements, color="#17becf", edgecolor="black", alpha=0.8)
        ax.axhline(0.5, color="orange", linestyle="--", label="Progress Threshold (0.5%)")

        ax.set_title("Relative Fitness Improvement (%) per Adaptive Generation", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generation / Iteration", fontsize=10)
        ax.set_ylabel("Relative Improvement (%)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        return cls._save_or_return(fig, save_path)


if __name__ == "__main__":
    print("=" * 70)
    print("  VISUALIZATION MODULE STANDALONE DEMO")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator

    gen = TaskGenerator(seed=42)
    sample_tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")

    out_dir = Path("results/graphs")
    fig1 = CloudSchedulerVisualizer.plot_task_cpu_demands(
        sample_tasks, out_dir / "test_cpu_demands.png"
    )
    plt.close(fig1)

    fig2 = CloudSchedulerVisualizer.plot_task_arrivals(
        sample_tasks, out_dir / "test_task_arrivals.png"
    )
    plt.close(fig2)

    print(f"Successfully generated sample charts to {out_dir.resolve()}")
    print("=" * 70)

