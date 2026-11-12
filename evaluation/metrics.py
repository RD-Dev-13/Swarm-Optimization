"""
Performance Evaluation Metrics Module for Cloud Resource Scheduling.

Implements all 10 academic benchmark metrics specified in Component 9:
1. Makespan (M): Maximum completion time across all VMs.
2. Average VM Utilization (U_avg): Cluster-wide mean CPU utilization percentage.
3. Per-VM Utilization (U_j): Individual CPU utilization percentage per host.
4. Load Imbalance: Degree of Imbalance (DI) and standard deviation of busy times.
5. Average Task Waiting Time (W_avg): Mean queue delay before execution starts.
6. Average Task Execution Time (E_avg): Mean computational processing duration.
7. Total Task Completion Time (T_total): Cumulative flow time / turnaround time.
8. Optimization Convergence Speed: Iterations to reach target threshold and AUC.
9. Algorithm Switching Frequency (f_switch): Controller transitions per optimization epoch.
10. Adaptation Response Time (T_adapt): Iteration latency to re-stabilize after an event.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from simulation.vm_simulator import VMSimulator, ScheduledTaskRecord


@dataclass
class PerformanceMetricsReport:
    """
    Comprehensive performance metrics report for a scheduling solution.
    """
    makespan_sec: float
    avg_utilization_pct: float
    per_vm_utilization: Dict[str, float]
    degree_of_imbalance: float
    busy_time_std_sec: float
    avg_waiting_time_sec: float
    avg_execution_time_sec: float
    total_flow_time_sec: float
    total_active_cpu_time_sec: float
    task_count: int
    vm_count: int
    convergence_speed_iterations: Optional[int] = None
    switching_frequency: Optional[float] = None
    adaptation_response_iterations: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize report to dictionary."""
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Format metrics as a single-row comparative DataFrame."""
        flat = {k: v for k, v in self.to_dict().items() if not isinstance(v, dict)}
        for vm_id, util in self.per_vm_utilization.items():
            flat[f"util_{vm_id}_pct"] = util
        return pd.DataFrame([flat])


class MetricsCalculator:
    """
    Computes rigorous mathematical evaluation metrics for cloud scheduling simulations.
    """

    @staticmethod
    def calculate_makespan(records: List[ScheduledTaskRecord]) -> float:
        """
        Calculate overall makespan (M).

        Formula:
            M = max_{i=1..N} (finish_time_i)

        Args:
            records: List of scheduled task records.

        Returns:
            Makespan in seconds (0.0 if empty).
        """
        if not records:
            return 0.0
        return round(float(max(r.finish_time for r in records)), 4)

    @staticmethod
    def calculate_utilizations(
        simulator: VMSimulator,
        makespan: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate average and per-VM CPU utilization percentages.

        Formula:
            U_j (%) = (busy_time_j / Makespan) * 100%
            U_avg (%) = (1 / m) * sum_{j=1..m} U_j

        Sequential single-core execution per VM guarantees U_j <= 100.0%.

        Args:
            simulator: VMSimulator cluster instance.
            makespan: Optional simulation makespan; if None, computed from cluster.

        Returns:
            Tuple of (average_utilization_pct, dict of {vm_id: util_pct}).
        """
        m = makespan if makespan is not None else simulator.get_makespan()
        if m <= 0.0:
            return 0.0, {vm.vm_id: 0.0 for vm in simulator.vm_list}

        per_vm = {
            vm.vm_id: vm.calculate_utilization(m)
            for vm in simulator.vm_list
        }
        avg_util = round(float(np.mean(list(per_vm.values()))), 2)
        return avg_util, per_vm

    @staticmethod
    def calculate_load_imbalance(
        simulator: VMSimulator
    ) -> Tuple[float, float]:
        """
        Calculate Degree of Imbalance (DI) and standard deviation of VM busy times.

        Formulas:
            DI = (B_max - B_min) / B_avg
            sigma_busy = sqrt( (1 / m) * sum_{j=1..m} (B_j - B_avg)^2 )

        Args:
            simulator: VMSimulator instance.

        Returns:
            Tuple of (degree_of_imbalance, busy_time_std_sec).
        """
        busy_times = [vm.busy_time for vm in simulator.vm_list]
        if not busy_times or sum(busy_times) == 0.0:
            return 0.0, 0.0

        b_arr = np.array(busy_times, dtype=float)
        b_max = float(np.max(b_arr))
        b_min = float(np.min(b_arr))
        b_avg = float(np.mean(b_arr))

        di = round((b_max - b_min) / b_avg, 4) if b_avg > 0 else 0.0
        std = round(float(np.std(b_arr)), 4)
        return di, std

    @staticmethod
    def calculate_task_time_metrics(
        records: List[ScheduledTaskRecord]
    ) -> Dict[str, float]:
        """
        Calculate average waiting time, average execution time, and total flow time.

        Formulas:
            W_avg = (1 / N) * sum_{i=1..N} (start_time_i - arrival_time_i)
            E_avg = (1 / N) * sum_{i=1..N} execution_time_i
            T_total = sum_{i=1..N} (finish_time_i - arrival_time_i)
            Total_Active_CPU = sum_{i=1..N} execution_time_i

        Args:
            records: Scheduled task records.

        Returns:
            Dictionary with computed time metrics in seconds.
        """
        if not records:
            return {
                "avg_waiting_time_sec": 0.0,
                "avg_execution_time_sec": 0.0,
                "total_flow_time_sec": 0.0,
                "total_active_cpu_time_sec": 0.0,
            }

        waits = [r.waiting_time for r in records]
        execs = [r.execution_time for r in records]
        turnarounds = [r.turnaround_time for r in records]

        return {
            "avg_waiting_time_sec": round(float(np.mean(waits)), 4),
            "avg_execution_time_sec": round(float(np.mean(execs)), 4),
            "total_flow_time_sec": round(float(np.sum(turnarounds)), 4),
            "total_active_cpu_time_sec": round(float(np.sum(execs)), 4),
        }

    @staticmethod
    def calculate_convergence_speed(
        fitness_history: List[float],
        tolerance_pct: float = 2.0
    ) -> int:
        """
        Determine the iteration at which optimization converged to within tolerance_pct
        of the final best fitness score.

        Formula:
            Iteration k* where fitness[k*] <= fitness_final * (1 + tolerance_pct / 100)

        Args:
            fitness_history: List of best fitness values recorded across iterations.
            tolerance_pct: Percentage band above final minimum (default: 2.0%).

        Returns:
            1-based iteration index of convergence, or total iterations if not reached.
        """
        if not fitness_history:
            return 0
        final_best = min(fitness_history)
        threshold = final_best * (1.0 + (tolerance_pct / 100.0))

        for idx, f in enumerate(fitness_history):
            if f <= threshold:
                return idx + 1
        return len(fitness_history)

    @staticmethod
    def calculate_switching_frequency(algorithm_history: List[str]) -> float:
        """
        Calculate normalized strategy switching frequency of the adaptive controller.

        Formula:
            f_switch = N_switches / (K_total - 1)
            where a switch occurs when strategy[k] != strategy[k-1].

        Args:
            algorithm_history: Sequence of algorithm identifiers chosen at each epoch.

        Returns:
            Switching frequency in [0.0, 1.0].
        """
        if not algorithm_history or len(algorithm_history) <= 1:
            return 0.0
        switches = sum(
            1 for i in range(1, len(algorithm_history))
            if algorithm_history[i] != algorithm_history[i - 1]
        )
        return round(float(switches) / (len(algorithm_history) - 1), 4)

    @staticmethod
    def calculate_adaptation_response_time(
        fitness_history: List[float],
        event_iteration: int,
        recovery_margin_pct: float = 5.0
    ) -> int:
        """
        Measure iteration latency required by the adaptive controller to re-converge
        following a disturbance event (e.g., VM degradation or workload surge).

        Args:
            fitness_history: Fitness history across iterations.
            event_iteration: Iteration index where disturbance occurred (1-based).
            recovery_margin_pct: Target margin relative to post-event stabilization.

        Returns:
            Number of iterations elapsed before stability was recovered.
        """
        total = len(fitness_history)
        if event_iteration >= total or event_iteration < 1:
            return 0

        post_event = fitness_history[event_iteration - 1:]
        if not post_event:
            return 0

        stabilized_val = min(post_event)
        target = stabilized_val * (1.0 + (recovery_margin_pct / 100.0))

        for i, val in enumerate(post_event):
            if val <= target:
                return i
        return len(post_event)

    @classmethod
    def generate_full_report(
        cls,
        records: List[ScheduledTaskRecord],
        simulator: VMSimulator,
        fitness_history: Optional[List[float]] = None,
        algorithm_history: Optional[List[str]] = None,
        event_iteration: Optional[int] = None,
    ) -> PerformanceMetricsReport:
        """
        Assemble comprehensive report encompassing all 10 scheduling and optimization metrics.

        Args:
            records: Scheduled task records.
            simulator: VMSimulator cluster instance.
            fitness_history: Optional convergence curve data.
            algorithm_history: Optional adaptive algorithm selection sequence.
            event_iteration: Optional iteration where disturbance event occurred.

        Returns:
            PerformanceMetricsReport populated with all evaluated values.
        """
        makespan = cls.calculate_makespan(records)
        avg_util, per_vm_utils = cls.calculate_utilizations(simulator, makespan)
        di, std_busy = cls.calculate_load_imbalance(simulator)
        task_times = cls.calculate_task_time_metrics(records)

        conv_speed = (
            cls.calculate_convergence_speed(fitness_history)
            if fitness_history is not None else None
        )
        switch_freq = (
            cls.calculate_switching_frequency(algorithm_history)
            if algorithm_history is not None else None
        )
        adapt_time = (
            cls.calculate_adaptation_response_time(fitness_history, event_iteration)
            if (fitness_history is not None and event_iteration is not None) else None
        )

        return PerformanceMetricsReport(
            makespan_sec=makespan,
            avg_utilization_pct=avg_util,
            per_vm_utilization=per_vm_utils,
            degree_of_imbalance=di,
            busy_time_std_sec=std_busy,
            avg_waiting_time_sec=task_times["avg_waiting_time_sec"],
            avg_execution_time_sec=task_times["avg_execution_time_sec"],
            total_flow_time_sec=task_times["total_flow_time_sec"],
            total_active_cpu_time_sec=task_times["total_active_cpu_time_sec"],
            task_count=len(records),
            vm_count=simulator.num_vms,
            convergence_speed_iterations=conv_speed,
            switching_frequency=switch_freq,
            adaptation_response_iterations=adapt_time,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("  PERFORMANCE METRICS MODULE STANDALONE DEMONSTRATION")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator
    from simulation.scheduler import BaselineScheduler

    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
    sched = BaselineScheduler()

    # Run MCT
    result = sched.schedule_mct(tasks)

    # Simulated optimization history
    mock_fitness_curve = [100.0, 85.0, 72.0, 65.0, 60.5, 59.8, 59.2, 59.0, 59.0, 59.0]
    mock_algo_history = ["PSO", "PSO", "GWO", "GWO", "ABC", "ABC", "PSO", "PSO", "PSO", "PSO"]

    report = MetricsCalculator.generate_full_report(
        records=result.scheduled_records,
        simulator=sched.simulator,
        fitness_history=mock_fitness_curve,
        algorithm_history=mock_algo_history,
        event_iteration=5,
    )

    print("\nEvaluated 10 Performance Metrics for MCT Schedule:")
    print(f"  1. Makespan:                 {report.makespan_sec:.2f} s")
    print(f"  2. Avg VM Utilization:       {report.avg_utilization_pct:.1f}%")
    print(f"  3. Per-VM Utilization:       {report.per_vm_utilization}")
    print(f"  4. Load Imbalance (DI):      {report.degree_of_imbalance:.4f} (std={report.busy_time_std_sec:.2f}s)")
    print(f"  5. Avg Task Waiting Time:    {report.avg_waiting_time_sec:.2f} s")
    print(f"  6. Avg Task Execution Time:  {report.avg_execution_time_sec:.2f} s")
    print(f"  7. Total Task Flow Time:     {report.total_flow_time_sec:.2f} s")
    print(f"  8. Convergence Speed:        Iteration {report.convergence_speed_iterations}")
    print(f"  9. Strategy Switching Freq:  {report.switching_frequency:.2f} switches/epoch")
    print(f" 10. Adaptation Response Time: {report.adaptation_response_iterations} iterations")
    print("=" * 70)

