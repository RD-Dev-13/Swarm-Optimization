"""
Fitness Evaluation Module for Cloud Resource Scheduling Optimization.

Calculates multi-objective fitness for task-to-VM assignments.
Objective:
    Fitness = Makespan + lambda * LoadImbalance (+ optional energy/cost/deadline penalties)

All metrics are scaled and dimensionally compatible (measured in seconds or normalized).
Supports both complete schedules and partial schedules for runtime adaptive optimization.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator, ScheduledTaskRecord
from config import DEFAULT_FITNESS_LAMBDA


@dataclass
class FitnessBreakdown:
    """
    Detailed breakdown of individual components contributing to total fitness.

    Attributes:
        fitness: Final combined scalar score (lower is better).
        makespan: Maximum completion time across all VMs in seconds.
        load_imbalance: Measure of workload variance among VMs (in seconds or dimensionless).
        imbalance_metric: The metric used ('std' for standard deviation in seconds, 'degree' for DI).
        energy_joules: Optional total energy consumption across all physical hosts in Joules.
        scheduling_cost: Optional monetary cost based on VM compute rates.
        deadline_violations: Optional count of tasks finishing past their deadline.
    """
    fitness: float
    makespan: float
    load_imbalance: float
    imbalance_metric: str
    energy_joules: Optional[float] = None
    scheduling_cost: Optional[float] = None
    deadline_violations: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize breakdown to dictionary."""
        return asdict(self)


class FitnessEvaluator:
    """
    Evaluates the quality of task-to-VM assignments for metaheuristic optimization.

    Minimization Objective:
        f(x) = Makespan + lambda * LoadImbalance

    Scale Compatibility:
        - When imbalance_metric='std', LoadImbalance is the sample standard deviation
          of VM busy times: sigma_busy = sqrt( (1/m) * sum((B_j - B_avg)^2) ) in seconds.
          Because both Makespan and sigma_busy are measured in seconds, dimensional
          coherence is preserved.
        - When imbalance_metric='range', LoadImbalance is (B_max - B_min) in seconds.
        - When imbalance_metric='degree', LoadImbalance is normalized DI * B_avg.

    Supports:
        - Full schedules (from t=0.0).
        - Partial schedules with warm VM ready clocks (for dynamic adaptation).
        - Optional extensions: energy consumption, financial cost, deadline penalties.
    """

    def __init__(
        self,
        lambda_imbalance: float = DEFAULT_FITNESS_LAMBDA,
        imbalance_metric: str = "std",
        enable_energy: bool = False,
        energy_weight: float = 0.001,
        enable_cost: bool = False,
        cost_weight: float = 0.01,
        enable_deadlines: bool = False,
        deadline_penalty: float = 10.0,
    ):
        """
        Initialize the FitnessEvaluator.

        Args:
            lambda_imbalance: Weight for workload imbalance penalty (>= 0.0).
            imbalance_metric: 'std' (standard deviation in seconds),
                              'range' (B_max - B_min in seconds),
                              or 'degree' (dimensionless degree of imbalance).
            enable_energy: Flag to activate optional energy consumption evaluation.
            energy_weight: Scaling factor for energy in fitness.
            enable_cost: Flag to activate optional economic scheduling cost.
            cost_weight: Scaling factor for cost in fitness.
            enable_deadlines: Flag to activate deadline violation penalties.
            deadline_penalty: Penalty added per deadline violation.
        """
        if lambda_imbalance < 0.0:
            raise ValueError(f"lambda_imbalance cannot be negative, got: {lambda_imbalance}")

        valid_metrics = {"std", "range", "degree"}
        norm_metric = str(imbalance_metric).strip().lower()
        if norm_metric not in valid_metrics:
            raise ValueError(f"Unknown imbalance_metric '{imbalance_metric}'. Choose from: {valid_metrics}")

        self.lambda_imbalance = float(lambda_imbalance)
        self.imbalance_metric = norm_metric
        self.enable_energy = bool(enable_energy)
        self.energy_weight = float(energy_weight)
        self.enable_cost = bool(enable_cost)
        self.cost_weight = float(cost_weight)
        self.enable_deadlines = bool(enable_deadlines)
        self.deadline_penalty = float(deadline_penalty)

        # Baseline VM power profiles (Watts): active and idle
        self._vm_power_profiles = {
            "VM1": {"active_w": 180.0, "idle_w": 75.0, "cost_per_sec": 0.002},
            "VM2": {"active_w": 240.0, "idle_w": 90.0, "cost_per_sec": 0.003},
            "VM3": {"active_w": 300.0, "idle_w": 110.0, "cost_per_sec": 0.004},
        }

    def _compute_load_imbalance(self, busy_times: List[float]) -> float:
        """
        Compute workload dispersion across VMs.

        Args:
            busy_times: List of active CPU busy times (in seconds) per VM.

        Returns:
            Computed load imbalance metric.
        """
        if not busy_times or len(busy_times) <= 1:
            return 0.0

        b_array = np.array(busy_times, dtype=float)
        b_mean = float(np.mean(b_array))

        if self.imbalance_metric == "std":
            # Standard deviation in seconds (dimensionally identical to makespan)
            return float(np.std(b_array))
        elif self.imbalance_metric == "range":
            # Difference between most and least loaded VM in seconds
            return float(np.max(b_array) - np.min(b_array))
        elif self.imbalance_metric == "degree":
            # Degree of Imbalance (DI): (B_max - B_min) / B_avg
            if b_mean == 0.0:
                return 0.0
            return float((np.max(b_array) - np.min(b_array)) / b_mean)

        return 0.0

    def evaluate_detailed(
        self,
        assignments: Union[List[int], np.ndarray, Dict[str, str]],
        tasks: List[Task],
        simulator: VMSimulator,
        initial_ready_times: Optional[Dict[str, float]] = None,
        initial_busy_times: Optional[Dict[str, float]] = None,
    ) -> FitnessBreakdown:
        """
        Simulate a schedule assignment and compute full fitness breakdown.

        Args:
            assignments: Mapping of tasks to VMs (list of VM indices or dict).
            tasks: List of Task objects.
            simulator: VMSimulator instance.
            initial_ready_times: Optional dict of initial VM ready clocks for partial scheduling.
            initial_busy_times: Optional dict of prior busy times for partial scheduling.

        Returns:
            FitnessBreakdown containing all computed objective values.
        """
        if len(tasks) == 0:
            return FitnessBreakdown(
                fitness=0.0,
                makespan=0.0,
                load_imbalance=0.0,
                imbalance_metric=self.imbalance_metric,
            )

        # Warm start VM clocks if this is a partial schedule
        is_partial = bool(initial_ready_times or initial_busy_times)
        simulator.reset_cluster()
        if initial_ready_times:
            for vm_id, r_time in initial_ready_times.items():
                if vm_id in simulator.vms:
                    simulator.vms[vm_id].ready_time = float(r_time)

        if initial_busy_times:
            for vm_id, b_time in initial_busy_times.items():
                if vm_id in simulator.vms:
                    simulator.vms[vm_id].busy_time = float(b_time)

        # Execute assignment simulation
        records = simulator.simulate_assignments(tasks, assignments, reset=not is_partial)

        makespan = simulator.get_makespan()
        busy_times = [vm.busy_time for vm in simulator.vm_list]
        load_imbalance = self._compute_load_imbalance(busy_times)

        # Base objective
        fitness = makespan + (self.lambda_imbalance * load_imbalance)

        # Optional Extensions
        energy_joules: Optional[float] = None
        scheduling_cost: Optional[float] = None
        deadline_violations: Optional[int] = None

        if self.enable_energy:
            total_energy = 0.0
            for vm in simulator.vm_list:
                profile = self._vm_power_profiles.get(
                    vm.vm_id, {"active_w": 200.0, "idle_w": 80.0}
                )
                active_e = profile["active_w"] * vm.busy_time
                idle_e = profile["idle_w"] * max(0.0, makespan - vm.busy_time)
                total_energy += (active_e + idle_e)
            energy_joules = round(total_energy, 2)
            fitness += (self.energy_weight * energy_joules)

        if self.enable_cost:
            total_cost = 0.0
            for vm in simulator.vm_list:
                rate = self._vm_power_profiles.get(vm.vm_id, {}).get("cost_per_sec", 0.002)
                total_cost += rate * makespan
            scheduling_cost = round(total_cost, 4)
            fitness += (self.cost_weight * scheduling_cost)

        if self.enable_deadlines:
            violations = 0
            for rec in records:
                # If task has a deadline attribute, check violation
                deadline = getattr(rec, "deadline", None)
                if deadline is not None and rec.finish_time > deadline:
                    violations += 1
            deadline_violations = violations
            fitness += (self.deadline_penalty * violations)

        return FitnessBreakdown(
            fitness=round(float(fitness), 4),
            makespan=round(float(makespan), 4),
            load_imbalance=round(float(load_imbalance), 4),
            imbalance_metric=self.imbalance_metric,
            energy_joules=energy_joules,
            scheduling_cost=scheduling_cost,
            deadline_violations=deadline_violations,
        )

    def evaluate(
        self,
        assignments: Union[List[int], np.ndarray, Dict[str, str]],
        tasks: List[Task],
        simulator: VMSimulator,
        initial_ready_times: Optional[Dict[str, float]] = None,
        initial_busy_times: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Fast evaluation returning scalar fitness score (for optimization loops).

        Args:
            assignments: Mapping of tasks to VMs.
            tasks: List of Task objects.
            simulator: VMSimulator instance.
            initial_ready_times: Optional dict of initial ready clocks.
            initial_busy_times: Optional dict of initial busy times.

        Returns:
            Scalar fitness float (to minimize).
        """
        breakdown = self.evaluate_detailed(
            assignments,
            tasks,
            simulator,
            initial_ready_times,
            initial_busy_times,
        )
        return breakdown.fitness


if __name__ == "__main__":
    print("=" * 70)
    print("  FITNESS EVALUATION STANDALONE DEMONSTRATION")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator

    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
    sim = VMSimulator()

    # Compare two different schedules:
    # 1. Round-Robin schedule
    # 2. Skewed schedule (all tasks sent to VM1)
    evaluator = FitnessEvaluator(lambda_imbalance=1.0, imbalance_metric="std")

    rr_assignments = [i % sim.num_vms for i in range(len(tasks))]
    breakdown_rr = evaluator.evaluate_detailed(rr_assignments, tasks, sim)

    skewed_assignments = [0] * len(tasks)
    breakdown_skewed = evaluator.evaluate_detailed(skewed_assignments, tasks, sim)

    print("\nSchedule A: Balanced Round Robin")
    print(f"  Makespan:       {breakdown_rr.makespan:.2f}s")
    print(f"  Load Imbalance: {breakdown_rr.load_imbalance:.2f}s (std dev)")
    print(f"  Total Fitness:  {breakdown_rr.fitness:.2f} (lower is better)")

    print("\nSchedule B: Degraded / Skewed (All on VM1)")
    print(f"  Makespan:       {breakdown_skewed.makespan:.2f}s")
    print(f"  Load Imbalance: {breakdown_skewed.load_imbalance:.2f}s (std dev)")
    print(f"  Total Fitness:  {breakdown_skewed.fitness:.2f} (lower is better)")

    assert breakdown_rr.fitness < breakdown_skewed.fitness, "Fitness function failed sanity test!"
    print("\nSanity Check Passed: Balanced schedule achieved superior fitness score.")
    print("=" * 70)
