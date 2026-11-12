"""
Baseline Scheduling Module for Cloud Resource Allocation.

Implements standard deterministic baseline heuristics:
1. First-Come, First-Served (FCFS): Assigns tasks to the earliest-available VM.
2. Round Robin (RR): Cyclically distributes tasks across VMs (k mod m).
3. Minimum Completion Time (MCT): Greedily allocates each task to the VM that achieves
   the earliest completion time.

All schedulers return structured ScheduleResult objects for direct comparison with
swarm intelligence and adaptive algorithms.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator, VirtualMachine, ScheduledTaskRecord


@dataclass
class ScheduleResult:
    """
    Encapsulates complete performance and schedule accounting for a scheduling run.

    Attributes:
        algorithm_name: Identifier of the scheduling strategy (e.g., 'FCFS', 'RR', 'MCT').
        assignments: List of integer VM indices corresponding to tasks in original order.
        scheduled_records: Detailed execution records per task.
        makespan: Maximum completion time across all VMs in seconds.
        average_utilization: Cluster-wide mean CPU utilization percentage [0-100%].
        vm_utilizations: Dictionary mapping vm_id -> utilization percentage.
        load_imbalance: Degree of Imbalance (DI) among VM busy times.
        total_execution_time: Sum of active computation across all tasks.
        average_waiting_time: Mean queue latency experienced by tasks.
        average_turnaround_time: Mean total lifecycle time (finish_time - arrival_time).
    """
    algorithm_name: str
    assignments: List[int]
    scheduled_records: List[ScheduledTaskRecord]
    makespan: float
    average_utilization: float
    vm_utilizations: Dict[str, float]
    load_imbalance: float
    total_execution_time: float
    average_waiting_time: float
    average_turnaround_time: float

    def to_dict(self) -> dict:
        """Serialize metrics to dictionary for reporting and logging."""
        return {
            "algorithm": self.algorithm_name,
            "makespan_sec": self.makespan,
            "avg_utilization_pct": self.average_utilization,
            "load_imbalance": self.load_imbalance,
            "total_execution_sec": self.total_execution_time,
            "avg_waiting_sec": self.average_waiting_time,
            "avg_turnaround_sec": self.average_turnaround_time,
            "task_count": len(self.scheduled_records),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert scheduled task execution records to a pandas DataFrame."""
        if not self.scheduled_records:
            return pd.DataFrame()
        return pd.DataFrame([rec.to_dict() for rec in self.scheduled_records])


class BaselineScheduler:
    """
    Executes and evaluates deterministic baseline scheduling heuristics under identical
    simulation workloads and VM cluster conditions.
    """

    def __init__(self, simulator: Optional[VMSimulator] = None):
        """
        Initialize BaselineScheduler.

        Args:
            simulator: VMSimulator instance. If None, instantiates default 3-VM cluster.
        """
        self.simulator = simulator if simulator is not None else VMSimulator()

    def _compile_result(
        self,
        algorithm_name: str,
        tasks: List[Task],
        assignments: List[int],
        records: List[ScheduledTaskRecord]
    ) -> ScheduleResult:
        """
        Compute aggregate performance metrics and compile into a ScheduleResult.

        Args:
            algorithm_name: Name of algorithm.
            tasks: Original input tasks.
            assignments: List of assigned VM indices.
            records: Scheduled task records.

        Returns:
            Fully populated ScheduleResult instance.
        """
        makespan = self.simulator.get_makespan()
        load_imbalance = self.simulator.get_load_imbalance()
        avg_utilization = self.simulator.get_average_utilization()

        vm_utils = {
            vm.vm_id: vm.calculate_utilization(makespan)
            for vm in self.simulator.vm_list
        }

        if records:
            total_exec = round(sum(r.execution_time for r in records), 4)
            avg_wait = round(float(np.mean([r.waiting_time for r in records])), 4)
            avg_turnaround = round(float(np.mean([r.turnaround_time for r in records])), 4)
        else:
            total_exec = 0.0
            avg_wait = 0.0
            avg_turnaround = 0.0

        return ScheduleResult(
            algorithm_name=algorithm_name,
            assignments=assignments,
            scheduled_records=records,
            makespan=makespan,
            average_utilization=avg_utilization,
            vm_utilizations=vm_utils,
            load_imbalance=load_imbalance,
            total_execution_time=total_exec,
            average_waiting_time=avg_wait,
            average_turnaround_time=avg_turnaround,
        )

    def schedule_fcfs(
        self,
        tasks: List[Task],
        simulator: Optional[VMSimulator] = None
    ) -> ScheduleResult:
        """
        First-Come, First-Served (FCFS) Scheduling.

        Logic:
            Tasks are considered chronologically in order of arrival time.
            Each task is dispatched to the VM that will be available earliest
            (i.e., minimum current ready_time). If multiple VMs are idle at the
            task's arrival time, the one with the highest MIPS capacity is selected.

        Args:
            tasks: List of Task objects.
            simulator: Optional simulator instance; if None, uses self.simulator.

        Returns:
            ScheduleResult with FCFS assignments and metrics.
        """
        sim = simulator if simulator is not None else self.simulator
        sim.reset_cluster()

        if not tasks:
            return self._compile_result("FCFS", tasks, [], [])

        num_vms = sim.num_vms
        vm_list = sim.vm_list

        # Work on sorted chronological order while preserving original indices
        sorted_indices = sorted(range(len(tasks)), key=lambda i: tasks[i].arrival_time)
        assignments = [0] * len(tasks)

        for idx in sorted_indices:
            task = tasks[idx]

            # Find VM that becomes available earliest
            # Tie breaker: Highest processing capacity (MIPS)
            best_vm_idx = 0
            best_avail_time = float("inf")
            best_mips = -1.0

            for v_idx, vm in enumerate(vm_list):
                if not vm.is_available:
                    continue
                # Availability timestamp from the perspective of this task
                avail = max(task.arrival_time, vm.ready_time)
                if avail < best_avail_time:
                    best_avail_time = avail
                    best_vm_idx = v_idx
                    best_mips = vm.mips
                elif abs(avail - best_avail_time) < 1e-6:
                    if vm.mips > best_mips:
                        best_avail_time = avail
                        best_vm_idx = v_idx
                        best_mips = vm.mips

            assignments[idx] = best_vm_idx
            vm_list[best_vm_idx].assign_task(task)

        records = [rec for vm in vm_list for rec in vm.assigned_tasks]
        records.sort(key=lambda r: (r.start_time, r.task_id))
        return self._compile_result("FCFS", tasks, assignments, records)

    def schedule_round_robin(
        self,
        tasks: List[Task],
        simulator: Optional[VMSimulator] = None
    ) -> ScheduleResult:
        """
        Round Robin (RR) Scheduling.

        Logic:
            Tasks are allocated cyclically across the available VMs in arrival sequence:
                VM_index = k mod num_vms
            Each VM executes its assigned queue sequentially.

        Args:
            tasks: List of Task objects.
            simulator: Optional simulator instance; if None, uses self.simulator.

        Returns:
            ScheduleResult with Round Robin assignments and metrics.
        """
        sim = simulator if simulator is not None else self.simulator
        sim.reset_cluster()

        if not tasks:
            return self._compile_result("RoundRobin", tasks, [], [])

        num_vms = sim.num_vms
        vm_list = sim.vm_list

        sorted_indices = sorted(range(len(tasks)), key=lambda i: tasks[i].arrival_time)
        assignments = [0] * len(tasks)

        # Distribute cyclically among active available VMs
        available_indices = [i for i, vm in enumerate(vm_list) if vm.is_available]
        if not available_indices:
            raise RuntimeError("No available VMs in cluster to schedule tasks.")

        for cycle_k, idx in enumerate(sorted_indices):
            target_vm_idx = available_indices[cycle_k % len(available_indices)]
            assignments[idx] = target_vm_idx
            vm_list[target_vm_idx].assign_task(tasks[idx])

        records = [rec for vm in vm_list for rec in vm.assigned_tasks]
        records.sort(key=lambda r: (r.start_time, r.task_id))
        return self._compile_result("RoundRobin", tasks, assignments, records)

    def schedule_mct(
        self,
        tasks: List[Task],
        simulator: Optional[VMSimulator] = None
    ) -> ScheduleResult:
        """
        Minimum Completion Time (MCT) Scheduling.

        Logic:
            For each task in arrival sequence, calculates prospective completion time
            across all VMs:
                CompletionTime(i, j) = max(arrival_time_i, ready_time_j) + (demand_i / mips_j)
            Assigns the task to the VM that achieves the minimum completion time:
                VM* = argmin_j (CompletionTime(i, j))

        Args:
            tasks: List of Task objects.
            simulator: Optional simulator instance; if None, uses self.simulator.

        Returns:
            ScheduleResult with MCT assignments and metrics.
        """
        sim = simulator if simulator is not None else self.simulator
        sim.reset_cluster()

        if not tasks:
            return self._compile_result("MCT", tasks, [], [])

        vm_list = sim.vm_list

        sorted_indices = sorted(range(len(tasks)), key=lambda i: tasks[i].arrival_time)
        assignments = [0] * len(tasks)

        for idx in sorted_indices:
            task = tasks[idx]

            best_vm_idx = 0
            best_completion_time = float("inf")

            for v_idx, vm in enumerate(vm_list):
                if not vm.is_available:
                    continue
                start_t = max(task.arrival_time, vm.ready_time)
                exec_t = vm.calculate_execution_time(task)
                comp_t = start_t + exec_t

                if comp_t < best_completion_time:
                    best_completion_time = comp_t
                    best_vm_idx = v_idx

            assignments[idx] = best_vm_idx
            vm_list[best_vm_idx].assign_task(task)

        records = [rec for vm in vm_list for rec in vm.assigned_tasks]
        records.sort(key=lambda r: (r.start_time, r.task_id))
        return self._compile_result("MCT", tasks, assignments, records)

    def run_all(
        self,
        tasks: List[Task],
        simulator: Optional[VMSimulator] = None
    ) -> Dict[str, ScheduleResult]:
        """
        Run all baseline heuristics under identical tasks and cluster setup.

        Args:
            tasks: List of tasks to schedule.
            simulator: Optional simulator instance.

        Returns:
            Dictionary mapping heuristic name -> ScheduleResult.
        """
        sim = simulator if simulator is not None else self.simulator
        return {
            "FCFS": self.schedule_fcfs(tasks, sim),
            "RoundRobin": self.schedule_round_robin(tasks, sim),
            "MCT": self.schedule_mct(tasks, sim),
        }

    def compare_baselines(
        self,
        tasks: List[Task],
        simulator: Optional[VMSimulator] = None
    ) -> pd.DataFrame:
        """
        Execute all baselines and format results into a comparative DataFrame.

        Args:
            tasks: List of tasks to schedule.
            simulator: Optional simulator instance.

        Returns:
            pd.DataFrame summarizing key scheduling performance metrics.
        """
        results = self.run_all(tasks, simulator)
        rows = [res.to_dict() for res in results.values()]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 75)
    print("  BASELINE SCHEDULERS STANDALONE DEMONSTRATION")
    print("=" * 75)

    from simulation.task_generator import TaskGenerator

    # Generate test workload of 20 normal tasks
    gen = TaskGenerator(seed=42)
    sample_tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")

    # Initialize BaselineScheduler
    scheduler = BaselineScheduler()
    df_comparison = scheduler.compare_baselines(sample_tasks)

    print("\nBaseline Algorithms Comparative Performance on 20 Tasks:")
    print(df_comparison.to_string(index=False))

    print("\nDetailed Per-VM Utilization Breakdown:")
    results = scheduler.run_all(sample_tasks)
    for name, res in results.items():
        print(f"\n[{name}] Total Makespan: {res.makespan:.2f}s | Avg Wait: {res.average_waiting_time:.2f}s")
        for vm_id, util in res.vm_utilizations.items():
            print(f"   {vm_id}: {util:.1f}% utilization")

    print("\n" + "=" * 75)

