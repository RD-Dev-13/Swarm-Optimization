"""
Virtual Machine Simulator Module for Cloud Resource Scheduling Simulation.

Models heterogeneous virtualized servers (VMs), task execution mechanics, sequential queuing,
runtime capacity degradation, workload accounting, and utilization metrics.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from simulation.task_generator import Task
from config import DEFAULT_VMS


@dataclass
class ScheduledTaskRecord:
    """
    Immutable execution record for a task dispatched to a specific virtual machine.

    Attributes:
        task_id: Unique identifier of the scheduled task.
        vm_id: Identifier of the assigned Virtual Machine.
        cpu_demand_mi: Task computational demand in Million Instructions.
        arrival_time: Submission time in seconds.
        start_time: Exact timestamp when task execution begins on the VM.
        finish_time: Exact timestamp when task execution finishes on the VM.
        execution_time: Duration of active CPU computation (finish_time - start_time).
        waiting_time: Delay between task arrival and start (start_time - arrival_time).
        turnaround_time: Total lifecycle duration in system (finish_time - arrival_time).
    """
    task_id: str
    vm_id: str
    cpu_demand_mi: float
    arrival_time: float
    start_time: float
    finish_time: float
    execution_time: float
    waiting_time: float
    turnaround_time: float

    def to_dict(self) -> dict:
        """Serialize scheduled record to dictionary."""
        return asdict(self)


class VirtualMachine:
    """
    Represents an isolated Virtual Machine (VM) hosted on cloud infrastructure.

    Execution Model:
        - Sequential Execution: Each VM processes one task at a time in FIFO order of assignment.
        - Non-Preemptive: Once a task begins execution, it runs to completion without interruption.
        - Arrival Respect: A task cannot start before its arrival time:
            start_time = max(task.arrival_time, current_vm_ready_time)
            finish_time = start_time + execution_time
            execution_time = task.cpu_demand_mi / effective_mips
    """

    def __init__(
        self,
        vm_id: str,
        mips: float,
        memory_mb: float = 4096.0,
        is_available: bool = True
    ):
        """
        Initialize a Virtual Machine instance.

        Args:
            vm_id: Unique VM string identifier (e.g., 'VM1').
            mips: Baseline CPU compute capacity in Million Instructions Per Second.
            memory_mb: Total RAM capacity in megabytes (default: 4096 MB).
            is_available: Initial operational status flag.
        """
        if mips <= 0:
            raise ValueError(f"MIPS capacity must be strictly positive, got: {mips}")
        if memory_mb <= 0:
            raise ValueError(f"Memory capacity must be strictly positive, got: {memory_mb}")

        self.vm_id: str = str(vm_id)
        self.base_mips: float = float(mips)
        self.mips: float = float(mips)
        self.memory_mb: float = float(memory_mb)
        self.is_available: bool = bool(is_available)
        self.degradation_factor: float = 1.0  # 1.0 = nominal 100% capacity

        # Dynamic simulation state
        self.assigned_tasks: List[ScheduledTaskRecord] = []
        self.current_workload_mi: float = 0.0
        self.ready_time: float = 0.0  # Earliest time the VM CPU becomes idle
        self.busy_time: float = 0.0   # Cumulative active CPU computation duration

    def calculate_execution_time(self, task: Union[Task, float]) -> float:
        """
        Calculate expected execution duration for a given task or CPU demand.

        Formula:
            Execution Time (s) = CPU Demand (MI) / Effective Capacity (MIPS)

        Args:
            task: Task object or float CPU demand in MI.

        Returns:
            Execution time in seconds rounded to four decimal places.
        """
        cpu_demand = task.cpu_demand_mi if isinstance(task, Task) else float(task)
        if cpu_demand < 0:
            raise ValueError(f"CPU demand cannot be negative, got: {cpu_demand}")
        return round(cpu_demand / self.mips, 4)

    def assign_task(self, task: Task) -> ScheduledTaskRecord:
        """
        Assign and schedule a task on this VM adhering to the sequential execution model.

        Respects task arrival times:
            start_time = max(task.arrival_time, self.ready_time)
            execution_time = task.cpu_demand_mi / self.mips
            finish_time = start_time + execution_time

        Args:
            task: Task object to schedule.

        Returns:
            ScheduledTaskRecord capturing all computed timing milestones.

        Raises:
            RuntimeError: If the VM is marked unavailable.
        """
        if not self.is_available:
            raise RuntimeError(f"Cannot assign task {task.task_id} to unavailable VM '{self.vm_id}'")

        # Scheduling timeline calculations
        exec_time = self.calculate_execution_time(task)
        start_time = round(max(float(task.arrival_time), self.ready_time), 4)
        finish_time = round(start_time + exec_time, 4)
        waiting_time = round(max(0.0, start_time - float(task.arrival_time)), 4)
        turnaround_time = round(finish_time - float(task.arrival_time), 4)

        # Update internal VM timeline & workload counters
        self.ready_time = finish_time
        self.busy_time = round(self.busy_time + exec_time, 4)
        self.current_workload_mi = round(self.current_workload_mi + task.cpu_demand_mi, 2)

        record = ScheduledTaskRecord(
            task_id=task.task_id,
            vm_id=self.vm_id,
            cpu_demand_mi=task.cpu_demand_mi,
            arrival_time=task.arrival_time,
            start_time=start_time,
            finish_time=finish_time,
            execution_time=exec_time,
            waiting_time=waiting_time,
            turnaround_time=turnaround_time,
        )
        self.assigned_tasks.append(record)
        return record

    def apply_degradation(self, factor: float) -> None:
        """
        Simulate an infrastructure degradation event (e.g., thermal throttling, noisy neighbors).

        Reduces effective processing capacity:
            effective_mips = base_mips * factor

        Args:
            factor: Capacity retention factor in (0.0, 1.0]. Example: 0.6 = 40% degradation.

        Raises:
            ValueError: If factor is not in (0.0, 1.0].
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"Degradation factor must be in range (0.0, 1.0], got: {factor}")
        self.degradation_factor = float(factor)
        self.mips = round(self.base_mips * self.degradation_factor, 2)

    def restore_capacity(self) -> None:
        """Restore VM capacity to its nominal base MIPS specification."""
        self.degradation_factor = 1.0
        self.mips = self.base_mips

    def set_availability(self, available: bool) -> None:
        """Toggle operational availability status."""
        self.is_available = bool(available)

    def calculate_utilization(self, makespan: float) -> float:
        """
        Calculate the percentage CPU utilization relative to the overall simulation makespan.

        Formula:
            Utilization (%) = (Busy Time / Makespan) * 100.0

        Sequential execution guarantees utilization never exceeds 100.0%.

        Args:
            makespan: Total simulation completion time in seconds.

        Returns:
            Percentage utilization bounded within [0.0, 100.0].
        """
        if makespan <= 0.0 or self.busy_time <= 0.0:
            return 0.0
        util = (self.busy_time / makespan) * 100.0
        return round(min(100.0, max(0.0, util)), 2)

    def reset(self) -> None:
        """Reset dynamic execution state while retaining static hardware configuration."""
        self.assigned_tasks.clear()
        self.current_workload_mi = 0.0
        self.ready_time = 0.0
        self.busy_time = 0.0

    def to_dict(self, makespan: Optional[float] = None) -> dict:
        """
        Export VM state summary.

        Args:
            makespan: Optional makespan to compute relative utilization.
        """
        effective_makespan = makespan if makespan is not None else self.ready_time
        return {
            "vm_id": self.vm_id,
            "base_mips": self.base_mips,
            "current_mips": self.mips,
            "memory_mb": self.memory_mb,
            "is_available": self.is_available,
            "degradation_factor": self.degradation_factor,
            "assigned_task_count": len(self.assigned_tasks),
            "current_workload_mi": self.current_workload_mi,
            "ready_time": self.ready_time,
            "busy_time": self.busy_time,
            "utilization_pct": self.calculate_utilization(effective_makespan),
        }

    def __repr__(self) -> str:
        return (
            f"VirtualMachine(id='{self.vm_id}', mips={self.mips:.0f}/{self.base_mips:.0f}, "
            f"tasks={len(self.assigned_tasks)}, busy={self.busy_time:.2f}s, "
            f"ready={self.ready_time:.2f}s, available={self.is_available})"
        )


class VMSimulator:
    """
    Manages the cluster of Virtual Machines and orchestrates task mapping simulations.

    Default configuration provides three heterogeneous VMs:
      - VM1: 1000 MIPS (standard baseline)
      - VM2: 1500 MIPS (medium performance)
      - VM3: 2000 MIPS (high compute)
    """

    def __init__(self, vm_configs: Optional[List[Dict]] = None):
        """
        Initialize the cloud VM cluster.

        Args:
            vm_configs: Optional list of dicts specifying [{'vm_id': str, 'mips': float, 'memory_mb': float}].
                        If None, uses DEFAULT_VMS from config.
        """
        configs = vm_configs if vm_configs is not None else DEFAULT_VMS
        if not (3 <= len(configs) <= 5):
            raise ValueError(f"Cluster size must be between 3 and 5 VMs, got {len(configs)}")

        self.vms: Dict[str, VirtualMachine] = {}
        for cfg in configs:
            vm = VirtualMachine(
                vm_id=cfg["vm_id"],
                mips=cfg["mips"],
                memory_mb=cfg.get("memory_mb", 4096.0),
                is_available=cfg.get("is_available", True),
            )
            self.vms[vm.vm_id] = vm

    @property
    def vm_list(self) -> List[VirtualMachine]:
        """Return ordered list of VirtualMachine instances."""
        return list(self.vms.values())

    @property
    def vm_ids(self) -> List[str]:
        """Return list of VM identifiers."""
        return list(self.vms.keys())

    @property
    def num_vms(self) -> int:
        """Return the number of VMs in the cluster."""
        return len(self.vms)

    def get_vm(self, vm_id: str) -> VirtualMachine:
        """Retrieve a specific VM by ID."""
        if vm_id not in self.vms:
            raise KeyError(f"VM '{vm_id}' not found in cluster. Available: {list(self.vms.keys())}")
        return self.vms[vm_id]

    def reset_cluster(self) -> None:
        """Reset dynamic state of all VMs in the cluster."""
        for vm in self.vms.values():
            vm.reset()

    def get_makespan(self) -> float:
        """
        Calculate total simulation makespan across all VMs.

        Makespan is the maximum finish / ready time among all VMs:
            Makespan = max_{j}(ready_time_j)
        """
        if not self.vms:
            return 0.0
        return round(max(vm.ready_time for vm in self.vms.values()), 4)

    def get_load_imbalance(self) -> float:
        """
        Calculate Degree of Imbalance (DI) among VM busy times.

        Formula:
            DI = (T_max - T_min) / T_avg
            where T_j = busy_time of VM_j.
        If T_avg == 0, returns 0.0.
        """
        busy_times = [vm.busy_time for vm in self.vms.values()]
        if not busy_times or sum(busy_times) == 0.0:
            return 0.0
        t_max = max(busy_times)
        t_min = min(busy_times)
        t_avg = sum(busy_times) / len(busy_times)
        return round((t_max - t_min) / t_avg, 4)

    def get_average_utilization(self) -> float:
        """Calculate mean CPU utilization percentage across all VMs."""
        makespan = self.get_makespan()
        if makespan <= 0.0:
            return 0.0
        utils = [vm.calculate_utilization(makespan) for vm in self.vms.values()]
        return round(float(np.mean(utils)), 2)

    def simulate_assignments(
        self,
        tasks: List[Task],
        assignments: Union[List[int], List[str], Dict[str, str]],
        reset: bool = True
    ) -> List[ScheduledTaskRecord]:
        """
        Simulate the scheduling of tasks according to an assignment vector or mapping.

        Args:
            tasks: List of Task objects.
            assignments: Either:
                - List of VM indices (0 to num_vms - 1) matching tasks length.
                - List of VM ID strings matching tasks length.
                - Dictionary mapping task_id -> vm_id.
            reset: If True, resets all VMs before simulation. If False, preserves existing
                   ready times and busy times (for partial dynamic scheduling).

        Returns:
            List of ScheduledTaskRecord objects representing full execution history.

        Raises:
            ValueError: If assignment dimensions or values are invalid.
        """
        if reset:
            self.reset_cluster()

        if len(tasks) == 0:
            return []

        all_records: List[ScheduledTaskRecord] = []
        vm_id_list = self.vm_ids

        # Ensure tasks are evaluated in arrival order
        sorted_indices = sorted(range(len(tasks)), key=lambda idx: tasks[idx].arrival_time)

        for idx in sorted_indices:
            task = tasks[idx]

            # Resolve target VM ID
            if isinstance(assignments, dict):
                target_vm_id = assignments[task.task_id]
            elif isinstance(assignments, (list, tuple, np.ndarray)):
                assigned_val = assignments[idx]
                if isinstance(assigned_val, (int, np.integer)):
                    if not (0 <= assigned_val < self.num_vms):
                        raise ValueError(f"VM index {assigned_val} out of range [0, {self.num_vms - 1}]")
                    target_vm_id = vm_id_list[int(assigned_val)]
                else:
                    target_vm_id = str(assigned_val)
            else:
                raise TypeError(f"Unsupported assignments format: {type(assignments)}")

            vm = self.get_vm(target_vm_id)
            record = vm.assign_task(task)
            all_records.append(record)

        return all_records

    def apply_vm_degradation(self, vm_id: str, factor: float) -> None:
        """
        Trigger capacity degradation on a specified VM.

        Args:
            vm_id: ID of the VM to degrade (e.g., 'VM2').
            factor: Capacity fraction in (0.0, 1.0].
        """
        vm = self.get_vm(vm_id)
        vm.apply_degradation(factor)

    def restore_vm_capacity(self, vm_id: str) -> None:
        """Restore degraded VM back to baseline capacity."""
        vm = self.get_vm(vm_id)
        vm.restore_capacity()

    def to_dataframe(self) -> pd.DataFrame:
        """Generate structured DataFrame summarizing status of all VMs in cluster."""
        makespan = self.get_makespan()
        rows = [vm.to_dict(makespan=makespan) for vm in self.vms.values()]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("  VM SIMULATOR STANDALONE VERIFICATION DEMONSTRATION")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator

    # Initialize generator and sample tasks
    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=10, workload_type="normal")

    # Initialize cluster with 3 default VMs (1000, 1500, 2000 MIPS)
    simulator = VMSimulator()
    print(f"\nInitialized Cluster with {simulator.num_vms} VMs:")
    for vm in simulator.vm_list:
        print(f"  - {vm}")

    # Assign tasks in round-robin fashion (0, 1, 2, 0, 1, 2, ...)
    round_robin_assignments = [i % simulator.num_vms for i in range(len(tasks))]
    records = simulator.simulate_assignments(tasks, round_robin_assignments)

    print(f"\nScheduled {len(records)} Tasks across VMs:")
    for rec in records[:6]:
        print(f"  Task {rec.task_id} -> {rec.vm_id} | Arrival: {rec.arrival_time:5.2f}s | "
              f"Start: {rec.start_time:5.2f}s | Exec: {rec.execution_time:5.2f}s | "
              f"Finish: {rec.finish_time:5.2f}s | Wait: {rec.waiting_time:5.2f}s")

    print("\nCluster Post-Execution Metrics:")
    df_cluster = simulator.to_dataframe()
    for _, row in df_cluster.iterrows():
        print(f"  {row['vm_id']}: Capacity={row['current_mips']} MIPS | Tasks={row['assigned_task_count']} | "
              f"Workload={row['current_workload_mi']:.1f} MI | Busy={row['busy_time']:.2f}s | "
              f"Util={row['utilization_pct']:.1f}%")

    print(f"\nOverall Simulation Makespan: {simulator.get_makespan():.2f}s")
    print(f"Degree of Load Imbalance:   {simulator.get_load_imbalance():.4f}")
    print(f"Average Cluster Utilization: {simulator.get_average_utilization():.2f}%")

    # Verify Degradation Event
    print("\nSimulating VM Degradation Event (VM2 degrades to 50% capacity)...")
    simulator.apply_vm_degradation("VM2", factor=0.5)
    print(f"  VM2 After Degradation: {simulator.get_vm('VM2')}")

    # Reschedule same tasks
    records_deg = simulator.simulate_assignments(tasks, round_robin_assignments)
    print(f"  New Makespan after VM2 degradation: {simulator.get_makespan():.2f}s "
          f"(increased as expected due to reduced capacity)")
    print("=" * 70)

