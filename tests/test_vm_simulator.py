"""
Automated unit tests for VirtualMachine and VMSimulator modules.

Verifies:
1. Calculates execution time correctly using task.cpu_demand_mi / mips.
2. Respects task arrival times (start_time >= arrival_time, waiting_time >= 0).
3. Tracks VM workloads and assigned task lists accurately.
4. Calculates utilization correctly (bounded within [0.0, 100.0]%).
5. Handles VM capacity degradation and restoration properly.
6. Handles VM availability toggling and rejects assignments to offline VMs.
7. Computes makespan and load imbalance correctly across the cluster.
"""

import pytest
import numpy as np

from simulation.task_generator import Task
from simulation.vm_simulator import VirtualMachine, VMSimulator, ScheduledTaskRecord


class TestVirtualMachine:
    """Unit tests for individual VirtualMachine instances."""

    @pytest.fixture
    def sample_vm(self) -> VirtualMachine:
        """Create a standard VM with 1000 MIPS."""
        return VirtualMachine(vm_id="VM1", mips=1000.0, memory_mb=4096.0)

    @pytest.fixture
    def sample_task(self) -> Task:
        """Create a typical test task."""
        return Task(
            task_id="T001",
            cpu_demand_mi=5000.0,
            arrival_time=2.0,
            priority=3,
            memory_mb=1024.0,
            workload_type="normal",
        )

    def test_vm_initialization(self, sample_vm: VirtualMachine):
        """Verifies default parameters and initial clean state."""
        assert sample_vm.vm_id == "VM1"
        assert sample_vm.base_mips == 1000.0
        assert sample_vm.mips == 1000.0
        assert sample_vm.memory_mb == 4096.0
        assert sample_vm.is_available is True
        assert sample_vm.ready_time == 0.0
        assert sample_vm.busy_time == 0.0
        assert sample_vm.current_workload_mi == 0.0
        assert len(sample_vm.assigned_tasks) == 0

    def test_invalid_vm_parameters(self):
        """Verifies rejection of non-positive MIPS or memory values."""
        with pytest.raises(ValueError, match="MIPS capacity must be strictly positive"):
            VirtualMachine(vm_id="VM_BAD", mips=0.0)

        with pytest.raises(ValueError, match="MIPS capacity must be strictly positive"):
            VirtualMachine(vm_id="VM_BAD", mips=-500.0)

        with pytest.raises(ValueError, match="Memory capacity must be strictly positive"):
            VirtualMachine(vm_id="VM_BAD", mips=1000.0, memory_mb=0.0)

    def test_execution_time_calculation(self, sample_vm: VirtualMachine, sample_task: Task):
        """Verifies formula: Execution Time = CPU Demand / MIPS."""
        # 5000 MI / 1000 MIPS = 5.0 seconds
        exec_time = sample_vm.calculate_execution_time(sample_task)
        assert exec_time == 5.0

        # Direct float input
        assert sample_vm.calculate_execution_time(2500.0) == 2.5

        # Rejection of negative demand
        with pytest.raises(ValueError, match="CPU demand cannot be negative"):
            sample_vm.calculate_execution_time(-100.0)

    def test_respects_task_arrival_time(self, sample_vm: VirtualMachine, sample_task: Task):
        """Verifies task cannot start before its arrival time when VM is idle."""
        record = sample_vm.assign_task(sample_task)

        assert record.arrival_time == 2.0
        assert record.start_time == 2.0  # VM was idle at t=0, task arrives at t=2
        assert record.execution_time == 5.0
        assert record.finish_time == 7.0
        assert record.waiting_time == 0.0
        assert record.turnaround_time == 5.0
        assert sample_vm.ready_time == 7.0
        assert sample_vm.busy_time == 5.0
        assert sample_vm.current_workload_mi == 5000.0

    def test_sequential_task_queuing_and_waiting_time(self, sample_vm: VirtualMachine):
        """Verifies sequential queue delay when task arrives while VM is currently busy."""
        # Task 1 arrives at t=1.0, demand 4000 MI (takes 4.0s) -> finishes at t=5.0
        t1 = Task("T001", 4000.0, arrival_time=1.0, priority=2, memory_mb=512, workload_type="normal")
        rec1 = sample_vm.assign_task(t1)
        assert rec1.start_time == 1.0
        assert rec1.finish_time == 5.0
        assert rec1.waiting_time == 0.0

        # Task 2 arrives at t=2.0 (while VM is busy until t=5.0), demand 2000 MI (takes 2.0s)
        t2 = Task("T002", 2000.0, arrival_time=2.0, priority=2, memory_mb=512, workload_type="normal")
        rec2 = sample_vm.assign_task(t2)
        assert rec2.start_time == 5.0       # Cannot start until t1 finishes
        assert rec2.finish_time == 7.0
        assert rec2.waiting_time == 3.0      # 5.0 - 2.0 = 3.0s waiting in queue
        assert rec2.turnaround_time == 5.0   # 7.0 - 2.0 = 5.0s turnaround

        assert sample_vm.ready_time == 7.0
        assert sample_vm.busy_time == 6.0    # 4.0s + 2.0s
        assert sample_vm.current_workload_mi == 6000.0
        assert len(sample_vm.assigned_tasks) == 2

    def test_utilization_calculation(self, sample_vm: VirtualMachine, sample_task: Task):
        """Verifies utilization percentage formula and boundaries."""
        sample_vm.assign_task(sample_task)  # busy_time = 5.0s, ready_time = 7.0s

        # If overall makespan = 10.0s, utilization = (5.0 / 10.0) * 100 = 50.0%
        util = sample_vm.calculate_utilization(makespan=10.0)
        assert util == 50.0

        # If makespan equals finish time 7.0s: (5.0 / 7.0) * 100 = 71.43%
        util_tight = sample_vm.calculate_utilization(makespan=7.0)
        assert util_tight == pytest.approx(71.43, abs=0.01)

        # Utilization never exceeds 100%
        assert sample_vm.calculate_utilization(makespan=2.0) == 100.0

        # Zero makespan returns 0.0%
        assert sample_vm.calculate_utilization(makespan=0.0) == 0.0

    def test_vm_capacity_degradation_and_restoration(self, sample_vm: VirtualMachine):
        """Verifies degradation reduces MIPS and slows task execution."""
        sample_vm.apply_degradation(factor=0.5)  # 50% capacity
        assert sample_vm.mips == 500.0
        assert sample_vm.degradation_factor == 0.5
        assert sample_vm.base_mips == 1000.0

        # 5000 MI on 500 MIPS takes 10.0 seconds instead of 5.0 seconds
        exec_time = sample_vm.calculate_execution_time(5000.0)
        assert exec_time == 10.0

        # Restore
        sample_vm.restore_capacity()
        assert sample_vm.mips == 1000.0
        assert sample_vm.degradation_factor == 1.0
        assert sample_vm.calculate_execution_time(5000.0) == 5.0

    def test_invalid_degradation_factor(self, sample_vm: VirtualMachine):
        """Verifies error handling on out-of-range degradation factors."""
        with pytest.raises(ValueError, match="Degradation factor must be in range"):
            sample_vm.apply_degradation(factor=0.0)

        with pytest.raises(ValueError, match="Degradation factor must be in range"):
            sample_vm.apply_degradation(factor=1.5)

    def test_vm_unavailability(self, sample_vm: VirtualMachine, sample_task: Task):
        """Verifies task cannot be scheduled on an offline VM."""
        sample_vm.set_availability(False)
        with pytest.raises(RuntimeError, match="Cannot assign task .* to unavailable VM"):
            sample_vm.assign_task(sample_task)

    def test_vm_reset(self, sample_vm: VirtualMachine, sample_task: Task):
        """Verifies reset cleans dynamic timeline without erasing hardware specs."""
        sample_vm.assign_task(sample_task)
        assert len(sample_vm.assigned_tasks) == 1
        assert sample_vm.ready_time > 0.0

        sample_vm.reset()
        assert len(sample_vm.assigned_tasks) == 0
        assert sample_vm.ready_time == 0.0
        assert sample_vm.busy_time == 0.0
        assert sample_vm.current_workload_mi == 0.0
        assert sample_vm.mips == 1000.0


class TestVMSimulatorCluster:
    """Integration tests for multi-VM cluster simulation."""

    @pytest.fixture
    def cluster(self) -> VMSimulator:
        """Create cluster with 3 default VMs (1000, 1500, 2000 MIPS)."""
        return VMSimulator()

    @pytest.fixture
    def task_batch(self) -> list:
        """Batch of 6 test tasks with diverse arrival times."""
        return [
            Task("T001", 3000.0, 0.0, priority=1, memory_mb=1024, workload_type="normal"),
            Task("T002", 4500.0, 1.0, priority=2, memory_mb=2048, workload_type="normal"),
            Task("T003", 6000.0, 2.0, priority=3, memory_mb=4096, workload_type="normal"),
            Task("T004", 2000.0, 3.0, priority=1, memory_mb=1024, workload_type="normal"),
            Task("T005", 3000.0, 4.0, priority=2, memory_mb=2048, workload_type="normal"),
            Task("T006", 4000.0, 5.0, priority=3, memory_mb=4096, workload_type="normal"),
        ]

    def test_default_cluster_setup(self, cluster: VMSimulator):
        """Verifies 3 VMs configured with expected MIPS."""
        assert cluster.num_vms == 3
        assert cluster.vm_ids == ["VM1", "VM2", "VM3"]
        assert cluster.get_vm("VM1").mips == 1000.0
        assert cluster.get_vm("VM2").mips == 1500.0
        assert cluster.get_vm("VM3").mips == 2000.0

    def test_cluster_size_constraints(self):
        """Verifies cluster size must be between 3 and 5 VMs."""
        with pytest.raises(ValueError, match="Cluster size must be between 3 and 5"):
            VMSimulator(vm_configs=[{"vm_id": "VM1", "mips": 1000.0}])

        # Valid 4 VM configuration
        vms_4 = [
            {"vm_id": f"VM{i}", "mips": 1000.0 * i} for i in range(1, 5)
        ]
        sim4 = VMSimulator(vm_configs=vms_4)
        assert sim4.num_vms == 4

    def test_simulate_assignments_list_indices(self, cluster: VMSimulator, task_batch: list):
        """Verifies simulation using integer VM indices [0, 1, 2, ...]."""
        assignments = [0, 1, 2, 0, 1, 2]
        records = cluster.simulate_assignments(task_batch, assignments)

        assert len(records) == 6
        # VM1 receives task 0 and task 3
        assert records[0].vm_id == "VM1"
        assert records[3].vm_id == "VM1"

        makespan = cluster.get_makespan()
        assert makespan > 0.0

        imbalance = cluster.get_load_imbalance()
        assert imbalance >= 0.0

        avg_util = cluster.get_average_utilization()
        assert 0.0 <= avg_util <= 100.0

    def test_simulate_assignments_dict_mapping(self, cluster: VMSimulator, task_batch: list):
        """Verifies simulation using task_id -> vm_id mapping dictionary."""
        mapping = {
            "T001": "VM1",
            "T002": "VM2",
            "T003": "VM3",
            "T004": "VM1",
            "T005": "VM2",
            "T006": "VM3",
        }
        records = cluster.simulate_assignments(task_batch, mapping)
        assert len(records) == 6
        assert records[0].task_id == "T001" and records[0].vm_id == "VM1"

    def test_cluster_degradation_increases_makespan(self, cluster: VMSimulator, task_batch: list):
        """Verifies degrading a busy VM increases execution time and affects makespan."""
        assignments = [0, 1, 2, 0, 1, 2]

        # Baseline run
        cluster.simulate_assignments(task_batch, assignments)
        baseline_makespan = cluster.get_makespan()
        vm1_baseline_busy = cluster.get_vm("VM1").busy_time

        # Degrade VM1 by 50%
        cluster.apply_vm_degradation("VM1", factor=0.5)
        assert cluster.get_vm("VM1").mips == 500.0

        # Rerun simulation
        cluster.simulate_assignments(task_batch, assignments)
        degraded_makespan = cluster.get_makespan()
        vm1_degraded_busy = cluster.get_vm("VM1").busy_time

        # VM1 busy time should double (same workload on half MIPS)
        assert vm1_degraded_busy == pytest.approx(vm1_baseline_busy * 2, abs=0.01)
        assert degraded_makespan >= baseline_makespan

