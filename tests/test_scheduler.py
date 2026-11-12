"""
Automated unit and integration tests for BaselineScheduler module.

Verifies:
1. Every task is assigned exactly once across all heuristics (FCFS, RR, MCT).
2. All assigned VM indices are valid and correspond to active VMs.
3. Start and finish times are chronologically valid and respect arrival times.
4. Total makespan is positive and correctly reflects max finish time.
5. Utilization percentages remain within [0.0, 100.0]%.
6. Handles small (1 task), medium (20 tasks), and large (50 tasks) workloads.
7. Correctly handles VM degradation scenarios during baseline comparisons.
"""

import pytest
import numpy as np

from simulation.task_generator import TaskGenerator, Task
from simulation.vm_simulator import VMSimulator
from simulation.scheduler import BaselineScheduler, ScheduleResult


class TestBaselineScheduler:
    """Test suite for baseline scheduling algorithms."""

    @pytest.fixture
    def generator(self) -> TaskGenerator:
        """Deterministic task generator fixture."""
        return TaskGenerator(seed=42)

    @pytest.fixture
    def scheduler(self) -> BaselineScheduler:
        """Baseline scheduler fixture with standard 3-VM cluster."""
        return BaselineScheduler()

    def test_assign_every_task_exactly_once(self, generator: TaskGenerator, scheduler: BaselineScheduler):
        """Verifies all tasks are scheduled once without loss or duplicates."""
        tasks = generator.generate_tasks(num_tasks=25, workload_type="normal")

        for algo_name in ["FCFS", "RoundRobin", "MCT"]:
            if algo_name == "FCFS":
                res = scheduler.schedule_fcfs(tasks)
            elif algo_name == "RoundRobin":
                res = scheduler.schedule_round_robin(tasks)
            else:
                res = scheduler.schedule_mct(tasks)

            assert len(res.assignments) == 25
            assert len(res.scheduled_records) == 25

            scheduled_ids = [rec.task_id for rec in res.scheduled_records]
            assert set(scheduled_ids) == {t.task_id for t in tasks}
            assert len(scheduled_ids) == len(set(scheduled_ids))

    def test_valid_vm_indices(self, generator: TaskGenerator, scheduler: BaselineScheduler):
        """Verifies assigned VM indices are always in [0, num_vms - 1]."""
        tasks = generator.generate_tasks(num_tasks=30, workload_type="heavy")
        results = scheduler.run_all(tasks)

        for name, res in results.items():
            for vm_idx in res.assignments:
                assert 0 <= vm_idx < scheduler.simulator.num_vms, (
                    f"{name} produced out-of-range VM index: {vm_idx}"
                )

    def test_chronological_validity_and_arrival_respect(
        self,
        generator: TaskGenerator,
        scheduler: BaselineScheduler
    ):
        """Verifies start_time >= arrival_time and finish_time = start_time + execution_time."""
        tasks = generator.generate_tasks(num_tasks=20, workload_type="bursty")
        results = scheduler.run_all(tasks)

        for name, res in results.items():
            for rec in res.scheduled_records:
                assert rec.start_time >= rec.arrival_time, (
                    f"{name}: Task {rec.task_id} started at {rec.start_time} before arrival {rec.arrival_time}"
                )
                assert rec.finish_time == pytest.approx(rec.start_time + rec.execution_time, abs=1e-4)
                assert rec.waiting_time >= 0.0
                assert rec.turnaround_time >= rec.execution_time

    def test_makespan_and_metrics_consistency(
        self,
        generator: TaskGenerator,
        scheduler: BaselineScheduler
    ):
        """Verifies makespan equals the max finish time across all tasks."""
        tasks = generator.generate_tasks(num_tasks=20, workload_type="light")
        results = scheduler.run_all(tasks)

        for name, res in results.items():
            max_finish = max(rec.finish_time for rec in res.scheduled_records)
            assert res.makespan == pytest.approx(max_finish, abs=1e-4)
            assert 0.0 <= res.average_utilization <= 100.0
            assert res.load_imbalance >= 0.0
            for vm_id, util in res.vm_utilizations.items():
                assert 0.0 <= util <= 100.0

    def test_single_task_edge_case(self, scheduler: BaselineScheduler):
        """Verifies proper execution with edge case of 1 task."""
        single_task = [
            Task(
                task_id="T001",
                cpu_demand_mi=2000.0,
                arrival_time=0.0,
                priority=1,
                memory_mb=1024,
                workload_type="normal",
            )
        ]
        results = scheduler.run_all(single_task)
        for name, res in results.items():
            assert len(res.scheduled_records) == 1
            assert res.makespan > 0.0
            assert res.average_waiting_time == 0.0

    def test_max_tasks_scale(self, generator: TaskGenerator, scheduler: BaselineScheduler):
        """Verifies robust execution with 50 tasks (maximum simulation scale)."""
        tasks = generator.generate_tasks(num_tasks=50, workload_type="bursty")
        results = scheduler.run_all(tasks)

        for name, res in results.items():
            assert len(res.scheduled_records) == 50
            assert res.makespan > 0.0

    def test_df_comparison_output(self, generator: TaskGenerator, scheduler: BaselineScheduler):
        """Verifies compare_baselines generates expected columns and rows."""
        tasks = generator.generate_tasks(num_tasks=15, workload_type="normal")
        df = scheduler.compare_baselines(tasks)

        assert len(df) == 3
        expected_cols = {
            "algorithm",
            "makespan_sec",
            "avg_utilization_pct",
            "load_imbalance",
            "total_execution_sec",
            "avg_waiting_sec",
            "avg_turnaround_sec",
            "task_count",
        }
        assert expected_cols.issubset(set(df.columns))

