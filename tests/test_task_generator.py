"""
Automated unit tests for TaskGenerator module.

Verifies:
1. Generates 20 tasks by default.
2. Generates 50 tasks (max limit).
3. Rejects invalid task counts (< 1, > 50, non-integer).
4. Generates unique task IDs.
5. Produces chronological arrival times (non-decreasing).
6. Reproduces results deterministically with identical seeds.
7. Supports all workload scenarios (light, normal, heavy, bursty) within bounds.
8. DataFrame conversion and CSV export integrity.
"""

import os
from pathlib import Path
import pytest
import numpy as np

from simulation.task_generator import TaskGenerator, Task
from config import (
    WORKLOAD_PROFILES,
    DEFAULT_NUM_TASKS,
    MAX_NUM_TASKS,
    MIN_NUM_TASKS,
)


class TestTaskGenerator:
    """Test suite for TaskGenerator class."""

    @pytest.fixture
    def generator(self) -> TaskGenerator:
        """Fixture providing a deterministic generator instance."""
        return TaskGenerator(seed=42)

    def test_default_generation_count(self, generator: TaskGenerator):
        """Verifies generator produces 20 tasks by default."""
        tasks = generator.generate_tasks()
        assert len(tasks) == DEFAULT_NUM_TASKS
        assert len(tasks) == 20

    def test_generate_fifty_tasks(self, generator: TaskGenerator):
        """Verifies generator correctly produces 50 tasks (maximum configured scale)."""
        tasks = generator.generate_tasks(num_tasks=MAX_NUM_TASKS)
        assert len(tasks) == 50
        assert tasks[-1].task_id == "T050"

    def test_rejects_invalid_task_counts(self, generator: TaskGenerator):
        """Verifies rejection of invalid task counts (negative, zero, > 50, non-integers)."""
        with pytest.raises(ValueError, match="Task count must be between"):
            generator.generate_tasks(num_tasks=0)

        with pytest.raises(ValueError, match="Task count must be between"):
            generator.generate_tasks(num_tasks=-5)

        with pytest.raises(ValueError, match="Task count must be between"):
            generator.generate_tasks(num_tasks=51)

        with pytest.raises(TypeError, match="num_tasks must be an integer"):
            # noinspection PyTypeChecker
            generator.generate_tasks(num_tasks="twenty")  # type: ignore

    def test_rejects_invalid_workload_type(self, generator: TaskGenerator):
        """Verifies error handling when an unsupported workload is requested."""
        with pytest.raises(ValueError, match="Unknown workload type"):
            generator.generate_tasks(num_tasks=20, workload_type="ultra_quantum")

    def test_unique_task_ids(self, generator: TaskGenerator):
        """Verifies every generated task receives a globally unique task_id."""
        tasks = generator.generate_tasks(num_tasks=40)
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids)), "Task IDs contain duplicates!"

    def test_chronological_arrival_times(self, generator: TaskGenerator):
        """Verifies task arrival times are strictly non-decreasing across all workloads."""
        for scenario in ["light", "normal", "heavy", "bursty"]:
            tasks = generator.generate_tasks(num_tasks=30, workload_type=scenario)
            arrivals = [t.arrival_time for t in tasks]
            # Must start at 0 or >= 0
            assert arrivals[0] >= 0.0
            # Non-decreasing condition: t[i] >= t[i-1]
            for i in range(1, len(arrivals)):
                assert arrivals[i] >= arrivals[i - 1], (
                    f"Arrival time inverted at task {i} in scenario '{scenario}': "
                    f"{arrivals[i-1]} > {arrivals[i]}"
                )

    def test_reproducibility_with_seed(self):
        """Verifies identical seeds yield identical tasks, and different seeds differ."""
        gen1 = TaskGenerator(seed=12345)
        tasks1 = gen1.generate_tasks(num_tasks=25, workload_type="normal")

        gen2 = TaskGenerator(seed=12345)
        tasks2 = gen2.generate_tasks(num_tasks=25, workload_type="normal")

        assert len(tasks1) == len(tasks2)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            assert t1.cpu_demand_mi == t2.cpu_demand_mi
            assert t1.arrival_time == t2.arrival_time
            assert t1.priority == t2.priority
            assert t1.memory_mb == t2.memory_mb

        # Different seed must produce different values
        gen3 = TaskGenerator(seed=99999)
        tasks3 = gen3.generate_tasks(num_tasks=25, workload_type="normal")
        different_demands = any(
            t1.cpu_demand_mi != t3.cpu_demand_mi for t1, t3 in zip(tasks1, tasks3)
        )
        assert different_demands, "Different seeds unexpectedly produced identical outputs"

    def test_all_workload_scenarios_bounds(self, generator: TaskGenerator):
        """Verifies CPU demands for each scenario strictly fall within defined bounds."""
        for scenario, profile in WORKLOAD_PROFILES.items():
            tasks = generator.generate_tasks(num_tasks=50, workload_type=scenario)
            for t in tasks:
                assert profile.cpu_min_mi <= t.cpu_demand_mi <= profile.cpu_max_mi, (
                    f"CPU demand {t.cpu_demand_mi} out of range for {scenario} "
                    f"[{profile.cpu_min_mi}, {profile.cpu_max_mi}]"
                )
                assert t.workload_type == scenario
                assert 1 <= t.priority <= 5
                assert t.memory_mb in [512, 1024, 2048, 4096]

    def test_dataframe_conversion(self, generator: TaskGenerator):
        """Verifies to_dataframe creates DataFrame with required columns and data types."""
        tasks = generator.generate_tasks(num_tasks=20, workload_type="bursty")
        df = generator.to_dataframe(tasks)

        expected_columns = {
            "task_id",
            "cpu_demand_mi",
            "arrival_time",
            "priority",
            "memory_mb",
            "workload_type",
        }
        assert set(df.columns) == expected_columns
        assert len(df) == 20
        assert not df.isnull().any().any()

    def test_csv_export(self, generator: TaskGenerator, tmp_path: Path):
        """Verifies CSV export writes valid file readable by pandas."""
        csv_path = tmp_path / "test_tasks.csv"
        tasks = generator.generate_tasks(num_tasks=20, workload_type="light")
        saved_file = generator.save_to_csv(csv_path, tasks)

        assert os.path.exists(saved_file)
        import pandas as pd
        loaded_df = pd.read_csv(saved_file)
        assert len(loaded_df) == 20
        assert loaded_df["task_id"].iloc[0] == "T001"
        assert loaded_df["cpu_demand_mi"].iloc[0] == tasks[0].cpu_demand_mi

