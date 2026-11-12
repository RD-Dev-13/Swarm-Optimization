"""
Automated unit tests for experimental evaluation harness and visualization module.

Verifies:
1. ExperimentRunner creates required directories and executes multi-algorithm trial.
2. Trial results contain metrics for all 7 algorithms (FCFS, RR, MCT, PSO, GWO, ABC, Adaptive).
3. Aggregation computes valid mean and standard deviations without NaN or null values.
4. CloudSchedulerVisualizer generates figures and saves PNG files to disk.
"""

from pathlib import Path
import pytest
import matplotlib.pyplot as plt
import pandas as pd

from simulation.task_generator import TaskGenerator
from simulation.vm_simulator import VMSimulator
from experiments.run_experiments import ExperimentRunner
from visualization.charts import CloudSchedulerVisualizer


class TestExperimentalEvaluation:
    """Test suite for ExperimentRunner class."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> ExperimentRunner:
        """Create runner using temporary output directories."""
        return ExperimentRunner(
            results_dir=str(tmp_path / "results"),
            graphs_dir=str(tmp_path / "graphs"),
            population_size=10,
            max_iterations=5,
        )

    def test_single_trial_executes_all_seven_algorithms(self, runner: ExperimentRunner):
        """Verifies all 7 algorithms are executed and evaluated in a single trial."""
        gen = TaskGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=10, workload_type="normal")
        sim = VMSimulator()

        trial_res = runner._execute_single_trial(tasks, sim, seed=42)

        expected_algos = {"FCFS", "RoundRobin", "MCT", "PSO", "GWO", "ABC", "Adaptive"}
        assert set(trial_res.keys()) == expected_algos

        for algo, metrics in trial_res.items():
            assert metrics["makespan"] > 0.0
            assert metrics["fitness"] > 0.0
            assert 0.0 <= metrics["utilization"] <= 100.0

    def test_aggregate_trials_computation(self, runner: ExperimentRunner):
        """Verifies aggregation computes valid means and standard deviations."""
        sample_trial_data = [
            {
                "FCFS": {"makespan": 30.0, "fitness": 33.0, "utilization": 80.0, "load_imbalance": 0.2, "waiting_time": 4.0},
                "Adaptive": {"makespan": 28.0, "fitness": 29.0, "utilization": 90.0, "load_imbalance": 0.05, "waiting_time": 3.5},
            },
            {
                "FCFS": {"makespan": 32.0, "fitness": 35.0, "utilization": 82.0, "load_imbalance": 0.22, "waiting_time": 4.2},
                "Adaptive": {"makespan": 29.0, "fitness": 30.0, "utilization": 88.0, "load_imbalance": 0.07, "waiting_time": 3.8},
            },
        ]
        df = runner._aggregate_trials("TestScenario", sample_trial_data)

        assert len(df) == 2
        assert "makespan_mean" in df.columns
        assert "makespan_std" in df.columns
        assert not df.isnull().any().any()


class TestVisualizationModule:
    """Test suite for CloudSchedulerVisualizer class."""

    @pytest.fixture
    def tasks(self):
        """Create sample task batch."""
        gen = TaskGenerator(seed=42)
        return gen.generate_tasks(num_tasks=15, workload_type="normal")

    def test_plot_task_cpu_demands(self, tasks, tmp_path: Path):
        """Verifies CPU demands plot generation."""
        out = tmp_path / "cpu.png"
        fig = CloudSchedulerVisualizer.plot_task_cpu_demands(tasks, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_plot_task_arrivals(self, tasks, tmp_path: Path):
        """Verifies task arrivals plot generation."""
        out = tmp_path / "arrivals.png"
        fig = CloudSchedulerVisualizer.plot_task_arrivals(tasks, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_plot_makespan_comparison(self, tmp_path: Path):
        """Verifies makespan comparison plot generation."""
        df = pd.DataFrame([
            {"algorithm": "FCFS", "makespan_sec": 30.0},
            {"algorithm": "Adaptive", "makespan_sec": 26.5},
        ])
        out = tmp_path / "comp.png"
        fig = CloudSchedulerVisualizer.plot_makespan_comparison(df, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

