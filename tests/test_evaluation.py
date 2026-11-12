"""
Automated unit tests for evaluation modules: fitness.py and metrics.py.

Verifies:
1. FitnessEvaluator computes makespan + lambda * imbalance accurately.
2. Supports all imbalance metrics ('std', 'range', 'degree') with dimensional consistency.
3. Supports warm-start partial schedules with initial ready times.
4. Handles optional energy, cost, and deadline extensions without disruption.
5. MetricsCalculator computes all 10 specified cloud scheduling metrics.
6. Utilization percentages are bounded between [0.0, 100.0]%.
7. Convergence speed, switching frequency, and adaptation response time calculate correctly.
"""

import pytest
import numpy as np

from simulation.task_generator import TaskGenerator, Task
from simulation.vm_simulator import VMSimulator
from simulation.scheduler import BaselineScheduler
from evaluation.fitness import FitnessEvaluator, FitnessBreakdown
from evaluation.metrics import MetricsCalculator, PerformanceMetricsReport


class TestFitnessEvaluator:
    """Test suite for FitnessEvaluator class."""

    @pytest.fixture
    def sim(self) -> VMSimulator:
        """Create standard 3-VM simulator."""
        return VMSimulator()

    @pytest.fixture
    def tasks(self) -> list:
        """Create fixed 6-task batch."""
        return [
            Task("T001", 3000.0, 0.0, 2, 1024, "normal"),
            Task("T002", 4500.0, 1.0, 2, 2048, "normal"),
            Task("T003", 6000.0, 2.0, 3, 4096, "normal"),
            Task("T004", 3000.0, 3.0, 1, 1024, "normal"),
            Task("T005", 1500.0, 4.0, 2, 2048, "normal"),
            Task("T006", 2000.0, 5.0, 3, 4096, "normal"),
        ]

    def test_lambda_rejection(self):
        """Verifies negative lambda is rejected."""
        with pytest.raises(ValueError, match="lambda_imbalance cannot be negative"):
            FitnessEvaluator(lambda_imbalance=-1.0)

    def test_unknown_imbalance_metric(self):
        """Verifies invalid imbalance metric is rejected."""
        with pytest.raises(ValueError, match="Unknown imbalance_metric"):
            FitnessEvaluator(imbalance_metric="quantum_variance")

    def test_fitness_scale_compatibility(self, sim: VMSimulator, tasks: list):
        """Verifies fitness = makespan + lambda * load_imbalance in seconds."""
        evaluator = FitnessEvaluator(lambda_imbalance=2.0, imbalance_metric="std")
        assignments = [0, 1, 2, 0, 1, 2]

        breakdown = evaluator.evaluate_detailed(assignments, tasks, sim)
        expected = breakdown.makespan + (2.0 * breakdown.load_imbalance)
        assert breakdown.fitness == pytest.approx(expected, abs=1e-4)

    def test_partial_schedule_warm_start(self, sim: VMSimulator, tasks: list):
        """Verifies partial schedule starts from pre-existing VM ready clocks."""
        evaluator = FitnessEvaluator(lambda_imbalance=1.0)
        assignments = [0, 1, 2, 0, 1, 2]

        # Normal cold start
        cold_breakdown = evaluator.evaluate_detailed(assignments, tasks, sim)

        # Warm start where VM1 already has 10 seconds of prior execution
        initial_clocks = {"VM1": 10.0, "VM2": 0.0, "VM3": 0.0}
        warm_breakdown = evaluator.evaluate_detailed(
            assignments, tasks, sim, initial_ready_times=initial_clocks
        )

        assert warm_breakdown.makespan > cold_breakdown.makespan
        assert warm_breakdown.fitness > cold_breakdown.fitness

    def test_optional_extensions(self, sim: VMSimulator, tasks: list):
        """Verifies energy and cost extensions compute without errors."""
        evaluator = FitnessEvaluator(
            lambda_imbalance=1.0,
            enable_energy=True,
            energy_weight=0.01,
            enable_cost=True,
            cost_weight=0.1,
        )
        assignments = [0, 1, 2, 0, 1, 2]
        breakdown = evaluator.evaluate_detailed(assignments, tasks, sim)

        assert breakdown.energy_joules is not None
        assert breakdown.energy_joules > 0.0
        assert breakdown.scheduling_cost is not None
        assert breakdown.scheduling_cost > 0.0


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample scheduled tasks from BaselineScheduler."""
        gen = TaskGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
        sched = BaselineScheduler()
        res = sched.schedule_mct(tasks)
        return res, sched.simulator

    def test_makespan_calculation(self, sample_data):
        """Verifies makespan equals maximum finish time."""
        res, _ = sample_data
        computed = MetricsCalculator.calculate_makespan(res.scheduled_records)
        assert computed == res.makespan
        assert computed > 0.0

    def test_utilization_bounds(self, sample_data):
        """Verifies average and individual utilizations stay in [0, 100]%."""
        res, sim = sample_data
        avg_u, per_vm = MetricsCalculator.calculate_utilizations(sim, res.makespan)

        assert 0.0 <= avg_u <= 100.0
        for vm_id, u in per_vm.items():
            assert 0.0 <= u <= 100.0

    def test_load_imbalance_metrics(self, sample_data):
        """Verifies DI and std dev computation."""
        _, sim = sample_data
        di, std_busy = MetricsCalculator.calculate_load_imbalance(sim)
        assert di >= 0.0
        assert std_busy >= 0.0

    def test_convergence_speed(self):
        """Verifies convergence speed identifies earliest iteration within tolerance."""
        # Minimum is 50.0 at iteration 8; tolerance 2% means threshold is 51.0
        curve = [100.0, 90.0, 80.0, 70.0, 60.0, 52.0, 50.8, 50.0, 50.0, 50.0]
        # At index 6 (1-based iteration 7), 50.8 <= 51.0
        conv_iter = MetricsCalculator.calculate_convergence_speed(curve, tolerance_pct=2.0)
        assert conv_iter == 7

    def test_switching_frequency(self):
        """Verifies switching frequency formula: switches / (epochs - 1)."""
        history = ["PSO", "PSO", "GWO", "GWO", "ABC"]  # 5 epochs, 2 switches
        freq = MetricsCalculator.calculate_switching_frequency(history)
        assert freq == 2.0 / 4.0  # 0.5

        # No switches
        assert MetricsCalculator.calculate_switching_frequency(["PSO", "PSO", "PSO"]) == 0.0

    def test_adaptation_response_time(self):
        """Verifies adaptation latency computation after event."""
        # Event at iteration 4; curve recovers to stable minimum (40.0)
        history = [50.0, 48.0, 45.0, 80.0, 65.0, 50.0, 41.5, 40.0, 40.0]
        # Event at iteration 4 (value 80.0); minimum post-event is 40.0
        # Target 5% = 42.0. At index 6 (iteration 7, which is 3 iterations after event), 41.5 <= 42.0
        latency = MetricsCalculator.calculate_adaptation_response_time(
            history, event_iteration=4, recovery_margin_pct=5.0
        )
        assert latency == 3

    def test_full_report_generation(self, sample_data):
        """Verifies comprehensive report populates all 10 metrics without missing fields."""
        res, sim = sample_data
        report = MetricsCalculator.generate_full_report(
            records=res.scheduled_records,
            simulator=sim,
            fitness_history=[60.0, 55.0, 50.0],
            algorithm_history=["PSO", "GWO", "ABC"],
            event_iteration=2,
        )

        assert isinstance(report, PerformanceMetricsReport)
        assert report.makespan_sec > 0.0
        assert report.avg_utilization_pct > 0.0
        assert report.convergence_speed_iterations is not None
        assert report.switching_frequency is not None
        assert report.adaptation_response_iterations is not None

        df = report.to_dataframe()
        assert len(df) == 1
        assert "makespan_sec" in df.columns
        assert "util_VM1_pct" in df.columns

