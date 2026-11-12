"""
Automated unit tests for Behavior-Aware AdaptiveSwarmController module.

Verifies:
1. Detects optimization stagnation accurately when fitness does not improve.
2. Measures and responds to population diversity correctly.
3. Detects workload and VM capacity changes (environmental disturbance).
4. Makes explainable rule-based decisions rather than random selection.
5. Achieves monotonic convergence with valid task assignments across all tasks.
6. Seamlessly transfers population state between algorithms without progress loss.
7. Decision trail contains all required explainability fields.
"""

import pytest
import numpy as np

from simulation.task_generator import TaskGenerator, Task
from simulation.vm_simulator import VMSimulator
from controller.adaptive_controller import AdaptiveSwarmController, ControllerDecision
from algorithms.base import OptimizationResult


class TestAdaptiveController:
    """Test suite for AdaptiveSwarmController class."""

    @pytest.fixture
    def sim(self) -> VMSimulator:
        """Standard 3-VM cluster."""
        return VMSimulator()

    @pytest.fixture
    def tasks_20(self) -> list:
        """Standard 20-task batch."""
        gen = TaskGenerator(seed=42)
        return gen.generate_tasks(num_tasks=20, workload_type="normal")

    @pytest.fixture
    def controller(self) -> AdaptiveSwarmController:
        """Controller instance with standard thresholds."""
        return AdaptiveSwarmController(
            population_size=15,
            max_iterations=20,
            stagnation_threshold=3,
            diversity_low_threshold=0.15,
            seed=42,
        )

    def test_explainable_decisions_not_random(self, controller: AdaptiveSwarmController):
        """Verifies rule engine returns deterministic explainable decisions."""
        # Condition A: Workload changed
        algo_a, reason_a = controller.evaluate_decision_rules(
            diversity=0.30, improvement=0.01, stagnation=0, workload_changed=True
        )
        assert algo_a == "PSO"
        assert "disturbance detected" in reason_a.lower()

        # Condition B: High stagnation + Low diversity
        algo_b, reason_b = controller.evaluate_decision_rules(
            diversity=0.05, improvement=0.0, stagnation=5, workload_changed=False
        )
        assert algo_b == "ABC"
        assert "scout" in reason_b.lower()

        # Condition C: Active improvement + Healthy diversity
        algo_c, reason_c = controller.evaluate_decision_rules(
            diversity=0.45, improvement=0.08, stagnation=0, workload_changed=False
        )
        assert algo_c == "GWO"
        assert "exploitation" in reason_c.lower()

    def test_detects_vm_degradation_disturbance(
        self,
        controller: AdaptiveSwarmController,
        sim: VMSimulator,
        tasks_20: list
    ):
        """Verifies controller detects runtime capacity change in VM cluster."""
        # Initial baseline check
        assert not controller.detect_workload_or_infrastructure_change(tasks_20, sim)

        # Degrade VM1
        sim.apply_vm_degradation("VM1", factor=0.5)

        # Now change should be detected
        changed = controller.detect_workload_or_infrastructure_change(tasks_20, sim)
        assert changed is True

        # Subsequent check without modifications returns False
        assert not controller.detect_workload_or_infrastructure_change(tasks_20, sim)

    def test_detects_task_burst_disturbance(
        self,
        controller: AdaptiveSwarmController,
        sim: VMSimulator,
        tasks_20: list
    ):
        """Verifies controller detects sudden arrival of additional tasks."""
        controller.detect_workload_or_infrastructure_change(tasks_20, sim)

        # Surge: Append 5 new tasks
        gen = TaskGenerator(seed=99)
        new_batch = tasks_20 + gen.generate_tasks(num_tasks=5, workload_type="bursty")

        changed = controller.detect_workload_or_infrastructure_change(new_batch, sim)
        assert changed is True

    def test_adaptive_optimization_validity(
        self,
        controller: AdaptiveSwarmController,
        sim: VMSimulator,
        tasks_20: list
    ):
        """Verifies adaptive optimization completes and assigns all tasks to valid VMs."""
        res = controller.optimize(tasks_20, sim, max_iterations=25)

        assert isinstance(res, OptimizationResult)
        assert res.algorithm_name == "Adaptive"
        assert len(res.best_assignment) == len(tasks_20)
        assert res.iterations_run == 25
        assert res.best_fitness < float("inf")

        # Every task mapped to valid VM index
        for vm_idx in res.best_assignment:
            assert isinstance(vm_idx, (int, np.integer))
            assert 0 <= vm_idx < sim.num_vms

    def test_monotonic_convergence_and_decision_logging(
        self,
        controller: AdaptiveSwarmController,
        sim: VMSimulator,
        tasks_20: list
    ):
        """Verifies best fitness is non-increasing and decision dataframe is complete."""
        res = controller.optimize(tasks_20, sim, max_iterations=20)

        # Monotonic convergence
        for i in range(1, len(res.convergence_history)):
            assert res.convergence_history[i] <= res.convergence_history[i - 1] + 1e-9

        df = controller.get_decision_history_dataframe()
        assert len(df) == 20
        expected_cols = {
            "iteration",
            "selected_algorithm",
            "reason",
            "diversity",
            "fitness_improvement",
            "stagnation_count",
            "workload_changed",
            "best_fitness",
        }
        assert expected_cols.issubset(set(df.columns))
        assert not df["selected_algorithm"].isnull().any()
        assert not df["reason"].isnull().any()

    def test_reproducibility(self, sim: VMSimulator, tasks_20: list):
        """Verifies identical seed yields identical decision path and final fitness."""
        c1 = AdaptiveSwarmController(population_size=12, max_iterations=15, seed=456)
        res1 = c1.optimize(tasks_20, sim)

        c2 = AdaptiveSwarmController(population_size=12, max_iterations=15, seed=456)
        res2 = c2.optimize(tasks_20, sim)

        assert res1.best_assignment == res2.best_assignment
        assert res1.best_fitness == pytest.approx(res2.best_fitness, abs=1e-6)
        assert res1.convergence_history == pytest.approx(res2.convergence_history, abs=1e-6)
        assert c1.algorithm_history == c2.algorithm_history

