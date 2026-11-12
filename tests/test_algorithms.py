"""
Automated unit tests for metaheuristic optimization algorithms (PSO, GWO, ABC).

Verifies:
1. Every task is assigned to a valid VM index in [0, num_vms - 1].
2. No task is lost, duplicated, or unassigned.
3. Best fitness is strictly non-increasing across iterations (monotonic convergence).
4. Population diversity remains strictly bounded within [0.0, 1.0].
5. Deterministic reproducibility when initialized with identical random seeds.
6. Proper support for custom warm-start initial populations.
7. Robustness across small (5 tasks) and large (50 tasks) problem instances.
"""

import pytest
import numpy as np

from simulation.task_generator import TaskGenerator, Task
from simulation.vm_simulator import VMSimulator
from algorithms.pso import ParticleSwarmOptimizer
from algorithms.gwo import GreyWolfOptimizer
from algorithms.abc import ArtificialBeeColony
from algorithms.base import BaseOptimizer, OptimizationResult


class TestPSO:
    """Test suite for Particle Swarm Optimization implementation."""

    @pytest.fixture
    def sim(self) -> VMSimulator:
        """Standard 3-VM cluster."""
        return VMSimulator()

    @pytest.fixture
    def tasks_20(self) -> list:
        """Standard 20-task batch."""
        gen = TaskGenerator(seed=42)
        return gen.generate_tasks(num_tasks=20, workload_type="normal")

    def test_pso_valid_discrete_assignments(self, sim: VMSimulator, tasks_20: list):
        """Verifies every task is assigned to a valid VM index in [0, num_vms - 1]."""
        pso = ParticleSwarmOptimizer(population_size=15, max_iterations=20, seed=123)
        res = pso.optimize(tasks_20, sim)

        assert isinstance(res, OptimizationResult)
        assert len(res.best_assignment) == len(tasks_20)
        for vm_idx in res.best_assignment:
            assert isinstance(vm_idx, (int, np.integer))
            assert 0 <= vm_idx < sim.num_vms

    def test_pso_monotonic_convergence(self, sim: VMSimulator, tasks_20: list):
        """Verifies best fitness history is non-increasing (f[t] <= f[t-1])."""
        pso = ParticleSwarmOptimizer(population_size=15, max_iterations=25, seed=42)
        res = pso.optimize(tasks_20, sim)

        history = res.convergence_history
        assert len(history) == 25
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-9, (
                f"Convergence violated at generation {i}: {history[i]} > {history[i-1]}"
            )

    def test_pso_diversity_bounds(self, sim: VMSimulator, tasks_20: list):
        """Verifies population diversity is within [0.0, 1.0]."""
        pso = ParticleSwarmOptimizer(population_size=15, max_iterations=20, seed=42)
        res = pso.optimize(tasks_20, sim)

        div_history = res.diversity_history
        assert len(div_history) == 20
        for d in div_history:
            assert 0.0 <= d <= 1.0

    def test_pso_reproducibility(self, sim: VMSimulator, tasks_20: list):
        """Verifies identical seed yields identical search path and solution."""
        pso1 = ParticleSwarmOptimizer(population_size=12, max_iterations=15, seed=777)
        res1 = pso1.optimize(tasks_20, sim)

        pso2 = ParticleSwarmOptimizer(population_size=12, max_iterations=15, seed=777)
        res2 = pso2.optimize(tasks_20, sim)

        assert res1.best_assignment == res2.best_assignment
        assert res1.best_fitness == pytest.approx(res2.best_fitness, abs=1e-6)
        assert res1.convergence_history == pytest.approx(res2.convergence_history, abs=1e-6)

    def test_pso_warm_start_population(self, sim: VMSimulator, tasks_20: list):
        """Verifies initial population injection is respected."""
        pop_size = 10
        num_tasks = len(tasks_20)
        custom_pop = np.full((pop_size, num_tasks), 1.5)  # All particles initially near VM1

        pso = ParticleSwarmOptimizer(population_size=pop_size, max_iterations=10, seed=42)
        res = pso.optimize(tasks_20, sim, initial_population=custom_pop)

        assert res.iterations_run == 10
        assert res.best_fitness < float("inf")

    def test_pso_scale_small_and_large(self, sim: VMSimulator):
        """Verifies scalability across 5 tasks and 50 tasks."""
        gen = TaskGenerator(seed=42)

        # Small scale: 5 tasks
        small_tasks = gen.generate_tasks(num_tasks=5, workload_type="light")
        pso_small = ParticleSwarmOptimizer(population_size=10, max_iterations=10, seed=42)
        res_small = pso_small.optimize(small_tasks, sim)
        assert len(res_small.best_assignment) == 5

        # Large scale: 50 tasks
        large_tasks = gen.generate_tasks(num_tasks=50, workload_type="heavy")
        pso_large = ParticleSwarmOptimizer(population_size=15, max_iterations=15, seed=42)
        res_large = pso_large.optimize(large_tasks, sim)
        assert len(res_large.best_assignment) == 50


class TestGWO:
    """Test suite for Grey Wolf Optimizer implementation."""

    @pytest.fixture
    def sim(self) -> VMSimulator:
        """Standard 3-VM cluster."""
        return VMSimulator()

    @pytest.fixture
    def tasks_20(self) -> list:
        """Standard 20-task batch."""
        gen = TaskGenerator(seed=42)
        return gen.generate_tasks(num_tasks=20, workload_type="normal")

    def test_gwo_valid_discrete_assignments(self, sim: VMSimulator, tasks_20: list):
        """Verifies GWO produces valid VM indices in [0, num_vms - 1] for all tasks."""
        gwo = GreyWolfOptimizer(population_size=15, max_iterations=20, seed=123)
        res = gwo.optimize(tasks_20, sim)

        assert isinstance(res, OptimizationResult)
        assert len(res.best_assignment) == len(tasks_20)
        for vm_idx in res.best_assignment:
            assert isinstance(vm_idx, (int, np.integer))
            assert 0 <= vm_idx < sim.num_vms

    def test_gwo_monotonic_convergence(self, sim: VMSimulator, tasks_20: list):
        """Verifies Alpha fitness history is strictly non-increasing."""
        gwo = GreyWolfOptimizer(population_size=15, max_iterations=25, seed=42)
        res = gwo.optimize(tasks_20, sim)

        history = res.convergence_history
        assert len(history) == 25
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-9, (
                f"Convergence violated at generation {i}: {history[i]} > {history[i-1]}"
            )

    def test_gwo_diversity_bounds(self, sim: VMSimulator, tasks_20: list):
        """Verifies GWO population diversity is in [0.0, 1.0]."""
        gwo = GreyWolfOptimizer(population_size=15, max_iterations=20, seed=42)
        res = gwo.optimize(tasks_20, sim)

        div_history = res.diversity_history
        assert len(div_history) == 20
        for d in div_history:
            assert 0.0 <= d <= 1.0

    def test_gwo_reproducibility(self, sim: VMSimulator, tasks_20: list):
        """Verifies identical seed yields identical results."""
        gwo1 = GreyWolfOptimizer(population_size=12, max_iterations=15, seed=888)
        res1 = gwo1.optimize(tasks_20, sim)

        gwo2 = GreyWolfOptimizer(population_size=12, max_iterations=15, seed=888)
        res2 = gwo2.optimize(tasks_20, sim)

        assert res1.best_assignment == res2.best_assignment
        assert res1.best_fitness == pytest.approx(res2.best_fitness, abs=1e-6)
        assert res1.convergence_history == pytest.approx(res2.convergence_history, abs=1e-6)

    def test_gwo_warm_start_population(self, sim: VMSimulator, tasks_20: list):
        """Verifies warm-start pack matrix is respected."""
        pop_size = 10
        num_tasks = len(tasks_20)
        custom_pack = np.full((pop_size, num_tasks), 2.1)

        gwo = GreyWolfOptimizer(population_size=pop_size, max_iterations=10, seed=42)
        res = gwo.optimize(tasks_20, sim, initial_population=custom_pack)

        assert res.iterations_run == 10
        assert res.best_fitness < float("inf")

    def test_gwo_scale_small_and_large(self, sim: VMSimulator):
        """Verifies scalability across 5 tasks and 50 tasks."""
        gen = TaskGenerator(seed=42)

        small_tasks = gen.generate_tasks(num_tasks=5, workload_type="light")
        gwo_small = GreyWolfOptimizer(population_size=10, max_iterations=10, seed=42)
        res_small = gwo_small.optimize(small_tasks, sim)
        assert len(res_small.best_assignment) == 5

        large_tasks = gen.generate_tasks(num_tasks=50, workload_type="heavy")
        gwo_large = GreyWolfOptimizer(population_size=15, max_iterations=15, seed=42)
        res_large = gwo_large.optimize(large_tasks, sim)
        assert len(res_large.best_assignment) == 50


class TestABC:
    """Test suite for Artificial Bee Colony algorithm implementation."""

    @pytest.fixture
    def sim(self) -> VMSimulator:
        """Standard 3-VM cluster."""
        return VMSimulator()

    @pytest.fixture
    def tasks_20(self) -> list:
        """Standard 20-task batch."""
        gen = TaskGenerator(seed=42)
        return gen.generate_tasks(num_tasks=20, workload_type="normal")

    def test_abc_valid_discrete_assignments(self, sim: VMSimulator, tasks_20: list):
        """Verifies ABC produces valid discrete VM indices in [0, num_vms - 1] for all tasks."""
        abc = ArtificialBeeColony(population_size=15, max_iterations=20, seed=123)
        res = abc.optimize(tasks_20, sim)

        assert isinstance(res, OptimizationResult)
        assert len(res.best_assignment) == len(tasks_20)
        for vm_idx in res.best_assignment:
            assert isinstance(vm_idx, (int, np.integer))
            assert 0 <= vm_idx < sim.num_vms

    def test_abc_monotonic_convergence(self, sim: VMSimulator, tasks_20: list):
        """Verifies best fitness history is strictly non-increasing."""
        abc = ArtificialBeeColony(population_size=15, max_iterations=25, seed=42)
        res = abc.optimize(tasks_20, sim)

        history = res.convergence_history
        assert len(history) == 25
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-9, (
                f"Convergence violated at generation {i}: {history[i]} > {history[i-1]}"
            )

    def test_abc_diversity_bounds(self, sim: VMSimulator, tasks_20: list):
        """Verifies ABC diversity is strictly in [0.0, 1.0]."""
        abc = ArtificialBeeColony(population_size=15, max_iterations=20, seed=42)
        res = abc.optimize(tasks_20, sim)

        div_history = res.diversity_history
        assert len(div_history) == 20
        for d in div_history:
            assert 0.0 <= d <= 1.0

    def test_abc_reproducibility(self, sim: VMSimulator, tasks_20: list):
        """Verifies identical seed yields identical results."""
        abc1 = ArtificialBeeColony(population_size=12, max_iterations=15, seed=999)
        res1 = abc1.optimize(tasks_20, sim)

        abc2 = ArtificialBeeColony(population_size=12, max_iterations=15, seed=999)
        res2 = abc2.optimize(tasks_20, sim)

        assert res1.best_assignment == res2.best_assignment
        assert res1.best_fitness == pytest.approx(res2.best_fitness, abs=1e-6)
        assert res1.convergence_history == pytest.approx(res2.convergence_history, abs=1e-6)

    def test_abc_warm_start_population(self, sim: VMSimulator, tasks_20: list):
        """Verifies warm-start food source matrix is respected."""
        pop_size = 10
        num_tasks = len(tasks_20)
        custom_foods = np.full((pop_size, num_tasks), 0.8)

        abc = ArtificialBeeColony(population_size=pop_size, max_iterations=10, seed=42)
        res = abc.optimize(tasks_20, sim, initial_population=custom_foods)

        assert res.iterations_run == 10
        assert res.best_fitness < float("inf")

    def test_abc_scale_small_and_large(self, sim: VMSimulator):
        """Verifies scalability across 5 tasks and 50 tasks."""
        gen = TaskGenerator(seed=42)

        small_tasks = gen.generate_tasks(num_tasks=5, workload_type="light")
        abc_small = ArtificialBeeColony(population_size=10, max_iterations=10, seed=42)
        res_small = abc_small.optimize(small_tasks, sim)
        assert len(res_small.best_assignment) == 5

        large_tasks = gen.generate_tasks(num_tasks=50, workload_type="heavy")
        abc_large = ArtificialBeeColony(population_size=15, max_iterations=15, seed=42)
        res_large = abc_large.optimize(large_tasks, sim)
        assert len(res_large.best_assignment) == 50


