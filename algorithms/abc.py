"""
Artificial Bee Colony (ABC) Algorithm for Cloud Resource Scheduling.

Simulates intelligent foraging behavior of honeybees:
- Employed Bees: Exploit neighborhood of assigned food sources.
- Onlooker Bees: Probabilistically recruit to high-nectar sources via roulette selection.
- Scout Bees: Abandon stagnated sources (trials >= limit) and discover new random solutions.
"""

from typing import List, Optional, Tuple, Union
import time
import numpy as np

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator
from evaluation.fitness import FitnessEvaluator, FitnessBreakdown
from algorithms.base import BaseOptimizer, OptimizationResult
from config import SWARM_POPULATION_SIZE, MAX_OPTIMIZATION_ITERATIONS


class ArtificialBeeColony(BaseOptimizer):
    """
    Continuous-to-Discrete Artificial Bee Colony (ABC) optimization engine.

    Structure:
        - Food Sources (SN = population_size): Candidate cloud schedules.
        - Employed Bees: Exactly SN bees, each attached to one food source.
        - Onlooker Bees: Exactly SN bees, selecting sources via fitness-proportional roulette.
        - Scout Bees: Replaces any food source that fails to improve after 'limit' trials.

    Discrete Neighborhood Operator:
        v_{i, d} = x_{i, d} + phi * (x_{i, d} - x_{k, d})
        where:
            k in {0, ..., SN - 1}, k != i
            d in {0, ..., num_tasks - 1} (random dimension)
            phi ~ U(-1.0, 1.0)
    """

    def __init__(
        self,
        population_size: int = SWARM_POPULATION_SIZE,
        max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
        abandonment_limit: Optional[int] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize the Artificial Bee Colony optimizer.

        Args:
            population_size: Number of food sources (default: 20).
            max_iterations: Maximum foraging cycles (default: 50).
            abandonment_limit: Stagnation trial threshold before scout abandonment.
                               If None, defaults to max(5, int(0.5 * population_size)).
            fitness_evaluator: FitnessEvaluator instance.
            seed: Deterministic random seed.
        """
        super().__init__(
            population_size=population_size,
            max_iterations=max_iterations,
            fitness_evaluator=fitness_evaluator,
            seed=seed,
        )

        # Abandonment limit: trials without improvement before scout triggers
        self.abandonment_limit = (
            int(abandonment_limit)
            if abandonment_limit is not None
            else max(5, int(0.5 * population_size))
        )

        # Colony State
        self.food_sources: Optional[np.ndarray] = None
        self.fitness_values: Optional[np.ndarray] = None
        self.trial_counters: Optional[np.ndarray] = None

        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf")
        self.abandoned_count_total: int = 0

    def initialize_colony(
        self,
        num_tasks: int,
        num_vms: int,
        initial_population: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize food sources, fitness, and trial counters.

        Args:
            num_tasks: Number of tasks (dimensionality).
            num_vms: Virtual machine count.
            initial_population: Optional warm-start food source matrix.
        """
        pop_size = self.population_size

        if initial_population is not None:
            if initial_population.shape != (pop_size, num_tasks):
                raise ValueError(
                    f"initial_population shape {initial_population.shape} must match "
                    f"({pop_size}, {num_tasks})"
                )
            self.food_sources = np.clip(initial_population.astype(float), 0.0, num_vms - 1e-4)
        else:
            self.food_sources = self._rng.uniform(
                0.0, float(num_vms) - 1e-4, size=(pop_size, num_tasks)
            )

        self.fitness_values = np.full(pop_size, float("inf"))
        self.trial_counters = np.zeros(pop_size, dtype=int)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.abandoned_count_total = 0

    def _calculate_nectar(self, fitness: float) -> float:
        """
        Convert minimization fitness to maximization nectar quality.

        Formula:
            nectar = 1.0 / (1.0 + fitness) if fitness >= 0 else 1.0 + abs(fitness)
        """
        if fitness >= 0.0:
            return 1.0 / (1.0 + fitness)
        return 1.0 + abs(fitness)

    def step(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        iteration_idx: int,
        total_iterations: int
    ) -> Tuple[float, float, int]:
        """
        Execute a single three-phase foraging cycle of ABC:
        1. Employed Bee Phase
        2. Onlooker Bee Phase
        3. Scout Bee Phase

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            iteration_idx: Current iteration index.
            total_iterations: Maximum planned generations.

        Returns:
            Tuple of (current_best_fitness, population_diversity, abandoned_count_in_step).
        """
        pop_size = self.population_size
        num_tasks = len(tasks)
        num_vms = simulator.num_vms

        # Initial evaluation if needed
        if np.isinf(self.fitness_values[0]):
            for i in range(pop_size):
                discrete = self.decode_continuous_to_discrete(self.food_sources[i], num_vms)
                fit = self.evaluator.evaluate(discrete, tasks, simulator)
                self.fitness_values[i] = fit
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = self.food_sources[i].copy()

        # ==========================================
        # 1. EMPLOYED BEE PHASE
        # ==========================================
        for i in range(pop_size):
            # Select partner k != i
            k = int(self._rng.integers(0, pop_size))
            while k == i:
                k = int(self._rng.integers(0, pop_size))

            # Select random task dimension d
            d = int(self._rng.integers(0, num_tasks))
            phi = float(self._rng.uniform(-1.0, 1.0))

            # Candidate mutated food source
            v = self.food_sources[i].copy()
            v[d] = np.clip(v[d] + phi * (v[d] - self.food_sources[k, d]), 0.0, float(num_vms) - 1e-4)

            # Greedy selection between current and candidate
            v_discrete = self.decode_continuous_to_discrete(v, num_vms)
            v_fit = self.evaluator.evaluate(v_discrete, tasks, simulator)

            if v_fit <= self.fitness_values[i]:
                self.food_sources[i] = v
                self.fitness_values[i] = v_fit
                self.trial_counters[i] = 0
                if v_fit < self.best_fitness:
                    self.best_fitness = v_fit
                    self.best_solution = v.copy()
            else:
                self.trial_counters[i] += 1

        # ==========================================
        # 2. ONLOOKER BEE PHASE (Roulette Selection)
        # ==========================================
        nectar_values = np.array([self._calculate_nectar(f) for f in self.fitness_values])
        total_nectar = np.sum(nectar_values)
        if total_nectar > 0:
            probs = nectar_values / total_nectar
        else:
            probs = np.full(pop_size, 1.0 / pop_size)

        onlookers_sent = 0
        while onlookers_sent < pop_size:
            # Roulette selection of source i
            i = int(self._rng.choice(pop_size, p=probs))

            # Neighbor partner k != i
            k = int(self._rng.integers(0, pop_size))
            while k == i:
                k = int(self._rng.integers(0, pop_size))

            d = int(self._rng.integers(0, num_tasks))
            phi = float(self._rng.uniform(-1.0, 1.0))

            v = self.food_sources[i].copy()
            v[d] = np.clip(v[d] + phi * (v[d] - self.food_sources[k, d]), 0.0, float(num_vms) - 1e-4)

            v_discrete = self.decode_continuous_to_discrete(v, num_vms)
            v_fit = self.evaluator.evaluate(v_discrete, tasks, simulator)

            if v_fit <= self.fitness_values[i]:
                self.food_sources[i] = v
                self.fitness_values[i] = v_fit
                self.trial_counters[i] = 0
                if v_fit < self.best_fitness:
                    self.best_fitness = v_fit
                    self.best_solution = v.copy()
            else:
                self.trial_counters[i] += 1

            onlookers_sent += 1

        # ==========================================
        # 3. SCOUT BEE PHASE (Abandonment Handling)
        # ==========================================
        abandoned_in_step = 0
        for i in range(pop_size):
            if self.trial_counters[i] >= self.abandonment_limit:
                # Replace stagnated source with fresh random exploratory position
                self.food_sources[i] = self._rng.uniform(
                    0.0, float(num_vms) - 1e-4, size=num_tasks
                )
                discrete = self.decode_continuous_to_discrete(self.food_sources[i], num_vms)
                fit = self.evaluator.evaluate(discrete, tasks, simulator)
                self.fitness_values[i] = fit
                self.trial_counters[i] = 0
                abandoned_in_step += 1

                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = self.food_sources[i].copy()

        self.abandoned_count_total += abandoned_in_step

        # 4. Compute Population Diversity
        discrete_pop = np.array([
            self.decode_continuous_to_discrete(fs, num_vms)
            for fs in self.food_sources
        ])
        diversity = self.calculate_population_diversity(discrete_pop)

        return self.best_fitness, diversity, abandoned_in_step

    def optimize(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        initial_population: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Execute full ABC search across all foraging generations.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            initial_population: Optional warm-start food source matrix.
            max_iterations: Optional override for generation limit.

        Returns:
            OptimizationResult containing best schedule, fitness, and diagnostics.
        """
        start_time = time.time()
        num_tasks = len(tasks)
        num_vms = simulator.num_vms
        total_iters = max_iterations if max_iterations is not None else self.max_iterations

        if num_tasks == 0:
            return OptimizationResult(
                algorithm_name="ABC",
                best_assignment=[],
                best_fitness=0.0,
                convergence_history=[],
                diversity_history=[],
                iterations_run=0,
                runtime_sec=0.0,
            )

        # Initialize Colony
        self.initialize_colony(num_tasks, num_vms, initial_population)

        convergence_history: List[float] = []
        diversity_history: List[float] = []

        for it in range(total_iters):
            best_fit, div, _ = self.step(tasks, simulator, it, total_iters)
            convergence_history.append(best_fit)
            diversity_history.append(div)

        runtime = time.time() - start_time

        # Final decode of best food source
        best_discrete = self.decode_continuous_to_discrete(
            self.best_solution, num_vms
        ).tolist()

        # Detailed fitness breakdown
        breakdown = self.evaluator.evaluate_detailed(best_discrete, tasks, simulator)

        return OptimizationResult(
            algorithm_name="ABC",
            best_assignment=best_discrete,
            best_fitness=self.best_fitness,
            convergence_history=convergence_history,
            diversity_history=diversity_history,
            iterations_run=total_iters,
            runtime_sec=runtime,
            population=self.food_sources.copy(),
            fitness_breakdown=breakdown,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("  ARTIFICIAL BEE COLONY (ABC) STANDALONE DEMO")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator

    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
    sim = VMSimulator()

    abc = ArtificialBeeColony(
        population_size=20,
        max_iterations=50,
        abandonment_limit=8,
        seed=42
    )

    print(f"\nRunning ABC on {len(tasks)} tasks across {sim.num_vms} VMs...")
    result = abc.optimize(tasks, sim)

    print(f"\nOptimization Finished in {result.runtime_sec:.3f} seconds.")
    print(f"  Best Combined Fitness:     {result.best_fitness:.2f}")
    if result.fitness_breakdown:
        print(f"  Best Makespan:            {result.fitness_breakdown.makespan:.2f} s")
        print(f"  Workload Imbalance:       {result.fitness_breakdown.load_imbalance:.2f} s")
    print(f"  Initial Best Fitness:     {result.convergence_history[0]:.2f}")
    print(f"  Final Best Fitness:       {result.convergence_history[-1]:.2f}")
    print(f"  Fitness Improvement:      {result.convergence_history[0] - result.convergence_history[-1]:.2f}")
    print(f"  Initial Diversity:        {result.diversity_history[0]:.4f}")
    print(f"  Final Diversity:          {result.diversity_history[-1]:.4f}")
    print(f"  Total Scout Abandonments: {abc.abandoned_count_total}")
    print(f"  Best Assignment (first 10): {result.best_assignment[:10]}")
    print("=" * 70)

