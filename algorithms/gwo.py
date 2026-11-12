"""
Grey Wolf Optimizer (GWO) for Cloud Resource Scheduling.

Simulates leadership hierarchy (Alpha, Beta, Delta, Omega) and cooperative hunting
mechanisms adapted to task-to-VM scheduling via a continuous-to-discrete decoder.
"""

from typing import List, Optional, Tuple, Union
import time
import numpy as np

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator
from evaluation.fitness import FitnessEvaluator, FitnessBreakdown
from algorithms.base import BaseOptimizer, OptimizationResult
from config import SWARM_POPULATION_SIZE, MAX_OPTIMIZATION_ITERATIONS


class GreyWolfOptimizer(BaseOptimizer):
    """
    Continuous-to-Discrete Grey Wolf Optimizer engine for cloud scheduling.

    Social Hierarchy:
        - Alpha (X_alpha): Fittest schedule found so far.
        - Beta (X_beta): Second fittest schedule.
        - Delta (X_delta): Third fittest schedule.
        - Omega (X_omega): Remaining wolves following the three leaders.

    Mathematical Model:
        Encircling & Hunting:
            D_alpha = |C_1 * X_alpha - X_i|;  X_1 = X_alpha - A_1 * D_alpha
            D_beta  = |C_2 * X_beta  - X_i|;  X_2 = X_beta  - A_2 * D_beta
            D_delta = |C_3 * X_delta - X_i|;  X_3 = X_delta - A_3 * D_delta

            X_i^{(t+1)} = (X_1 + X_2 + X_3) / 3

        Control Parameters:
            A = 2 * a * r_1 - a
            C = 2 * r_2
            a(t) = 2 - 2 * (t / T)  (Decreases linearly from 2.0 to 0.0)
            r_1, r_2 ~ U(0, 1)

    Exploration vs Exploitation:
        - |A| > 1: Wolves diverge to explore new regions (encircling).
        - |A| < 1: Wolves converge to attack the prey (exploitation).
    """

    def __init__(
        self,
        population_size: int = SWARM_POPULATION_SIZE,
        max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize the Grey Wolf Optimizer.

        Args:
            population_size: Pack size (number of wolves, default: 20).
            max_iterations: Maximum optimization generations (default: 50).
            fitness_evaluator: FitnessEvaluator instance.
            seed: Deterministic random seed.
        """
        super().__init__(
            population_size=population_size,
            max_iterations=max_iterations,
            fitness_evaluator=fitness_evaluator,
            seed=seed,
        )

        # Wolf pack state
        self.positions: Optional[np.ndarray] = None

        # Leader wolves
        self.alpha_pos: Optional[np.ndarray] = None
        self.alpha_fitness: float = float("inf")

        self.beta_pos: Optional[np.ndarray] = None
        self.beta_fitness: float = float("inf")

        self.delta_pos: Optional[np.ndarray] = None
        self.delta_fitness: float = float("inf")

    def initialize_pack(
        self,
        num_tasks: int,
        num_vms: int,
        initial_population: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize wolf pack positions and leaders.

        Args:
            num_tasks: Number of tasks (dimensions).
            num_vms: Number of virtual machines.
            initial_population: Optional warm-start positions (pop_size x num_tasks).
        """
        pop_size = self.population_size

        if initial_population is not None:
            if initial_population.shape != (pop_size, num_tasks):
                raise ValueError(
                    f"initial_population shape {initial_population.shape} must match "
                    f"({pop_size}, {num_tasks})"
                )
            self.positions = np.clip(initial_population.astype(float), 0.0, num_vms - 1e-4)
        else:
            # Uniform random distribution across continuous VM coordinate space [0, m)
            self.positions = self._rng.uniform(0.0, float(num_vms) - 1e-4, size=(pop_size, num_tasks))

        self.alpha_pos = None
        self.alpha_fitness = float("inf")
        self.beta_pos = None
        self.beta_fitness = float("inf")
        self.delta_pos = None
        self.delta_fitness = float("inf")

    def step(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        iteration_idx: int,
        total_iterations: int
    ) -> Tuple[float, float]:
        """
        Execute a single generation step of GWO.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            iteration_idx: Current 0-based generation index.
            total_iterations: Maximum planned generations.

        Returns:
            Tuple of (current_best_fitness, population_diversity).
        """
        pop_size = self.population_size
        num_tasks = len(tasks)
        num_vms = simulator.num_vms

        # 1. Evaluate Fitness of Current Positions and Update Hierarchy
        fitnesses = np.zeros(pop_size)
        discrete_pop = np.zeros_like(self.positions, dtype=int)

        for i in range(pop_size):
            discrete_assignment = self.decode_continuous_to_discrete(self.positions[i], num_vms)
            discrete_pop[i] = discrete_assignment
            fit = self.evaluator.evaluate(discrete_assignment, tasks, simulator)
            fitnesses[i] = fit

            # Update Alpha, Beta, Delta
            if fit < self.alpha_fitness:
                # Demote Alpha -> Beta, Beta -> Delta
                self.delta_fitness = self.beta_fitness
                self.delta_pos = None if self.beta_pos is None else self.beta_pos.copy()

                self.beta_fitness = self.alpha_fitness
                self.beta_pos = None if self.alpha_pos is None else self.alpha_pos.copy()

                self.alpha_fitness = fit
                self.alpha_pos = self.positions[i].copy()

            elif fit < self.beta_fitness:
                # Demote Beta -> Delta
                self.delta_fitness = self.beta_fitness
                self.delta_pos = None if self.beta_pos is None else self.beta_pos.copy()

                self.beta_fitness = fit
                self.beta_pos = self.positions[i].copy()

            elif fit < self.delta_fitness:
                self.delta_fitness = fit
                self.delta_pos = self.positions[i].copy()

        # 2. Population Diversity
        diversity = self.calculate_population_diversity(discrete_pop)

        # 3. Linearly Decreasing Parameter a from 2.0 to 0.0
        a = 2.0 - (2.0 * iteration_idx / max(1, total_iterations - 1))

        # 4. Update Wolf Positions based on Alpha, Beta, and Delta Leaders
        new_positions = np.zeros_like(self.positions)

        for i in range(pop_size):
            # Wolf 1 (guided by Alpha)
            r1_1 = self._rng.uniform(0.0, 1.0, size=num_tasks)
            r2_1 = self._rng.uniform(0.0, 1.0, size=num_tasks)
            A1 = 2.0 * a * r1_1 - a
            C1 = 2.0 * r2_1
            D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
            X1 = self.alpha_pos - A1 * D_alpha

            # Wolf 2 (guided by Beta)
            r1_2 = self._rng.uniform(0.0, 1.0, size=num_tasks)
            r2_2 = self._rng.uniform(0.0, 1.0, size=num_tasks)
            A2 = 2.0 * a * r1_2 - a
            C2 = 2.0 * r2_2
            target_beta = self.beta_pos if self.beta_pos is not None else self.alpha_pos
            D_beta = np.abs(C2 * target_beta - self.positions[i])
            X2 = target_beta - A2 * D_beta

            # Wolf 3 (guided by Delta)
            r1_3 = self._rng.uniform(0.0, 1.0, size=num_tasks)
            r2_3 = self._rng.uniform(0.0, 1.0, size=num_tasks)
            A3 = 2.0 * a * r1_3 - a
            C3 = 2.0 * r2_3
            target_delta = self.delta_pos if self.delta_pos is not None else self.alpha_pos
            D_delta = np.abs(C3 * target_delta - self.positions[i])
            X3 = target_delta - A3 * D_delta

            # Position average
            new_pos = (X1 + X2 + X3) / 3.0
            new_positions[i] = np.clip(new_pos, 0.0, float(num_vms) - 1e-4)

        self.positions = new_positions
        return self.alpha_fitness, diversity

    def optimize(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        initial_population: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Execute full GWO search across all generations.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            initial_population: Optional warm-start pack matrix.
            max_iterations: Optional override for generation limit.

        Returns:
            OptimizationResult containing best solution and convergence diagnostics.
        """
        start_time = time.time()
        num_tasks = len(tasks)
        num_vms = simulator.num_vms
        total_iters = max_iterations if max_iterations is not None else self.max_iterations

        if num_tasks == 0:
            return OptimizationResult(
                algorithm_name="GWO",
                best_assignment=[],
                best_fitness=0.0,
                convergence_history=[],
                diversity_history=[],
                iterations_run=0,
                runtime_sec=0.0,
            )

        # Initialize Pack
        self.initialize_pack(num_tasks, num_vms, initial_population)

        convergence_history: List[float] = []
        diversity_history: List[float] = []

        for it in range(total_iters):
            best_fit, div = self.step(tasks, simulator, it, total_iters)
            convergence_history.append(best_fit)
            diversity_history.append(div)

        runtime = time.time() - start_time

        # Final decode of Alpha best
        best_discrete = self.decode_continuous_to_discrete(
            self.alpha_pos, num_vms
        ).tolist()

        # Full metric breakdown
        breakdown = self.evaluator.evaluate_detailed(best_discrete, tasks, simulator)

        return OptimizationResult(
            algorithm_name="GWO",
            best_assignment=best_discrete,
            best_fitness=self.alpha_fitness,
            convergence_history=convergence_history,
            diversity_history=diversity_history,
            iterations_run=total_iters,
            runtime_sec=runtime,
            population=self.positions.copy(),
            fitness_breakdown=breakdown,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("  GREY WOLF OPTIMIZER (GWO) STANDALONE DEMO")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator

    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
    sim = VMSimulator()

    gwo = GreyWolfOptimizer(
        population_size=20,
        max_iterations=50,
        seed=42
    )

    print(f"\nRunning GWO on {len(tasks)} tasks across {sim.num_vms} VMs...")
    result = gwo.optimize(tasks, sim)

    print(f"\nOptimization Finished in {result.runtime_sec:.3f} seconds.")
    print(f"  Best Combined Fitness:  {result.best_fitness:.2f}")
    if result.fitness_breakdown:
        print(f"  Best Makespan:         {result.fitness_breakdown.makespan:.2f} s")
        print(f"  Workload Imbalance:    {result.fitness_breakdown.load_imbalance:.2f} s")
    print(f"  Initial Best Fitness:  {result.convergence_history[0]:.2f}")
    print(f"  Final Best Fitness:    {result.convergence_history[-1]:.2f}")
    print(f"  Fitness Improvement:   {result.convergence_history[0] - result.convergence_history[-1]:.2f}")
    print(f"  Initial Diversity:     {result.diversity_history[0]:.4f}")
    print(f"  Final Diversity:       {result.diversity_history[-1]:.4f}")
    print(f"  Best Assignment (first 10): {result.best_assignment[:10]}")
    print("=" * 70)

