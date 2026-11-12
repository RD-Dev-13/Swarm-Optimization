"""
Particle Swarm Optimization (PSO) for Cloud Resource Scheduling.

Maps continuous particle trajectories in R^N into discrete task-to-VM assignments
using a floor-clamping decoder. Implements dynamic inertia damping, cognitive and
social acceleration, velocity clamping, personal and global best tracking, and
population diversity profiling.
"""

from typing import List, Optional, Tuple, Union
import time
import numpy as np

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator
from evaluation.fitness import FitnessEvaluator, FitnessBreakdown
from algorithms.base import BaseOptimizer, OptimizationResult
from config import SWARM_POPULATION_SIZE, MAX_OPTIMIZATION_ITERATIONS


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Continuous-to-Discrete Particle Swarm Optimization engine for cloud scheduling.

    Representation:
        - Dimension D = Number of tasks (N).
        - Particle position x_i in [0.0, m): each dimension encodes continuous VM allocation.
        - Discrete Mapping: vm_d = clip(floor(x_{i, d}), 0, m - 1).
        - Velocity v_i in [-V_max, V_max]: direction and magnitude of assignment migration.

    Equations:
        v_{i, d}^{(t+1)} = w(t) * v_{i, d}^{(t)} + c_1 * r_1 * (p_{i, d} - x_{i, d}^{(t)})
                                                 + c_2 * r_2 * (g_d - x_{i, d}^{(t)})
        x_{i, d}^{(t+1)} = clip(x_{i, d}^{(t)} + v_{i, d}^{(t+1)}, 0.0, m - 1e-4)

        where:
            w(t) = w_max - (t / T) * (w_max - w_min)   (Linear Inertia Damping)
            c_1: Cognitive acceleration (nostalgia / personal memory weight)
            c_2: Social acceleration (swarm cooperation weight)
            r_1, r_2 ~ U(0, 1): Stochastic exploration coefficients
    """

    def __init__(
        self,
        population_size: int = SWARM_POPULATION_SIZE,
        max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max_factor: float = 0.5,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize the PSO scheduler.

        Args:
            population_size: Number of particles in the swarm (default: 20).
            max_iterations: Maximum search iterations (default: 50).
            w_max: Initial inertia weight favoring global exploration (default: 0.9).
            w_min: Final inertia weight favoring localized exploitation (default: 0.4).
            c1: Cognitive acceleration coefficient (default: 2.0).
            c2: Social acceleration coefficient (default: 2.0).
            v_max_factor: Maximum velocity as fraction of VM range (default: 0.5).
            fitness_evaluator: FitnessEvaluator instance.
            seed: Deterministic random seed.
        """
        super().__init__(
            population_size=population_size,
            max_iterations=max_iterations,
            fitness_evaluator=fitness_evaluator,
            seed=seed,
        )
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.v_max_factor = float(v_max_factor)

        # Swarm internal state
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.personal_bests: Optional[np.ndarray] = None
        self.personal_best_fitness: Optional[np.ndarray] = None
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float("inf")
        self.current_iteration: int = 0

    def initialize_swarm(
        self,
        num_tasks: int,
        num_vms: int,
        initial_population: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize positions, velocities, and personal bests.

        Args:
            num_tasks: Problem dimensionality (number of tasks).
            num_vms: Available virtual machine count.
            initial_population: Optional warm-start positions (pop_size x num_tasks).
        """
        pop_size = self.population_size
        v_max = self.v_max_factor * (num_vms - 1)

        if initial_population is not None:
            if initial_population.shape != (pop_size, num_tasks):
                raise ValueError(
                    f"initial_population shape {initial_population.shape} must match "
                    f"({pop_size}, {num_tasks})"
                )
            self.positions = np.clip(initial_population.astype(float), 0.0, num_vms - 1e-4)
        else:
            # Uniform random spread across continuous VM coordinate space [0, m)
            self.positions = self._rng.uniform(0.0, float(num_vms) - 1e-4, size=(pop_size, num_tasks))

        # Initial velocities sampled uniformly in [-v_max, v_max]
        self.velocities = self._rng.uniform(-v_max, v_max, size=(pop_size, num_tasks))

        # Personal bests start at initial positions
        self.personal_bests = self.positions.copy()
        self.personal_best_fitness = np.full(pop_size, float("inf"))

        self.global_best_position = None
        self.global_best_fitness = float("inf")
        self.current_iteration = 0

    def step(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        iteration_idx: int,
        total_iterations: int
    ) -> Tuple[float, float]:
        """
        Execute a single generation step of PSO.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            iteration_idx: Current 0-based generation index.
            total_iterations: Maximum planned generations.

        Returns:
            Tuple of (current_best_fitness, population_diversity).
        """
        pop_size = self.population_size
        num_vms = simulator.num_vms
        v_max = self.v_max_factor * (num_vms - 1)

        # 1. Dynamic Inertia Weight Damping
        w = self.w_max - (iteration_idx / max(1, total_iterations - 1)) * (self.w_max - self.w_min)

        # 2. Evaluate Fitness of Current Positions
        discrete_pop = np.zeros_like(self.positions, dtype=int)
        for i in range(pop_size):
            discrete_assignment = self.decode_continuous_to_discrete(self.positions[i], num_vms)
            discrete_pop[i] = discrete_assignment

            current_fit = self.evaluator.evaluate(discrete_assignment, tasks, simulator)

            # Update Personal Best
            if current_fit < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = current_fit
                self.personal_bests[i] = self.positions[i].copy()

            # Update Global Best
            if current_fit < self.global_best_fitness:
                self.global_best_fitness = current_fit
                self.global_best_position = self.positions[i].copy()

        # 3. Compute Population Diversity
        diversity = self.calculate_population_diversity(discrete_pop)

        # 4. Update Velocities and Positions
        r1 = self._rng.uniform(0.0, 1.0, size=self.positions.shape)
        r2 = self._rng.uniform(0.0, 1.0, size=self.positions.shape)

        cognitive = self.c1 * r1 * (self.personal_bests - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)

        self.velocities = (w * self.velocities) + cognitive + social
        self.velocities = np.clip(self.velocities, -v_max, v_max)

        self.positions = self.positions + self.velocities
        self.positions = np.clip(self.positions, 0.0, float(num_vms) - 1e-4)

        return self.global_best_fitness, diversity

    def optimize(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        initial_population: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Execute full PSO search across all generations.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            initial_population: Optional warm-start population matrix.
            max_iterations: Optional override for generation count.

        Returns:
            OptimizationResult containing best schedule, fitness, and diagnostics.
        """
        start_time = time.time()
        num_tasks = len(tasks)
        num_vms = simulator.num_vms
        total_iters = max_iterations if max_iterations is not None else self.max_iterations

        if num_tasks == 0:
            return OptimizationResult(
                algorithm_name="PSO",
                best_assignment=[],
                best_fitness=0.0,
                convergence_history=[],
                diversity_history=[],
                iterations_run=0,
                runtime_sec=0.0,
            )

        # Initialize Swarm
        self.initialize_swarm(num_tasks, num_vms, initial_population)

        convergence_history: List[float] = []
        diversity_history: List[float] = []

        for it in range(total_iters):
            best_fit, div = self.step(tasks, simulator, it, total_iters)
            convergence_history.append(best_fit)
            diversity_history.append(div)

        runtime = time.time() - start_time

        # Final decode of global best
        best_discrete = self.decode_continuous_to_discrete(
            self.global_best_position, num_vms
        ).tolist()

        # Compute full breakdown
        breakdown = self.evaluator.evaluate_detailed(best_discrete, tasks, simulator)

        return OptimizationResult(
            algorithm_name="PSO",
            best_assignment=best_discrete,
            best_fitness=self.global_best_fitness,
            convergence_history=convergence_history,
            diversity_history=diversity_history,
            iterations_run=total_iters,
            runtime_sec=runtime,
            population=self.positions.copy(),
            fitness_breakdown=breakdown,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("  PARTICLE SWARM OPTIMIZATION (PSO) STANDALONE DEMO")
    print("=" * 70)

    from simulation.task_generator import TaskGenerator

    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
    sim = VMSimulator()

    pso = ParticleSwarmOptimizer(
        population_size=20,
        max_iterations=50,
        seed=42
    )

    print(f"\nRunning PSO on {len(tasks)} tasks across {sim.num_vms} VMs...")
    result = pso.optimize(tasks, sim)

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

