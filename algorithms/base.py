"""
Base module for metaheuristic swarm intelligence optimization algorithms.

Defines the shared OptimizationResult data structure, population diversity metrics,
and the common BaseOptimizer interface for PSO, GWO, and ABC.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator
from evaluation.fitness import FitnessEvaluator, FitnessBreakdown


@dataclass
class OptimizationResult:
    """
    Standardized result object returned by all optimization algorithms.

    Attributes:
        algorithm_name: Identifier of the algorithm ('PSO', 'GWO', 'ABC', 'Adaptive').
        best_assignment: List of integer VM indices corresponding to tasks.
        best_fitness: Minimum fitness score achieved.
        convergence_history: Best fitness recorded at each iteration.
        diversity_history: Population diversity measurement at each iteration.
        iterations_run: Total iterations executed.
        runtime_sec: Wall-clock execution time in seconds.
        population: Final continuous or discrete population matrix (pop_size x num_tasks).
        fitness_breakdown: Optional detailed fitness component accounting.
    """
    algorithm_name: str
    best_assignment: List[int]
    best_fitness: float
    convergence_history: List[float]
    diversity_history: List[float]
    iterations_run: int
    runtime_sec: float
    population: Optional[np.ndarray] = None
    fitness_breakdown: Optional[FitnessBreakdown] = None

    def to_dict(self) -> dict:
        """Serialize result metrics to dictionary."""
        return {
            "algorithm": self.algorithm_name,
            "best_fitness": self.best_fitness,
            "iterations_run": self.iterations_run,
            "runtime_sec": round(self.runtime_sec, 4),
            "final_diversity": self.diversity_history[-1] if self.diversity_history else 0.0,
            "best_makespan": self.fitness_breakdown.makespan if self.fitness_breakdown else None,
            "best_imbalance": self.fitness_breakdown.load_imbalance if self.fitness_breakdown else None,
        }


class BaseOptimizer(ABC):
    """
    Abstract base class providing common mechanics across PSO, GWO, and ABC:
    - Continuous-to-discrete mapping decoder.
    - Normalized pairwise Hamming population diversity calculation.
    - Unified execution interface.
    """

    def __init__(
        self,
        population_size: int = 20,
        max_iterations: int = 50,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize base optimizer settings.

        Args:
            population_size: Swarm / population count (particles, wolves, food sources).
            max_iterations: Maximum optimization generations.
            fitness_evaluator: FitnessEvaluator instance. If None, default instance is created.
            seed: Deterministic random seed.
        """
        if population_size < 4:
            raise ValueError(f"Population size must be at least 4, got: {population_size}")
        if max_iterations < 1:
            raise ValueError(f"Max iterations must be at least 1, got: {max_iterations}")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else FitnessEvaluator()
        self.seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Reset internal pseudo-random number generator."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def decode_continuous_to_discrete(
        continuous_vector: np.ndarray,
        num_vms: int
    ) -> np.ndarray:
        """
        Map a continuous coordinate vector x in R^N into valid discrete VM indices in {0, ..., num_vms - 1}.

        Decoder Formula:
            vm_idx = clip(floor(x_d), 0, num_vms - 1)

        This standard floor-clamping mapping ensures:
        - Every task receives a valid VM index.
        - No task is lost, duplicated, or unassigned.
        - Smooth gradient-like exploration in continuous space maps uniformly to discrete VMs.

        Args:
            continuous_vector: 1D array of floats representing task assignments.
            num_vms: Number of available virtual machines.

        Returns:
            1D array of integer VM indices in [0, num_vms - 1].
        """
        discrete = np.floor(continuous_vector).astype(int)
        return np.clip(discrete, 0, num_vms - 1)

    @staticmethod
    def calculate_population_diversity(population_discrete: np.ndarray) -> float:
        """
        Calculate normalized pairwise Hamming diversity across the discrete population.

        Formula:
            D = (2 / (S * (S - 1))) * sum_{a=1}^{S-1} sum_{b=a+1}^S [ (1 / N) * sum_{d=0}^{N-1} I(x_{a, d} != x_{b, d}) ]

        Properties:
            - D in [0.0, 1.0].
            - D = 0.0: Complete swarm collapse / stagnation (all individuals identical).
            - D > 0.5: High exploratory dispersion.

        Args:
            population_discrete: 2D array of shape (pop_size, num_tasks) containing integer VM indices.

        Returns:
            Scalar diversity value in [0.0, 1.0].
        """
        pop_size, num_tasks = population_discrete.shape
        if pop_size <= 1 or num_tasks == 0:
            return 0.0

        total_distance = 0.0
        pair_count = 0

        for a in range(pop_size - 1):
            diffs = population_discrete[a] != population_discrete[a + 1:]
            total_distance += np.sum(diffs) / num_tasks
            pair_count += (pop_size - 1 - a)

        return round(float(total_distance / pair_count), 4) if pair_count > 0 else 0.0

    @abstractmethod
    def optimize(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        initial_population: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Execute full optimization run on given tasks and VM cluster.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            initial_population: Optional warm-start population matrix (pop_size x num_tasks).
            max_iterations: Optional override for generation limit.

        Returns:
            OptimizationResult containing best solution and convergence diagnostics.
        """
        pass

