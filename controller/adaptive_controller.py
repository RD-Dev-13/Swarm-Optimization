"""
Behavior-Aware Adaptive Swarm Controller for Dynamic Cloud Resource Scheduling.

Core Research Contribution:
Monitors runtime optimization behavior and environmental dynamics across four key indicators:
1. Population Diversity (D_t in [0.0, 1.0])
2. Relative Fitness Improvement (Delta f_t)
3. Optimization Stagnation Counter (S_t)
4. Workload & Infrastructure Variation (Omega_t)

Executes explainable rule-based strategy switching among PSO, GWO, and ABC, seamlessly
transferring population state between algorithms to maximize search efficiency and escape local optima.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union
import time
import numpy as np
import pandas as pd

from simulation.task_generator import Task
from simulation.vm_simulator import VMSimulator
from evaluation.fitness import FitnessEvaluator, FitnessBreakdown
from algorithms.base import BaseOptimizer, OptimizationResult
from algorithms.pso import ParticleSwarmOptimizer
from algorithms.gwo import GreyWolfOptimizer
from algorithms.abc import ArtificialBeeColony
from config import (
    SWARM_POPULATION_SIZE,
    MAX_OPTIMIZATION_ITERATIONS,
    STAGNATION_THRESHOLD,
)


@dataclass
class ControllerDecision:
    """
    Explainable runtime decision record logged at every adaptive generation.

    Attributes:
        iteration: 1-based generation index.
        selected_algorithm: Strategy chosen ('PSO', 'GWO', 'ABC').
        reason: Clear, explainable rationale for the selection decision.
        diversity: Measured pairwise Hamming population diversity in [0.0, 1.0].
        fitness_improvement: Relative improvement percentage over the observation window.
        stagnation_count: Consecutive iterations without meaningful fitness improvement.
        workload_changed: Flag indicating whether an external disturbance was detected.
        best_fitness: Minimum fitness score discovered so far.
    """
    iteration: int
    selected_algorithm: str
    reason: str
    diversity: float
    fitness_improvement: float
    stagnation_count: int
    workload_changed: bool
    best_fitness: float

    def to_dict(self) -> dict:
        """Serialize decision record to standard dictionary."""
        return asdict(self)


class AdaptiveSwarmController:
    """
    Behavior-Aware Adaptive Controller for Cloud Resource Scheduling.

    Distinction:
        - Static Selection: Picks one algorithm at start based on initial input size.
        - Fixed Hybrid: Interleaves algorithms in fixed sequence (e.g., 10 PSO then 10 GWO).
        - Behavior-Aware Adaptive (Proposed): Continuously measures population diversity,
          fitness trajectory, stagnation, and environment state to dynamically deploy the
          most suitable metaheuristic operator.

    Selection Rules:
        - Rule 1 (Environmental Disturbance): If VM degrades or burst arrives -> Deploy PSO.
        - Rule 2 (Stagnation & Low Diversity): If Stagnation >= Thresh & Diversity <= Low -> Deploy ABC.
        - Rule 3 (Stagnation with Diversity): If Stagnation >= Thresh & Diversity > Low -> Alternate Strategy.
        - Rule 4 (Active Progress & Healthy Diversity): If Improvement >= Thresh & Diversity > Low -> Deploy GWO.
        - Rule 5 (Progress with Narrowing Diversity): If Improvement >= Thresh & Diversity <= Low -> Retain Current.
    """

    def __init__(
        self,
        population_size: int = SWARM_POPULATION_SIZE,
        max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
        stagnation_threshold: int = STAGNATION_THRESHOLD,
        diversity_low_threshold: float = 0.15,
        improvement_threshold: float = 0.005,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize the Behavior-Aware Adaptive Controller.

        Args:
            population_size: Swarm size (default: 20).
            max_iterations: Maximum optimization generations (default: 50).
            stagnation_threshold: Generations without progress before stagnation triggers (default: 5).
            diversity_low_threshold: Threshold below which diversity is deemed critically low (default: 0.15).
            improvement_threshold: Minimum relative improvement ratio considered progress (default: 0.005 = 0.5%).
            fitness_evaluator: FitnessEvaluator instance.
            seed: Deterministic random seed.
        """
        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.stagnation_threshold = int(stagnation_threshold)
        self.diversity_low_threshold = float(diversity_low_threshold)
        self.improvement_threshold = float(improvement_threshold)
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else FitnessEvaluator()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Optimization Engines
        self.pso = ParticleSwarmOptimizer(
            population_size=self.population_size,
            fitness_evaluator=self.evaluator,
            seed=seed,
        )
        self.gwo = GreyWolfOptimizer(
            population_size=self.population_size,
            fitness_evaluator=self.evaluator,
            seed=seed,
        )
        self.abc = ArtificialBeeColony(
            population_size=self.population_size,
            abandonment_limit=max(4, self.stagnation_threshold),
            fitness_evaluator=self.evaluator,
            seed=seed,
        )

        # Dynamic Controller State
        self.current_population: Optional[np.ndarray] = None
        self.current_best_position: Optional[np.ndarray] = None
        self.current_best_fitness: float = float("inf")
        self.stagnation_counter: int = 0
        self.active_strategy: str = "PSO"

        # History diagnostics
        self.decision_history: List[ControllerDecision] = []
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.algorithm_history: List[str] = []

        # Infrastructure Tracking Checksum
        self._last_infrastructure_signature: Optional[str] = None

    def _get_infrastructure_signature(self, tasks: List[Task], simulator: VMSimulator) -> str:
        """Generate fingerprint of current task demands and VM capacities to detect changes."""
        task_sum = sum(t.cpu_demand_mi for t in tasks)
        task_count = len(tasks)
        vm_mips = sum(vm.mips for vm in simulator.vm_list if vm.is_available)
        return f"T{task_count}:{task_sum:.1f}_VM:{vm_mips:.1f}"

    def detect_workload_or_infrastructure_change(
        self,
        tasks: List[Task],
        simulator: VMSimulator
    ) -> bool:
        """
        Detect runtime changes in task volume, CPU demands, or VM capacities.

        Args:
            tasks: Current task list.
            simulator: VMSimulator instance.

        Returns:
            True if infrastructure or workload changed since last check, False otherwise.
        """
        current_sig = self._get_infrastructure_signature(tasks, simulator)
        if self._last_infrastructure_signature is None:
            self._last_infrastructure_signature = current_sig
            return False

        changed = (current_sig != self._last_infrastructure_signature)
        self._last_infrastructure_signature = current_sig
        return changed

    def compute_relative_fitness_improvement(self, current_fitness: float) -> float:
        """
        Calculate relative improvement between the current best fitness and previous iteration.

        Formula:
            Delta f = (f_prev - f_curr) / max(|f_prev|, 1e-6)

        Args:
            current_fitness: Newly evaluated best fitness score.

        Returns:
            Relative improvement scalar (>= 0.0 if improved, 0.0 otherwise).
        """
        if not self.fitness_history:
            return 0.0

        prev_fitness = self.fitness_history[-1]
        if prev_fitness <= 0.0:
            return 0.0

        improvement = (prev_fitness - current_fitness) / prev_fitness
        return max(0.0, float(improvement))

    def evaluate_decision_rules(
        self,
        diversity: float,
        improvement: float,
        stagnation: int,
        workload_changed: bool,
    ) -> Tuple[str, str]:
        """
        Evaluate explainable decision logic to select the next optimization strategy.

        Args:
            diversity: Current population diversity in [0.0, 1.0].
            improvement: Relative fitness improvement ratio.
            stagnation: Consecutive stagnation counter.
            workload_changed: Whether an environmental event occurred.

        Returns:
            Tuple of (selected_algorithm, reason_explanation).
        """
        # RULE 1: Workload or Infrastructure Event Detected
        if workload_changed:
            return (
                "PSO",
                "Environmental disturbance detected (VM capacity shift or task arrival). "
                "Deploying PSO with momentum inertia to re-stabilize and explore new constraints."
            )

        # RULE 2: Critical Stagnation AND Diversity Collapse
        if stagnation >= self.stagnation_threshold and diversity <= self.diversity_low_threshold:
            return (
                "ABC",
                f"Critical stagnation ({stagnation} >= {self.stagnation_threshold} gen) "
                f"and low diversity (D={diversity:.3f} <= {self.diversity_low_threshold}). "
                "Switching to ABC to trigger scout bee abandonment and escape local minima."
            )

        # RULE 3: Stagnation with Adequate Diversity
        if stagnation >= self.stagnation_threshold and diversity > self.diversity_low_threshold:
            # Alternate away from currently stalled strategy
            next_strat = "ABC" if self.active_strategy == "GWO" else (
                "GWO" if self.active_strategy == "PSO" else "PSO"
            )
            return (
                next_strat,
                f"Stagnation threshold reached ({stagnation} gen) with adequate diversity (D={diversity:.3f}). "
                f"Alternating strategy to {next_strat} to perturb search trajectory."
            )

        # RULE 4: Steady Improvement with Healthy Diversity
        if improvement >= self.improvement_threshold and diversity > self.diversity_low_threshold:
            return (
                "GWO",
                f"Steady fitness improvement (Δf={improvement:.2%}) and healthy diversity (D={diversity:.3f}). "
                "Deploying GWO for rapid alpha-guided exploitation."
            )

        # RULE 5: Improving despite Narrowing Diversity
        if improvement >= self.improvement_threshold and diversity <= self.diversity_low_threshold:
            return (
                self.active_strategy,
                f"Active improvement (Δf={improvement:.2%}) despite low diversity (D={diversity:.3f}). "
                f"Retaining {self.active_strategy} to complete local descent."
            )

        # Default / Exploration Maintenance
        return (
            self.active_strategy,
            f"Stable optimization conditions (D={diversity:.3f}, Stagnation={stagnation}). "
            f"Continuing {self.active_strategy}."
        )

    def optimize(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        max_iterations: Optional[int] = None,
        initial_population: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Execute Behavior-Aware Adaptive Swarm Optimization.

        At each generation:
        1. Checks environment for dynamic task or VM disturbances.
        2. Evaluates the 4 runtime behavioral indicators.
        3. Executes explainable decision rules to select/switch strategy.
        4. Transfers living population state seamlessly to the active optimizer.
        5. Steps the active optimizer for one generation.
        6. Logs diagnostics and repeats until generation budget is exhausted.

        Args:
            tasks: List of Task objects.
            simulator: VMSimulator cluster instance.
            max_iterations: Generation budget (default: self.max_iterations).
            initial_population: Optional warm-start population matrix.

        Returns:
            OptimizationResult containing best schedule, full diagnostics, and decision trail.
        """
        start_time = time.time()
        num_tasks = len(tasks)
        num_vms = simulator.num_vms
        total_iters = max_iterations if max_iterations is not None else self.max_iterations

        if num_tasks == 0:
            return OptimizationResult(
                algorithm_name="Adaptive",
                best_assignment=[],
                best_fitness=0.0,
                convergence_history=[],
                diversity_history=[],
                iterations_run=0,
                runtime_sec=0.0,
            )

        # 1. Initialize Controller State
        if initial_population is not None:
            self.current_population = np.clip(
                initial_population.astype(float), 0.0, num_vms - 1e-4
            )
        else:
            self.current_population = self._rng.uniform(
                0.0, float(num_vms) - 1e-4, size=(self.population_size, num_tasks)
            )

        self.current_best_position = None
        self.current_best_fitness = float("inf")
        self.stagnation_counter = 0
        self.active_strategy = "PSO"

        self.decision_history.clear()
        self.fitness_history.clear()
        self.diversity_history.clear()
        self.algorithm_history.clear()
        self._last_infrastructure_signature = self._get_infrastructure_signature(tasks, simulator)

        # Initial baseline evaluation of population
        discrete_init = np.array([
            BaseOptimizer.decode_continuous_to_discrete(self.current_population[i], num_vms)
            for i in range(self.population_size)
        ])
        for i in range(self.population_size):
            fit = self.evaluator.evaluate(discrete_init[i], tasks, simulator)
            if fit < self.current_best_fitness:
                self.current_best_fitness = fit
                self.current_best_position = self.current_population[i].copy()

        current_diversity = BaseOptimizer.calculate_population_diversity(discrete_init)
        current_improvement = 0.0

        # Prime optimizer engines with the shared population
        self.pso.initialize_swarm(num_tasks, num_vms, self.current_population)
        self.gwo.initialize_pack(num_tasks, num_vms, self.current_population)
        self.abc.initialize_colony(num_tasks, num_vms, self.current_population)

        # 2. Main Adaptive Optimization Loop
        for it in range(total_iters):
            # Check Environmental Perturbations
            workload_changed = self.detect_workload_or_infrastructure_change(tasks, simulator)

            # Evaluate Decision Rules
            selected_algo, reason = self.evaluate_decision_rules(
                diversity=current_diversity,
                improvement=current_improvement,
                stagnation=self.stagnation_counter,
                workload_changed=workload_changed,
            )

            # Strategy Switching & State Transfer
            if selected_algo != self.active_strategy or it == 0 or workload_changed:
                self.active_strategy = selected_algo

                # Ensure global best is seeded into position 0 of current population
                if self.current_best_position is not None:
                    self.current_population[0] = self.current_best_position.copy()

                # Transfer population state to selected engine
                if selected_algo == "PSO":
                    self.pso.initialize_swarm(num_tasks, num_vms, self.current_population)
                elif selected_algo == "GWO":
                    self.gwo.initialize_pack(num_tasks, num_vms, self.current_population)
                elif selected_algo == "ABC":
                    self.abc.initialize_colony(num_tasks, num_vms, self.current_population)

            # Execute One Generation of the Selected Algorithm
            prev_best = self.current_best_fitness

            if self.active_strategy == "PSO":
                step_fit, step_div = self.pso.step(tasks, simulator, it, total_iters)
                self.current_population = self.pso.positions.copy()
            elif self.active_strategy == "GWO":
                step_fit, step_div = self.gwo.step(tasks, simulator, it, total_iters)
                self.current_population = self.gwo.positions.copy()
            else:  # ABC
                step_fit, step_div, _ = self.abc.step(tasks, simulator, it, total_iters)
                self.current_population = self.abc.food_sources.copy()

            # Update Global Best if step improved
            if step_fit < self.current_best_fitness:
                self.current_best_fitness = step_fit
                # Update best position from active engine
                if self.active_strategy == "PSO":
                    self.current_best_position = self.pso.global_best_position.copy()
                elif self.active_strategy == "GWO":
                    self.current_best_position = self.gwo.alpha_pos.copy()
                else:
                    self.current_best_position = self.abc.best_solution.copy()

            # Compute Relative Improvement
            if prev_best > 0 and prev_best < float("inf"):
                current_improvement = max(0.0, (prev_best - self.current_best_fitness) / prev_best)
            else:
                current_improvement = 0.0

            # Update Stagnation Counter
            if current_improvement >= self.improvement_threshold:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            current_diversity = step_div

            # Log Generation Diagnostics
            decision = ControllerDecision(
                iteration=it + 1,
                selected_algorithm=self.active_strategy,
                reason=reason,
                diversity=round(current_diversity, 4),
                fitness_improvement=round(current_improvement, 4),
                stagnation_count=self.stagnation_counter,
                workload_changed=workload_changed,
                best_fitness=round(self.current_best_fitness, 4),
            )
            self.decision_history.append(decision)
            self.fitness_history.append(self.current_best_fitness)
            self.diversity_history.append(current_diversity)
            self.algorithm_history.append(self.active_strategy)

        runtime = time.time() - start_time

        # Final decode of best schedule
        best_discrete = BaseOptimizer.decode_continuous_to_discrete(
            self.current_best_position, num_vms
        ).tolist()

        breakdown = self.evaluator.evaluate_detailed(best_discrete, tasks, simulator)

        return OptimizationResult(
            algorithm_name="Adaptive",
            best_assignment=best_discrete,
            best_fitness=self.current_best_fitness,
            convergence_history=self.fitness_history,
            diversity_history=self.diversity_history,
            iterations_run=total_iters,
            runtime_sec=runtime,
            population=self.current_population.copy(),
            fitness_breakdown=breakdown,
        )

    def get_decision_history_dataframe(self) -> pd.DataFrame:
        """Export controller decision trail as a pandas DataFrame."""
        if not self.decision_history:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in self.decision_history])


if __name__ == "__main__":
    print("=" * 75)
    print("  BEHAVIOR-AWARE ADAPTIVE CONTROLLER STANDALONE DEMO")
    print("=" * 75)

    from simulation.task_generator import TaskGenerator

    gen = TaskGenerator(seed=42)
    tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
    sim = VMSimulator()

    controller = AdaptiveSwarmController(
        population_size=20,
        max_iterations=50,
        stagnation_threshold=4,
        diversity_low_threshold=0.15,
        seed=42,
    )

    print(f"\nExecuting Adaptive Swarm Optimization on {len(tasks)} tasks...")
    res = controller.optimize(tasks, sim)

    print(f"\nOptimization Finished in {res.runtime_sec:.3f} seconds.")
    print(f"  Final Best Combined Fitness: {res.best_fitness:.2f}")
    if res.fitness_breakdown:
        print(f"  Final Makespan:              {res.fitness_breakdown.makespan:.2f} s")
        print(f"  Final Load Imbalance:        {res.fitness_breakdown.load_imbalance:.2f} s")

    df_decisions = controller.get_decision_history_dataframe()
    print("\nStrategy Distribution Across 50 Generations:")
    print(df_decisions["selected_algorithm"].value_counts().to_string())

    print("\nSample Decision History Trail (First 6 Iterations):")
    sample_cols = ["iteration", "selected_algorithm", "diversity", "stagnation_count", "best_fitness"]
    print(df_decisions[sample_cols].head(6).to_string(index=False))

    print("\nFirst Decision Rationale:")
    print(f"  Iter 1: {df_decisions['reason'].iloc[0]}")
    if len(df_decisions) > 5:
        print(f"  Iter 5: {df_decisions['reason'].iloc[4]}")
    print("=" * 75)

