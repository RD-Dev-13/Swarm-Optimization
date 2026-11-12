"""
Experimental Evaluation Framework for Behavior-Aware Adaptive Cloud Scheduling.

Executes all four benchmark research scenarios across multiple independent Monte Carlo trials:
- Scenario 1: Normal Workload (Steady arrivals, standard CPU demand, 3 stable VMs)
- Scenario 2: Sudden Workload Burst (Surge cluster arrival, burst detection, re-evaluation)
- Scenario 3: VM Capacity Degradation (Thermal throttling / noisy neighbor event, capacity drop)
- Scenario 4: Mixed Workload (Heterogeneous tasks across 4 VMs)

Compares:
1. First-Come, First-Served (FCFS)
2. Round Robin (RR)
3. Minimum Completion Time (MCT)
4. Particle Swarm Optimization (PSO)
5. Grey Wolf Optimizer (GWO)
6. Artificial Bee Colony (ABC)
7. Behavior-Aware Adaptive Controller (Adaptive)

Computes statistical means, standard deviations, generates publication-ready comparison
figures, and exports reproducible CSV and JSON records into results/experiments/ and results/graphs/.
"""

import os
import tempfile
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation.task_generator import TaskGenerator, Task
from simulation.vm_simulator import VMSimulator, VirtualMachine
from simulation.scheduler import BaselineScheduler
from algorithms.pso import ParticleSwarmOptimizer
from algorithms.gwo import GreyWolfOptimizer
from algorithms.abc import ArtificialBeeColony
from controller.adaptive_controller import AdaptiveSwarmController
from evaluation.fitness import FitnessEvaluator
from evaluation.metrics import MetricsCalculator, PerformanceMetricsReport
from visualization.charts import CloudSchedulerVisualizer
from config import DEFAULT_VMS


class ExperimentRunner:
    """
    Automated experimental testbed managing benchmark scenarios and statistical evaluations.
    """

    def __init__(
        self,
        results_dir: str = "results/experiments",
        graphs_dir: str = "results/graphs",
        population_size: int = 20,
        max_iterations: int = 40,
    ):
        """
        Initialize the ExperimentRunner.

        Args:
            results_dir: Directory for storing output CSVs and JSONs.
            graphs_dir: Directory for saving generated visualization plots.
            population_size: Swarm population size for metaheuristics.
            max_iterations: Maximum optimization generations.
        """
        self.results_dir = Path(results_dir).resolve()
        self.graphs_dir = Path(graphs_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.evaluator = FitnessEvaluator(lambda_imbalance=1.0)

    def _execute_single_trial(
        self,
        tasks: List[Task],
        simulator: VMSimulator,
        seed: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Execute all 7 scheduling methods under identical tasks and VM state for one trial.

        Returns:
            Dictionary mapping algorithm_name -> performance metrics dict.
        """
        trial_results: Dict[str, Dict[str, float]] = {}

        # 1. Baseline Heuristics
        base_sched = BaselineScheduler(simulator)
        base_results = base_sched.run_all(tasks, simulator)

        for name, s_res in base_results.items():
            report = MetricsCalculator.generate_full_report(
                s_res.scheduled_records, simulator
            )
            trial_results[name] = {
                "makespan": report.makespan_sec,
                "utilization": report.avg_utilization_pct,
                "load_imbalance": report.degree_of_imbalance,
                "imbalance_std": report.busy_time_std_sec,
                "waiting_time": report.avg_waiting_time_sec,
                "execution_time": report.avg_execution_time_sec,
                "fitness": round(report.makespan_sec + (1.0 * report.busy_time_std_sec), 4),
            }

        # 2. Particle Swarm Optimization (PSO)
        pso = ParticleSwarmOptimizer(
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            fitness_evaluator=self.evaluator,
            seed=seed,
        )
        pso_res = pso.optimize(tasks, simulator)
        pso_records = simulator.simulate_assignments(tasks, pso_res.best_assignment)
        pso_report = MetricsCalculator.generate_full_report(pso_records, simulator)
        trial_results["PSO"] = {
            "makespan": pso_report.makespan_sec,
            "utilization": pso_report.avg_utilization_pct,
            "load_imbalance": pso_report.degree_of_imbalance,
            "imbalance_std": pso_report.busy_time_std_sec,
            "waiting_time": pso_report.avg_waiting_time_sec,
            "execution_time": pso_report.avg_execution_time_sec,
            "fitness": pso_res.best_fitness,
        }

        # 3. Grey Wolf Optimizer (GWO)
        gwo = GreyWolfOptimizer(
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            fitness_evaluator=self.evaluator,
            seed=seed,
        )
        gwo_res = gwo.optimize(tasks, simulator)
        gwo_records = simulator.simulate_assignments(tasks, gwo_res.best_assignment)
        gwo_report = MetricsCalculator.generate_full_report(gwo_records, simulator)
        trial_results["GWO"] = {
            "makespan": gwo_report.makespan_sec,
            "utilization": gwo_report.avg_utilization_pct,
            "load_imbalance": gwo_report.degree_of_imbalance,
            "imbalance_std": gwo_report.busy_time_std_sec,
            "waiting_time": gwo_report.avg_waiting_time_sec,
            "execution_time": gwo_report.avg_execution_time_sec,
            "fitness": gwo_res.best_fitness,
        }

        # 4. Artificial Bee Colony (ABC)
        abc = ArtificialBeeColony(
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            fitness_evaluator=self.evaluator,
            seed=seed,
        )
        abc_res = abc.optimize(tasks, simulator)
        abc_records = simulator.simulate_assignments(tasks, abc_res.best_assignment)
        abc_report = MetricsCalculator.generate_full_report(abc_records, simulator)
        trial_results["ABC"] = {
            "makespan": abc_report.makespan_sec,
            "utilization": abc_report.avg_utilization_pct,
            "load_imbalance": abc_report.degree_of_imbalance,
            "imbalance_std": abc_report.busy_time_std_sec,
            "waiting_time": abc_report.avg_waiting_time_sec,
            "execution_time": abc_report.avg_execution_time_sec,
            "fitness": abc_res.best_fitness,
        }

        # 5. Behavior-Aware Adaptive Controller
        controller = AdaptiveSwarmController(
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            fitness_evaluator=self.evaluator,
            seed=seed,
        )
        adapt_res = controller.optimize(tasks, simulator)
        adapt_records = simulator.simulate_assignments(tasks, adapt_res.best_assignment)
        adapt_report = MetricsCalculator.generate_full_report(adapt_records, simulator)
        trial_results["Adaptive"] = {
            "makespan": adapt_report.makespan_sec,
            "utilization": adapt_report.avg_utilization_pct,
            "load_imbalance": adapt_report.degree_of_imbalance,
            "imbalance_std": adapt_report.busy_time_std_sec,
            "waiting_time": adapt_report.avg_waiting_time_sec,
            "execution_time": adapt_report.avg_execution_time_sec,
            "fitness": adapt_res.best_fitness,
        }

        return trial_results

    def _aggregate_trials(
        self,
        scenario_name: str,
        trial_data: List[Dict[str, Dict[str, float]]]
    ) -> pd.DataFrame:
        """Aggregate multiple trials into mean and std dev metrics."""
        algorithms = list(trial_data[0].keys())
        metric_keys = ["makespan", "fitness", "utilization", "load_imbalance", "waiting_time"]

        summary_rows = []
        for algo in algorithms:
            row = {"scenario": scenario_name, "algorithm": algo}
            for m in metric_keys:
                vals = [t[algo][m] for t in trial_data]
                row[f"{m}_mean"] = round(float(np.mean(vals)), 2)
                row[f"{m}_std"] = round(float(np.std(vals)), 2)
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def run_scenario_1_normal(self, trials: int = 3, seed_base: int = 42) -> pd.DataFrame:
        """
        Scenario 1: Normal Workload.
        - 20 standard tasks (2,000–10,000 MI, 1.0s mean inter-arrival).
        - 3 stable VMs (1000, 1500, 2000 MIPS).
        """
        print("\n" + "=" * 70)
        print("  RUNNING SCENARIO 1: NORMAL WORKLOAD")
        print("=" * 70)

        trial_records = []
        for trial in range(trials):
            seed = seed_base + (trial * 17)
            gen = TaskGenerator(seed=seed)
            tasks = gen.generate_tasks(num_tasks=20, workload_type="normal")
            sim = VMSimulator()

            print(f"  [Scenario 1] Executing Trial {trial + 1}/{trials} (Seed: {seed})...")
            res = self._execute_single_trial(tasks, sim, seed)
            trial_records.append(res)

        df_summary = self._aggregate_trials("Scenario 1 (Normal)", trial_records)
        csv_file = self.results_dir / "scenario_1_normal_results.csv"
        df_summary.to_csv(csv_file, index=False)
        print(f"  Saved Scenario 1 summary to {csv_file}")
        return df_summary

    def run_scenario_2_burst(self, trials: int = 3, seed_base: int = 142) -> pd.DataFrame:
        """
        Scenario 2: Sudden Workload Burst.
        - Starts with 15 normal tasks, followed by a sudden burst surge of 15 bursty tasks.
        - Total 30 tasks with rapid surge arrivals.
        """
        print("\n" + "=" * 70)
        print("  RUNNING SCENARIO 2: SUDDEN WORKLOAD BURST")
        print("=" * 70)

        trial_records = []
        for trial in range(trials):
            seed = seed_base + (trial * 23)
            gen = TaskGenerator(seed=seed)
            # 30 tasks in bursty profile simulating surge traffic
            tasks = gen.generate_tasks(num_tasks=30, workload_type="bursty")
            sim = VMSimulator()

            print(f"  [Scenario 2] Executing Trial {trial + 1}/{trials} (Seed: {seed}, Tasks: 30)...")
            res = self._execute_single_trial(tasks, sim, seed)
            trial_records.append(res)

        df_summary = self._aggregate_trials("Scenario 2 (Bursty)", trial_records)
        csv_file = self.results_dir / "scenario_2_burst_results.csv"
        df_summary.to_csv(csv_file, index=False)
        print(f"  Saved Scenario 2 summary to {csv_file}")
        return df_summary

    def run_scenario_3_degradation(self, trials: int = 3, seed_base: int = 242) -> pd.DataFrame:
        """
        Scenario 3: Virtual Machine Degradation.
        - 25 heavy tasks.
        - VM2 capacity degraded by 50% (1500 MIPS -> 750 MIPS).
        - Tests adaptation under constrained computing capacity.
        """
        print("\n" + "=" * 70)
        print("  RUNNING SCENARIO 3: VM CAPACITY DEGRADATION")
        print("=" * 70)

        trial_records = []
        for trial in range(trials):
            seed = seed_base + (trial * 31)
            gen = TaskGenerator(seed=seed)
            tasks = gen.generate_tasks(num_tasks=25, workload_type="heavy")

            # Initialize cluster with VM2 degraded to 50%
            sim = VMSimulator()
            sim.apply_vm_degradation("VM2", factor=0.5)

            print(f"  [Scenario 3] Executing Trial {trial + 1}/{trials} (VM2 degraded to 750 MIPS)...")
            res = self._execute_single_trial(tasks, sim, seed)
            trial_records.append(res)

        df_summary = self._aggregate_trials("Scenario 3 (Degradation)", trial_records)
        csv_file = self.results_dir / "scenario_3_degradation_results.csv"
        df_summary.to_csv(csv_file, index=False)
        print(f"  Saved Scenario 3 summary to {csv_file}")
        return df_summary

    def run_scenario_4_mixed(self, trials: int = 3, seed_base: int = 342) -> pd.DataFrame:
        """
        Scenario 4: Mixed Heterogeneous Workload.
        - 35 tasks with diverse compute profiles across 4 heterogeneous VMs.
        - VM1: 1000, VM2: 1500, VM3: 2000, VM4: 2500 MIPS.
        """
        print("\n" + "=" * 70)
        print("  RUNNING SCENARIO 4: MIXED HETEROGENEOUS WORKLOAD")
        print("=" * 70)

        vm_configs_4 = [
            {"vm_id": "VM1", "mips": 1000.0, "memory_mb": 4096.0},
            {"vm_id": "VM2", "mips": 1500.0, "memory_mb": 8192.0},
            {"vm_id": "VM3", "mips": 2000.0, "memory_mb": 16384.0},
            {"vm_id": "VM4", "mips": 2500.0, "memory_mb": 32768.0},
        ]

        trial_records = []
        for trial in range(trials):
            seed = seed_base + (trial * 41)
            gen = TaskGenerator(seed=seed)
            # Mixed batch: 35 tasks
            tasks = gen.generate_tasks(num_tasks=35, workload_type="normal")
            sim = VMSimulator(vm_configs=vm_configs_4)

            print(f"  [Scenario 4] Executing Trial {trial + 1}/{trials} (4 VMs, 35 Tasks)...")
            res = self._execute_single_trial(tasks, sim, seed)
            trial_records.append(res)

        df_summary = self._aggregate_trials("Scenario 4 (Mixed)", trial_records)
        csv_file = self.results_dir / "scenario_4_mixed_results.csv"
        df_summary.to_csv(csv_file, index=False)
        print(f"  Saved Scenario 4 summary to {csv_file}")
        return df_summary

    def run_all_experiments(self, trials_per_scenario: int = 3) -> Dict[str, Any]:
        """
        Orchestrate complete benchmark evaluation of all 4 scenarios.

        Returns:
            Dictionary containing compiled benchmark summaries and paths.
        """
        t0 = time.time()
        print("\n" + "=" * 75)
        print("  STARTING COMPREHENSIVE EXPERIMENTAL EVALUATION SUITE")
        print(f"  Trials per Scenario: {trials_per_scenario} | Population: {self.population_size} | Iterations: {self.max_iterations}")
        print("=" * 75)

        s1_df = self.run_scenario_1_normal(trials=trials_per_scenario)
        s2_df = self.run_scenario_2_burst(trials=trials_per_scenario)
        s3_df = self.run_scenario_3_degradation(trials=trials_per_scenario)
        s4_df = self.run_scenario_4_mixed(trials=trials_per_scenario)

        # Combine into master benchmark dataframe
        all_df = pd.concat([s1_df, s2_df, s3_df, s4_df], ignore_index=True)
        master_csv = self.results_dir / "comprehensive_benchmark_summary.csv"
        all_df.to_csv(master_csv, index=False)

        # Generate comparative charts
        s1_comp_df = s1_df.rename(columns={"makespan_mean": "makespan_sec", "load_imbalance_mean": "load_imbalance"})
        fig_makespan = CloudSchedulerVisualizer.plot_makespan_comparison(
            s1_comp_df, self.graphs_dir / "benchmark_makespan_comparison.png"
        )
        plt.close(fig_makespan)

        fig_imbalance = CloudSchedulerVisualizer.plot_load_imbalance_comparison(
            s1_comp_df, self.graphs_dir / "benchmark_imbalance_comparison.png"
        )
        plt.close(fig_imbalance)

        # Generate sample single-run convergence curves & adaptive timeline for report (50 iterations)
        gen = TaskGenerator(seed=42)
        tasks_demo = gen.generate_tasks(num_tasks=20, workload_type="normal")
        sim_demo = VMSimulator()

        evaluator = self.evaluator
        demo_iters = 50
        pso_res = ParticleSwarmOptimizer(20, demo_iters, fitness_evaluator=evaluator, seed=42).optimize(tasks_demo, sim_demo)
        gwo_res = GreyWolfOptimizer(20, demo_iters, fitness_evaluator=evaluator, seed=42).optimize(tasks_demo, sim_demo)
        abc_res = ArtificialBeeColony(20, demo_iters, fitness_evaluator=evaluator, seed=42).optimize(tasks_demo, sim_demo)
        ctrl = AdaptiveSwarmController(20, demo_iters, fitness_evaluator=evaluator, seed=42)
        ctrl_res = ctrl.optimize(tasks_demo, sim_demo)

        res_dict = {"PSO": pso_res, "GWO": gwo_res, "ABC": abc_res, "Adaptive": ctrl_res}
        fig_conv = CloudSchedulerVisualizer.plot_convergence_curves(res_dict, self.graphs_dir / "benchmark_convergence_curves.png")
        plt.close(fig_conv)

        fig_div = CloudSchedulerVisualizer.plot_diversity_over_time(res_dict, self.graphs_dir / "benchmark_diversity_trajectories.png")
        plt.close(fig_div)

        decision_df = ctrl.get_decision_history_dataframe()
        fig_timeline = CloudSchedulerVisualizer.plot_adaptive_strategy_timeline(
            decision_df, self.graphs_dir / "benchmark_adaptive_strategy_timeline.png"
        )
        plt.close(fig_timeline)

        fig_imp = CloudSchedulerVisualizer.plot_fitness_improvement_over_time(
            decision_df, self.graphs_dir / "benchmark_fitness_improvement.png"
        )
        plt.close(fig_imp)

        # Scenario 3 (Degradation) Makespan Comparison Chart
        s3_comp_df = s3_df.rename(columns={"makespan_mean": "makespan_sec", "load_imbalance_mean": "load_imbalance"})
        fig_s3_ms = CloudSchedulerVisualizer.plot_makespan_comparison(
            s3_comp_df, self.graphs_dir / "scenario3_degradation_makespan.png"
        )
        plt.close(fig_s3_ms)

        # VM Utilization for Adaptive Schedule
        sim_demo.simulate_assignments(tasks_demo, ctrl_res.best_assignment)
        makespan_demo = sim_demo.get_makespan()
        vm_utils = {vm.vm_id: vm.calculate_utilization(makespan_demo) for vm in sim_demo.vm_list}
        fig_util = CloudSchedulerVisualizer.plot_vm_utilization(
            vm_utils, self.graphs_dir / "benchmark_vm_utilization.png"
        )
        plt.close(fig_util)

        # Export JSON metadata
        duration = round(time.time() - t0, 2)
        summary_meta = {
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmark_duration_sec": duration,
            "trials_per_scenario": trials_per_scenario,
            "algorithms_evaluated": ["FCFS", "RoundRobin", "MCT", "PSO", "GWO", "ABC", "Adaptive"],
            "scenarios": [
                "Scenario 1: Normal Workload (20 tasks, 3 stable VMs)",
                "Scenario 2: Sudden Workload Burst (30 bursty tasks, surge arrivals)",
                "Scenario 3: VM Degradation (25 heavy tasks, VM2 at 50% capacity)",
                "Scenario 4: Mixed Heterogeneous Workload (35 tasks, 4 VMs)",
            ],
            "master_csv_path": str(master_csv),
        }
        json_file = self.results_dir / "comprehensive_benchmark_summary.json"
        with open(json_file, "w") as f:
            json.dump(summary_meta, f, indent=2)

        print("\n" + "=" * 75)
        print("  EXPERIMENTAL BENCHMARK SUITE COMPLETED SUCCESSFULLY")
        print(f"  Total Duration: {duration} seconds")
        print(f"  Summary CSV:    {master_csv}")
        print(f"  Summary JSON:   {json_file}")
        print(f"  Figures:        {self.graphs_dir}")
        print("=" * 75)

        return summary_meta


if __name__ == "__main__":
    runner = ExperimentRunner(population_size=15, max_iterations=25)
    runner.run_all_experiments(trials_per_scenario=2)

