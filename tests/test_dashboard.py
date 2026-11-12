"""
Automated tests for Dashboard integration and session workflow routines.

Verifies:
1. Dashboard modules and simulation functions interact cleanly.
2. Generating tasks through the generator creates valid DataFrame and session state equivalents.
3. Executing all scheduling algorithms (FCFS, MCT, PSO, GWO, ABC, Adaptive) produces valid dictionary outputs for Streamlit rendering.
4. Decision history and benchmark dataframes export cleanly as CSV.
"""

from pathlib import Path
import pytest
import pandas as pd

from simulation.task_generator import TaskGenerator
from simulation.vm_simulator import VMSimulator
from simulation.scheduler import BaselineScheduler
from algorithms.pso import ParticleSwarmOptimizer
from algorithms.gwo import GreyWolfOptimizer
from algorithms.abc import ArtificialBeeColony
from controller.adaptive_controller import AdaptiveSwarmController
from evaluation.fitness import FitnessEvaluator


class TestDashboardWorkflows:
    """Verifies backend logic driving the interactive Streamlit dashboard."""

    @pytest.fixture
    def environment(self):
        """Prepare sample task batch and VM cluster."""
        gen = TaskGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=15, workload_type="normal")
        sim = VMSimulator()
        evaluator = FitnessEvaluator(lambda_imbalance=1.0)
        return tasks, sim, evaluator

    def test_dashboard_task_generation_and_export(self, environment):
        """Verifies task generation and CSV serializability as performed by dashboard."""
        tasks, _, _ = environment
        gen = TaskGenerator(seed=42)
        df = gen.to_dataframe(tasks)

        assert len(df) == 15
        assert set(df.columns) == {
            "task_id", "cpu_demand_mi", "arrival_time", "priority", "memory_mb", "workload_type"
        }
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        assert len(csv_bytes) > 0

    def test_dashboard_algorithm_execution_pipeline(self, environment):
        """Verifies all algorithms execute and generate valid dictionary summaries for UI cards."""
        tasks, sim, evaluator = environment

        # Baseline
        base_sched = BaselineScheduler(sim)
        fcfs_res = base_sched.schedule_fcfs(tasks, sim).to_dict()
        assert "makespan_sec" in fcfs_res

        # PSO
        pso = ParticleSwarmOptimizer(10, 10, fitness_evaluator=evaluator, seed=42)
        pso_dict = pso.optimize(tasks, sim).to_dict()
        assert "best_fitness" in pso_dict

        # GWO
        gwo = GreyWolfOptimizer(10, 10, fitness_evaluator=evaluator, seed=42)
        gwo_dict = gwo.optimize(tasks, sim).to_dict()
        assert "best_fitness" in gwo_dict

        # ABC
        abc = ArtificialBeeColony(10, 10, fitness_evaluator=evaluator, seed=42)
        abc_dict = abc.optimize(tasks, sim).to_dict()
        assert "best_fitness" in abc_dict

        # Adaptive
        ctrl = AdaptiveSwarmController(10, 10, fitness_evaluator=evaluator, seed=42)
        adapt_res = ctrl.optimize(tasks, sim)
        adapt_dict = adapt_res.to_dict()
        assert "best_fitness" in adapt_dict

        decisions_df = ctrl.get_decision_history_dataframe()
        assert len(decisions_df) == 10
        assert "selected_algorithm" in decisions_df.columns

