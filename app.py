"""
Behavior-Aware Adaptive Swarm Optimization for Dynamic Cloud Resource Scheduling
Interactive Research Dashboard powered by Streamlit and Plotly.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Optional

from simulation.task_generator import TaskGenerator, Task
from simulation.vm_simulator import VMSimulator, VirtualMachine
from simulation.scheduler import BaselineScheduler, ScheduleResult
from algorithms.pso import ParticleSwarmOptimizer
from algorithms.gwo import GreyWolfOptimizer
from algorithms.abc import ArtificialBeeColony
from controller.adaptive_controller import AdaptiveSwarmController
from evaluation.fitness import FitnessEvaluator
from evaluation.metrics import MetricsCalculator, PerformanceMetricsReport
from config import (
    WORKLOAD_PROFILES,
    DEFAULT_NUM_TASKS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VMS,
    DEFAULT_FITNESS_LAMBDA,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Cloud Resource Scheduler",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "tasks" not in st.session_state:
    gen = TaskGenerator(seed=DEFAULT_RANDOM_SEED)
    st.session_state.tasks = gen.generate_tasks(num_tasks=DEFAULT_NUM_TASKS, workload_type="normal")
    st.session_state.task_df = gen.to_dataframe()

if "vm_configs" not in st.session_state:
    st.session_state.vm_configs = [
        {"vm_id": "VM1", "mips": 1000.0, "memory_mb": 4096.0, "degraded": False},
        {"vm_id": "VM2", "mips": 1500.0, "memory_mb": 8192.0, "degraded": False},
        {"vm_id": "VM3", "mips": 2000.0, "memory_mb": 16384.0, "degraded": False},
    ]

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = {}

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = {}

if "adaptive_decisions_df" not in st.session_state:
    st.session_state.adaptive_decisions_df = None

# Helper to build current simulator
def get_current_simulator() -> VMSimulator:
    configs = []
    for cfg in st.session_state.vm_configs:
        effective_mips = cfg["mips"] * (0.5 if cfg.get("degraded", False) else 1.0)
        configs.append({
            "vm_id": cfg["vm_id"],
            "mips": effective_mips,
            "memory_mb": cfg["memory_mb"],
        })
    return VMSimulator(vm_configs=configs)


# -----------------------------------------------------------------------------
# SIDEBAR: CONFIGURATION PANEL
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Configuration Panel")

st.sidebar.subheader("1. Workload Specification")
selected_workload = st.sidebar.selectbox(
    "Workload Profile",
    options=["normal", "light", "heavy", "bursty"],
    index=0,
    help="Select statistical distribution for task arrivals and computational demands."
)

task_count = st.sidebar.slider(
    "Number of Tasks (N)",
    min_value=5,
    max_value=50,
    value=len(st.session_state.tasks) if st.session_state.tasks else 20,
    step=1,
)

random_seed = st.sidebar.number_input(
    "Random Seed (Reproducibility)",
    min_value=0,
    max_value=999999,
    value=DEFAULT_RANDOM_SEED,
    step=1,
)

if st.sidebar.button("⚡ Generate New Task Batch", use_container_width=True):
    generator = TaskGenerator(seed=int(random_seed))
    new_tasks = generator.generate_tasks(num_tasks=int(task_count), workload_type=selected_workload)
    st.session_state.tasks = new_tasks
    st.session_state.task_df = generator.to_dataframe()
    st.session_state.benchmark_results.clear()
    st.session_state.optimization_results.clear()
    st.session_state.adaptive_decisions_df = None
    st.sidebar.success(f"Generated {len(new_tasks)} {selected_workload.upper()} tasks!")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Cluster & Swarm Hyperparameters")
pop_size = st.sidebar.slider("Swarm Population Size", min_value=10, max_value=40, value=20, step=5)
max_iters = st.sidebar.slider("Optimization Generations", min_value=10, max_value=100, value=40, step=5)
lambda_weight = st.sidebar.slider("Load Imbalance Weight (λ)", min_value=0.0, max_value=3.0, value=1.0, step=0.25)
stagnation_thresh = st.sidebar.slider("Stagnation Limit (generations)", min_value=2, max_value=10, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Runtime VM Degradation")
for i, vm_cfg in enumerate(st.session_state.vm_configs):
    degraded = st.sidebar.checkbox(
        f"Degrade {vm_cfg['vm_id']} (-50% MIPS)",
        value=vm_cfg.get("degraded", False),
        key=f"deg_toggle_{i}",
    )
    st.session_state.vm_configs[i]["degraded"] = degraded

# -----------------------------------------------------------------------------
# MAIN APPLICATION LAYOUT
# -----------------------------------------------------------------------------
st.title("☁️ Behavior-Aware Adaptive Swarm Optimization")
st.markdown(
    "**A Dynamic Cloud Resource Scheduling Research Simulator**  \n"
    "*Modular discrete-event scheduling engine featuring Particle Swarm Optimization (PSO), "
    "Grey Wolf Optimizer (GWO), Artificial Bee Colony (ABC), and an Explainable Behavior-Aware Controller.*"
)

tabs = st.tabs([
    "📋 Overview & Architecture",
    "📦 Task Generation",
    "🖥️ Virtual Machines",
    "🚀 Heuristic Execution",
    "🧠 Adaptive Controller",
    "📊 Comparative Benchmark",
])

# =============================================================================
# TAB 1: OVERVIEW & ARCHITECTURE
# =============================================================================
with tabs[0]:
    st.header("Project Overview & Mathematical Architecture")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Problem Statement
        In heterogeneous cloud data centers, task scheduling is an **NP-hard** multi-objective combinatorial optimization problem.
        Static dispatching heuristics (e.g. Round Robin, FCFS) ignore workload heterogeneity, while static metaheuristics suffer from:
        - **Premature Convergence:** GWO rapidly exploits but collapses population diversity to zero.
        - **Exploration Latency:** ABC maintains strong exploration via scout bees but refines local solutions gradually.
        - **Static Hyperparameter Mismatch:** Fixed algorithms cannot adapt when tasks surge or VM capacities degrade.

        ### Proposed Adaptive System Architecture
        This prototype introduces a **Behavior-Aware Adaptive Controller** that monitors four runtime indicators:
        1. **Population Diversity ($D_t \in [0, 1]$):** Pairwise Hamming distance measuring schedule dispersion.
        2. **Relative Fitness Improvement ($\Delta f_t$):** Generation-over-generation optimization rate.
        3. **Optimization Stagnation ($S_t$):** Consecutive generations without progress.
        4. **Workload Variation ($\Omega_t$):** Online detection of task surges and VM capacity drops.

        The controller executes explainable rules to switch between **PSO**, **GWO**, and **ABC**, transferring population state without losing optimization history.
        """)
    with col2:
        st.info("💡 **Quick Start Guide**\n\n"
                "1. Check the **Task Generation** tab to inspect the current synthetic job batch.\n"
                "2. Monitor VM capacities and toggle degradation in **Virtual Machines**.\n"
                "3. Execute individual algorithms or the full adaptive solver in **Heuristic Execution**.\n"
                "4. Analyze explainable decision trails in **Adaptive Controller**.\n"
                "5. View multi-metric performance graphs in **Comparative Benchmark**.")

        st.metric("Tasks Loaded", len(st.session_state.tasks))
        st.metric("VMs Configured", len(st.session_state.vm_configs))
        degraded_count = sum(1 for v in st.session_state.vm_configs if v.get("degraded"))
        st.metric("Degraded VMs", degraded_count, delta="-50% capacity" if degraded_count > 0 else "Nominal")

# =============================================================================
# TAB 2: TASK GENERATION
# =============================================================================
with tabs[1]:
    st.header("Synthetic Cloud Task Workload")
    df_tasks = st.session_state.task_df

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", len(df_tasks))
    col2.metric("Mean CPU Demand", f"{df_tasks['cpu_demand_mi'].mean():.1f} MI")
    col3.metric("Peak CPU Demand", f"{df_tasks['cpu_demand_mi'].max():.1f} MI")
    col4.metric("Arrival Duration", f"{df_tasks['arrival_time'].max():.2f} s")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_cpu = px.bar(
            df_tasks,
            x="task_id",
            y="cpu_demand_mi",
            color="priority",
            title="Task CPU Demand (MI) by Priority Rank",
            labels={"cpu_demand_mi": "Demand (MI)", "task_id": "Task ID", "priority": "Priority"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_cpu, use_container_width=True)

    with col_right:
        fig_arr = px.line(
            df_tasks,
            x="arrival_time",
            y=list(range(1, len(df_tasks) + 1)),
            markers=True,
            title="Chronological Task Arrival Timeline",
            labels={"arrival_time": "Arrival Time (seconds)", "y": "Cumulative Tasks"},
        )
        st.plotly_chart(fig_arr, use_container_width=True)

    st.subheader("Task Specifications Table")
    st.dataframe(df_tasks, use_container_width=True)

    csv_data = df_tasks.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Tasks as CSV", csv_data, "cloud_tasks.csv", "text/csv")

# =============================================================================
# TAB 3: VIRTUAL MACHINES
# =============================================================================
with tabs[2]:
    st.header("Virtual Machine Infrastructure & Status")
    sim = get_current_simulator()
    df_vms = sim.to_dataframe()

    col_vm1, col_vm2 = st.columns([3, 2])
    with col_vm1:
        st.subheader("Active VM Specifications")
        st.dataframe(df_vms[["vm_id", "base_mips", "current_mips", "memory_mb", "is_available"]], use_container_width=True)

        fig_mips = px.bar(
            df_vms,
            x="vm_id",
            y=["current_mips", "base_mips"],
            barmode="group",
            title="Nominal vs. Current Effective Processing Capacity (MIPS)",
            labels={"value": "Processing Rate (MIPS)", "vm_id": "VM Identifier"},
            color_discrete_map={"current_mips": "#2ca02c", "base_mips": "#aec7e8"}
        )
        st.plotly_chart(fig_mips, use_container_width=True)

    with col_vm2:
        st.subheader("Degradation Anomaly Controls")
        st.write("Toggle simulated VM degradation events in real time:")
        for idx, cfg in enumerate(st.session_state.vm_configs):
            is_deg = cfg.get("degraded", False)
            btn_label = f"Restore {cfg['vm_id']} to 100%" if is_deg else f"Simulate Degradation on {cfg['vm_id']} (-50%)"
            if st.button(btn_label, key=f"btn_deg_{idx}"):
                st.session_state.vm_configs[idx]["degraded"] = not is_deg
                st.rerun()

# =============================================================================
# TAB 4: HEURISTIC EXECUTION
# =============================================================================
with tabs[3]:
    st.header("Optimization & Scheduling Execution")
    sim = get_current_simulator()
    evaluator = FitnessEvaluator(lambda_imbalance=lambda_weight)

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    run_all = col_btn1.button("▶️ Run Full Comparative Suite", use_container_width=True, type="primary")
    run_adaptive = col_btn2.button("🧠 Run Adaptive Controller Only", use_container_width=True)
    run_pso = col_btn3.button("🐝 Run PSO Only", use_container_width=True)
    run_gwo = col_btn4.button("🐺 Run GWO Only", use_container_width=True)

    if run_all or run_adaptive or run_pso or run_gwo:
        tasks = st.session_state.tasks
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_dict = {}

        # 1. Baselines
        if run_all:
            status_text.text("Executing Baseline Schedulers (FCFS, RoundRobin, MCT)...")
            base_sched = BaselineScheduler(sim)
            base_res = base_sched.run_all(tasks, sim)
            for name, r in base_res.items():
                results_dict[name] = r.to_dict()
            progress_bar.progress(25)

        # 2. PSO
        if run_all or run_pso:
            status_text.text("Running Particle Swarm Optimization (PSO)...")
            pso = ParticleSwarmOptimizer(pop_size, max_iters, fitness_evaluator=evaluator, seed=int(random_seed))
            pso_res = pso.optimize(tasks, sim)
            st.session_state.optimization_results["PSO"] = pso_res
            results_dict["PSO"] = pso_res.to_dict()
            progress_bar.progress(50)

        # 3. GWO
        if run_all or run_gwo:
            status_text.text("Running Grey Wolf Optimizer (GWO)...")
            gwo = GreyWolfOptimizer(pop_size, max_iters, fitness_evaluator=evaluator, seed=int(random_seed))
            gwo_res = gwo.optimize(tasks, sim)
            st.session_state.optimization_results["GWO"] = gwo_res
            results_dict["GWO"] = gwo_res.to_dict()
            progress_bar.progress(70)

        # 4. ABC
        if run_all:
            status_text.text("Running Artificial Bee Colony (ABC)...")
            abc = ArtificialBeeColony(pop_size, max_iters, abandonment_limit=stagnation_thresh, fitness_evaluator=evaluator, seed=int(random_seed))
            abc_res = abc.optimize(tasks, sim)
            st.session_state.optimization_results["ABC"] = abc_res
            results_dict["ABC"] = abc_res.to_dict()
            progress_bar.progress(85)

        # 5. Adaptive Controller
        if run_all or run_adaptive:
            status_text.text("Running Behavior-Aware Adaptive Controller...")
            ctrl = AdaptiveSwarmController(
                population_size=pop_size,
                max_iterations=max_iters,
                stagnation_threshold=stagnation_thresh,
                fitness_evaluator=evaluator,
                seed=int(random_seed),
            )
            ctrl_res = ctrl.optimize(tasks, sim)
            st.session_state.optimization_results["Adaptive"] = ctrl_res
            st.session_state.adaptive_decisions_df = ctrl.get_decision_history_dataframe()
            results_dict["Adaptive"] = ctrl_res.to_dict()
            progress_bar.progress(100)

        st.session_state.benchmark_results.update(results_dict)
        status_text.text("✅ Execution complete!")

    # Display Current Benchmark Results
    if st.session_state.benchmark_results:
        st.subheader("Latest Execution Benchmark Summary")
        df_bench = pd.DataFrame(list(st.session_state.benchmark_results.values()))
        st.dataframe(df_bench, use_container_width=True)

# =============================================================================
# TAB 5: ADAPTIVE CONTROLLER MONITORING
# =============================================================================
with tabs[4]:
    st.header("Adaptive Controller Runtime Monitoring")

    df_decisions = st.session_state.adaptive_decisions_df
    if df_decisions is not None and not df_decisions.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        strat_counts = df_decisions["selected_algorithm"].value_counts().to_dict()
        col_m1.metric("Final Best Fitness", f"{df_decisions['best_fitness'].iloc[-1]:.2f}")
        col_m2.metric("PSO Generations", strat_counts.get("PSO", 0))
        col_m3.metric("GWO Generations", strat_counts.get("GWO", 0))
        col_m4.metric("ABC Generations", strat_counts.get("ABC", 0))

        # Timeline Chart
        algo_map = {"PSO": 0, "GWO": 1, "ABC": 2}
        y_vals = [algo_map[a] for a in df_decisions["selected_algorithm"]]

        fig_trail = go.Figure()
        fig_trail.add_trace(go.Scatter(
            x=df_decisions["iteration"],
            y=y_vals,
            mode="lines+markers",
            marker=dict(size=9, color=y_vals, colorscale=["#1f77b4", "#ff7f0e", "#2ca02c"]),
            line=dict(shape="hv", dash="dot"),
            name="Strategy Choice",
            text=df_decisions["reason"],
            hovertemplate="<b>Gen %{x}</b>: %{text}<extra></extra>"
        ))
        fig_trail.update_layout(
            title="Adaptive Strategy Selection Timeline",
            xaxis_title="Generation",
            yaxis=dict(tickvals=[0, 1, 2], ticktext=["PSO", "GWO", "ABC"]),
            height=300
        )
        st.plotly_chart(fig_trail, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_div = px.line(
                df_decisions,
                x="iteration",
                y="diversity",
                title="Population Diversity Trajectory ($D_t$)",
                labels={"diversity": "Normalized Diversity [0, 1]", "iteration": "Generation"},
            )
            fig_div.add_hline(y=0.15, line_dash="dash", line_color="red", annotation_text="Low Threshold (0.15)")
            st.plotly_chart(fig_div, use_container_width=True)

        with col_d2:
            fig_imp = px.bar(
                df_decisions,
                x="iteration",
                y=df_decisions["fitness_improvement"] * 100.0,
                title="Relative Fitness Improvement (%) per Generation",
                labels={"y": "Improvement (%)", "iteration": "Generation"},
            )
            fig_imp.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="0.5% Threshold")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Explainable Decision Log")
        st.dataframe(df_decisions[["iteration", "selected_algorithm", "diversity", "stagnation_count", "reason"]], use_container_width=True)
    else:
        st.info("Run the **Adaptive Controller** in the 'Heuristic Execution' tab to generate and inspect live decision monitoring.")

# =============================================================================
# TAB 6: COMPARATIVE BENCHMARK & GRAPHS
# =============================================================================
with tabs[5]:
    st.header("Comparative Experimental Benchmark")

    # Load stored benchmark summary if available
    summary_csv = Path("results/experiments/comprehensive_benchmark_summary.csv")
    if summary_csv.exists():
        st.subheader("Multi-Scenario Benchmark Results (Stored Experiments)")
        df_all_bench = pd.read_csv(summary_csv)
        st.dataframe(df_all_bench, use_container_width=True)

        selected_scenario = st.selectbox(
            "Select Scenario to Plot",
            options=df_all_bench["scenario"].unique()
        )
        sub_df = df_all_bench[df_all_bench["scenario"] == selected_scenario]

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_ms = px.bar(
                sub_df,
                x="algorithm",
                y="makespan_mean",
                error_y="makespan_std",
                title=f"Makespan Comparison: {selected_scenario}",
                labels={"makespan_mean": "Makespan (seconds)", "algorithm": "Algorithm"},
                color="algorithm"
            )
            st.plotly_chart(fig_ms, use_container_width=True)

        with col_c2:
            fig_imb = px.bar(
                sub_df,
                x="algorithm",
                y="load_imbalance_mean",
                error_y="load_imbalance_std",
                title=f"Degree of Load Imbalance: {selected_scenario}",
                labels={"load_imbalance_mean": "Degree of Imbalance", "algorithm": "Algorithm"},
                color="algorithm"
            )
            st.plotly_chart(fig_imb, use_container_width=True)

    # Multi-algorithm Convergence Curves
    if st.session_state.optimization_results:
        st.subheader("Convergence Curve Comparison")
        fig_conv = go.Figure()
        for name, r in st.session_state.optimization_results.items():
            if hasattr(r, "convergence_history") and r.convergence_history:
                fig_conv.add_trace(go.Scatter(
                    x=list(range(1, len(r.convergence_history) + 1)),
                    y=r.convergence_history,
                    mode="lines",
                    name=f"{name} (Best: {r.best_fitness:.2f})"
                ))
        fig_conv.update_layout(
            title="Convergence Descent Comparison",
            xaxis_title="Generation",
            yaxis_title="Best Objective Fitness",
            height=400
        )
        st.plotly_chart(fig_conv, use_container_width=True)

st.markdown("---")
st.caption("Behavior-Aware Adaptive Cloud Resource Scheduling Research System | Built with Python, NumPy, Pandas, Streamlit, and Plotly.")

