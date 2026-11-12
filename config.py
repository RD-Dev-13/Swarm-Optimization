"""
Configuration Module for Behavior-Aware Adaptive Cloud Resource Scheduling System.

Defines simulation constants, workload parameters, VM profiles, and optimization hyperparameters.
All settings are designed to be reproducible and configurable via the Streamlit dashboard.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class WorkloadProfile:
    """Workload specification defining CPU demand ranges and inter-arrival timing."""
    name: str
    cpu_min_mi: float
    cpu_max_mi: float
    mean_inter_arrival_sec: float
    description: str


WORKLOAD_PROFILES: Dict[str, WorkloadProfile] = {
    "light": WorkloadProfile(
        name="light",
        cpu_min_mi=1000.0,
        cpu_max_mi=5000.0,
        mean_inter_arrival_sec=0.5,
        description="Light compute tasks with high arrival frequency (0.5s inter-arrival)."
    ),
    "normal": WorkloadProfile(
        name="normal",
        cpu_min_mi=2000.0,
        cpu_max_mi=10000.0,
        mean_inter_arrival_sec=1.0,
        description="Balanced standard cloud tasks with 1.0s mean inter-arrival."
    ),
    "heavy": WorkloadProfile(
        name="heavy",
        cpu_min_mi=8000.0,
        cpu_max_mi=20000.0,
        mean_inter_arrival_sec=1.5,
        description="Compute-intensive batch tasks with 1.5s mean inter-arrival."
    ),
    "bursty": WorkloadProfile(
        name="bursty",
        cpu_min_mi=2000.0,
        cpu_max_mi=15000.0,
        mean_inter_arrival_sec=0.1,  # Burst cluster scale
        description="Bursty traffic with alternating dense clusters and quiescent periods."
    ),
}

# Task Generator Defaults & Limits
DEFAULT_NUM_TASKS: int = 20
MIN_NUM_TASKS: int = 1
MAX_NUM_TASKS: int = 50
DEFAULT_RANDOM_SEED: int = 42

# Priority Range (1: Low, 2: Normal, 3: High, 4: Urgent, 5: Critical)
MIN_PRIORITY: int = 1
MAX_PRIORITY: int = 5

# Memory Allocation Options (in MB)
MEMORY_CHOICES_MB: tuple = (512, 1024, 2048, 4096)

# Default Virtual Machine Specifications (Phase 2 preview)
DEFAULT_VMS = [
    {"vm_id": "VM1", "mips": 1000.0, "memory_mb": 4096},
    {"vm_id": "VM2", "mips": 1500.0, "memory_mb": 8192},
    {"vm_id": "VM3", "mips": 2000.0, "memory_mb": 16384},
]

# Optimization Hyperparameters (Phases 5-8 preview)
SWARM_POPULATION_SIZE: int = 20
MAX_OPTIMIZATION_ITERATIONS: int = 50
STAGNATION_THRESHOLD: int = 5
DEFAULT_FITNESS_LAMBDA: float = 1.0

