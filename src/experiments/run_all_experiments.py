"""
Orchestration Pipeline to Run All 5 Research Gap Experiments (RG1-RG5).
Generates empirical data for observation.md and Q1 paper manuscript.
"""

import os
import sys
import time
import json

from src.experiments.rg1_credit_assignment import run_rg1_experiment
from src.experiments.rg2_topological_bias import run_rg2_experiment
from src.experiments.rg3_architectural_asymmetry import run_rg3_experiment
from src.experiments.rg4_search_scaling import run_rg4_experiment
from src.experiments.rg5_heuristic_optimization import run_rg5_experiment


def main():
    print("=" * 70)
    print("  BAGH-CHAL RESEARCH GAP EXPERIMENTAL PIPELINE (RG1 - RG5)")
    print("  Venue Target: Springer Nature Q1 Journal")
    print("=" * 70)
    t_start = time.time()

    # Stage 1: RG1
    rg1_path = "data/experimental_results/rg1_credit_assignment/rg1_results_summary.json"
    if os.path.exists(rg1_path):
        print(f"\n>>> STAGE 1/5: RG1 already completed. Loaded from {rg1_path}")
    else:
        print("\n>>> STAGE 1/5: RG1 (Asymmetric Credit Assignment & Reward Sparsity)")
        run_rg1_experiment(num_episodes=30, seeds=[42, 101])

    # Stage 2: RG2
    rg2_path = "data/experimental_results/rg2_topological_bias/rg2_results_summary.json"
    if os.path.exists(rg2_path):
        print(f"\n>>> STAGE 2/5: RG2 already completed. Loaded from {rg2_path}")
    else:
        print("\n>>> STAGE 2/5: RG2 (Topological Inductive Bias: GNN vs CNN vs MLP)")
        run_rg2_experiment(num_episodes=20, seeds=[42, 101])

    # Stage 3: RG3
    rg3_path = "data/experimental_results/rg3_architectural_asymmetry/rg3_results_summary.json"
    if os.path.exists(rg3_path):
        print(f"\n>>> STAGE 3/5: RG3 already completed. Loaded from {rg3_path}")
    else:
        print("\n>>> STAGE 3/5: RG3 (Architectural Asymmetry: Single vs Dual Network)")
        run_rg3_experiment(num_episodes=20, seeds=[42, 101])

    # Stage 4: RG4
    rg4_path = "data/experimental_results/rg4_search_scaling/rg4_results_summary.json"
    if os.path.exists(rg4_path):
        print(f"\n>>> STAGE 4/5: RG4 already completed. Loaded from {rg4_path}")
    else:
        print("\n>>> STAGE 4/5: RG4 (Bounded Rationality & Search Scaling Paradox)")
        run_rg4_experiment(depths=[1, 2, 3], rounds_per_cell=3)

    # Stage 5: RG5
    rg5_path = "data/experimental_results/rg5_heuristic_optimization/rg5_results_summary.json"
    if os.path.exists(rg5_path):
        print(f"\n>>> STAGE 5/5: RG5 already completed. Loaded from {rg5_path}")
    else:
        print("\n>>> STAGE 5/5: RG5 (Handcrafted Heuristics vs Metaheuristic Optimization)")
        run_rg5_experiment(population_size=6, generations=3)

    elapsed_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 70)
    print(f"  ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed_min:.2f} MINUTES!")
    print("  Empirical logs stored in data/experimental_results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
