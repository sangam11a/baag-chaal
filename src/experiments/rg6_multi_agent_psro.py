"""
RG6: Empirical Game-Theoretic Analysis (EGTA) and Multi-Agent PSRO for Bagh-Chal.

Applies:
1. Policy-Space Response Oracles (PSRO) over heterogeneous population of heuristic,
   search, and neural agents.
2. Linear Programming (LP) Exact Zero-Sum Nash Equilibrium Solver.
3. Nash Exploitability & Fictitious Play Convergence.
4. Non-parametric Bootstrap Confidence Intervals & Cohen's d Effect Sizes.
"""

from __future__ import annotations
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("."))

from typing import Dict, Any, List
import numpy as np

from src.environment.board import TIGER, GOAT
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_tiger import GreedyTigerAgent, PositionalTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.asymmetric_mcts import AsymmetricParityMCTSAgent
from src.game_theory.meta_solver import ZeroSumNashSolver, FictitiousPlaySolver
from src.game_theory.statistical_metrics import StatisticianMetrics
from src.game_theory.psro import PSROMetaEngine


def run_rg6_experiment(
    games_per_cell: int = 4,
    output_dir: str = "data/experimental_results/rg6_multi_agent_psro",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("  RG6: MULTI-AGENT EMPIRICAL GAME-THEORETIC ANALYSIS & PSRO (Bagh-Chal)")
    print("  Perspective: Statistician & Game Theorist (Von Neumann - Nash - Lanctot)")
    print("=" * 75)

    # 1. Define Population of Asymmetric Agents
    tigers = [
        RandomAgent("Random-Tiger", side=TIGER),
        GreedyTigerAgent("Greedy-Tiger", side=TIGER, seed=42),
        PositionalTigerAgent("Positional-Tiger", side=TIGER, seed=42),
        MinimaxAgent("Minimax-d2-Tiger", side=TIGER, max_depth=2),
        AsymmetricParityMCTSAgent("AP-MCTS-Tiger", side=TIGER, num_simulations=35, seed=42),
    ]

    goats = [
        RandomAgent("Random-Goat", side=GOAT),
        DefensiveGoatAgent("Defensive-Goat", side=GOAT, seed=42),
        BlockerGoatAgent("Blocker-Goat", side=GOAT, seed=42),
        MinimaxAgent("Minimax-d2-Goat", side=GOAT, max_depth=2),
        AsymmetricParityMCTSAgent("AP-MCTS-Goat", side=GOAT, num_simulations=35, seed=42),
    ]

    m, n = len(tigers), len(goats)
    print(f"\nPopulation: {m} Tigers x {n} Goats ({m * n} Cross-Matchup Cells)")
    print(f"Games per Cell: {games_per_cell * 2} (Paired role-reversal to eliminate bias)")

    # 2. Build Empirical Payoff Matrix M
    psro = PSROMetaEngine(initial_tigers=tigers, initial_goats=goats, games_per_cell=games_per_cell)
    M = psro.payoff_matrix

    print("\n--- Empirical Payoff Matrix M (Tiger Perspective: +1 Win, -1 Loss, 0 Draw) ---")
    header = f"{'Tiger \\ Goat':<20}" + "".join([f"{g.name[:12]:>14}" for g in goats])
    print(header)
    print("-" * len(header))
    for i, t in enumerate(tigers):
        row_str = f"{t.name:<20}" + "".join([f"{M[i, j]:>14.2f}" for j in range(n)])
        print(row_str)

    # 3. Solve for Exact Zero-Sum Nash Equilibrium via LP
    sigma_t, sigma_g, game_val = ZeroSumNashSolver.solve_lp(M)
    exploitability = ZeroSumNashSolver.compute_exploitability(M, sigma_t, sigma_g)
    entropy_t = ZeroSumNashSolver.compute_strategy_entropy(sigma_t)
    entropy_g = ZeroSumNashSolver.compute_strategy_entropy(sigma_g)

    print("\n" + "=" * 60)
    print("  EXACT GAME-THEORETIC NASH EQUILIBRIUM (Linear Programming)")
    print("=" * 60)
    print(f"Meta-Game Value (v*)       : {game_val:+.4f} (Draw Equilibrium = 0.0)")
    print(f"Nash Exploitability (E)     : {exploitability:.6f}")
    print(f"Tiger Strategy Entropy H(T) : {entropy_t:.3f} nats")
    print(f"Goat Strategy Entropy H(G)  : {entropy_g:.3f} nats\n")

    print("Optimal Tiger Meta-Strategy (sigma_T*):")
    for i, t in enumerate(tigers):
        if sigma_t[i] > 1e-4:
            print(f"  * {t.name:<22}: {sigma_t[i] * 100:6.2f}%")

    print("\nOptimal Goat Meta-Strategy (sigma_G*):")
    for j, g in enumerate(goats):
        if sigma_g[j] > 1e-4:
            print(f"  * {g.name:<22}: {sigma_g[j] * 100:6.2f}%")

    # 4. Fictitious Play Meta-Solver Convergence
    fp_solver = FictitiousPlaySolver(M)
    fp_res = fp_solver.run(iterations=300)
    print(f"\nFictitious Play Converged Exploitability: {fp_res['final_exploitability']:.4f}")

    # 5. Statistical Rigor: Bootstrapped CIs & Effect Size of AP-MCTS vs Minimax
    ap_mcts_idx = 4
    minimax_idx = 3

    # Compare AP-MCTS-Tiger vs Minimax-d2-Tiger across Goat population
    ap_tiger_payoffs = M[ap_mcts_idx, :]
    minimax_tiger_payoffs = M[minimax_idx, :]
    d_tiger = StatisticianMetrics.cohens_d(ap_tiger_payoffs, minimax_tiger_payoffs)

    # Compare AP-MCTS-Goat vs Minimax-d2-Goat across Tiger population (lower payoff is better for goat)
    ap_goat_payoffs = -M[:, ap_mcts_idx]
    minimax_goat_payoffs = -M[:, minimax_idx]
    d_goat = StatisticianMetrics.cohens_d(ap_goat_payoffs, minimax_goat_payoffs)

    # Bootstrapped CI for Game Value
    all_payoffs = M.flatten()
    mean_val, ci_low, ci_high = StatisticianMetrics.bootstrap_confidence_interval(all_payoffs, num_bootstrap=1000)

    print("\n" + "=" * 60)
    print("  STATISTICIAN'S RIGOR: BOOTSTRAP CIs & EFFECT SIZES")
    print("=" * 60)
    print(f"Empirical Payoff Mean       : {mean_val:+.3f} (95% BCa CI: [{ci_low:+.3f}, {ci_high:+.3f}])")
    print(f"Cohen's d (AP-MCTS vs Minimax Tiger): {d_tiger:+.3f}")
    print(f"Cohen's d (AP-MCTS vs Minimax Goat) : {d_goat:+.3f}")

    results = {
        "payoff_matrix": M.tolist(),
        "tiger_names": [t.name for t in tigers],
        "goat_names": [g.name for g in goats],
        "nash_equilibrium": {
            "game_value": game_val,
            "exploitability": exploitability,
            "tiger_meta_strategy": {t.name: float(sigma_t[i]) for i, t in enumerate(tigers)},
            "goat_meta_strategy": {g.name: float(sigma_g[j]) for j, g in enumerate(goats)},
            "tiger_entropy": entropy_t,
            "goat_entropy": entropy_g,
        },
        "fictitious_play": {
            "final_game_value": fp_res["final_game_value"],
            "final_exploitability": fp_res["final_exploitability"],
        },
        "statistical_metrics": {
            "bootstrap_mean_payoff": mean_val,
            "bootstrap_95_ci": [ci_low, ci_high],
            "cohens_d_tiger_ap_mcts_vs_minimax": d_tiger,
            "cohens_d_goat_ap_mcts_vs_minimax": d_goat,
        },
    }

    summary_path = os.path.join(output_dir, "rg6_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] RG6 Completed! Results saved to {summary_path}\n")
    return results


if __name__ == "__main__":
    run_rg6_experiment(games_per_cell=3)
