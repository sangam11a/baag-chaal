"""
RG4: Bounded Rationality & Search Scaling Paradox in Asymmetric Games.

Investigates the win rate landscape W(d_T, d_G) across asymmetric search depths.
Hypothesis:
1. Low-depth search heavily biases toward the predator (Tigers) due to immediate capture visibility.
2. Equal search depth at or above depth 2 triggers a sharp phase transition toward Draws,
   empirically confirming the Lim-Nievergelt equilibrium under bounded rationality.
3. An asymmetric depth advantage (d_G > d_T) rapidly shifts win probability to Goats.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any

from src.environment.board import TIGER, GOAT
from src.agents.minimax_agent import MinimaxAgent
from src.agents.mcts_agent import MCTSAgent
from src.evaluation.tournament import play_game


def run_rg4_experiment(
    depths: List[int] = [1, 2, 3],
    rounds_per_cell: int = 5,
    output_dir: str = "data/experimental_results/rg4_search_scaling",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "depth_grid": {},
        "mcts_vs_minimax": {},
    }

    print("=" * 60)
    print("RG4: Executing Search Scaling & Bounded Rationality Experiment")
    print(f"Depths: {depths} | Rounds per Cell: {rounds_per_cell}")
    print("=" * 60)

    # 1. Minimax Cross-Depth Grid: Tiger Depth vs Goat Depth
    for d_tiger in depths:
        for d_goat in depths:
            cell_key = f"T_d{d_tiger}_vs_G_d{d_goat}"
            tiger_agent = MinimaxAgent(name=f"Tiger-d{d_tiger}", side=TIGER, max_depth=d_tiger)
            goat_agent = MinimaxAgent(name=f"Goat-d{d_goat}", side=GOAT, max_depth=d_goat)

            t_wins = 0
            g_wins = 0
            draws = 0
            plies_list = []
            t_times = []
            g_times = []

            for r in range(rounds_per_cell):
                seed = 7000 + r
                rec = play_game(tiger_agent=tiger_agent, goat_agent=goat_agent, seed=seed)
                if rec.winner == TIGER:
                    t_wins += 1
                elif rec.winner == GOAT:
                    g_wins += 1
                else:
                    draws += 1
                plies_list.append(rec.total_moves)
                if rec.tiger_decision_times_ms:
                    t_times.append(np.mean(rec.tiger_decision_times_ms))
                if rec.goat_decision_times_ms:
                    g_times.append(np.mean(rec.goat_decision_times_ms))

            results["depth_grid"][cell_key] = {
                "d_tiger": d_tiger,
                "d_goat": d_goat,
                "tiger_win_rate": t_wins / rounds_per_cell,
                "goat_win_rate": g_wins / rounds_per_cell,
                "draw_rate": draws / rounds_per_cell,
                "avg_plies": float(np.mean(plies_list)),
                "avg_tiger_time_ms": float(np.mean(t_times)) if t_times else 0.0,
                "avg_goat_time_ms": float(np.mean(g_times)) if g_times else 0.0,
            }

            print(
                f"[{cell_key}] Tiger WR: {t_wins/rounds_per_cell:.2f} | "
                f"Goat WR: {g_wins/rounds_per_cell:.2f} | "
                f"Draw: {draws/rounds_per_cell:.2f} | "
                f"Avg Plies: {np.mean(plies_list):.1f} | "
                f"Lat: T={np.mean(t_times):.1f}ms, G={np.mean(g_times):.1f}ms"
            )

    # 2. MCTS Scaling vs Fixed Minimax Baseline
    print("\n--- Evaluating MCTS Simulation Scaling vs Minimax-d2 ---")
    mcts_sim_levels = [25, 50, 100]
    minimax_baseline_goat = MinimaxAgent("Minimax-d2-Goat", side=GOAT, max_depth=2)
    minimax_baseline_tiger = MinimaxAgent("Minimax-d2-Tiger", side=TIGER, max_depth=2)

    for sims in mcts_sim_levels:
        mcts_tiger = MCTSAgent(f"MCTS-{sims}-Tiger", side=TIGER, num_simulations=sims, seed=42)
        mcts_goat = MCTSAgent(f"MCTS-{sims}-Goat", side=GOAT, num_simulations=sims, seed=42)

        # MCTS as Tiger vs Minimax Goat
        t_wins = 0
        t_draws = 0
        for r in range(rounds_per_cell):
            rec = play_game(tiger_agent=mcts_tiger, goat_agent=minimax_baseline_goat, seed=8000 + r)
            if rec.winner == TIGER:
                t_wins += 1
            elif rec.winner == 0:
                t_draws += 1

        # Minimax Tiger vs MCTS as Goat
        g_wins = 0
        g_draws = 0
        for r in range(rounds_per_cell):
            rec = play_game(tiger_agent=minimax_baseline_tiger, goat_agent=mcts_goat, seed=8000 + r)
            if rec.winner == GOAT:
                g_wins += 1
            elif rec.winner == 0:
                g_draws += 1

        results["mcts_vs_minimax"][f"sims_{sims}"] = {
            "simulations": sims,
            "mcts_as_tiger_wr": t_wins / rounds_per_cell,
            "mcts_as_tiger_draw": t_draws / rounds_per_cell,
            "mcts_as_goat_wr": g_wins / rounds_per_cell,
            "mcts_as_goat_draw": g_draws / rounds_per_cell,
        }
        print(
            f"[MCTS {sims} Sims] vs Minimax-d2: Tiger WR={t_wins/rounds_per_cell:.2f} (Draw={t_draws/rounds_per_cell:.2f}) | "
            f"Goat WR={g_wins/rounds_per_cell:.2f} (Draw={g_draws/rounds_per_cell:.2f})"
        )

    summary_path = os.path.join(output_dir, "rg4_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRG4 Completed! Results saved to {summary_path}\n")
    return results


if __name__ == "__main__":
    run_rg4_experiment(depths=[1, 2, 3], rounds_per_cell=4)
