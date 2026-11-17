"""
Benchmark Tournament Script for Baseline and Heuristic Bagh-Chal Agents.
"""

import os
import argparse
from typing import List, Dict
from collections import defaultdict

from src.environment.board import TIGER, GOAT
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_tiger import GreedyTigerAgent, PositionalTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.agents.minimax_agent import MinimaxAgent
from src.evaluation.tournament import play_game
from src.evaluation.logger import GameLogger


def run_benchmark(rounds_per_matchup: int = 10, log_dir: str = "data/benchmark_results"):
    print(f"=== Starting Bagh-Chal Benchmark Tournament ({rounds_per_matchup} rounds/matchup) ===")
    logger = GameLogger(log_dir)

    tiger_agents = [
        RandomAgent(name="Random-Tiger", side=TIGER),
        GreedyTigerAgent(name="Greedy-Tiger", side=TIGER),
        PositionalTigerAgent(name="Positional-Tiger", side=TIGER),
        MinimaxAgent(name="Minimax-d2-Tiger", side=TIGER, max_depth=2),
    ]

    goat_agents = [
        RandomAgent(name="Random-Goat", side=GOAT),
        DefensiveGoatAgent(name="Defensive-Goat", side=GOAT),
        BlockerGoatAgent(name="Blocker-Goat", side=GOAT),
        MinimaxAgent(name="Minimax-d2-Goat", side=GOAT, max_depth=2),
    ]

    # Stats tracking
    tiger_wins = defaultdict(int)
    goat_wins = defaultdict(int)
    draws = defaultdict(int)
    total_games = 0

    for tiger in tiger_agents:
        for goat in goat_agents:
            print(f"\n--- Matchup: {tiger.name} vs {goat.name} ---")
            for r in range(rounds_per_matchup):
                seed = 1000 + r
                tiger.rng = None if not hasattr(tiger, "rng") or tiger.rng is None else tiger.rng
                rec = play_game(tiger_agent=tiger, goat_agent=goat, seed=seed)
                logger.log_game(rec)
                total_games += 1

                if rec.winner == TIGER:
                    tiger_wins[(tiger.name, goat.name)] += 1
                elif rec.winner == GOAT:
                    goat_wins[(tiger.name, goat.name)] += 1
                else:
                    draws[(tiger.name, goat.name)] += 1

            t_w = tiger_wins[(tiger.name, goat.name)]
            g_w = goat_wins[(tiger.name, goat.name)]
            d_w = draws[(tiger.name, goat.name)]
            print(f"Result ({rounds_per_matchup} games): Tiger wins={t_w}, Goat wins={g_w}, Draws={d_w}")

    print("\n" + "="*50)
    print(f"Benchmark Complete! Total games played: {total_games}")
    print(f"Results archived to {log_dir}/tournament_summary.csv")
    print("="*50)


if __name__ == "__main__":
    run_benchmark(rounds_per_matchup=5)

