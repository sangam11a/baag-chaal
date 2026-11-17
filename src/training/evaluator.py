"""
Standardized Evaluation Engine for Trained Neural Policies in Bagh-Chal.

Evaluates neural agents against:
- PositionalTigerAgent / GreedyTigerAgent
- DefensiveGoatAgent / BlockerGoatAgent
- Minimax-d2 Agent

Computes:
- Tiger Win Rate, Goat Win Rate, Draw Rate
- Average Game Plies
- Competitive Score (CS) = (W + 0.5 * D) / N
- Wilson Score 95% Confidence Intervals
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple
import torch

from src.environment.board import TIGER, GOAT
from src.agents.neural_agent import NeuralAgent
from src.agents.heuristic_tiger import PositionalTigerAgent, GreedyTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.agents.minimax_agent import MinimaxAgent
from src.evaluation.tournament import play_game


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score 95% confidence interval for a binomial proportion.
    """
    if total == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 1.645
    p = successes / total
    denominator = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denominator
    margin = (z * math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * total)) / total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


class NeuralEvaluator:
    """
    Orchestrates standardized evaluation of neural models during or post-training.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device

    def evaluate_model(
        self,
        model: torch.nn.Module,
        representation: str = "gnn",
        architecture: str = "single",
        num_games_per_opponent: int = 10,
        base_seed: int = 5000,
        include_minimax: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation against the benchmark pool.
        """
        model.eval()

        # Build Neural Agents for Tiger and Goat
        if architecture == "single":
            tiger_neural = NeuralAgent(
                "Neural-Tiger", side=TIGER, model=model, representation=representation,
                temperature=0.0, device=self.device
            )
            goat_neural = NeuralAgent(
                "Neural-Goat", side=GOAT, model=model, representation=representation,
                temperature=0.0, device=self.device
            )
        else:
            tiger_neural = NeuralAgent(
                "Dual-Tiger", side=TIGER, model=model.tiger_net, representation=representation,
                temperature=0.0, device=self.device
            )
            goat_neural = NeuralAgent(
                "Dual-Goat", side=GOAT, model=model.goat_net, representation=representation,
                temperature=0.0, device=self.device
            )

        # Benchmark opponents
        tiger_opponents = [
            PositionalTigerAgent("Benchmark-PositionalTiger", side=TIGER, seed=base_seed),
            GreedyTigerAgent("Benchmark-GreedyTiger", side=TIGER, seed=base_seed + 1),
        ]
        goat_opponents = [
            DefensiveGoatAgent("Benchmark-DefensiveGoat", side=GOAT, seed=base_seed),
            BlockerGoatAgent("Benchmark-BlockerGoat", side=GOAT, seed=base_seed + 1),
        ]

        if include_minimax:
            tiger_opponents.append(MinimaxAgent("Benchmark-Minimax-d2-Tiger", side=TIGER, max_depth=2))
            goat_opponents.append(MinimaxAgent("Benchmark-Minimax-d2-Goat", side=GOAT, max_depth=2))

        # 1. Evaluate Neural as Tiger vs Benchmark Goats
        t_wins = 0
        t_losses = 0
        t_draws = 0
        t_plies = []

        total_t_games = 0
        for g_opp in goat_opponents:
            for g in range(num_games_per_opponent):
                seed = base_seed + total_t_games
                rec = play_game(tiger_agent=tiger_neural, goat_agent=g_opp, seed=seed)
                if rec.winner == TIGER:
                    t_wins += 1
                elif rec.winner == GOAT:
                    t_losses += 1
                else:
                    t_draws += 1
                t_plies.append(rec.total_moves)
                total_t_games += 1

        # 2. Evaluate Neural as Goat vs Benchmark Tigers
        g_wins = 0
        g_losses = 0
        g_draws = 0
        g_plies = []

        total_g_games = 0
        for t_opp in tiger_opponents:
            for g in range(num_games_per_opponent):
                seed = base_seed + 1000 + total_g_games
                rec = play_game(tiger_agent=t_opp, goat_agent=goat_neural, seed=seed)
                if rec.winner == GOAT:
                    g_wins += 1
                elif rec.winner == TIGER:
                    g_losses += 1
                else:
                    g_draws += 1
                g_plies.append(rec.total_moves)
                total_g_games += 1

        total_games = total_t_games + total_g_games
        overall_wins = t_wins + g_wins
        overall_draws = t_draws + g_draws
        competitive_score = (overall_wins + 0.5 * overall_draws) / max(1, total_games)

        t_wr = t_wins / max(1, total_t_games)
        g_wr = g_wins / max(1, total_g_games)
        t_ci_low, t_ci_high = wilson_score_interval(t_wins, total_t_games)
        g_ci_low, g_ci_high = wilson_score_interval(g_wins, total_g_games)

        return {
            "total_games": total_games,
            "neural_as_tiger": {
                "wins": t_wins,
                "losses": t_losses,
                "draws": t_draws,
                "win_rate": t_wr,
                "draw_rate": t_draws / max(1, total_t_games),
                "avg_plies": float(sum(t_plies) / max(1, len(t_plies))),
                "ci_95": (t_ci_low, t_ci_high),
            },
            "neural_as_goat": {
                "wins": g_wins,
                "losses": g_losses,
                "draws": g_draws,
                "win_rate": g_wr,
                "draw_rate": g_draws / max(1, total_g_games),
                "avg_plies": float(sum(g_plies) / max(1, len(g_plies))),
                "ci_95": (g_ci_low, g_ci_high),
            },
            "competitive_score": competitive_score,
            "tiger_bias_index": t_wr - g_wr,
            "role_balance_index": 1.0 - abs(t_wr - g_wr) / max(t_wr, g_wr, 1e-6),
        }
