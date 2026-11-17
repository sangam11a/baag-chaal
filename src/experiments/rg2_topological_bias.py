"""
RG2: Topological Inductive Bias: GNN vs CNN vs Flat MLP on Irregular Game Boards.

Hypothesis: Because Bagh-Chal features irregular diagonal connectivity (only on even-parity vertices),
standard 2D CNNs suffer from spatial distortion / spurious diagonal correlations, whereas a
Graph Neural Network (GNN) with degree-normalized adjacency delivers superior sample efficiency,
lower policy cross-entropy loss, and higher competitive win rates against benchmark agents.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any

from src.environment.board import TIGER, GOAT
from src.environment.state import CANONICAL_ACTIONS, TOTAL_ACTIONS
from src.environment.env import BaghChalEnv
from src.training.ppo import PPOTrainer
from src.agents.neural_agent import NeuralAgent
from src.agents.heuristic_tiger import PositionalTigerAgent
from src.agents.heuristic_goat import BlockerGoatAgent
from src.evaluation.tournament import play_game


def evaluate_agent_against_heuristics(
    agent_model: torch.nn.Module,
    representation: str,
    num_eval_games: int = 10,
    seed_offset: int = 5000,
) -> Dict[str, float]:
    """
    Evaluates a trained neural agent against established domain heuristics:
    1. Neural Agent as Tiger vs Blocker-Goat
    2. Positional-Tiger vs Neural Agent as Goat
    """
    tiger_neural = NeuralAgent(
        name=f"Neural-{representation.upper()}-Tiger",
        side=TIGER,
        model=agent_model,
        representation=representation,
        temperature=0.05,
    )
    goat_neural = NeuralAgent(
        name=f"Neural-{representation.upper()}-Goat",
        side=GOAT,
        model=agent_model,
        representation=representation,
        temperature=0.05,
    )

    heuristic_tiger = PositionalTigerAgent(name="Benchmark-Tiger", side=TIGER, seed=seed_offset)
    heuristic_goat = BlockerGoatAgent(name="Benchmark-Goat", side=GOAT, seed=seed_offset)

    # 1. Neural Tiger vs Heuristic Goat
    t_wins = 0
    t_draws = 0
    for g in range(num_eval_games):
        rec = play_game(tiger_agent=tiger_neural, goat_agent=heuristic_goat, seed=seed_offset + g)
        if rec.winner == TIGER:
            t_wins += 1
        elif rec.winner == 0:
            t_draws += 1

    # 2. Heuristic Tiger vs Neural Goat
    g_wins = 0
    g_draws = 0
    for g in range(num_eval_games):
        rec = play_game(tiger_agent=heuristic_tiger, goat_agent=goat_neural, seed=seed_offset + g)
        if rec.winner == GOAT:
            g_wins += 1
        elif rec.winner == 0:
            g_draws += 1

    return {
        "neural_tiger_wr": t_wins / num_eval_games,
        "neural_tiger_draw_rate": t_draws / num_eval_games,
        "neural_goat_wr": g_wins / num_eval_games,
        "neural_goat_draw_rate": g_draws / num_eval_games,
        "overall_competitive_score": (t_wins + g_wins + 0.5 * (t_draws + g_draws)) / (2 * num_eval_games),
    }


def run_rg2_experiment(
    num_episodes: int = 40,
    seeds: List[int] = [42, 101, 202],
    output_dir: str = "data/experimental_results/rg2_topological_bias",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    representations = ["flat", "cnn", "gnn"]

    print("=" * 60)
    print("RG2: Executing Topological Inductive Bias Experiment (GNN vs CNN vs MLP)")
    print(f"Representations: {representations} | Episodes: {num_episodes} | Seeds: {seeds}")
    print("=" * 60)

    for rep in representations:
        results[rep] = {
            "training_losses": [],
            "policy_losses": [],
            "value_losses": [],
            "eval_tiger_wr": [],
            "eval_goat_wr": [],
            "competitive_scores": [],
        }

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            trainer = PPOTrainer(
                representation=rep,
                architecture="single",
                lr=3e-4,
                batch_size=32,
                ppo_epochs=3,
            )
            env = BaghChalEnv()

            losses = []
            p_losses = []
            v_losses = []

            # Training iterations
            for batch_idx in range(num_episodes // 10):
                ep_stats = trainer.collect_episodes(env, num_episodes=10)
                train_stats = trainer.train_step()
                losses.append(train_stats.get("loss", 0.0))
                p_losses.append(train_stats.get("policy_loss", 0.0))
                v_losses.append(train_stats.get("value_loss", 0.0))

            # Evaluate against benchmark heuristics
            eval_metrics = evaluate_agent_against_heuristics(
                agent_model=trainer.model,
                representation=rep,
                num_eval_games=10,
                seed_offset=seed,
            )

            results[rep]["training_losses"].append(float(np.mean(losses)))
            results[rep]["policy_losses"].append(float(np.mean(p_losses)))
            results[rep]["value_losses"].append(float(np.mean(v_losses)))
            results[rep]["eval_tiger_wr"].append(eval_metrics["neural_tiger_wr"])
            results[rep]["eval_goat_wr"].append(eval_metrics["neural_goat_wr"])
            results[rep]["competitive_scores"].append(eval_metrics["overall_competitive_score"])

            print(
                f"[{rep.upper()} | Seed {seed}] Loss: {results[rep]['training_losses'][-1]:.4f} | "
                f"Policy Loss: {results[rep]['policy_losses'][-1]:.4f} | "
                f"Eval Tiger WR: {eval_metrics['neural_tiger_wr']:.2f} | "
                f"Eval Goat WR: {eval_metrics['neural_goat_wr']:.2f} | "
                f"Competitive Score: {eval_metrics['overall_competitive_score']:.2f}"
            )

    summary_path = os.path.join(output_dir, "rg2_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRG2 Completed! Results saved to {summary_path}\n")
    return results


if __name__ == "__main__":
    run_rg2_experiment(num_episodes=30, seeds=[42, 101, 202])

