"""
RG3: Policy Representation in Role-Asymmetric Games: Single vs Dual Network Architectures.

Hypothesis: In strongly asymmetric games, a single unified policy network suffers from
representational interference (gradient conflict between predator attack policies and
prey encirclement policies). A Dual-Network architecture (role-specialized weights)
achieves higher training stability, greater role balance (RBI), and superior head-to-head
win rates compared to single-network models under identical training iterations.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Any

from src.environment.board import TIGER, GOAT
from src.environment.env import BaghChalEnv
from src.training.ppo import PPOTrainer
from src.agents.neural_agent import NeuralAgent
from src.evaluation.tournament import play_paired_match


def run_rg3_experiment(
    num_episodes: int = 40,
    seeds: List[int] = [42, 101, 202],
    output_dir: str = "data/experimental_results/rg3_architectural_asymmetry",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "single": {"train_losses": [], "tbi": [], "rbi": []},
        "dual": {"train_losses": [], "tbi": [], "rbi": []},
        "head_to_head": {
            "single_as_tiger_wins": 0,
            "dual_as_tiger_wins": 0,
            "draws": 0,
            "total_games": 0,
        },
    }

    print("=" * 60)
    print("RG3: Executing Architectural Asymmetry Experiment (Single vs Dual Network)")
    print(f"Episodes: {num_episodes} | Seeds: {seeds}")
    print("=" * 60)

    for arch in ["single", "dual"]:
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            trainer = PPOTrainer(
                representation="flat",
                architecture=arch,
                lr=3e-4,
                batch_size=32,
                ppo_epochs=3,
            )
            env = BaghChalEnv()

            losses = []
            tiger_wins = 0
            goat_wins = 0
            draws = 0

            for batch_idx in range(num_episodes // 10):
                ep_stats = trainer.collect_episodes(env, num_episodes=10)
                train_stats = trainer.train_step()
                losses.append(train_stats.get("loss", 0.0))

                tiger_wins += ep_stats["tiger_win_rate"] * 10
                goat_wins += ep_stats["goat_win_rate"] * 10
                draws += ep_stats["draw_rate"] * 10

            tw_rate = tiger_wins / num_episodes
            gw_rate = goat_wins / num_episodes
            tbi = tw_rate - gw_rate
            max_wr = max(tw_rate, gw_rate, 1e-6)
            rbi = 1.0 - (abs(tw_rate - gw_rate) / max_wr)

            results[arch]["train_losses"].append(float(np.mean(losses)))
            results[arch]["tbi"].append(tbi)
            results[arch]["rbi"].append(rbi)

            print(
                f"[{arch.upper()} | Seed {seed}] Loss: {results[arch]['train_losses'][-1]:.4f} | "
                f"Tiger WR: {tw_rate:.2f} | Goat WR: {gw_rate:.2f} | RBI: {rbi:.2f}"
            )

    # Direct Head-to-Head Evaluation: Single-Net vs Dual-Net
    print("\n--- Running Head-to-Head Tournament: Single-Net vs Dual-Net ---")
    trainer_single = PPOTrainer(representation="flat", architecture="single", lr=3e-4)
    trainer_dual = PPOTrainer(representation="flat", architecture="dual", lr=3e-4)

    env = BaghChalEnv()
    trainer_single.collect_episodes(env, num_episodes=20)
    trainer_single.train_step()
    trainer_dual.collect_episodes(env, num_episodes=20)
    trainer_dual.train_step()

    # Create Neural Agents
    # For single network:
    agent_single_tiger = NeuralAgent("Single-Tiger", side=TIGER, model=trainer_single.model, representation="flat")
    agent_single_goat = NeuralAgent("Single-Goat", side=GOAT, model=trainer_single.model, representation="flat")

    # For dual network:
    agent_dual_tiger = NeuralAgent("Dual-Tiger", side=TIGER, model=trainer_dual.model.tiger_net, representation="flat")
    agent_dual_goat = NeuralAgent("Dual-Goat", side=GOAT, model=trainer_dual.model.goat_net, representation="flat")

    # Paired matches:
    # Match A: Single as Tiger, Dual as Goat
    # Match B: Dual as Tiger, Single as Goat
    num_matches = 15
    single_wins = 0
    dual_wins = 0
    draws_total = 0

    for m in range(num_matches):
        seed_m = 9000 + m
        # Match A: Single Tiger vs Dual Goat
        rec_a, rec_b = play_paired_match(agent_single_tiger, agent_dual_goat, seed=seed_m)
        if rec_a.winner == TIGER:
            single_wins += 1
        elif rec_a.winner == GOAT:
            dual_wins += 1
        else:
            draws_total += 1

        # Match B: Dual Tiger vs Single Goat
        rec_c, rec_d = play_paired_match(agent_dual_tiger, agent_single_goat, seed=seed_m)
        if rec_c.winner == TIGER:
            dual_wins += 1
        elif rec_c.winner == GOAT:
            single_wins += 1
        else:
            draws_total += 1

    total_games = num_matches * 2
    results["head_to_head"]["single_wins"] = single_wins
    results["head_to_head"]["dual_wins"] = dual_wins
    results["head_to_head"]["draws"] = draws_total
    results["head_to_head"]["total_games"] = total_games
    results["head_to_head"]["dual_win_rate"] = dual_wins / total_games
    results["head_to_head"]["single_win_rate"] = single_wins / total_games

    print(
        f"Head-to-Head Results ({total_games} games): Dual Wins={dual_wins} ({dual_wins/total_games:.1%}) | "
        f"Single Wins={single_wins} ({single_wins/total_games:.1%}) | Draws={draws_total} ({draws_total/total_games:.1%})"
    )

    summary_path = os.path.join(output_dir, "rg3_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRG3 Completed! Results saved to {summary_path}\n")
    return results


if __name__ == "__main__":
    run_rg3_experiment(num_episodes=30, seeds=[42, 101, 202])

