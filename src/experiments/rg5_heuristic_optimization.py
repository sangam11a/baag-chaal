"""
RG5: Handcrafted Expert Heuristics vs Metaheuristic Parameter Optimization.

Investigates whether evolutionary optimization (genetic algorithm) of domain evaluation
weights outperforms handcrafted heuristics and matches deep neural policies at 1000x lower latency.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any

from src.environment.board import NUM_POINTS, EMPTY, TIGER, GOAT, ADJACENCY_LIST, JUMPS_BY_SOURCE, pos_to_coord
from src.environment.state import BaghChalState
from src.agents.minimax_agent import MinimaxAgent
from src.agents.heuristic_tiger import PositionalTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.evaluation.tournament import play_game


def make_custom_eval_fn(weights: List[float]):
    """
    weights: [w_cap, w_mob, w_trapped, w_center, w_cluster]
    """
    w_cap, w_mob, w_trapped, w_center, w_cluster = weights

    def eval_fn(state: BaghChalState) -> float:
        material = state.goats_captured * w_cap
        b = state.board
        total_mobility = 0
        tigers_trapped = 0
        tiger_center_dist = 0

        for pos in range(NUM_POINTS):
            if b[pos] == TIGER:
                mob = 0
                for nbr in ADJACENCY_LIST[pos]:
                    if b[nbr] == EMPTY:
                        mob += 1
                for mid, dest in JUMPS_BY_SOURCE[pos]:
                    if b[mid] == GOAT and b[dest] == EMPTY:
                        mob += 3
                if mob == 0:
                    tigers_trapped += 1
                total_mobility += mob

                r, c = pos_to_coord(pos)
                tiger_center_dist += abs(r - 2) + abs(c - 2)

        goat_clusters = 0
        for pos in range(NUM_POINTS):
            if b[pos] == GOAT:
                for nbr in ADJACENCY_LIST[pos]:
                    if b[nbr] == GOAT:
                        goat_clusters += 1

        score = (
            material
            + total_mobility * w_mob
            - tigers_trapped * w_trapped
            - tiger_center_dist * w_center
            - goat_clusters * w_cluster
        )
        return score

    return eval_fn


def evaluate_weight_vector(weights: List[float], num_test_games: int = 4) -> float:
    """
    Evaluate fitness of weight vector:
    Plays as Tiger vs DefensiveGoat and BlockerGoat, and as Goat vs PositionalTiger.
    """
    custom_fn = make_custom_eval_fn(weights)
    test_agent_tiger = MinimaxAgent("Custom-Tiger", side=TIGER, max_depth=2, eval_fn=custom_fn)
    test_agent_goat = MinimaxAgent("Custom-Goat", side=GOAT, max_depth=2, eval_fn=custom_fn)

    def_goat = DefensiveGoatAgent(seed=42)
    pos_tiger = PositionalTigerAgent(seed=42)

    fitness = 0.0
    for g in range(num_test_games):
        # As Tiger
        rec1 = play_game(test_agent_tiger, def_goat, seed=1000 + g)
        if rec1.winner == TIGER:
            fitness += 1.0
        elif rec1.winner == 0:
            fitness += 0.5

        # As Goat
        rec2 = play_game(pos_tiger, test_agent_goat, seed=2000 + g)
        if rec2.winner == GOAT:
            fitness += 1.0
        elif rec2.winner == 0:
            fitness += 0.5

    return fitness / (2.0 * num_test_games)


def run_rg5_experiment(
    population_size: int = 8,
    generations: int = 4,
    output_dir: str = "data/experimental_results/rg5_heuristic_optimization",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("RG5: Executing Metaheuristic Parameter Optimization Experiment")
    print(f"Population: {population_size} | Generations: {generations}")
    print("=" * 60)

    # Initial default handcrafted weights: [2500, 40, 1500, 15, 5]
    default_weights = [2500.0, 40.0, 1500.0, 15.0, 5.0]
    default_fitness = evaluate_weight_vector(default_weights, num_test_games=6)
    print(f"Handcrafted Default Weights Fitness: {default_fitness:.3f}")

    # Initialize population around default weights with perturbations
    random.seed(42)
    np.random.seed(42)

    population = [list(default_weights)]
    for _ in range(population_size - 1):
        ind = [
            max(100.0, default_weights[0] * random.uniform(0.5, 2.0)),
            max(1.0, default_weights[1] * random.uniform(0.5, 2.0)),
            max(100.0, default_weights[2] * random.uniform(0.5, 2.0)),
            max(1.0, default_weights[3] * random.uniform(0.5, 2.0)),
            max(1.0, default_weights[4] * random.uniform(0.5, 2.0)),
        ]
        population.append(ind)

    generation_history = []
    best_weights = list(default_weights)
    best_fitness = default_fitness

    for gen in range(generations):
        fitnesses = [evaluate_weight_vector(ind, num_test_games=4) for ind in population]
        best_gen_idx = int(np.argmax(fitnesses))
        best_gen_fit = fitnesses[best_gen_idx]

        if best_gen_fit > best_fitness:
            best_fitness = best_gen_fit
            best_weights = list(population[best_gen_idx])

        generation_history.append({
            "generation": gen,
            "best_fitness": best_gen_fit,
            "mean_fitness": float(np.mean(fitnesses)),
            "best_weights": list(population[best_gen_idx]),
        })

        print(f"Gen {gen + 1}/{generations}: Best Fitness={best_gen_fit:.3f} | Mean={np.mean(fitnesses):.3f}")

        # Selection & Mutation for next gen
        sorted_indices = np.argsort(fitnesses)[::-1]
        survivors = [population[i] for i in sorted_indices[:population_size // 2]]

        next_pop = [list(w) for w in survivors]
        while len(next_pop) < population_size:
            parent = random.choice(survivors)
            child = [
                val * random.uniform(0.8, 1.25) for val in parent
            ]
            next_pop.append(child)
        population = next_pop

    # Final validation tournament: Evolved vs Default Handcrafted
    print("\n--- Final Head-to-Head Tournament: Evolved Heuristic vs Handcrafted Default ---")
    evolved_fn = make_custom_eval_fn(best_weights)
    default_fn = make_custom_eval_fn(default_weights)

    evolved_tiger = MinimaxAgent("Evolved-Tiger", side=TIGER, max_depth=2, eval_fn=evolved_fn)
    default_goat = MinimaxAgent("Default-Goat", side=GOAT, max_depth=2, eval_fn=default_fn)

    default_tiger = MinimaxAgent("Default-Tiger", side=TIGER, max_depth=2, eval_fn=default_fn)
    evolved_goat = MinimaxAgent("Evolved-Goat", side=GOAT, max_depth=2, eval_fn=evolved_fn)

    evolved_wins = 0
    default_wins = 0
    draws = 0
    n_validation = 10

    # Match A: Evolved Tiger vs Default Goat
    for i in range(n_validation):
        rec = play_game(evolved_tiger, default_goat, seed=3000 + i)
        if rec.winner == TIGER:
            evolved_wins += 1
        elif rec.winner == GOAT:
            default_wins += 1
        else:
            draws += 1

    # Match B: Default Tiger vs Evolved Goat
    for i in range(n_validation):
        rec = play_game(default_tiger, evolved_goat, seed=4000 + i)
        if rec.winner == GOAT:
            evolved_wins += 1
        elif rec.winner == TIGER:
            default_wins += 1
        else:
            draws += 1

    total_matches = 2 * n_validation
    print(
        f"Validation Results ({total_matches} games): Evolved Wins={evolved_wins} | "
        f"Default Wins={default_wins} | Draws={draws}"
    )

    results = {
        "handcrafted_default_weights": default_weights,
        "handcrafted_fitness": default_fitness,
        "best_evolved_weights": best_weights,
        "best_evolved_fitness": best_fitness,
        "generation_history": generation_history,
        "validation_head_to_head": {
            "total_games": total_matches,
            "evolved_wins": evolved_wins,
            "default_wins": default_wins,
            "draws": draws,
            "evolved_win_rate": evolved_wins / total_matches,
        },
    }

    summary_path = os.path.join(output_dir, "rg5_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRG5 Completed! Results saved to {summary_path}\n")
    return results


if __name__ == "__main__":
    run_rg5_experiment(population_size=6, generations=3)

