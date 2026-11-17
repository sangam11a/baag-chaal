"""
RG1: Asymmetric Credit Assignment & Reward Sparsity in Bagh-Chal Self-Play.

Investigates:
1. Sparse Terminal Reward
2. Naive Dense Reward
3. Potential-Based Reward Shaping (PBRS) - Ng et al. (1999) policy-invariant formulation

Hypothesis: Naive dense rewards induce sub-optimal local equilibria (tiger capture fixation
without encirclement defense), whereas PBRS stabilizes goat survival and accelerates
equilibration without distorting the optimal policy.
"""

from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from src.environment.board import NUM_POINTS, EMPTY, TIGER, GOAT, ADJACENCY_LIST, JUMPS_BY_SOURCE
from src.environment.state import BaghChalState, Action, TOTAL_ACTIONS, CANONICAL_ACTIONS
from src.environment.rules import BaghChalRules
from src.environment.env import BaghChalEnv
from src.training.networks import MLPPolicyValueNet
from src.training.ppo import PPOTrainer, RolloutBuffer


def compute_state_potential(state: BaghChalState, role: int) -> float:
    """
    Computes potential function Phi(s) for role.
    Ng et al. (1999) Potential-Based Reward Shaping.
    """
    b = state.board
    total_mobility = 0
    tigers_trapped = 0
    for pos in range(NUM_POINTS):
        if b[pos] == TIGER:
            mob = 0
            for nbr in ADJACENCY_LIST[pos]:
                if b[nbr] == EMPTY:
                    mob += 1
            for mid, dest in JUMPS_BY_SOURCE[pos]:
                if b[mid] == GOAT and b[dest] == EMPTY:
                    mob += 2
            if mob == 0:
                tigers_trapped += 1
            total_mobility += mob

    goat_clusters = 0
    for pos in range(NUM_POINTS):
        if b[pos] == GOAT:
            for nbr in ADJACENCY_LIST[pos]:
                if b[nbr] == GOAT:
                    goat_clusters += 1

    if role == TIGER:
        # Tiger potential: high when goats captured, high mobility, low trapped tigers
        phi = (
            (state.goats_captured / 5.0) * 1.0
            + (total_mobility / 30.0) * 0.3
            - (tigers_trapped / 4.0) * 0.5
        )
    else:
        # Goat potential: high when few goats captured, low tiger mobility, high trapped tigers, high clustering
        phi = (
            ((5 - state.goats_captured) / 5.0) * 1.0
            - (total_mobility / 30.0) * 0.4
            + (tigers_trapped / 4.0) * 0.8
            + (goat_clusters / 40.0) * 0.2
        )
    return phi


def run_rg1_experiment(
    num_episodes: int = 60,
    seeds: List[int] = [42, 101, 202],
    output_dir: str = "data/experimental_results/rg1_credit_assignment",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    reward_modes = ["sparse", "naive_dense", "pbrs"]
    gamma = 0.99

    print("=" * 60)
    print("RG1: Executing Asymmetric Credit Assignment Experiment")
    print(f"Modes: {reward_modes} | Episodes: {num_episodes} | Seeds: {seeds}")
    print("=" * 60)

    for mode in reward_modes:
        results[mode] = {
            "tiger_win_rates": [],
            "goat_win_rates": [],
            "draw_rates": [],
            "avg_plies": [],
            "policy_losses": [],
            "value_losses": [],
            "final_tiger_bias": [],
        }

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            trainer = PPOTrainer(
                representation="flat",
                architecture="single",
                lr=3e-4,
                gamma=gamma,
                batch_size=32,
                ppo_epochs=3,
            )
            env = BaghChalEnv()

            ep_plies = []
            tiger_wins = 0
            goat_wins = 0
            draws = 0

            # Collect episodes with specific reward scheme
            for ep in range(num_episodes):
                env.reset(seed=seed + ep)
                done = False
                plies = 0

                prev_state = env.state
                prev_turn = prev_state.turn

                while not done:
                    legal = env.get_legal_actions()
                    mask_np = env.get_action_mask()
                    mask_tensor = torch.from_numpy(mask_np).to(trainer.device).unsqueeze(0)
                    acting_turn = env.state.turn

                    logits, value, raw_obs = trainer._get_obs_and_forward(env, mask_tensor)
                    logits = logits.squeeze(0)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action_idx, device=trainer.device)).item()

                    act = CANONICAL_ACTIONS[action_idx]
                    old_state = env.state
                    old_cap = old_state.goats_captured

                    _, terminal_rew, term, trunc, info = env.step(act)
                    done = term or trunc
                    new_state = env.state
                    plies += 1

                    # Compute custom reward based on mode
                    if mode == "sparse":
                        reward = terminal_rew
                    elif mode == "naive_dense":
                        reward = terminal_rew
                        if acting_turn == TIGER and new_state.goats_captured > old_cap:
                            reward += 0.2
                        elif acting_turn == GOAT:
                            reward += 0.005
                    elif mode == "pbrs":
                        # Ng et al. 1999: F = gamma * Phi(s') - Phi(s)
                        phi_old = compute_state_potential(old_state, acting_turn)
                        phi_new = compute_state_potential(new_state, acting_turn) if not done else 0.0
                        shaping_bonus = gamma * phi_new - phi_old
                        reward = terminal_rew + 0.05 * shaping_bonus

                    trainer.buffer.observations.append(raw_obs)
                    trainer.buffer.actions.append(action_idx)
                    trainer.buffer.log_probs.append(log_prob)
                    trainer.buffer.rewards.append(reward)
                    trainer.buffer.values.append(value.item())
                    trainer.buffer.masks.append(mask_np)
                    trainer.buffer.dones.append(done)
                    trainer.buffer.roles.append(acting_turn)

                ep_plies.append(plies)
                if info["winner"] == TIGER:
                    tiger_wins += 1
                elif info["winner"] == GOAT:
                    goat_wins += 1
                else:
                    draws += 1

                # Train periodically
                if (ep + 1) % 15 == 0:
                    train_stats = trainer.train_step()
                    trainer.buffer.clear()

            tw_rate = tiger_wins / num_episodes
            gw_rate = goat_wins / num_episodes
            dr_rate = draws / num_episodes
            avg_p = float(np.mean(ep_plies))
            tbi = tw_rate - gw_rate  # Tiger Bias Index: +1 all tiger, -1 all goat, 0 balanced

            results[mode]["tiger_win_rates"].append(tw_rate)
            results[mode]["goat_win_rates"].append(gw_rate)
            results[mode]["draw_rates"].append(dr_rate)
            results[mode]["avg_plies"].append(avg_p)
            results[mode]["final_tiger_bias"].append(tbi)

            print(
                f"[{mode.upper()} | Seed {seed}] Tiger WR: {tw_rate:.2f} | Goat WR: {gw_rate:.2f} | "
                f"Draw: {dr_rate:.2f} | Avg Plies: {avg_p:.1f} | Tiger Bias: {tbi:+.2f}"
            )

    # Save summary json
    summary_path = os.path.join(output_dir, "rg1_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRG1 Completed! Results saved to {summary_path}\n")
    return results


if __name__ == "__main__":
    run_rg1_experiment(num_episodes=45, seeds=[42, 101, 202])

