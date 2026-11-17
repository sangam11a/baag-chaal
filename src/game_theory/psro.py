"""
Policy-Space Response Oracles (PSRO) for Asymmetric Two-Player Zero-Sum Games.

Implements Lanctot et al. (NeurIPS 2017) / Balduzzi et al. (2019):
1. Maintains empirical populations of role-specialized policies: Pi_T and Pi_G
2. Evaluates empirical payoff matrix M via paired matches
3. Solves for Nash meta-distribution (sigma_T, sigma_G) via Linear Programming
4. Generates Best Responses (BR) against the mixed meta-strategy
5. Tracks Nash Exploitability decay across population expansion rounds
"""

from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.environment.board import TIGER, GOAT
from src.agents.base_agent import BaseAgent
from src.evaluation.tournament import play_paired_match
from src.game_theory.meta_solver import ZeroSumNashSolver


class PSROMetaEngine:
    """
    Empirical Game-Theoretic Analysis (EGTA) engine with Policy-Space Response Oracles.
    """
    def __init__(
        self,
        initial_tigers: List[BaseAgent],
        initial_goats: List[BaseAgent],
        games_per_cell: int = 4,
        base_seed: int = 7000,
    ):
        self.tiger_population: List[BaseAgent] = list(initial_tigers)
        self.goat_population: List[BaseAgent] = list(initial_goats)
        self.games_per_cell = games_per_cell
        self.base_seed = base_seed

        # Payoff matrix M: (num_tigers, num_goats)
        # Entry M[i, j] in [-1, +1] is expected payoff from Tiger's perspective (+1=Tiger win, -1=Goat win, 0=draw)
        self.payoff_matrix: np.ndarray = np.zeros((len(self.tiger_population), len(self.goat_population)))
        self._populate_full_matrix()

        # Track history of meta-game equilibria and exploitability
        self.history: List[Dict[str, Any]] = []

    def _evaluate_matchup(self, tiger: BaseAgent, goat: BaseAgent, seed_offset: int) -> float:
        """
        Runs paired evaluation matches between a tiger and goat agent.
        Returns average payoff in [-1.0, +1.0] from Tiger's zero-sum perspective.
        """
        payoffs = []
        for g in range(self.games_per_cell):
            seed = self.base_seed + seed_offset + g
            rec_a, rec_b = play_paired_match(tiger, goat, seed=seed)
            for rec in [rec_a, rec_b]:
                if rec.winner == TIGER:
                    payoffs.append(1.0)
                elif rec.winner == GOAT:
                    payoffs.append(-1.0)
                else:
                    payoffs.append(0.0)
        return float(np.mean(payoffs))

    def _populate_full_matrix(self) -> None:
        """
        Computes all entries of the empirical payoff matrix.
        """
        m, n = len(self.tiger_population), len(self.goat_population)
        self.payoff_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                offset = (i * n + j) * self.games_per_cell * 2
                self.payoff_matrix[i, j] = self._evaluate_matchup(
                    self.tiger_population[i], self.goat_population[j], seed_offset=offset
                )

    def solve_meta_game(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Solves the empirical payoff matrix for the current Nash Equilibrium.
        Returns:
            (sigma_T, sigma_G, game_value, exploitability)
        """
        sigma_t, sigma_g, val = ZeroSumNashSolver.solve_lp(self.payoff_matrix)
        exploitability = ZeroSumNashSolver.compute_exploitability(self.payoff_matrix, sigma_t, sigma_g)
        return sigma_t, sigma_g, val, exploitability

    def add_oracle_agents(
        self,
        new_tiger: Optional[BaseAgent] = None,
        new_goat: Optional[BaseAgent] = None,
    ) -> Dict[str, Any]:
        """
        Expands the empirical population with new response oracle agents.
        Evaluates only the newly added rows and columns to maintain efficiency.
        """
        old_m, old_n = self.payoff_matrix.shape

        if new_tiger is not None:
            self.tiger_population.append(new_tiger)
        if new_goat is not None:
            self.goat_population.append(new_goat)

        new_m, new_n = len(self.tiger_population), len(self.goat_population)
        new_M = np.zeros((new_m, new_n))
        # Copy over previously computed payoffs
        new_M[:old_m, :old_n] = self.payoff_matrix

        # Evaluate new tiger rows against all goats
        for i in range(old_m, new_m):
            for j in range(new_n):
                offset = (i * new_n + j) * self.games_per_cell * 2
                new_M[i, j] = self._evaluate_matchup(
                    self.tiger_population[i], self.goat_population[j], seed_offset=offset
                )

        # Evaluate all tigers against new goat columns
        for i in range(old_m):
            for j in range(old_n, new_n):
                offset = (i * new_n + j) * self.games_per_cell * 2
                new_M[i, j] = self._evaluate_matchup(
                    self.tiger_population[i], self.goat_population[j], seed_offset=offset
                )

        self.payoff_matrix = new_M
        sigma_t, sigma_g, val, exp = self.solve_meta_game()

        step_record = {
            "round": len(self.history) + 1,
            "population_size": (new_m, new_n),
            "game_value": val,
            "exploitability": exp,
            "tiger_strategy": sigma_t.tolist(),
            "goat_strategy": sigma_g.tolist(),
            "tiger_names": [a.name for a in self.tiger_population],
            "goat_names": [a.name for a in self.goat_population],
        }
        self.history.append(step_record)
        return step_record

