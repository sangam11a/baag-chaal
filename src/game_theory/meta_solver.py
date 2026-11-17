"""
Game-Theoretic Meta-Solvers for Asymmetric Zero-Sum Matrix Games.

Implements:
1. Exact Linear Programming (LP) Nash Equilibrium Solver via HiGHS
2. Nash Exploitability Computation
3. Fictitious Play (Iterative Best Response Dynamics)
4. Strategic Entropy & Policy Diversity Metrics
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linprog


class ZeroSumNashSolver:
    """
    Computes exact Minimax / Nash Equilibrium for two-player zero-sum matrix games.
    Payoff matrix M: (m, n) where row player (Tiger) maximizes and column player (Goat) minimizes.
    """

    @staticmethod
    def solve_lp(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solves zero-sum matrix game M via Linear Programming.
        Returns:
            p: optimal mixed strategy for Row Player (m,)
            q: optimal mixed strategy for Column Player (n,)
            game_value: game-theoretic value v*
        """
        m, n = M.shape
        if m == 0 or n == 0:
            raise ValueError("Payoff matrix must have non-zero dimensions.")

        if m == 1 and n == 1:
            return np.array([1.0]), np.array([1.0]), float(M[0, 0])

        # Shift payoffs to ensure all M_ij > 0 for standard zero-sum LP
        shift = float(np.min(M))
        c_shift = abs(shift) + 1.0 if shift <= 0 else 0.0
        M_shifted = M + c_shift

        # -------------------------------------------------------------
        # 1. Row Player (Maximizer):
        # min sum(x_i) s.t. M_shifted^T x >= 1, x >= 0
        # p_i = x_i / sum(x_i), v* = 1 / sum(x_i) - c_shift
        # -------------------------------------------------------------
        c_row = np.ones(m)
        A_ub_row = -M_shifted.T  # (n, m)
        b_ub_row = -np.ones(n)
        bounds_row = [(0.0, None) for _ in range(m)]

        res_row = linprog(c_row, A_ub=A_ub_row, b_ub=b_ub_row, bounds=bounds_row, method="highs")
        if not res_row.success:
            # Fallback to uniform if degenerate
            p = np.ones(m) / m
            v_val = float(p @ M @ (np.ones(n) / n))
        else:
            x = res_row.x
            sum_x = np.sum(x)
            p = x / sum_x if sum_x > 0 else np.ones(m) / m
            v_val = (1.0 / sum_x) - c_shift if sum_x > 0 else 0.0

        # -------------------------------------------------------------
        # 2. Column Player (Minimizer):
        # max sum(y_j) s.t. M_shifted y <= 1, y >= 0
        # or min -sum(y_j) s.t. M_shifted y <= 1, y >= 0
        # q_j = y_j / sum(y_j)
        # -------------------------------------------------------------
        c_col = -np.ones(n)
        A_ub_col = M_shifted  # (m, n)
        b_ub_col = np.ones(m)
        bounds_col = [(0.0, None) for _ in range(n)]

        res_col = linprog(c_col, A_ub=A_ub_col, b_ub=b_ub_col, bounds=bounds_col, method="highs")
        if not res_col.success:
            q = np.ones(n) / n
        else:
            y = res_col.x
            sum_y = np.sum(y)
            q = y / sum_y if sum_y > 0 else np.ones(n) / n

        # Clean numerical zero noise
        p = np.clip(p, 0.0, 1.0)
        p = p / np.sum(p)
        q = np.clip(q, 0.0, 1.0)
        q = q / np.sum(q)

        return p, q, float(v_val)

    @staticmethod
    def compute_exploitability(M: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        """
        Computes Nash Exploitability:
        E(p, q) = 0.5 * [(max_i (M q)_i - p^T M q) + (p^T M q - min_j (p^T M)_j)]
        Zero exploitability indicates an exact Nash Equilibrium.
        """
        expected_payoff = float(p @ M @ q)
        best_row_response_val = float(np.max(M @ q))
        best_col_response_val = float(np.min(p @ M))

        row_exploitability = best_row_response_val - expected_payoff
        col_exploitability = expected_payoff - best_col_response_val

        total_exploitability = 0.5 * (row_exploitability + col_exploitability)
        return max(0.0, float(total_exploitability))

    @staticmethod
    def compute_strategy_entropy(prob: np.ndarray) -> float:
        """
        Computes Shannon entropy of mixed strategy: H(p) = -sum p_i log(p_i).
        Measures strategy diversity / exploration breadth.
        """
        p_clean = prob[prob > 1e-8]
        if len(p_clean) == 0:
            return 0.0
        return float(-np.sum(p_clean * np.log(p_clean)))


class FictitiousPlaySolver:
    """
    Simulates Fictitious Play (Brown, 1951) to analyze iterative best-response convergence.
    """
    def __init__(self, M: np.ndarray):
        self.M = M
        self.m, self.n = M.shape

    def run(self, iterations: int = 1000) -> Dict[str, Any]:
        row_counts = np.zeros(self.m)
        col_counts = np.zeros(self.n)

        # Initialize with first strategy
        row_counts[0] = 1.0
        col_counts[0] = 1.0

        exploitability_trace = []
        game_values = []

        for t in range(1, iterations + 1):
            p = row_counts / t
            q = col_counts / t

            # Best responses to current empirical distributions
            row_br = int(np.argmax(self.M @ q))
            col_br = int(np.argmin(p @ self.M))

            row_counts[row_br] += 1.0
            col_counts[col_br] += 1.0

            if t % 50 == 0 or t == iterations:
                exp = ZeroSumNashSolver.compute_exploitability(self.M, p, q)
                val = float(p @ self.M @ q)
                exploitability_trace.append(exp)
                game_values.append(val)

        p_final = row_counts / iterations
        q_final = col_counts / iterations
        return {
            "p_final": p_final,
            "q_final": q_final,
            "final_game_value": float(p_final @ self.M @ q_final),
            "final_exploitability": ZeroSumNashSolver.compute_exploitability(self.M, p_final, q_final),
            "exploitability_trace": exploitability_trace,
            "game_values": game_values,
        }

