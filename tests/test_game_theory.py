"""
Unit Tests for Game Theory, PSRO, Statistical Metrics, and Asymmetric Parity MCTS.
"""

from dataclasses import replace
import pytest
import numpy as np

from src.environment.board import TIGER, GOAT
from src.environment.rules import BaghChalRules
from src.environment.env import BaghChalEnv
from src.agents.heuristic_tiger import GreedyTigerAgent, PositionalTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.agents.asymmetric_mcts import AsymmetricParityMCTSAgent
from src.game_theory.meta_solver import ZeroSumNashSolver, FictitiousPlaySolver
from src.game_theory.statistical_metrics import StatisticianMetrics
from src.game_theory.psro import PSROMetaEngine


def test_zero_sum_lp_matching_pennies():
    # Matching Pennies:
    # Row Player (Tiger) wins if match (+1), Col Player (Goat) wins if mismatch (-1)
    M = np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    p, q, val = ZeroSumNashSolver.solve_lp(M)
    assert np.allclose(p, [0.5, 0.5], atol=1e-3)
    assert np.allclose(q, [0.5, 0.5], atol=1e-3)
    assert pytest.approx(val, abs=1e-3) == 0.0

    exp = ZeroSumNashSolver.compute_exploitability(M, p, q)
    assert pytest.approx(exp, abs=1e-3) == 0.0


def test_zero_sum_lp_rock_paper_scissors():
    M = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])
    p, q, val = ZeroSumNashSolver.solve_lp(M)
    assert np.allclose(p, [1.0/3, 1.0/3, 1.0/3], atol=1e-3)
    assert np.allclose(q, [1.0/3, 1.0/3, 1.0/3], atol=1e-3)
    assert pytest.approx(val, abs=1e-3) == 0.0


def test_zero_sum_lp_biased_game():
    # Game where Row Player has dominant strategy (row 0 gives +2 or +3)
    M = np.array([
        [2.0, 3.0],
        [0.0, 1.0],
    ])
    p, q, val = ZeroSumNashSolver.solve_lp(M)
    assert pytest.approx(p[0], abs=1e-3) == 1.0
    assert pytest.approx(val, abs=1e-3) == 2.0


def test_fictitious_play_convergence():
    M = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])
    solver = FictitiousPlaySolver(M)
    res = solver.run(iterations=200)
    assert len(res["p_final"]) == 3
    assert len(res["q_final"]) == 3
    assert res["final_exploitability"] < 0.25


def test_statistical_metrics():
    # 1. Bootstrap CI
    data = [1.0, 0.8, 0.9, 1.0, 0.7, 0.95]
    mean, ci_low, ci_high = StatisticianMetrics.bootstrap_confidence_interval(data, num_bootstrap=500)
    assert ci_low <= mean <= ci_high

    # 2. Cohen's d
    g1 = [10.0, 11.0, 12.0, 10.5]
    g2 = [5.0, 6.0, 4.5, 5.5]
    d = StatisticianMetrics.cohens_d(g1, g2)
    assert d > 0.8  # Large effect size

    d_identical = StatisticianMetrics.cohens_d(g1, g1)
    assert pytest.approx(d_identical, abs=1e-4) == 0.0

    # 3. Wilson interval
    w_low, w_high = StatisticianMetrics.wilson_interval(8, 10)
    assert 0.0 <= w_low <= 0.8 <= w_high <= 1.0


def test_psro_engine_initialization_and_solve():
    tigers = [GreedyTigerAgent(name="Greedy-T", side=TIGER, seed=42)]
    goats = [DefensiveGoatAgent(name="Defensive-G", side=GOAT, seed=42)]

    psro = PSROMetaEngine(initial_tigers=tigers, initial_goats=goats, games_per_cell=1)
    assert psro.payoff_matrix.shape == (1, 1)

    sigma_t, sigma_g, val, exp = psro.solve_meta_game()
    assert len(sigma_t) == 1
    assert len(sigma_g) == 1
    assert pytest.approx(exp, abs=1e-4) == 0.0


def test_asymmetric_parity_mcts_selects_legal_action():
    env = BaghChalEnv()
    env.reset(seed=123)
    agent_tiger = AsymmetricParityMCTSAgent(name="AP-MCTS-Tiger", side=TIGER, num_simulations=15, seed=123)
    legal = env.get_legal_actions()
    action = agent_tiger.select_action(env.state, legal)
    assert action in legal


def test_asymmetric_parity_mcts_finds_instant_capture():
    # Setup state where tiger can capture goat on move 1
    env = BaghChalEnv()
    state = env.state
    # Place tiger at 0, goat at 1, empty at 2
    b = list(state.board)
    b[0] = TIGER
    b[1] = GOAT
    b[2] = 0
    state = replace(state, board=tuple(b), turn=TIGER)

    agent = AsymmetricParityMCTSAgent(name="AP-MCTS", side=TIGER, num_simulations=10)
    legal = [a for a in BaghChalRules.get_legal_actions(state) if a.action_type.name == "JUMP" and a.from_pos == 0]
    chosen = agent.select_action(state, legal)
    assert chosen.action_type.name == "JUMP"
    assert chosen.from_pos == 0 and chosen.to_pos == 2
