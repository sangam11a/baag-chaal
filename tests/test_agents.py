"""
Unit tests for Bagh-Chal Baseline and Heuristic Agents:
Random, Greedy-Tiger, Positional-Tiger, Defensive-Goat, Blocker-Goat, and Minimax-AlphaBeta.
"""

import pytest
from src.environment.board import (
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    coord_to_pos,
)
from src.environment.state import (
    BaghChalState,
    Action,
    ActionType,
    Phase,
)
from src.environment.rules import BaghChalRules
from src.environment.env import BaghChalEnv
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_tiger import GreedyTigerAgent, PositionalTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.mcts_agent import MCTSAgent


def test_random_agent_selects_legal_action():
    agent = RandomAgent(seed=42)
    s = BaghChalState.initial_state()
    legal = BaghChalRules.get_legal_actions(s)
    act = agent.select_action(s, legal)
    assert act in legal


def test_greedy_tiger_takes_immediate_capture():
    agent = GreedyTigerAgent(seed=42)
    # Setup state with Tiger at 0, Goat at 1, Empty at 2
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = GOAT
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=19,
        goats_captured=0,
        move_count=1,
        plies_since_capture=1,
        zobrist_hash=0,
    )
    legal = BaghChalRules.get_legal_actions(s)
    chosen_act = agent.select_action(s, legal)
    assert chosen_act.action_type == ActionType.JUMP
    assert chosen_act.from_pos == 0
    assert chosen_act.captured_pos == 1
    assert chosen_act.to_pos == 2


def test_defensive_goat_avoids_suicidal_placements():
    agent = DefensiveGoatAgent(seed=42)
    # State where Tiger at 0 can jump over 1 if 1 has a goat and 2 is empty
    # Goat should avoid placing at 1!
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=GOAT,
        phase=Phase.PLACEMENT,
        goats_unplaced=20,
        goats_captured=0,
        move_count=0,
        plies_since_capture=0,
        zobrist_hash=0,
    )
    legal = BaghChalRules.get_legal_actions(s)
    act = agent.select_action(s, legal)
    assert act in legal

    # Verify that the placement does not allow immediate jump on next turn
    next_s = s.apply_action(act)
    tiger_actions = BaghChalRules.get_legal_actions(next_s)
    jumps = [a for a in tiger_actions if a.action_type == ActionType.JUMP]
    assert len(jumps) == 0, f"Defensive goat allowed immediate jump {jumps} with placement {act}"


def test_minimax_finds_instant_winning_capture():
    # Tiger has captured 4 goats; an immediate jump capture is available for the 5th
    agent = MinimaxAgent(side=TIGER, max_depth=2)
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = GOAT
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=15,
        goats_captured=4,  # One more to win!
        move_count=9,
        plies_since_capture=1,
        zobrist_hash=0,
    )
    legal = BaghChalRules.get_legal_actions(s)
    chosen = agent.select_action(s, legal)

    assert chosen.action_type == ActionType.JUMP
    assert chosen.captured_pos == 1
    # Check that after this action, tiger wins immediately
    next_s = s.apply_action(chosen)
    is_term, winner, _ = BaghChalRules.get_game_status(next_s)
    assert is_term is True
    assert winner == TIGER


def test_minimax_depth_node_scaling():
    agent_d1 = MinimaxAgent(side=TIGER, max_depth=1)
    agent_d2 = MinimaxAgent(side=TIGER, max_depth=2)

    s = BaghChalState.initial_state()
    # Goat places piece
    s = s.apply_action(Action(ActionType.PLACE, from_pos=None, to_pos=12))

    legal = BaghChalRules.get_legal_actions(s)
    agent_d1.select_action(s, legal)
    nodes_d1 = agent_d1.nodes_searched

    agent_d2.select_action(s, legal)
    nodes_d2 = agent_d2.nodes_searched

    assert nodes_d2 > nodes_d1, f"Depth 2 nodes ({nodes_d2}) should exceed Depth 1 nodes ({nodes_d1})"


def test_agent_versus_agent_smoke_match():
    # Run a quick 10-ply game between Positional-Tiger and Blocker-Goat to verify compatibility
    tiger = PositionalTigerAgent(seed=42)
    goat = BlockerGoatAgent(seed=42)
    env = BaghChalEnv()

    for ply in range(10):
        if env._terminated:
            break
        legal = env.get_legal_actions()
        if env.state.turn == GOAT:
            act = goat.select_action(env.state, legal)
        else:
            act = tiger.select_action(env.state, legal)
        env.step(act)

    assert env.state.move_count == 10 or env._terminated


def test_mcts_selects_legal_action():
    agent = MCTSAgent(name="MCTS-Test", side=GOAT, num_simulations=30, seed=42)
    s = BaghChalState.initial_state()
    legal = BaghChalRules.get_legal_actions(s)
    act = agent.select_action(s, legal)
    assert act in legal


def test_mcts_tiger_finds_winning_capture():
    # 4 goats captured, immediate winning jump available
    agent = MCTSAgent(name="MCTS-Tiger", side=TIGER, num_simulations=50, seed=42)
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = GOAT
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=15,
        goats_captured=4,
        move_count=9,
        plies_since_capture=1,
        zobrist_hash=0,
    )
    legal = BaghChalRules.get_legal_actions(s)
    act = agent.select_action(s, legal)
    assert act.action_type == ActionType.JUMP
    assert act.captured_pos == 1
    next_s = s.apply_action(act)
    is_term, winner, _ = BaghChalRules.get_game_status(next_s)
    assert is_term is True
    assert winner == TIGER
