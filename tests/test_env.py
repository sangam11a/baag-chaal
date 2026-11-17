"""
Unit tests for BaghChalEnv Gym-compatible environment, actions masks, and representations.
"""

import pytest
import numpy as np

from src.environment.board import EMPTY, TIGER, GOAT, NUM_POINTS, BOARD_SIZE
from src.environment.state import (
    TOTAL_ACTIONS,
    CANONICAL_ACTIONS,
    ACTION_TO_INDEX,
    Action,
    ActionType,
    Phase,
)
from src.environment.env import BaghChalEnv


def test_env_initialization():
    env = BaghChalEnv(reward_type="sparse")
    assert env.state.turn == GOAT
    assert env.state.phase == Phase.PLACEMENT
    assert env.state.goats_unplaced == 20
    assert env.state.goats_captured == 0
    assert not env._terminated
    assert env._winner is None


def test_action_mask_validity():
    env = BaghChalEnv()
    mask = env.get_action_mask()
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask.dtype == np.float32

    legal = env.get_legal_actions()
    assert np.sum(mask) == len(legal)
    for act in legal:
        idx = ACTION_TO_INDEX[act]
        assert mask[idx] == 1.0


def test_step_by_action_object_and_index():
    env = BaghChalEnv()
    legal = env.get_legal_actions()
    first_action = legal[0]

    # Step with Action object
    next_s, rew, term, trunc, info = env.step(first_action)
    assert next_s.turn == TIGER
    assert rew == 0.0
    assert not term
    assert not trunc
    assert info["turn"] == TIGER

    # Next step with integer action index
    tiger_legal = env.get_legal_actions()
    tiger_act = tiger_legal[0]
    tiger_idx = ACTION_TO_INDEX[tiger_act]

    next_s2, rew2, term2, trunc2, info2 = env.step(tiger_idx)
    assert next_s2.turn == GOAT
    assert rew2 == 0.0


def test_invalid_action_rejected():
    env = BaghChalEnv()
    # Tiger move on Goat's turn is illegal
    tiger_move = Action(ActionType.MOVE, from_pos=0, to_pos=1)
    with pytest.raises(ValueError):
        env.step(tiger_move)

    # Out of range integer
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(TOTAL_ACTIONS)


def test_step_after_termination_rejected():
    env = BaghChalEnv()
    env._terminated = True
    legal = env.get_legal_actions()
    assert len(legal) == 0
    mask = env.get_action_mask()
    assert np.sum(mask) == 0.0

    place_act = Action(ActionType.PLACE, from_pos=None, to_pos=1)
    with pytest.raises(RuntimeError):
        env.step(place_act)


def test_flat_observation_contract():
    env = BaghChalEnv()
    obs = env.get_flat_observation()
    assert obs.shape == (78,)
    assert obs.dtype == np.float32

    # Check turn scalar (index 75): GOAT = -1.0
    assert obs[75] == -1.0
    # goats_unplaced (index 76): 20/20 = 1.0
    assert obs[76] == 1.0
    # goats_captured (index 77): 0/5 = 0.0
    assert obs[77] == 0.0


def test_cnn_observation_contract():
    env = BaghChalEnv()
    obs = env.get_cnn_observation()
    assert obs.shape == (5, 5, 6)
    assert obs.dtype == np.float32

    # 4 corner tigers in plane 0
    assert obs[0, 0, 0] == 1.0
    assert obs[0, 4, 0] == 1.0
    assert obs[4, 0, 0] == 1.0
    assert obs[4, 4, 0] == 1.0
    # 21 empty positions in plane 2
    assert np.sum(obs[:, :, 2]) == 21.0


def test_gnn_observation_contract():
    env = BaghChalEnv()
    gnn_data = env.get_gnn_observation()

    assert "node_features" in gnn_data
    assert "edge_index" in gnn_data
    assert "global_features" in gnn_data

    nodes = gnn_data["node_features"]
    edges = gnn_data["edge_index"]
    globals_ = gnn_data["global_features"]

    assert nodes.shape == (NUM_POINTS, 5)
    assert edges.shape[0] == 2
    assert globals_.shape == (4,)

    # globals: turn, goats_unplaced/20, goats_captured/5, phase
    assert globals_[0] == -1.0  # GOAT
    assert globals_[1] == 1.0   # 20/20
    assert globals_[2] == 0.0   # 0/5
    assert globals_[3] == 0.0   # PLACEMENT phase


def test_dense_reward_shaping():
    env = BaghChalEnv(reward_type="dense")
    # Goat places piece
    _, rew, _, _, _ = env.step(Action(ActionType.PLACE, from_pos=None, to_pos=12))
    assert rew == 0.005  # Step survival incentive for goats


def test_render_ansi():
    env = BaghChalEnv()
    render_text = env.render(mode="ansi")
    assert isinstance(render_text, str)
    assert "Turn: GOAT" in render_text
    assert "Phase: PLACEMENT" in render_text
    assert "T" in render_text
    assert "." in render_text

