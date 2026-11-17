"""
Unit tests for Deep RL Policy-Value Networks, Action Masking, and PPO Pipeline.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.environment.state import TOTAL_ACTIONS
from src.environment.board import NUM_POINTS, BOARD_SIZE, TIGER, GOAT
from src.environment.env import BaghChalEnv
from src.training.networks import (
    MLPPolicyValueNet,
    CNNPolicyValueNet,
    GNNPolicyValueNet,
    DualNetworkWrapper,
    apply_action_mask,
)
from src.training.ppo import PPOTrainer


def test_action_mask_strict_zero_probability():
    logits = torch.randn(2, 10)
    # Mask: only action 2 and 4 legal in batch 0; only action 0 in batch 1
    mask = torch.zeros(2, 10)
    mask[0, 2] = 1.0
    mask[0, 4] = 1.0
    mask[1, 0] = 1.0

    masked_logits = apply_action_mask(logits, mask)
    probs = F.softmax(masked_logits, dim=-1)

    # Illegal actions must have exact 0.0 probability
    assert probs[0, 0].item() == 0.0
    assert probs[0, 1].item() == 0.0
    assert probs[0, 2].item() > 0.0
    assert probs[0, 3].item() == 0.0
    assert probs[0, 4].item() > 0.0
    assert abs(probs[0, 2].item() + probs[0, 4].item() - 1.0) < 1e-6

    assert probs[1, 0].item() == 1.0
    assert probs[1, 1].item() == 0.0


def test_mlp_network_forward_shape():
    net = MLPPolicyValueNet(in_features=78, action_dim=TOTAL_ACTIONS)
    x = torch.randn(4, 78)
    mask = torch.ones(4, TOTAL_ACTIONS)
    logits, value = net(x, action_mask=mask)

    assert logits.shape == (4, TOTAL_ACTIONS)
    assert value.shape == (4, 1)
    assert (-1.0 <= value).all() and (value <= 1.0).all()


def test_cnn_network_forward_shape():
    net = CNNPolicyValueNet(in_channels=6, action_dim=TOTAL_ACTIONS)
    x = torch.randn(3, 5, 5, 6)
    mask = torch.ones(3, TOTAL_ACTIONS)
    logits, value = net(x, action_mask=mask)

    assert logits.shape == (3, TOTAL_ACTIONS)
    assert value.shape == (3, 1)


def test_gnn_network_forward_shape():
    net = GNNPolicyValueNet(node_features_dim=5, global_features_dim=4, action_dim=TOTAL_ACTIONS)
    node_feats = torch.randn(2, NUM_POINTS, 5)
    global_feats = torch.randn(2, 4)
    mask = torch.ones(2, TOTAL_ACTIONS)

    logits, value = net(node_feats, global_feats, action_mask=mask)
    assert logits.shape == (2, TOTAL_ACTIONS)
    assert value.shape == (2, 1)


def test_dual_network_wrapper_routing():
    tiger_net = MLPPolicyValueNet(in_features=78, action_dim=TOTAL_ACTIONS)
    goat_net = MLPPolicyValueNet(in_features=78, action_dim=TOTAL_ACTIONS)
    wrapper = DualNetworkWrapper(tiger_net, goat_net)

    assert wrapper.get_network_for_role(TIGER) is tiger_net
    assert wrapper.get_network_for_role(GOAT) is goat_net


def test_ppo_trainer_single_episode_and_update():
    trainer = PPOTrainer(
        representation="flat",
        architecture="single",
        batch_size=16,
        ppo_epochs=1,
    )
    env = BaghChalEnv()
    stats = trainer.collect_episodes(env, num_episodes=1)
    assert stats["total_steps"] > 0

    train_stats = trainer.train_step()
    assert "loss" in train_stats
    assert "policy_loss" in train_stats
    assert "value_loss" in train_stats


def test_ppo_trainer_dual_network_training():
    trainer = PPOTrainer(
        representation="flat",
        architecture="dual",
        batch_size=16,
        ppo_epochs=1,
    )
    env = BaghChalEnv()
    stats = trainer.collect_episodes(env, num_episodes=1)
    assert stats["total_steps"] > 0

    train_stats = trainer.train_step()
    assert "loss" in train_stats
    assert "policy_loss" in train_stats
    assert "value_loss" in train_stats
