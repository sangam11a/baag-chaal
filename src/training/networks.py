"""
Deep Neural Network Architectures for Bagh-Chal Reinforcement Learning.

Implements the 3 representation paradigms from the consensus plan:
1. Flat MLP (78-dim input)
2. Spatial CNN with residual blocks (5x5x6 input)
3. Graph Neural Network (GNN) on the 25-node irregular board topology

Supports single-network and dual-network (role-specialized) architectures.
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.environment.state import TOTAL_ACTIONS
from src.environment.board import NUM_POINTS, BOARD_SIZE, TIGER, GOAT


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply legal action mask to logits.
    Mask is 1.0 for legal actions, 0.0 for illegal.
    Sets illegal action logits to -1e8 prior to softmax.
    """
    neg_inf = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
    masked_logits = torch.where(mask > 0.5, logits, neg_inf)
    return masked_logits


class MLPPolicyValueNet(nn.Module):
    """
    Flat Multi-Layer Perceptron Policy & Value Network.
    Input: (B, 78) flat observation vector.
    """
    def __init__(self, in_features: int = 78, action_dim: int = TOTAL_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)

        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        logits = self.policy_head(h)
        if action_mask is not None:
            logits = apply_action_mask(logits, action_mask)

        value = torch.tanh(self.value_head(h))
        return logits, value


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class CNNPolicyValueNet(nn.Module):
    """
    Spatial Convolutional Neural Network with Residual Blocks.
    Input: (B, 5, 5, 6) -> permuted to (B, 6, 5, 5).
    """
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        action_dim: int = TOTAL_ACTIONS,
    ):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D(hidden_channels) for _ in range(num_res_blocks)
        ])
        flat_size = hidden_channels * BOARD_SIZE * BOARD_SIZE  # 64 * 25 = 1600
        self.dense = nn.Linear(flat_size, 256)

        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is (B, 5, 5, 6) -> permute to (B, 6, 5, 5)
        if x.dim() == 4 and x.shape[-1] == 6:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and x.shape[-1] == 6:
            x = x.permute(2, 0, 1).unsqueeze(0)

        h = self.initial_conv(x)
        for block in self.res_blocks:
            h = block(h)

        flat = h.reshape(h.size(0), -1)
        feat = F.relu(self.dense(flat))

        logits = self.policy_head(feat)
        if action_mask is not None:
            logits = apply_action_mask(logits, action_mask)

        value = torch.tanh(self.value_head(feat))
        return logits, value


class GraphConvolutionLayer(nn.Module):
    """
    Pure PyTorch Graph Convolution (GCN) Layer.
    Computes message passing along board adjacency graph without external C++ libs.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_self = nn.Linear(in_features, out_features, bias=False)
        self.weight_nbr = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, node_feats: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        node_feats: (B, 25, in_features)
        adj_norm: (25, 25) normalized adjacency matrix
        """
        h_self = self.weight_self(node_feats)
        # Aggregation from neighbors: adj_norm @ node_feats
        # Batched matmul: (B, 25, in_feats) with (25, 25)
        h_agg = torch.matmul(adj_norm, node_feats)
        h_nbr = self.weight_nbr(h_agg)
        out = h_self + h_nbr + self.bias
        return F.relu(out)


class GNNPolicyValueNet(nn.Module):
    """
    Graph Neural Network for Bagh-Chal's Irregular Board Topology.
    Operates on the 25-node board graph with exact diagonal edge connectivity.
    """
    def __init__(
        self,
        node_features_dim: int = 5,
        global_features_dim: int = 4,
        hidden_dim: int = 64,
        action_dim: int = TOTAL_ACTIONS,
    ):
        super().__init__()
        self.gcn1 = GraphConvolutionLayer(node_features_dim, hidden_dim)
        self.gcn2 = GraphConvolutionLayer(hidden_dim, hidden_dim)

        self.global_mlp = nn.Sequential(
            nn.Linear(global_features_dim, 32),
            nn.ReLU(),
        )

        # Combined features: pooled node features (mean + max = 2 * hidden_dim) + global (32)
        combined_dim = 2 * hidden_dim + 32
        self.head_dense = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        # Precompute normalized adjacency matrix for the 25 nodes
        self.register_buffer("adj_norm", self._build_normalized_adjacency())

    def _build_normalized_adjacency(self) -> torch.Tensor:
        from src.environment.board import ADJACENCY_LIST
        adj = torch.zeros((NUM_POINTS, NUM_POINTS), dtype=torch.float32)
        for u in range(NUM_POINTS):
            for v in ADJACENCY_LIST[u]:
                adj[u, v] = 1.0

        # Degree normalization: D^(-0.5) * A * D^(-0.5)
        degrees = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_mat = torch.diag(deg_inv_sqrt)
        norm_adj = torch.matmul(torch.matmul(d_mat, adj), d_mat)
        return norm_adj

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_features: (B, 25, 5) or (25, 5)
        global_features: (B, 4) or (4,)
        """
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        # Graph Convolutions
        h = self.gcn1(node_features, self.adj_norm)
        h = self.gcn2(h, self.adj_norm)

        # Graph Readout: Global Mean + Global Max pooling across the 25 nodes
        mean_pool = h.mean(dim=1)  # (B, hidden_dim)
        max_pool = h.max(dim=1)[0] # (B, hidden_dim)

        # Global features
        g_feat = self.global_mlp(global_features) # (B, 32)

        combined = torch.cat([mean_pool, max_pool, g_feat], dim=-1)
        feat = self.head_dense(combined)

        logits = self.policy_head(feat)
        if action_mask is not None:
            logits = apply_action_mask(logits, action_mask)

        value = torch.tanh(self.value_head(feat))
        return logits, value


class DualNetworkWrapper(nn.Module):
    """
    Dual-Network Architecture Wrapper for Asymmetric Role Specialization.
    Maintains separate networks for Tiger and Goat roles.
    """
    def __init__(self, tiger_net: nn.Module, goat_net: nn.Module):
        super().__init__()
        self.tiger_net = tiger_net
        self.goat_net = goat_net

    def get_network_for_role(self, role: int) -> nn.Module:
        return self.tiger_net if role == TIGER else self.goat_net

    def forward(
        self,
        obs: Any,
        role: Optional[Any] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if role is None:
            # Default to tiger if role not specified
            role = TIGER

        if isinstance(role, int):
            net = self.get_network_for_role(role)
            if isinstance(obs, dict):
                return net(obs["node_features"], obs["global_features"], action_mask)
            return net(obs, action_mask)

        # role is Tensor of shape (B,) with mixed TIGER/GOAT
        batch_size = role.shape[0]
        device = role.device
        action_dim = TOTAL_ACTIONS

        out_logits = torch.zeros((batch_size, action_dim), device=device, dtype=torch.float32)
        out_values = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

        tiger_idx = (role == TIGER).nonzero(as_tuple=True)[0]
        goat_idx = (role == GOAT).nonzero(as_tuple=True)[0]

        if len(tiger_idx) > 0:
            mask_t = action_mask[tiger_idx] if action_mask is not None else None
            if isinstance(obs, dict):
                l_t, v_t = self.tiger_net(obs["node_features"][tiger_idx], obs["global_features"][tiger_idx], mask_t)
            else:
                l_t, v_t = self.tiger_net(obs[tiger_idx], mask_t)
            out_logits[tiger_idx] = l_t
            out_values[tiger_idx] = v_t

        if len(goat_idx) > 0:
            mask_g = action_mask[goat_idx] if action_mask is not None else None
            if isinstance(obs, dict):
                l_g, v_g = self.goat_net(obs["node_features"][goat_idx], obs["global_features"][goat_idx], mask_g)
            else:
                l_g, v_g = self.goat_net(obs[goat_idx], mask_g)
            out_logits[goat_idx] = l_g
            out_values[goat_idx] = v_g

        return out_logits, out_values

