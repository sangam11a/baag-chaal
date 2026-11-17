"""
Neural Network Agent for Bagh-Chal.

Wraps trained policy-value networks (MLP, CNN, GNN) for action selection and inference.
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.environment.board import TIGER, GOAT
from src.environment.state import BaghChalState, Action, ACTION_TO_INDEX, CANONICAL_ACTIONS, TOTAL_ACTIONS
from src.agents.base_agent import BaseAgent
from src.training.networks import apply_action_mask


class NeuralAgent(BaseAgent):
    """
    Agent that uses a trained PyTorch policy network for decision making.
    """
    def __init__(
        self,
        name: str,
        side: int,
        model: torch.nn.Module,
        representation: str = "flat",  # "flat", "cnn", or "gnn"
        temperature: float = 0.0,      # 0.0 for argmax greedy, >0 for probabilistic sampling
        device: Optional[str] = None,
    ):
        super().__init__(name=name, side=side)
        if device is None:
            import os
            try:
                param_device = next(model.parameters()).device
                device = str(param_device)
            except Exception:
                device = os.getenv("BAGH_CHAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.representation = representation
        self.temperature = temperature
        self.device = device

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Construct binary action mask
        mask_np = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        for act in legal_actions:
            mask_np[ACTION_TO_INDEX[act]] = 1.0
        mask_tensor = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)

        # Forward pass based on representation type
        with torch.no_grad():
            if self.representation == "flat":
                # Compute flat observation from state
                from src.environment.env import BaghChalEnv
                # Minimal fast extraction
                obs_np = np.zeros(78, dtype=np.float32)
                for i, piece in enumerate(state.board):
                    obs_np[i * 3 + piece] = 1.0
                obs_np[75] = 1.0 if state.turn == TIGER else -1.0
                obs_np[76] = state.goats_unplaced / 20.0
                obs_np[77] = state.goats_captured / 5.0
                obs_tensor = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)
                logits, value = self.model(obs_tensor, action_mask=mask_tensor)

            elif self.representation == "cnn":
                from src.environment.board import BOARD_SIZE, coord_to_pos
                tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        p = state.board[coord_to_pos(r, c)]
                        if p == TIGER:
                            tensor[r, c, 0] = 1.0
                        elif p == GOAT:
                            tensor[r, c, 1] = 1.0
                        else:
                            tensor[r, c, 2] = 1.0
                        tensor[r, c, 3] = 1.0 if state.turn == TIGER else 0.0
                        tensor[r, c, 4] = state.goats_unplaced / 20.0
                        tensor[r, c, 5] = state.goats_captured / 5.0
                obs_tensor = torch.from_numpy(tensor).to(self.device).unsqueeze(0)
                logits, value = self.model(obs_tensor, action_mask=mask_tensor)

            elif self.representation == "gnn":
                from src.environment.board import NUM_POINTS, pos_to_coord, is_diagonal_active, get_neighbors
                node_feats = np.zeros((NUM_POINTS, 5), dtype=np.float32)
                for pos in range(NUM_POINTS):
                    p = state.board[pos]
                    r, c = pos_to_coord(pos)
                    node_feats[pos, p] = 1.0
                    node_feats[pos, 3] = 1.0 if is_diagonal_active(r, c) else 0.0
                    node_feats[pos, 4] = len(get_neighbors(pos)) / 8.0

                global_feats = np.array([
                    1.0 if state.turn == TIGER else -1.0,
                    state.goats_unplaced / 20.0,
                    state.goats_captured / 5.0,
                    float(state.phase),
                ], dtype=np.float32)

                n_t = torch.from_numpy(node_feats).to(self.device).unsqueeze(0)
                g_t = torch.from_numpy(global_feats).to(self.device).unsqueeze(0)
                logits, value = self.model(n_t, g_t, action_mask=mask_tensor)

            logits = logits.squeeze(0)

        # Action selection
        if self.temperature <= 1e-4:
            chosen_idx = int(torch.argmax(logits).item())
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            dist = torch.distributions.Categorical(probs)
            chosen_idx = int(dist.sample().item())

        chosen_action = CANONICAL_ACTIONS[chosen_idx]
        if chosen_action not in legal_actions:
            # Fallback safety
            return legal_actions[0]
        return chosen_action

