"""
BaghChalEnv: Gym-Compatible Environment for Bagh-Chal.

Supports:
- Discrete action space with action masking (TOTAL_ACTIONS)
- Multi-representation observations (Flat 78-dim, CNN 5x5x6, GNN graph)
- Reward shaping options (Sparse terminal, captured-scaled)
- Exact repetition tracking and draw detection
- ANSI board visualization
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from src.environment.board import (
    BOARD_SIZE,
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    TOTAL_GOATS,
    MAX_CAPTURES_TO_WIN,
    pos_to_coord,
    coord_to_pos,
    is_diagonal_active,
    get_neighbors,
    get_graph_edges,
)
from src.environment.state import (
    BaghChalState,
    Action,
    ActionType,
    Phase,
    TOTAL_ACTIONS,
    CANONICAL_ACTIONS,
    ACTION_TO_INDEX,
)
from src.environment.rules import BaghChalRules


class BaghChalEnv:
    """
    Gym-style environment for two-player zero-sum game Bagh-Chal.
    """
    def __init__(self, reward_type: str = "sparse"):
        """
        Args:
            reward_type: "sparse" (+1 win, -1 loss, 0 draw)
                         or "dense" (includes capture bonuses for tiger / survival for goat)
        """
        self.reward_type = reward_type
        self.state: BaghChalState = BaghChalState.initial_state()
        self.repetition_history: Dict[int, int] = {}
        self.move_history: List[Action] = []
        self._terminated: bool = False
        self._winner: Optional[int] = None
        self._termination_reason: str = "ongoing"

        # Static edge representation for GNNs
        self._graph_edges = get_graph_edges()
        # Directed edge index for PyTorch Geometric / standard GNN: (2, 2 * num_undirected_edges)
        src_nodes = []
        dst_nodes = []
        for u, v in self._graph_edges:
            src_nodes.extend([u, v])
            dst_nodes.extend([v, u])
        self.edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[BaghChalState, Dict[str, Any]]:
        """Reset environment to canonical start state."""
        self.state = BaghChalState.initial_state()
        self.repetition_history = {self.state.zobrist_hash: 1}
        self.move_history = []
        self._terminated = False
        self._winner = None
        self._termination_reason = "ongoing"
        return self.state, self._get_info()

    def get_legal_actions(self) -> List[Action]:
        """Return list of legal actions for current player."""
        if self._terminated:
            return []
        return BaghChalRules.get_legal_actions(self.state)

    def get_action_mask(self) -> np.ndarray:
        """
        Return binary mask of size TOTAL_ACTIONS.
        1 for legal action, 0 for illegal.
        """
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        if not self._terminated:
            legal = self.get_legal_actions()
            for act in legal:
                idx = ACTION_TO_INDEX[act]
                mask[idx] = 1.0
        return mask

    def step(
        self, action: Union[Action, int]
    ) -> Tuple[BaghChalState, float, bool, bool, Dict[str, Any]]:
        """
        Execute one ply in the environment.

        Args:
            action: Action object or integer action index

        Returns:
            (next_state, reward, terminated, truncated, info)
            Reward is from the perspective of the player who just took the action!
        """
        if self._terminated:
            raise RuntimeError("Cannot step in a terminated environment. Call reset().")

        if isinstance(action, (int, np.integer)):
            if not (0 <= action < TOTAL_ACTIONS):
                raise ValueError(f"Action index {action} out of range [0..{TOTAL_ACTIONS-1}]")
            action_obj = CANONICAL_ACTIONS[action]
        else:
            action_obj = action

        # Verify action legality
        legal = self.get_legal_actions()
        if action_obj not in legal:
            raise ValueError(f"Illegal action {action_obj} in state.")

        acting_player = self.state.turn
        prev_captured = self.state.goats_captured

        # Transition state
        self.state = self.state.apply_action(action_obj)
        self.move_history.append(action_obj)

        # Update repetition history
        h = self.state.zobrist_hash
        self.repetition_history[h] = self.repetition_history.get(h, 0) + 1
        rep_count = self.repetition_history[h]

        # Check terminal status
        terminated, winner, reason = BaghChalRules.get_game_status(self.state, rep_count)
        self._terminated = terminated
        self._winner = winner
        self._termination_reason = reason

        # Reward computation from perspective of acting_player
        reward = 0.0
        if terminated:
            if winner == acting_player:
                reward = 1.0
            elif winner == 0:
                reward = 0.0  # Draw
            else:
                reward = -1.0
        elif self.reward_type == "dense":
            # Intermediate dense reward shaping
            if acting_player == TIGER and self.state.goats_captured > prev_captured:
                reward += 0.2  # Immediate capture incentive
            elif acting_player == GOAT:
                reward += 0.005  # Step survival incentive for goats

        info = self._get_info()
        truncated = False

        return self.state, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        return {
            "turn": self.state.turn,
            "phase": self.state.phase,
            "goats_unplaced": self.state.goats_unplaced,
            "goats_captured": self.state.goats_captured,
            "move_count": self.state.move_count,
            "winner": self._winner,
            "termination_reason": self._termination_reason,
            "repetition_count": self.repetition_history.get(self.state.zobrist_hash, 1),
        }

    # ==========================================
    # State Representations for Neural Networks
    # ==========================================

    def get_flat_observation(self) -> np.ndarray:
        """
        78-dimensional flat feature vector:
        - 25 x 3 one-hot board (EMPTY, TIGER, GOAT): 75 floats
        - 1 float: turn (+1.0 if TIGER, -1.0 if GOAT)
        - 1 float: goats_unplaced / 20.0
        - 1 float: goats_captured / 5.0
        """
        obs = np.zeros(78, dtype=np.float32)
        # One-hot board
        for i, piece in enumerate(self.state.board):
            obs[i * 3 + piece] = 1.0

        obs[75] = 1.0 if self.state.turn == TIGER else -1.0
        obs[76] = self.state.goats_unplaced / float(TOTAL_GOATS)
        obs[77] = self.state.goats_captured / float(MAX_CAPTURES_TO_WIN)
        return obs

    def get_cnn_observation(self) -> np.ndarray:
        """
        5x5x6 spatial tensor:
        Plane 0: Tiger positions (binary)
        Plane 1: Goat positions (binary)
        Plane 2: Empty positions (binary)
        Plane 3: Turn plane (+1.0 for Tiger, 0.0 for Goat)
        Plane 4: Normalized goats unplaced (broadcast)
        Plane 5: Normalized goats captured (broadcast)
        """
        tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.state.board[coord_to_pos(r, c)]
                if piece == TIGER:
                    tensor[r, c, 0] = 1.0
                elif piece == GOAT:
                    tensor[r, c, 1] = 1.0
                else:
                    tensor[r, c, 2] = 1.0

                tensor[r, c, 3] = 1.0 if self.state.turn == TIGER else 0.0
                tensor[r, c, 4] = self.state.goats_unplaced / float(TOTAL_GOATS)
                tensor[r, c, 5] = self.state.goats_captured / float(MAX_CAPTURES_TO_WIN)
        return tensor

    def get_gnn_observation(self) -> Dict[str, np.ndarray]:
        """
        Graph representation for Graph Neural Networks:
        - node_features: (25, 5) float array:
            [is_empty, is_tiger, is_goat, is_diagonal_active, normalized_degree]
        - edge_index: (2, 2 * num_undirected_edges) int64 array
        - global_features: (4,) float array:
            [turn (+1 Tiger / -1 Goat), goats_unplaced/20, goats_captured/5, phase (0/1)]
        """
        node_features = np.zeros((NUM_POINTS, 5), dtype=np.float32)
        for pos in range(NUM_POINTS):
            piece = self.state.board[pos]
            r, c = pos_to_coord(pos)
            node_features[pos, piece] = 1.0
            node_features[pos, 3] = 1.0 if is_diagonal_active(r, c) else 0.0
            node_features[pos, 4] = len(get_neighbors(pos)) / 8.0

        global_features = np.array([
            1.0 if self.state.turn == TIGER else -1.0,
            self.state.goats_unplaced / float(TOTAL_GOATS),
            self.state.goats_captured / float(MAX_CAPTURES_TO_WIN),
            float(self.state.phase),
        ], dtype=np.float32)

        return {
            "node_features": node_features,
            "edge_index": self.edge_index,
            "global_features": global_features,
        }

    def render(self, mode: str = "ansi") -> str:
        """
        Render ASCII/ANSI representation of the Bagh-Chal board.
        Symbols:
          T = Tiger
          G = Goat
          . = Empty intersection
        """
        lines = []
        lines.append(f"Turn: {'TIGER' if self.state.turn == TIGER else 'GOAT'} | "
                     f"Phase: {self.state.phase.name} | "
                     f"Goats Unplaced: {self.state.goats_unplaced} | "
                     f"Captured: {self.state.goats_captured}/5")
        lines.append("  0 1 2 3 4")
        for r in range(BOARD_SIZE):
            row_str = f"{r} "
            for c in range(BOARD_SIZE):
                p = self.state.board[coord_to_pos(r, c)]
                sym = "." if p == EMPTY else ("T" if p == TIGER else "G")
                row_str += sym + ("-" if c < 4 else "")
            lines.append(row_str)
            if r < 4:
                # Intermediary connection line (with diagonal crosses where appropriate)
                diag_line = "  "
                for c in range(BOARD_SIZE):
                    if is_diagonal_active(r, c):
                        diag_line += "|\\" if c % 2 == 0 and c < 4 else ("|/" if c < 4 else "|")
                    else:
                        diag_line += "|/" if c % 2 == 1 and c < 4 else ("|\\" if c < 4 else "|")
                lines.append(diag_line)
        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output

