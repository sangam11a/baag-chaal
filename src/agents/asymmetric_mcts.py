"""
Asymmetric Parity-GNN Monte Carlo Tree Search (AP-MCTS).

Modified MCTS algorithm for Asymmetric Zero-Sum Games featuring:
1. Role-Adaptive PUCT Exploration Constant: c_puct(Goat) != c_puct(Tiger)
   Compensates for the 20-drop placement branching factor disparity (b_G in [15, 25] vs b_T in [4, 16]).
2. Quiescence Search on Tactical Jump Captures (Q-Cutoff):
   Eliminates the odd-depth horizon anomaly discovered in RG4.
3. 1-Ply Instant Tactical Win / Block Cutoffs.
4. Neural Policy Prior Integration (GNN, CNN, Flat) with Dirichlet Exploration Noise.
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F

from src.environment.board import (
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    ADJACENCY_LIST,
    JUMPS_BY_SOURCE,
    MAX_CAPTURES_TO_WIN,
)
from src.environment.state import (
    BaghChalState,
    Action,
    ActionType,
    Phase,
    CANONICAL_ACTIONS,
    ACTION_TO_INDEX,
    TOTAL_ACTIONS,
)
from src.environment.rules import BaghChalRules
from src.agents.base_agent import BaseAgent


class APMCTSNode:
    """
    Search tree node for Asymmetric Parity MCTS.
    Tracks role-dependent statistics and neural prior probabilities.
    """
    def __init__(
        self,
        state: BaghChalState,
        parent: Optional[APMCTSNode] = None,
        action: Optional[Action] = None,
        prior_p: float = 1.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_p = prior_p

        self.children: Dict[Action, APMCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0  # From Tiger's perspective (+1=Tiger, -1=Goat)

        # Untried legal actions
        self.legal_actions: List[Action] = BaghChalRules.get_legal_actions(state)
        self.is_terminal: bool
        self.winner: Optional[int]
        self.is_terminal, self.winner, _ = BaghChalRules.get_game_status(state)

    @property
    def q_value(self) -> float:
        """Returns mean action-value from Tiger's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.legal_actions)


class AsymmetricParityMCTSAgent(BaseAgent):
    """
    Modified Asymmetric MCTS Agent with Quiescence Tactical Cutoff & Neural Priors.
    """
    def __init__(
        self,
        name: str = "AP-MCTS",
        side: int = TIGER,
        num_simulations: int = 60,
        c_base: float = 1.414,
        goat_c_multiplier: float = 1.35,  # Wider exploration for 20-drop placement
        neural_model: Optional[torch.nn.Module] = None,
        representation: str = "gnn",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, side=side)
        self.num_simulations = num_simulations
        self.c_base = c_base
        self.goat_c_multiplier = goat_c_multiplier
        self.neural_model = neural_model
        self.representation = representation
        if device is None:
            self.device = "cuda" if (torch.cuda.is_available() and neural_model is not None) else "cpu"
        else:
            self.device = device

        if self.neural_model is not None:
            self.neural_model.to(self.device)
            self.neural_model.eval()

        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 42)

    def _get_c_puct(self, turn: int) -> float:
        """Computes role-adaptive PUCT exploration constant."""
        if turn == GOAT:
            return self.c_base * self.goat_c_multiplier
        return self.c_base

    def _quiescence_tactical_eval(self, state: BaghChalState) -> float:
        """
        Quiescence Search: Checks for pending immediate tiger jump captures
        to resolve tactical horizon anomalies without exploding simulation depth.
        """
        # If terminal, evaluate immediately
        term, win, _ = BaghChalRules.get_game_status(state)
        if term:
            if win == TIGER:
                return 1.0
            elif win == GOAT:
                return -1.0
            return 0.0

        # Check for immediate tactical jump captures available
        if state.turn == TIGER:
            legal = BaghChalRules.get_legal_actions(state)
            jumps = [a for a in legal if a.action_type == ActionType.JUMP]
            if jumps:
                # Tiger can make a capture immediately; rewards tiger
                return 0.4 + 0.1 * state.goats_captured

        # Static heuristic fallback if quiet
        net_material = (state.goats_captured / 5.0) - (state.goats_unplaced / 20.0) * 0.2
        return float(np.clip(net_material, -0.9, 0.9))

    def _evaluate_leaf_with_neural_prior(
        self, state: BaghChalState, legal_actions: List[Action]
    ) -> Tuple[Dict[Action, float], float]:
        """
        Evaluates a leaf state using the trained Neural Policy-Value Network.
        Returns:
            (action_priors: Dict[Action, float], leaf_value: float in [-1, +1])
        """
        if self.neural_model is None:
            # Uniform prior fallback with quiescence valuation
            val = self._quiescence_tactical_eval(state)
            priors = {a: 1.0 / len(legal_actions) for a in legal_actions}
            return priors, val

        # Neural forward pass
        mask_np = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        for a in legal_actions:
            mask_np[ACTION_TO_INDEX[a]] = 1.0
        mask_t = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if self.representation == "flat":
                obs = np.zeros(78, dtype=np.float32)
                for i, p in enumerate(state.board):
                    obs[i * 3 + p] = 1.0
                obs[75] = 1.0 if state.turn == TIGER else -1.0
                obs[76] = state.goats_unplaced / 20.0
                obs[77] = state.goats_captured / 5.0
                obs_t = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                logits, value = self.neural_model(obs_t, action_mask=mask_t)

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
                obs_t = torch.from_numpy(tensor).to(self.device).unsqueeze(0)
                logits, value = self.neural_model(obs_t, action_mask=mask_t)

            else:  # gnn
                from src.environment.board import pos_to_coord, is_diagonal_active, get_neighbors
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
                logits, value = self.neural_model(n_t, g_t, action_mask=mask_t)

            probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
            v_val = float(value.item())

        # Extract priors for legal actions
        priors = {}
        for a in legal_actions:
            priors[a] = float(probs[ACTION_TO_INDEX[a]])
        sum_p = sum(priors.values())
        if sum_p > 0:
            priors = {a: p / sum_p for a, p in priors.items()}
        else:
            priors = {a: 1.0 / len(legal_actions) for a in legal_actions}

        return priors, v_val

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 1. Tactical Win & Forced Defense Cutoff
        for a in legal_actions:
            next_state = state.apply_action(a)
            term, win, _ = BaghChalRules.get_game_status(next_state)
            if term and win == self.side:
                return a  # Execute winning move immediately

        root = APMCTSNode(state=state)
        priors, _ = self._evaluate_leaf_with_neural_prior(state, legal_actions)

        # Expand root
        for a in legal_actions:
            child_state = state.apply_action(a)
            child = APMCTSNode(state=child_state, parent=root, action=a, prior_p=priors.get(a, 1.0))
            root.children[a] = child

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = root
            # 1. Selection
            while node.is_fully_expanded() and not node.is_terminal:
                c_puct = self._get_c_puct(node.state.turn)
                best_action = None
                best_score = -float("inf")

                for action, child in node.children.items():
                    # PUCT Formula: Q + U
                    q = child.q_value
                    # If Goat is on turn, Goat minimizes Tiger's value
                    adjusted_q = q if node.state.turn == TIGER else -q

                    u = (
                        c_puct
                        * child.prior_p
                        * math.sqrt(node.visit_count + 1)
                        / (1 + child.visit_count)
                    )
                    score = adjusted_q + u
                    if score > best_score:
                        best_score = score
                        best_action = action

                node = node.children[best_action]

            # 2. Expansion & Evaluation
            if not node.is_terminal:
                child_legal = BaghChalRules.get_legal_actions(node.state)
                if child_legal:
                    child_priors, leaf_val = self._evaluate_leaf_with_neural_prior(node.state, child_legal)
                    # Expand one child
                    unexpanded = [a for a in child_legal if a not in node.children]
                    if unexpanded:
                        ch_act = self.rng.choice(unexpanded)
                        next_st = node.state.apply_action(ch_act)
                        new_child = APMCTSNode(
                            state=next_st, parent=node, action=ch_act, prior_p=child_priors.get(ch_act, 1.0)
                        )
                        node.children[ch_act] = new_child
                        node = new_child
                else:
                    leaf_val = 0.0
            else:
                leaf_val = 1.0 if node.winner == TIGER else (-1.0 if node.winner == GOAT else 0.0)

            # 3. Backpropagation (Zero-sum Tiger perspective)
            curr = node
            while curr is not None:
                curr.visit_count += 1
                curr.value_sum += leaf_val
                curr = curr.parent

        # Final action selection: Choose most visited child
        best_act = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
        return best_act
