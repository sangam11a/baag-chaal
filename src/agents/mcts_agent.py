"""
Monte Carlo Tree Search (MCTS) Agent with UCT for Bagh-Chal.

Implements standard UCT search for two-player zero-sum deterministic games:
- Tree policy with UCB1 selection (c = sqrt(2))
- Node expansion
- Fast rollout simulation
- Value backpropagation (Tiger perspective: +1 Tiger win, -1 Goat win, 0 Draw)
"""

from typing import List, Optional, Dict, Tuple
import math
import random
import time

from src.environment.board import TIGER, GOAT
from src.environment.state import BaghChalState, Action
from src.environment.rules import BaghChalRules
from src.agents.base_agent import BaseAgent


class MCTSNode:
    """
    Node in the MCTS tree.
    """
    def __init__(
        self,
        state: BaghChalState,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Action] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken by parent to reach this node
        self.children: Dict[Action, "MCTSNode"] = {}
        self.untried_actions: List[Action] = BaghChalRules.get_legal_actions(state)
        self.visits: int = 0
        self.total_value: float = 0.0  # From Tiger's perspective (+1 Tiger win, -1 Goat win)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        is_term, _, _ = BaghChalRules.get_game_status(self.state)
        return is_term

    def best_child(self, exploration_constant: float = math.sqrt(2)) -> "MCTSNode":
        """
        Select child using UCB1.
        If current node is TIGER turn: tiger wants to MAXIMIZE tiger value.
        If current node is GOAT turn: goat wants to MINIMIZE tiger value.
        """
        is_tiger_turn = (self.state.turn == TIGER)
        best_score = -float("inf") if is_tiger_turn else float("inf")
        best_nodes: List["MCTSNode"] = []

        log_parent_visits = math.log(max(1, self.visits))

        for action, child in self.children.items():
            if child.visits == 0:
                score = float("inf") if is_tiger_turn else -float("inf")
            else:
                q_value = child.total_value / child.visits
                u_exploration = exploration_constant * math.sqrt(log_parent_visits / child.visits)
                score = (q_value + u_exploration) if is_tiger_turn else (q_value - u_exploration)

            if is_tiger_turn:
                if score > best_score:
                    best_score = score
                    best_nodes = [child]
                elif abs(score - best_score) < 1e-7:
                    best_nodes.append(child)
            else:
                if score < best_score:
                    best_score = score
                    best_nodes = [child]
                elif abs(score - best_score) < 1e-7:
                    best_nodes.append(child)

        return random.choice(best_nodes)


class MCTSAgent(BaseAgent):
    """
    Pure MCTS Agent with UCT selection for Bagh-Chal.
    """
    def __init__(
        self,
        name: str = "MCTS-UCT",
        side: int = TIGER,
        num_simulations: int = 200,
        c_param: float = math.sqrt(2),
        max_rollout_depth: int = 25,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, side=side)
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.max_rollout_depth = max_rollout_depth
        self.rng = random.Random(seed)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB1 until an unexpanded or terminal node is reached."""
        curr = node
        while not curr.is_terminal():
            if not curr.is_fully_expanded():
                return self._expand(curr)
            curr = curr.best_child(self.c_param)
        return curr

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand one untried action from the current node."""
        action = self.rng.choice(node.untried_actions)
        node.untried_actions.remove(action)
        next_state = node.state.apply_action(action)
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children[action] = child_node
        return child_node

    def _rollout(self, state: BaghChalState) -> float:
        """
        Fast simulation rollout from state up to max_rollout_depth plies.
        Returns outcome value from TIGER perspective:
        +1.0 (Tiger win), -1.0 (Goat win), 0.0 (Draw).
        """
        curr = state
        depth = 0

        while depth < self.max_rollout_depth:
            is_term, winner, _ = BaghChalRules.get_game_status(curr)
            if is_term:
                if winner == TIGER:
                    return 1.0
                elif winner == GOAT:
                    return -1.0
                else:
                    return 0.0

            legal = BaghChalRules.get_legal_actions(curr)
            if not legal:
                # Stalemate
                return -1.0 if curr.turn == TIGER else 1.0

            # Quick heuristic rollout: prioritize captures for tiger, safe moves for goat
            act = self.rng.choice(legal)
            curr = curr.apply_action(act)
            depth += 1

        # Truncation: evaluate based on incremental captures achieved during rollout
        cap_diff = curr.goats_captured - state.goats_captured
        return min(1.0, max(-1.0, cap_diff * 0.2))

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate simulation value (+1 Tiger, -1 Goat) back to tree root."""
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.total_value += value
            curr = curr.parent

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 1. Immediate winning move check (tactical cutoff)
        for act in legal_actions:
            next_s = state.apply_action(act)
            is_term, winner, _ = BaghChalRules.get_game_status(next_s)
            if is_term and winner == self.side:
                return act

        root = MCTSNode(state=state)

        for _ in range(self.num_simulations):
            selected_node = self._select(root)
            value = self._rollout(selected_node.state)
            self._backpropagate(selected_node, value)

        # Action selection: pick child with highest visit count (most robust)
        most_visited_act = None
        max_visits = -1

        for act, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                most_visited_act = act

        return most_visited_act if most_visited_act is not None else self.rng.choice(legal_actions)

