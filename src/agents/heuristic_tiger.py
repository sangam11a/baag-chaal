"""
Heuristic Tiger Agents for Bagh-Chal.

Implements:
1. GreedyTigerAgent: Strictly maximizes immediate captures, then maximizes post-move mobility.
2. PositionalTigerAgent: Domain-expert positional heuristic combining capture priority,
   threat creation, central control, and corner trap avoidance.
"""

from typing import List, Optional, Tuple
import random
from src.environment.board import (
    BOARD_SIZE,
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    pos_to_coord,
    coord_to_pos,
    ADJACENCY_LIST,
    JUMPS_BY_SOURCE,
    INITIAL_TIGER_POSITIONS,
)
from src.environment.state import BaghChalState, Action, ActionType, Phase
from src.agents.base_agent import BaseAgent


def _compute_total_tiger_mobility(state: BaghChalState) -> int:
    """Count total legal single steps and jumps available to all tigers."""
    b = state.board
    mobility = 0
    for pos in range(NUM_POINTS):
        if b[pos] == TIGER:
            for nbr in ADJACENCY_LIST[pos]:
                if b[nbr] == EMPTY:
                    mobility += 1
            for mid, dest in JUMPS_BY_SOURCE[pos]:
                if b[mid] == GOAT and b[dest] == EMPTY:
                    mobility += 2  # Weight jump capability higher
    return mobility


def _count_potential_jump_threats(state: BaghChalState) -> int:
    """Count how many goats are directly threatened with capture on the next move."""
    b = state.board
    threats = 0
    for pos in range(NUM_POINTS):
        if b[pos] == TIGER:
            for mid, dest in JUMPS_BY_SOURCE[pos]:
                if b[mid] == GOAT and b[dest] == EMPTY:
                    threats += 1
    return threats


class GreedyTigerAgent(BaseAgent):
    """
    Greedy Tiger:
    - If jump captures exist, execute the jump that leaves tigers with highest mobility.
    - If no jump, execute normal step that maximizes immediate tiger mobility.
    """
    def __init__(self, name: str = "Greedy-Tiger", side: int = TIGER, seed: Optional[int] = None):
        super().__init__(name=name, side=side)
        self.rng = random.Random(seed)

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        jumps = [a for a in legal_actions if a.action_type == ActionType.JUMP]
        if jumps:
            # Score each jump by subsequent state mobility
            best_score = -1
            best_jumps = []
            for act in jumps:
                next_s = state.apply_action(act)
                mob = _compute_total_tiger_mobility(next_s)
                if mob > best_score:
                    best_score = mob
                    best_jumps = [act]
                elif mob == best_score:
                    best_jumps.append(act)
            return self.rng.choice(best_jumps)

        # No jump: score moves by mobility
        best_score = -1
        best_moves = []
        for act in legal_actions:
            next_s = state.apply_action(act)
            mob = _compute_total_tiger_mobility(next_s)
            if mob > best_score:
                best_score = mob
                best_moves = [act]
            elif mob == best_score:
                best_moves.append(act)

        return self.rng.choice(best_moves) if best_moves else self.rng.choice(legal_actions)


class PositionalTigerAgent(BaseAgent):
    """
    Positional Tiger:
    Balances capture opportunities with positional control:
    - High priority to captures
    - Bonus for threatening multiple goats (tiger forks)
    - Bonus for occupying high-degree center/diamond nodes
    - Penalty for lingering in low-degree corners where goats easily trap
    """
    def __init__(
        self,
        name: str = "Positional-Tiger",
        side: int = TIGER,
        seed: Optional[int] = None,
        w_capture: float = 1000.0,
        w_threat: float = 40.0,
        w_mobility: float = 10.0,
        w_center: float = 15.0,
        w_corner_penalty: float = 25.0,
    ):
        super().__init__(name=name, side=side)
        self.rng = random.Random(seed)
        self.w_capture = w_capture
        self.w_threat = w_threat
        self.w_mobility = w_mobility
        self.w_center = w_center
        self.w_corner_penalty = w_corner_penalty

    def _score_action(self, state: BaghChalState, action: Action) -> float:
        score = 0.0

        # Immediate capture bonus
        if action.action_type == ActionType.JUMP:
            score += self.w_capture

        next_s = state.apply_action(action)

        # 1. Resulting mobility
        score += self.w_mobility * _compute_total_tiger_mobility(next_s)

        # 2. Resulting jump threats
        score += self.w_threat * _count_potential_jump_threats(next_s)

        # 3. Positional distribution of all 4 tigers
        for pos in range(NUM_POINTS):
            if next_s.board[pos] == TIGER:
                r, c = pos_to_coord(pos)
                dist_from_center = abs(r - 2) + abs(c - 2)
                # Center bonus (distance 0 = max bonus)
                score += self.w_center * (4 - dist_from_center)
                # Corner penalty (corners have pos in 0, 4, 20, 24)
                if pos in INITIAL_TIGER_POSITIONS:
                    score -= self.w_corner_penalty

        return score

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        best_score = -float("inf")
        best_actions = []

        for act in legal_actions:
            score = self._score_action(state, act)
            if score > best_score:
                best_score = score
                best_actions = [act]
            elif abs(score - best_score) < 1e-6:
                best_actions.append(act)

        return self.rng.choice(best_actions)

