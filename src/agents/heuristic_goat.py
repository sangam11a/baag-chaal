"""
Heuristic Goat Agents for Bagh-Chal.

Implements:
1. DefensiveGoatAgent: Minimizes vulnerable goats by avoiding placements or moves
   that create immediate jump opportunities for any tiger.
2. BlockerGoatAgent (Trapping / Ring Goat): Focuses on aggressive encirclement to
   constrict tiger mobility to zero while maintaining defensive clustering.
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
    JUMP_TABLE,
)
from src.environment.state import BaghChalState, Action, ActionType, Phase
from src.agents.base_agent import BaseAgent


def _count_tiger_captures_possible(state: BaghChalState) -> int:
    """Count how many immediate jump captures tigers can make in state."""
    b = state.board
    captures = 0
    for pos in range(NUM_POINTS):
        if b[pos] == TIGER:
            for mid, dest in JUMPS_BY_SOURCE[pos]:
                if b[mid] == GOAT and b[dest] == EMPTY:
                    captures += 1
    return captures


def _compute_tiger_total_freedom(state: BaghChalState) -> int:
    """Sum of empty adjacent neighbors for all 4 tigers."""
    b = state.board
    freedom = 0
    for pos in range(NUM_POINTS):
        if b[pos] == TIGER:
            for nbr in ADJACENCY_LIST[pos]:
                if b[nbr] == EMPTY:
                    freedom += 1
    return freedom


class DefensiveGoatAgent(BaseAgent):
    """
    Defensive Goat:
    Safety-first strategy:
    - Never willingly sacrifice a goat if a safe move exists.
    - Heavily penalizes moves that expose any goat to a jump capture.
    - Rewards placing goats adjacent to friendly goats (clustering defends against jumps).
    """
    def __init__(self, name: str = "Defensive-Goat", side: int = GOAT, seed: Optional[int] = None):
        super().__init__(name=name, side=side)
        self.rng = random.Random(seed)

    def _score_action(self, state: BaghChalState, action: Action) -> float:
        next_s = state.apply_action(action)

        # 1. Major penalty for any immediate jump capture granted to tigers
        tiger_captures = _count_tiger_captures_possible(next_s)
        penalty = tiger_captures * 500.0

        # 2. Penalty for tiger mobility (lower tiger mobility is better)
        tiger_mob = _compute_tiger_total_freedom(next_s)

        # 3. Clustering bonus: count friendly neighbors of placed/moved goat
        cluster_bonus = 0.0
        pos = action.to_pos
        for nbr in ADJACENCY_LIST[pos]:
            if next_s.board[nbr] == GOAT:
                cluster_bonus += 5.0

        return -penalty - (tiger_mob * 3.0) + cluster_bonus

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


class BlockerGoatAgent(BaseAgent):
    """
    Blocker Goat (Trapping / Ring Goat):
    Offensive containment strategy:
    - Actively minimizes tiger legal mobility
    - Penalizes allowing tiger captures (a captured goat reduces blocking workforce)
    - Positions goats to surround and immobilize the most constrained tiger
    """
    def __init__(
        self,
        name: str = "Blocker-Goat",
        side: int = GOAT,
        seed: Optional[int] = None,
        w_capture_penalty: float = 600.0,
        w_tiger_freedom: float = 20.0,
        w_cluster: float = 4.0,
    ):
        super().__init__(name=name, side=side)
        self.rng = random.Random(seed)
        self.w_capture_penalty = w_capture_penalty
        self.w_tiger_freedom = w_tiger_freedom
        self.w_cluster = w_cluster

    def _score_action(self, state: BaghChalState, action: Action) -> float:
        next_s = state.apply_action(action)

        # 1. Heavy penalty if action allows a tiger capture
        captures_allowed = _count_tiger_captures_possible(next_s)
        score = - (captures_allowed * self.w_capture_penalty)

        # 2. Constricting tiger mobility (primary objective)
        tiger_freedom = _compute_tiger_total_freedom(next_s)
        score -= (tiger_freedom * self.w_tiger_freedom)

        # 3. Adjacent friendly protection
        for nbr in ADJACENCY_LIST[action.to_pos]:
            if next_s.board[nbr] == GOAT:
                score += self.w_cluster

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

