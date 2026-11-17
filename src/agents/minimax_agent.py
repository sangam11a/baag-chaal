"""
Minimax Agent with Alpha-Beta Pruning, Transposition Table, and Move Ordering.

High-performance search agent for Bagh-Chal:
- Asymmetric heuristic evaluation function
- Zobrist-keyed transposition table
- Move ordering (captures & TT-best moves first)
- Configurable search depth
"""

from typing import List, Optional, Tuple, Dict
import math
import time

from src.environment.board import (
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    pos_to_coord,
    ADJACENCY_LIST,
    JUMPS_BY_SOURCE,
    MAX_CAPTURES_TO_WIN,
    INITIAL_TIGER_POSITIONS,
)
from src.environment.state import BaghChalState, Action, ActionType, Phase
from src.environment.rules import BaghChalRules
from src.agents.base_agent import BaseAgent

# Transposition Table Entry Flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

WIN_VAL = 100000.0


def default_eval_function(state: BaghChalState) -> float:
    """
    Asymmetric board evaluation from the perspective of TIGER (Maximizer).
    Positive = Tiger advantage. Negative = Goat advantage.
    """
    # 1. Material score (captures are massive for tiger)
    material = state.goats_captured * 2500.0

    # 2. Tiger mobility & trapped count
    b = state.board
    total_mobility = 0
    tigers_trapped = 0
    tiger_center_dist = 0

    for pos in range(NUM_POINTS):
        if b[pos] == TIGER:
            mob = 0
            for nbr in ADJACENCY_LIST[pos]:
                if b[nbr] == EMPTY:
                    mob += 1
            for mid, dest in JUMPS_BY_SOURCE[pos]:
                if b[mid] == GOAT and b[dest] == EMPTY:
                    mob += 3  # Jumps carry high mobility weight
            if mob == 0:
                tigers_trapped += 1
            total_mobility += mob

            r, c = pos_to_coord(pos)
            tiger_center_dist += abs(r - 2) + abs(c - 2)

    # 3. Goat clustering / safety
    goat_clusters = 0
    for pos in range(NUM_POINTS):
        if b[pos] == GOAT:
            for nbr in ADJACENCY_LIST[pos]:
                if b[nbr] == GOAT:
                    goat_clusters += 1

    # Tiger rewards: mobility, captures.
    # Tiger penalties: trapped tigers, corner confinement, dense goat clusters.
    score = (
        material
        + total_mobility * 40.0
        - tigers_trapped * 1500.0
        - tiger_center_dist * 15.0
        - goat_clusters * 5.0
    )
    return score


class MinimaxAgent(BaseAgent):
    """
    Minimax agent with Alpha-Beta pruning, Transposition Table, and Move Ordering.
    """
    def __init__(
        self,
        name: str = "Minimax-AlphaBeta",
        side: int = TIGER,
        max_depth: int = 4,
        eval_fn = default_eval_function,
        time_limit_sec: Optional[float] = None,
    ):
        super().__init__(name=name, side=side)
        self.max_depth = max_depth
        self.eval_fn = eval_fn
        self.time_limit_sec = time_limit_sec
        # Transposition table: hash -> (depth, flag, value, best_action)
        self.transposition_table: Dict[int, Tuple[int, int, float, Optional[Action]]] = {}
        self.nodes_searched = 0

    def reset(self) -> None:
        self.transposition_table.clear()
        self.nodes_searched = 0

    def _order_moves(
        self,
        state: BaghChalState,
        actions: List[Action],
        tt_move: Optional[Action] = None,
    ) -> List[Action]:
        """
        Move ordering for maximal alpha-beta cutoffs:
        1. Transposition table principal variation move first
        2. Tiger jumps (captures)
        3. Normal moves
        """
        def move_priority(act: Action) -> int:
            if act == tt_move:
                return 1000
            if act.action_type == ActionType.JUMP:
                return 500
            return 0

        return sorted(actions, key=move_priority, reverse=True)

    def _alpha_beta(
        self,
        state: BaghChalState,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> Tuple[float, Optional[Action]]:
        self.nodes_searched += 1
        orig_alpha = alpha
        h = state.zobrist_hash

        # Transposition table lookup
        tt_action = None
        if h in self.transposition_table:
            tt_depth, tt_flag, tt_val, tt_act = self.transposition_table[h]
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_val, tt_act
                elif tt_flag == TT_LOWERBOUND:
                    alpha = max(alpha, tt_val)
                elif tt_flag == TT_UPPERBOUND:
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val, tt_act
            tt_action = tt_act

        # Terminal state check
        is_term, winner, _ = BaghChalRules.get_game_status(state)
        if is_term:
            if winner == TIGER:
                return WIN_VAL - ply, None
            elif winner == GOAT:
                return -WIN_VAL + ply, None
            else:
                return 0.0, None  # Draw

        if depth == 0:
            return self.eval_fn(state), None

        legal_actions = BaghChalRules.get_legal_actions(state)
        if not legal_actions:
            # Stalemate / trapped
            if state.turn == TIGER:
                return -WIN_VAL + ply, None
            else:
                return WIN_VAL - ply, None

        ordered_actions = self._order_moves(state, legal_actions, tt_action)
        best_action = ordered_actions[0]

        if state.turn == TIGER:
            # Maximizer
            max_eval = -math.inf
            for act in ordered_actions:
                next_s = state.apply_action(act)
                val, _ = self._alpha_beta(next_s, depth - 1, alpha, beta, ply + 1)
                if val > max_eval:
                    max_eval = val
                    best_action = act
                alpha = max(alpha, val)
                if beta <= alpha:
                    break  # Beta cutoff
            eval_result = max_eval
        else:
            # Minimizer (Goat)
            min_eval = math.inf
            for act in ordered_actions:
                next_s = state.apply_action(act)
                val, _ = self._alpha_beta(next_s, depth - 1, alpha, beta, ply + 1)
                if val < min_eval:
                    min_eval = val
                    best_action = act
                beta = min(beta, val)
                if beta <= alpha:
                    break  # Alpha cutoff
            eval_result = min_eval

        # Transposition table store
        if eval_result <= orig_alpha:
            flag = TT_UPPERBOUND
        elif eval_result >= beta:
            flag = TT_LOWERBOUND
        else:
            flag = TT_EXACT
        self.transposition_table[h] = (depth, flag, eval_result, best_action)

        return eval_result, best_action

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        self.nodes_searched = 0
        if not legal_actions:
            raise ValueError("No legal actions.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Iterative deepening from depth 1 to max_depth
        best_act = legal_actions[0]
        start_time = time.time()

        for d in range(1, self.max_depth + 1):
            val, act = self._alpha_beta(state, depth=d, alpha=-math.inf, beta=math.inf, ply=0)
            if act is not None:
                best_act = act
            if self.time_limit_sec and (time.time() - start_time) >= self.time_limit_sec:
                break

        return best_act

