"""
Bagh-Chal Rules Engine, Legal Action Generation, and Terminal State Evaluation.

Enforces game rules:
- Goat placement & movement
- Tiger movement & jump captures
- Win conditions: 5 captures (Tiger win), 4 tigers immobilized (Goat win)
- Draw conditions: 3-fold repetition, 100 plies without capture (movement phase)
"""

from typing import List, Optional, Dict
from src.environment.board import (
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    MAX_CAPTURES_TO_WIN,
    ADJACENCY_LIST,
    JUMPS_BY_SOURCE,
)
from src.environment.state import BaghChalState, Action, ActionType, Phase


class BaghChalRules:
    # Maximum plies without a capture before declaring a draw (50 full moves)
    MAX_PLIES_WITHOUT_CAPTURE: int = 100
    MAX_TOTAL_PLIES: int = 400

    @staticmethod
    def get_legal_actions(state: BaghChalState) -> List[Action]:
        """
        Generate all legal actions for the player whose turn it is.
        """
        legal: List[Action] = []
        b = state.board

        if state.turn == GOAT:
            if state.phase == Phase.PLACEMENT:
                # In placement phase, goats place one piece onto any empty intersection
                for pos in range(NUM_POINTS):
                    if b[pos] == EMPTY:
                        legal.append(Action(ActionType.PLACE, from_pos=None, to_pos=pos))
            else:
                # In movement phase, goats slide one piece to an adjacent empty intersection
                for pos in range(NUM_POINTS):
                    if b[pos] == GOAT:
                        for neighbor in ADJACENCY_LIST[pos]:
                            if b[neighbor] == EMPTY:
                                legal.append(Action(ActionType.MOVE, from_pos=pos, to_pos=neighbor))

        elif state.turn == TIGER:
            for pos in range(NUM_POINTS):
                if b[pos] == TIGER:
                    # 1. Normal steps to adjacent empty positions
                    for neighbor in ADJACENCY_LIST[pos]:
                        if b[neighbor] == EMPTY:
                            legal.append(Action(ActionType.MOVE, from_pos=pos, to_pos=neighbor))

                    # 2. Jump captures over an adjacent goat into an empty position
                    for mid_pos, to_pos in JUMPS_BY_SOURCE[pos]:
                        if b[mid_pos] == GOAT and b[to_pos] == EMPTY:
                            legal.append(
                                Action(ActionType.JUMP, from_pos=pos, to_pos=to_pos, captured_pos=mid_pos)
                            )

        return legal

    @staticmethod
    def is_tiger_win(state: BaghChalState) -> bool:
        """Tigers win immediately upon capturing 5 or more goats."""
        return state.goats_captured >= MAX_CAPTURES_TO_WIN

    @staticmethod
    def is_goat_win(state: BaghChalState, legal_actions: Optional[List[Action]] = None) -> bool:
        """
        Goats win if it is Tiger's turn and the tigers have NO legal moves
        (all four tigers are completely immobilized).
        """
        if state.turn != TIGER:
            return False
        if legal_actions is None:
            legal_actions = BaghChalRules.get_legal_actions(state)
        return len(legal_actions) == 0

    @classmethod
    def is_draw(
        cls,
        state: BaghChalState,
        repetition_count: int = 1,
        total_plies: Optional[int] = None,
    ) -> bool:
        """
        Check if current game state represents a draw via:
        1. 3-fold repetition
        2. 100 plies without a capture (in movement phase)
        3. Hard ply ceiling to prevent infinite games in self-play / search
        """
        if repetition_count >= 3:
            return True
        if state.phase == Phase.MOVEMENT and state.plies_since_capture >= cls.MAX_PLIES_WITHOUT_CAPTURE:
            return True
        plies = total_plies if total_plies is not None else state.move_count
        if plies >= cls.MAX_TOTAL_PLIES:
            return True
        return False

    @classmethod
    def get_game_status(
        cls,
        state: BaghChalState,
        repetition_count: int = 1,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Determine if the game is terminal and who won.
        Returns:
            (is_terminal: bool, winner: Optional[int], reason: str)
            winner: TIGER (1), GOAT (2), 0 (DRAW), or None (ongoing)
        """
        # 1. Tiger capture win
        if cls.is_tiger_win(state):
            return True, TIGER, "tiger_captured_5_goats"

        # 2. Check legal moves
        legal = cls.get_legal_actions(state)

        # 3. Goat immobilization win
        if state.turn == TIGER and len(legal) == 0:
            return True, GOAT, "tigers_immobilized"

        # 4. Rare stalemate: Goat on turn with no legal moves in movement phase
        if state.turn == GOAT and len(legal) == 0:
            # If goats cannot move, tigers win by immobilizing the flock
            return True, TIGER, "goats_immobilized"

        # 5. Draw conditions
        if cls.is_draw(state, repetition_count):
            if repetition_count >= 3:
                return True, 0, "threefold_repetition"
            if state.phase == Phase.MOVEMENT and state.plies_since_capture >= cls.MAX_PLIES_WITHOUT_CAPTURE:
                return True, 0, "plies_without_capture_exceeded"
            return True, 0, "max_total_plies_exceeded"

        return False, None, "ongoing"

