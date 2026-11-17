"""
Bagh-Chal State Representation, Action Definitions, and Canonical Indexing.

Implements immutable state representation, Zobrist hashing for O(1) state lookup,
canonical action encoding/decoding, and state transformation mechanics.
"""

from dataclasses import dataclass
from enum import IntEnum
import random
from typing import Tuple, Optional, List, Dict

from src.environment.board import (
    BOARD_SIZE,
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    INITIAL_TIGER_POSITIONS,
    TOTAL_GOATS,
    MAX_CAPTURES_TO_WIN,
    ADJACENCY_LIST,
    JUMP_TABLE,
    pos_to_coord,
    coord_to_pos,
)


class ActionType(IntEnum):
    PLACE = 0
    MOVE = 1
    JUMP = 2


class Phase(IntEnum):
    PLACEMENT = 0
    MOVEMENT = 1


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    from_pos: Optional[int]
    to_pos: int
    captured_pos: Optional[int] = None

    def __str__(self) -> str:
        if self.action_type == ActionType.PLACE:
            r, c = pos_to_coord(self.to_pos)
            return f"PLACE({r},{c})"
        elif self.action_type == ActionType.MOVE:
            r1, c1 = pos_to_coord(self.from_pos)
            r2, c2 = pos_to_coord(self.to_pos)
            return f"MOVE(({r1},{c1})->({r2},{c2}))"
        else:
            r1, c1 = pos_to_coord(self.from_pos)
            r2, c2 = pos_to_coord(self.to_pos)
            rc, cc = pos_to_coord(self.captured_pos)
            return f"JUMP(({r1},{c1})->({r2},{c2}) x ({rc},{cc}))"


def _build_action_space() -> Tuple[List[Action], Dict[Action, int]]:
    """
    Construct complete canonical action space for Bagh-Chal.
    1. 25 placement actions (one for each empty board position)
    2. Directed movement actions along valid adjacent edges
    3. Directed jump actions along valid straight lines
    """
    actions: List[Action] = []

    # 1. Placement actions
    for p in range(NUM_POINTS):
        actions.append(Action(ActionType.PLACE, from_pos=None, to_pos=p))

    # 2. Movement actions
    for u in range(NUM_POINTS):
        for v in ADJACENCY_LIST[u]:
            actions.append(Action(ActionType.MOVE, from_pos=u, to_pos=v))

    # 3. Jump actions
    for (u, mid), v in JUMP_TABLE.items():
        actions.append(Action(ActionType.JUMP, from_pos=u, to_pos=v, captured_pos=mid))

    action_to_idx: Dict[Action, int] = {act: idx for idx, act in enumerate(actions)}
    return actions, action_to_idx


CANONICAL_ACTIONS, ACTION_TO_INDEX = _build_action_space()
TOTAL_ACTIONS: int = len(CANONICAL_ACTIONS)


class ZobristHasher:
    """
    Zobrist hashing for O(1) state hash updates and fast cycle/repetition detection.
    """
    def __init__(self, seed: int = 1337):
        rng = random.Random(seed)
        # 25 points x 3 states (EMPTY=0, TIGER=1, GOAT=2)
        self.table = [
            [rng.getrandbits(64) for _ in range(3)]
            for _ in range(NUM_POINTS)
        ]
        # Turn keys: 0 for TIGER, 1 for GOAT
        self.turn_keys = [rng.getrandbits(64) for _ in range(3)]
        # Phase keys: 0 for PLACEMENT, 1 for MOVEMENT
        self.phase_keys = [rng.getrandbits(64) for _ in range(2)]
        # Goats unplaced keys: 0..20
        self.goats_unplaced_keys = [rng.getrandbits(64) for _ in range(TOTAL_GOATS + 1)]
        # Goats captured keys: 0..5
        self.goats_captured_keys = [rng.getrandbits(64) for _ in range(MAX_CAPTURES_TO_WIN + 1)]

    def compute_full_hash(
        self,
        board: Tuple[int, ...],
        turn: int,
        phase: Phase,
        goats_unplaced: int,
        goats_captured: int,
    ) -> int:
        h = 0
        for pos, piece in enumerate(board):
            h ^= self.table[pos][piece]
        h ^= self.turn_keys[turn]
        h ^= self.phase_keys[int(phase)]
        h ^= self.goats_unplaced_keys[goats_unplaced]
        h ^= self.goats_captured_keys[min(goats_captured, MAX_CAPTURES_TO_WIN)]
        return h


GLOBAL_ZOBRIST = ZobristHasher(seed=42)


@dataclass(frozen=True)
class BaghChalState:
    """
    Immutable state of a Bagh-Chal board.
    """
    board: Tuple[int, ...]
    turn: int                     # TIGER (1) or GOAT (2)
    phase: Phase                  # PLACEMENT (0) or MOVEMENT (1)
    goats_unplaced: int           # 0..20
    goats_captured: int           # 0..5
    move_count: int               # Total plies played
    plies_since_capture: int      # Plies since last goat capture (for 100-ply draw rule)
    zobrist_hash: int

    @classmethod
    def initial_state(cls) -> "BaghChalState":
        """Create the canonical initial state of Bagh-Chal."""
        b = [EMPTY] * NUM_POINTS
        for pos in INITIAL_TIGER_POSITIONS:
            b[pos] = TIGER
        board_tuple = tuple(b)
        turn = GOAT
        phase = Phase.PLACEMENT
        goats_unplaced = TOTAL_GOATS
        goats_captured = 0
        move_count = 0
        plies_since_capture = 0
        z_hash = GLOBAL_ZOBRIST.compute_full_hash(
            board_tuple, turn, phase, goats_unplaced, goats_captured
        )
        return cls(
            board=board_tuple,
            turn=turn,
            phase=phase,
            goats_unplaced=goats_unplaced,
            goats_captured=goats_captured,
            move_count=move_count,
            plies_since_capture=plies_since_capture,
            zobrist_hash=z_hash,
        )

    def apply_action(self, action: Action) -> "BaghChalState":
        """
        Transition function: return new immutable state after applying action.
        """
        b = list(self.board)
        next_turn = TIGER if self.turn == GOAT else GOAT
        next_unplaced = self.goats_unplaced
        next_captured = self.goats_captured
        next_phase = self.phase
        plies_since_cap = self.plies_since_capture + 1

        if action.action_type == ActionType.PLACE:
            if self.turn != GOAT:
                raise ValueError("Only Goats can place pieces")
            if b[action.to_pos] != EMPTY:
                raise ValueError(f"Cannot place on non-empty position {action.to_pos}")
            b[action.to_pos] = GOAT
            next_unplaced -= 1
            if next_unplaced == 0:
                next_phase = Phase.MOVEMENT

        elif action.action_type == ActionType.MOVE:
            piece = b[action.from_pos]
            if piece != self.turn:
                raise ValueError(f"Cannot move piece {piece} on turn of {self.turn}")
            if b[action.to_pos] != EMPTY:
                raise ValueError(f"Destination {action.to_pos} not empty")
            b[action.from_pos] = EMPTY
            b[action.to_pos] = piece

        elif action.action_type == ActionType.JUMP:
            if self.turn != TIGER:
                raise ValueError("Only Tigers can jump/capture")
            if b[action.from_pos] != TIGER:
                raise ValueError("Source must have a Tiger")
            if b[action.to_pos] != EMPTY:
                raise ValueError("Destination must be EMPTY")
            if b[action.captured_pos] != GOAT:
                raise ValueError("Jumped piece must be a GOAT")
            b[action.from_pos] = EMPTY
            b[action.captured_pos] = EMPTY
            b[action.to_pos] = TIGER
            next_captured += 1
            plies_since_cap = 0  # Reset ply counter on capture

        else:
            raise ValueError(f"Unknown action type {action.action_type}")

        board_tuple = tuple(b)
        z_hash = GLOBAL_ZOBRIST.compute_full_hash(
            board_tuple, next_turn, next_phase, next_unplaced, next_captured
        )

        return BaghChalState(
            board=board_tuple,
            turn=next_turn,
            phase=next_phase,
            goats_unplaced=next_unplaced,
            goats_captured=next_captured,
            move_count=self.move_count + 1,
            plies_since_capture=plies_since_cap,
            zobrist_hash=z_hash,
        )

    def count_pieces(self) -> Tuple[int, int]:
        """Count (num_tigers, num_goats_on_board)."""
        tigers = sum(1 for p in self.board if p == TIGER)
        goats = sum(1 for p in self.board if p == GOAT)
        return tigers, goats

