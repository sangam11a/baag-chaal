"""
Bagh-Chal Game Environment Package.
Formalization of the traditional Nepali asymmetric board game Bagh-Chal (Tigers and Goats).
"""

from src.environment.board import (
    BOARD_SIZE,
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    pos_to_coord,
    coord_to_pos,
    get_neighbors,
    get_jump_destinations,
    SYMMETRIES,
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
from src.environment.env import BaghChalEnv

__all__ = [
    "BOARD_SIZE",
    "NUM_POINTS",
    "EMPTY",
    "TIGER",
    "GOAT",
    "pos_to_coord",
    "coord_to_pos",
    "get_neighbors",
    "get_jump_destinations",
    "SYMMETRIES",
    "BaghChalState",
    "Action",
    "ActionType",
    "Phase",
    "BaghChalRules",
    "BaghChalEnv",
]
