"""
Base Agent Interface for Bagh-Chal AI Agents.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from src.environment.state import BaghChalState, Action, ACTION_TO_INDEX


class BaseAgent(ABC):
    """
    Abstract Base Class for all Bagh-Chal agents.
    """
    def __init__(self, name: str, side: int):
        """
        Args:
            name: Human-readable agent identifier
            side: TIGER (1) or GOAT (2)
        """
        self.name = name
        self.side = side

    @abstractmethod
    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        """
        Choose an action from the list of legal actions in the given state.
        """
        pass

    def select_action_index(self, state: BaghChalState, legal_actions: List[Action]) -> int:
        """
        Return the integer index (0..TOTAL_ACTIONS-1) of the chosen action.
        """
        act = self.select_action(state, legal_actions)
        return ACTION_TO_INDEX[act]

    def reset(self) -> None:
        """Reset internal state/caches between games if applicable."""
        pass

