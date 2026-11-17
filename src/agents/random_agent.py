"""
Uniform Random Agent Baseline for Bagh-Chal.
"""

import random
from typing import List, Optional
from src.environment.state import BaghChalState, Action
from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Selects legal actions uniformly at random.
    Serves as the bottom-tier baseline benchmark.
    """
    def __init__(self, name: str = "Random", side: int = 1, seed: Optional[int] = None):
        super().__init__(name=name, side=side)
        self.rng = random.Random(seed)

    def select_action(self, state: BaghChalState, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        return self.rng.choice(legal_actions)

