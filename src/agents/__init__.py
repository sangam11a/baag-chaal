"""
Bagh-Chal AI Agents Package.
Includes baseline random, rule-based heuristics, and search-based Minimax agents.
"""

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_tiger import GreedyTigerAgent, PositionalTigerAgent
from src.agents.heuristic_goat import DefensiveGoatAgent, BlockerGoatAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.mcts_agent import MCTSAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "GreedyTigerAgent",
    "PositionalTigerAgent",
    "DefensiveGoatAgent",
    "BlockerGoatAgent",
    "MinimaxAgent",
    "MCTSAgent",
]
