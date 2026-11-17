"""
Tournament Runner and Evaluation Engine for Bagh-Chal AI Agents.

Supports:
- Paired match evaluation (swapping Tiger/Goat roles on identical seeds)
- Round-robin tournaments
- Metrics aggregation: win rates, draw rates, average game length, decision latency
"""

import uuid
import time
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone

from src.environment.board import TIGER, GOAT
from src.environment.state import Action
from src.environment.env import BaghChalEnv
from src.agents.base_agent import BaseAgent
from src.evaluation.logger import GameRecord, GameLogger


def play_game(
    tiger_agent: BaseAgent,
    goat_agent: BaseAgent,
    seed: int = 42,
    max_plies: int = 400,
) -> GameRecord:
    """
    Simulate a single game of Bagh-Chal between tiger_agent and goat_agent.
    """
    env = BaghChalEnv()
    env.reset(seed=seed)
    tiger_agent.reset()
    goat_agent.reset()

    game_id = str(uuid.uuid4())
    t_times: List[float] = []
    g_times: List[float] = []
    move_strings: List[str] = []

    done = False
    while not done:
        legal = env.get_legal_actions()
        acting_turn = env.state.turn

        t0 = time.perf_counter()
        if acting_turn == GOAT:
            action = goat_agent.select_action(env.state, legal)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            g_times.append(dt_ms)
        else:
            action = tiger_agent.select_action(env.state, legal)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            t_times.append(dt_ms)

        move_strings.append(str(action))
        _, _, term, trunc, info = env.step(action)
        done = term or trunc

    winner = info["winner"]
    if winner == TIGER:
        result = "tiger_win"
    elif winner == GOAT:
        result = "goat_win"
    else:
        result = "draw"

    return GameRecord(
        game_id=game_id,
        seed=seed,
        tiger_agent=tiger_agent.name,
        goat_agent=goat_agent.name,
        result=result,
        winner=winner,
        termination_reason=info["termination_reason"],
        goats_captured=info["goats_captured"],
        goats_placed=20 - info["goats_unplaced"],
        total_moves=info["move_count"],
        tiger_moves=len(t_times),
        goat_moves=len(g_times),
        final_board_state=list(env.state.board),
        move_history=move_strings,
        tiger_decision_times_ms=t_times,
        goat_decision_times_ms=g_times,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def play_paired_match(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    seed: int = 42,
) -> Tuple[GameRecord, GameRecord]:
    """
    Play 2 games on the same seed with roles reversed:
    Game 1: Agent A as Tiger, Agent B as Goat
    Game 2: Agent B as Tiger, Agent A as Goat
    """
    game1 = play_game(tiger_agent=agent_a, goat_agent=agent_b, seed=seed)
    game2 = play_game(tiger_agent=agent_b, goat_agent=agent_a, seed=seed)
    return game1, game2

