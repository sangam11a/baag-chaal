"""
Data Logging Engine for Bagh-Chal Tournaments and AI Evaluation.

Implements the JSON Lines and CSV logging schema defined in the consensus deliverables.
"""

from dataclasses import dataclass, asdict
import json
import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class GameRecord:
    game_id: str
    seed: int
    tiger_agent: str
    goat_agent: str
    result: str  # "tiger_win" | "goat_win" | "draw"
    winner: int  # 1 (TIGER), 2 (GOAT), 0 (DRAW)
    termination_reason: str
    goats_captured: int
    goats_placed: int
    total_moves: int
    tiger_moves: int
    goat_moves: int
    final_board_state: List[int]
    move_history: List[str]
    tiger_decision_times_ms: List[float]
    goat_decision_times_ms: List[float]
    timestamp_utc: str


class GameLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.games_file = os.path.join(log_dir, "tournament_games.jsonl")
        self.summary_csv = os.path.join(log_dir, "tournament_summary.csv")

        # Initialize summary CSV if not existing
        if not os.path.exists(self.summary_csv):
            with open(self.summary_csv, "w", encoding="utf-8") as f:
                f.write(
                    "game_id,seed,tiger_agent,goat_agent,result,winner,termination_reason,"
                    "goats_captured,total_moves,avg_tiger_time_ms,avg_goat_time_ms,timestamp_utc\n"
                )

    def log_game(self, record: GameRecord) -> None:
        # Append to JSON Lines
        with open(self.games_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        # Append to summary CSV
        avg_t_time = (
            sum(record.tiger_decision_times_ms) / len(record.tiger_decision_times_ms)
            if record.tiger_decision_times_ms else 0.0
        )
        avg_g_time = (
            sum(record.goat_decision_times_ms) / len(record.goat_decision_times_ms)
            if record.goat_decision_times_ms else 0.0
        )

        with open(self.summary_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{record.game_id},{record.seed},{record.tiger_agent},{record.goat_agent},"
                f"{record.result},{record.winner},{record.termination_reason},"
                f"{record.goats_captured},{record.total_moves},{avg_t_time:.3f},{avg_g_time:.3f},"
                f"{record.timestamp_utc}\n"
            )

