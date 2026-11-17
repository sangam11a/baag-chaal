"""
Unit tests for Bagh-Chal rules, legal action generation, transitions, and win/draw conditions.
"""

import pytest
from src.environment.board import (
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    INITIAL_TIGER_POSITIONS,
    TOTAL_GOATS,
    MAX_CAPTURES_TO_WIN,
    coord_to_pos,
)
from src.environment.state import (
    BaghChalState,
    Action,
    ActionType,
    Phase,
    CANONICAL_ACTIONS,
    ACTION_TO_INDEX,
)
from src.environment.rules import BaghChalRules


def test_initial_state_setup():
    s = BaghChalState.initial_state()
    assert s.turn == GOAT
    assert s.phase == Phase.PLACEMENT
    assert s.goats_unplaced == 20
    assert s.goats_captured == 0
    assert s.move_count == 0
    assert s.plies_since_capture == 0

    # 4 tigers in corners
    for p in INITIAL_TIGER_POSITIONS:
        assert s.board[p] == TIGER
    # 21 empty points
    empty_count = sum(1 for p in s.board if p == EMPTY)
    assert empty_count == 21


def test_initial_legal_actions_all_placements():
    s = BaghChalState.initial_state()
    legal = BaghChalRules.get_legal_actions(s)
    # 21 empty positions, all must be PLACE actions
    assert len(legal) == 21
    for act in legal:
        assert act.action_type == ActionType.PLACE
        assert act.from_pos is None
        assert s.board[act.to_pos] == EMPTY
        assert act.to_pos not in INITIAL_TIGER_POSITIONS


def test_goats_cannot_move_in_placement_phase():
    s = BaghChalState.initial_state()
    # Place one goat at pos 12 (center)
    s = s.apply_action(Action(ActionType.PLACE, from_pos=None, to_pos=12))
    assert s.turn == TIGER
    # Tiger moves 0 -> 1
    s = s.apply_action(Action(ActionType.MOVE, from_pos=0, to_pos=1))
    assert s.turn == GOAT
    assert s.phase == Phase.PLACEMENT

    # Legal actions for goat must ONLY be PLACE, not MOVE
    legal = BaghChalRules.get_legal_actions(s)
    for act in legal:
        assert act.action_type == ActionType.PLACE


def test_tiger_normal_moves_in_placement():
    s = BaghChalState.initial_state()
    # Goat places at pos 12
    s = s.apply_action(Action(ActionType.PLACE, from_pos=None, to_pos=12))
    assert s.turn == TIGER

    tiger_legal = BaghChalRules.get_legal_actions(s)
    assert len(tiger_legal) > 0
    # Each corner tiger has 3 neighbors, all are empty in initial state
    # 4 tigers x 3 neighbors = 12 legal moves
    assert len(tiger_legal) == 12
    for act in tiger_legal:
        assert act.action_type == ActionType.MOVE
        assert act.from_pos in INITIAL_TIGER_POSITIONS


def test_tiger_jump_capture_mechanics():
    # Setup state where Tiger at 0 (0,0) can jump over Goat at 1 (0,1) landing at 2 (0,2)
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = GOAT
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=19,
        goats_captured=0,
        move_count=1,
        plies_since_capture=1,
        zobrist_hash=0,
    )

    legal = BaghChalRules.get_legal_actions(s)
    jump_actions = [a for a in legal if a.action_type == ActionType.JUMP]
    assert len(jump_actions) == 1
    jump = jump_actions[0]
    assert jump.from_pos == 0
    assert jump.captured_pos == 1
    assert jump.to_pos == 2

    # Execute jump
    next_s = s.apply_action(jump)
    assert next_s.board[0] == EMPTY
    assert next_s.board[1] == EMPTY
    assert next_s.board[2] == TIGER
    assert next_s.goats_captured == 1
    assert next_s.plies_since_capture == 0
    assert next_s.turn == GOAT


def test_phase_transition_to_movement():
    s = BaghChalState.initial_state()
    # Simulate 20 goat placements and alternating tiger moves
    for i in range(20):
        assert s.turn == GOAT
        empty_positions = [p for p in range(NUM_POINTS) if s.board[p] == EMPTY]
        place_pos = empty_positions[0]
        s = s.apply_action(Action(ActionType.PLACE, from_pos=None, to_pos=place_pos))

        if i < 19:
            assert s.turn == TIGER
            tiger_actions = BaghChalRules.get_legal_actions(s)
            s = s.apply_action(tiger_actions[0])

    # After 20th goat is placed, goats_unplaced == 0, phase must be MOVEMENT
    assert s.goats_unplaced == 0
    assert s.phase == Phase.MOVEMENT


def test_movement_phase_goats_move_only():
    # Directly construct a MOVEMENT phase state
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER
    # Place 16 goats
    for p in range(5, 20):
        b[p] = GOAT

    s = BaghChalState(
        board=tuple(b),
        turn=GOAT,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=4,
        move_count=40,
        plies_since_capture=5,
        zobrist_hash=0,
    )

    legal = BaghChalRules.get_legal_actions(s)
    assert len(legal) > 0
    for act in legal:
        assert act.action_type == ActionType.MOVE
        assert act.from_pos is not None
        assert s.board[act.from_pos] == GOAT
        assert s.board[act.to_pos] == EMPTY


def test_tiger_win_5_captures():
    s = BaghChalState.initial_state()
    # Artificially set 5 captures
    s5 = BaghChalState(
        board=s.board,
        turn=GOAT,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=5,
        move_count=50,
        plies_since_capture=0,
        zobrist_hash=0,
    )
    is_term, winner, reason = BaghChalRules.get_game_status(s5)
    assert is_term is True
    assert winner == TIGER
    assert reason == "tiger_captured_5_goats"


def test_goat_win_tigers_immobilized():
    # Surround a tiger at 0 (0,0): neighbors are (0,1)=1, (1,0)=5, (1,1)=6
    # And place goats such that tiger at 0 cannot jump:
    # jumps would land on (0,2)=2, (2,0)=10, (2,2)=12
    # If 2, 10, 12 are also occupied by GOAT, tiger cannot jump!
    # Trap all 4 tigers in their corners
    b = [EMPTY] * NUM_POINTS
    # 4 tigers at corners
    b[0] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    # Fill all neighbors and jump landing spots with goats
    for r in range(5):
        for c in range(5):
            pos = coord_to_pos(r, c)
            if pos not in INITIAL_TIGER_POSITIONS:
                b[pos] = GOAT  # 21 goats (or full board)

    s_trapped = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=0,
        move_count=40,
        plies_since_capture=40,
        zobrist_hash=0,
    )

    legal = BaghChalRules.get_legal_actions(s_trapped)
    assert len(legal) == 0

    is_term, winner, reason = BaghChalRules.get_game_status(s_trapped)
    assert is_term is True
    assert winner == GOAT
    assert reason == "tigers_immobilized"


def test_threefold_repetition_draw():
    s = BaghChalState.initial_state()
    is_term, winner, reason = BaghChalRules.get_game_status(s, repetition_count=3)
    assert is_term is True
    assert winner == 0
    assert reason == "threefold_repetition"


def test_plies_without_capture_draw():
    # Construct a valid movement phase state with 18 goats and 4 tigers
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER
    for p in range(5, 23):  # 18 goats on board
        b[p] = GOAT

    s_draw = BaghChalState(
        board=tuple(b),
        turn=GOAT,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=2,
        move_count=120,
        plies_since_capture=100,
        zobrist_hash=0,
    )
    is_term, winner, reason = BaghChalRules.get_game_status(s_draw)
    assert is_term is True
    assert winner == 0
    assert reason == "plies_without_capture_exceeded"
