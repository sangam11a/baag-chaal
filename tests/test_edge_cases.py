"""
Exhaustive edge-case and boundary condition unit tests for Bagh-Chal AI engine.
Tests illegal jump configurations, multi-piece obstructions, Zobrist hash integrity,
symmetry invariance, and full game rollouts.
"""

import pytest
import random
from src.environment.board import (
    BOARD_SIZE,
    NUM_POINTS,
    EMPTY,
    TIGER,
    GOAT,
    coord_to_pos,
    pos_to_coord,
    SYMMETRIES,
)
from src.environment.state import (
    BaghChalState,
    Action,
    ActionType,
    Phase,
    GLOBAL_ZOBRIST,
)
from src.environment.rules import BaghChalRules
from src.environment.env import BaghChalEnv


def test_tiger_cannot_jump_over_another_tiger():
    # Board with Tiger at 0 (0,0), another Tiger at 1 (0,1), empty at 2 (0,2)
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = TIGER
    b[4] = TIGER
    b[20] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=20,
        goats_captured=0,
        move_count=1,
        plies_since_capture=1,
        zobrist_hash=0,
    )

    legal = BaghChalRules.get_legal_actions(s)
    for act in legal:
        assert act.action_type != ActionType.JUMP or act.captured_pos != 1


def test_tiger_cannot_jump_over_empty_space():
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = EMPTY
    b[2] = EMPTY
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=20,
        goats_captured=0,
        move_count=1,
        plies_since_capture=1,
        zobrist_hash=0,
    )

    legal = BaghChalRules.get_legal_actions(s)
    jump_actions = [a for a in legal if a.action_type == ActionType.JUMP]
    assert len(jump_actions) == 0


def test_tiger_cannot_jump_if_destination_occupied():
    # Goat at 1, another piece (Goat or Tiger) at 2
    for blocking_piece in [GOAT, TIGER]:
        b = [EMPTY] * NUM_POINTS
        b[0] = TIGER
        b[1] = GOAT
        b[2] = blocking_piece
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
        jump_actions = [a for a in legal if a.action_type == ActionType.JUMP and a.from_pos == 0]
        assert len(jump_actions) == 0


def test_tiger_cannot_jump_diagonally_from_odd_parity_position():
    # Position (0, 1) = pos 1 has odd parity (r+c = 1) -> NO diagonals allowed
    # Even if pieces align diagonally: Tiger at (0,1), Goat at (1,2), Empty at (2,3)
    p_tiger = coord_to_pos(0, 1)
    p_goat = coord_to_pos(1, 2)
    p_landing = coord_to_pos(2, 3)

    b = [EMPTY] * NUM_POINTS
    b[p_tiger] = TIGER
    b[p_goat] = GOAT
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
    illegal_diagonal_jump = [
        a for a in legal
        if a.action_type == ActionType.JUMP and a.from_pos == p_tiger and a.to_pos == p_landing
    ]
    assert len(illegal_diagonal_jump) == 0


def test_tiger_valid_diagonal_jump():
    # Position (0, 0) = pos 0 has even parity (r+c = 0) -> Diagonal jump IS allowed:
    # Tiger at (0,0), Goat at (1,1), Empty at (2,2)
    p_tiger = coord_to_pos(0, 0)
    p_goat = coord_to_pos(1, 1)
    p_landing = coord_to_pos(2, 2)

    b = [EMPTY] * NUM_POINTS
    b[p_tiger] = TIGER
    b[p_goat] = GOAT
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
    diag_jump = [
        a for a in legal
        if a.action_type == ActionType.JUMP and a.from_pos == p_tiger and a.to_pos == p_landing
    ]
    assert len(diag_jump) == 1
    assert diag_jump[0].captured_pos == p_goat


def test_tiger_cannot_step_onto_goat():
    # Tigers can only capture by jumping, never by stepping onto an adjacent goat!
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
    direct_step_onto_goat = [
        a for a in legal
        if a.action_type == ActionType.MOVE and a.from_pos == 0 and a.to_pos == 1
    ]
    assert len(direct_step_onto_goat) == 0


def test_zobrist_hash_determinism_and_uniqueness():
    s1 = BaghChalState.initial_state()
    s2 = BaghChalState.initial_state()
    assert s1.zobrist_hash == s2.zobrist_hash

    # Making a move alters the hash
    act = BaghChalRules.get_legal_actions(s1)[0]
    next_s = s1.apply_action(act)
    assert next_s.zobrist_hash != s1.zobrist_hash


def test_full_random_playout_terminates_cleanly():
    # Run 5 complete random games to verify no crashes, deadlocks, or illegal states
    random.seed(42)
    env = BaghChalEnv()

    for game_idx in range(5):
        env.reset()
        done = False
        step_count = 0

        while not done:
            legal = env.get_legal_actions()
            assert len(legal) > 0, "Non-terminal state has 0 legal actions"
            act = random.choice(legal)
            _, _, term, trunc, info = env.step(act)
            done = term or trunc
            step_count += 1
            assert step_count <= 450, "Game exceeded hard ceiling"

        assert info["winner"] in [TIGER, GOAT, 0]


def test_repetition_tracking_in_env():
    # Make a back-and-forth movement cycle in movement phase to confirm 3-fold repetition triggers draw
    env = BaghChalEnv()
    # Construct a movement state
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER
    # Place 16 goats in fixed positions
    for p in range(6, 20):
        b[p] = GOAT

    env.state = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=4,
        move_count=40,
        plies_since_capture=0,
        zobrist_hash=GLOBAL_ZOBRIST.compute_full_hash(tuple(b), TIGER, Phase.MOVEMENT, 0, 4),
    )
    env.repetition_history = {env.state.zobrist_hash: 1}

    # Tiger moves 0 -> 1
    env.step(Action(ActionType.MOVE, from_pos=0, to_pos=1))
    # Goat moves 6 -> 5
    env.step(Action(ActionType.MOVE, from_pos=6, to_pos=5))
    # Tiger moves 1 -> 0
    env.step(Action(ActionType.MOVE, from_pos=1, to_pos=0))
    # Goat moves 5 -> 6 (back to original state! Occurrence 2)
    env.step(Action(ActionType.MOVE, from_pos=5, to_pos=6))
    assert not env._terminated

    # Repeat cycle once more
    env.step(Action(ActionType.MOVE, from_pos=0, to_pos=1))
    env.step(Action(ActionType.MOVE, from_pos=6, to_pos=5))
    env.step(Action(ActionType.MOVE, from_pos=1, to_pos=0))
    _, _, term, _, info = env.step(Action(ActionType.MOVE, from_pos=5, to_pos=6))

    assert term is True
    assert info["winner"] == 0
    assert info["termination_reason"] == "threefold_repetition"


def test_tiger_double_jump_attempt_rejected():
    # Setup state with 2 collinear jump opportunities
    # Tiger at 0, Goat at 1, Goat at 3. Tiger jumps 0 -> 2. Can Tiger jump 2 -> 4 in same ply?
    # Bagh-Chal rules strictly limit to ONE jump per ply.
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[1] = GOAT
    b[3] = GOAT
    b[4] = EMPTY
    b[20] = TIGER
    b[24] = TIGER

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.PLACEMENT,
        goats_unplaced=18,
        goats_captured=0,
        move_count=1,
        plies_since_capture=1,
        zobrist_hash=0,
    )

    legal = BaghChalRules.get_legal_actions(s)
    # The jump from 0 must land on 2
    jumps_from_0 = [a for a in legal if a.action_type == ActionType.JUMP and a.from_pos == 0]
    assert len(jumps_from_0) == 1
    assert jumps_from_0[0].to_pos == 2

    # After jump to 2, turn MUST switch to GOAT
    next_s = s.apply_action(jumps_from_0[0])
    assert next_s.turn == GOAT
    assert next_s.board[2] == TIGER
    assert next_s.board[3] == GOAT  # Second goat was NOT captured


def test_action_roundtrip_mapping():
    from src.environment.state import TOTAL_ACTIONS, CANONICAL_ACTIONS, ACTION_TO_INDEX
    assert len(CANONICAL_ACTIONS) == TOTAL_ACTIONS
    assert len(ACTION_TO_INDEX) == TOTAL_ACTIONS
    for idx, act in enumerate(CANONICAL_ACTIONS):
        assert ACTION_TO_INDEX[act] == idx
        assert CANONICAL_ACTIONS[idx] == act


def test_state_immutability():
    s = BaghChalState.initial_state()
    original_board = s.board
    legal = BaghChalRules.get_legal_actions(s)
    s_next = s.apply_action(legal[0])
    # s must remain unchanged
    assert s.board == original_board
    assert s.board is not s_next.board
    assert s.move_count == 0
    assert s_next.move_count == 1


def test_all_8_symmetries_preserve_piece_counts():
    s = BaghChalState.initial_state()
    # Place a couple goats
    s = s.apply_action(Action(ActionType.PLACE, from_pos=None, to_pos=12))
    s = s.apply_action(Action(ActionType.MOVE, from_pos=0, to_pos=1))
    s = s.apply_action(Action(ActionType.PLACE, from_pos=None, to_pos=6))

    orig_tigers, orig_goats = s.count_pieces()

    for perm in SYMMETRIES:
        permuted_board = [EMPTY] * NUM_POINTS
        for pos in range(NUM_POINTS):
            permuted_board[perm[pos]] = s.board[pos]
        tigers = sum(1 for p in permuted_board if p == TIGER)
        goats = sum(1 for p in permuted_board if p == GOAT)
        assert tigers == orig_tigers
        assert goats == orig_goats


def test_env_reset_clears_history():
    env = BaghChalEnv()
    legal = env.get_legal_actions()
    env.step(legal[0])
    assert len(env.move_history) == 1
    assert env.state.move_count == 1

    env.reset()
    assert len(env.move_history) == 0
    assert env.state.move_count == 0
    assert len(env.repetition_history) == 1


def test_max_total_plies_triggers_draw():
    b = [EMPTY] * NUM_POINTS
    b[0] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER
    for p in range(5, 21):
        b[p] = GOAT

    s_max = BaghChalState(
        board=tuple(b),
        turn=GOAT,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=0,
        move_count=BaghChalRules.MAX_TOTAL_PLIES,
        plies_since_capture=10,
        zobrist_hash=0,
    )
    is_term, winner, reason = BaghChalRules.get_game_status(s_max)
    assert is_term is True
    assert winner == 0
    assert reason == "max_total_plies_exceeded"


def test_tiger_trapped_at_center_vs_corner():
    # Corner tiger at 0 needs only 3 surrounding pieces to be blocked
    # Center tiger at 12 needs all 8 surrounding positions blocked
    b = [EMPTY] * NUM_POINTS
    b[12] = TIGER
    b[4] = TIGER
    b[20] = TIGER
    b[24] = TIGER

    from src.environment.board import ADJACENCY_LIST, JUMPS_BY_SOURCE
    # Surround center tiger (pos 12) with goats
    for neighbor in ADJACENCY_LIST[12]:
        b[neighbor] = GOAT
    # Also block jump landing spots for center tiger
    for mid, dest in JUMPS_BY_SOURCE[12]:
        b[dest] = GOAT

    s = BaghChalState(
        board=tuple(b),
        turn=TIGER,
        phase=Phase.MOVEMENT,
        goats_unplaced=0,
        goats_captured=0,
        move_count=50,
        plies_since_capture=0,
        zobrist_hash=0,
    )
    legal = BaghChalRules.get_legal_actions(s)
    center_tiger_moves = [a for a in legal if a.from_pos == 12]
    assert len(center_tiger_moves) == 0
