"""
Bagh-Chal Board Representation and Graph Topology.

Defines the 5x5 board geometry, irregular connectivity graph (orthogonal and
diagonals on even parity vertices), jump paths, and D4 dihedral symmetry mappings.
"""

from typing import List, Tuple, Dict, Set

BOARD_SIZE: int = 5
NUM_POINTS: int = 25

# Piece identifiers
EMPTY: int = 0
TIGER: int = 1
GOAT: int = 2

# Initial tiger starting positions on the four board corners
INITIAL_TIGER_POSITIONS: Tuple[int, int, int, int] = (0, 4, 20, 24)

# Phase definitions
TOTAL_GOATS: int = 20
MAX_CAPTURES_TO_WIN: int = 5


def pos_to_coord(pos: int) -> Tuple[int, int]:
    """Convert flat 1D position [0..24] to 2D coordinate (row, col)."""
    if not (0 <= pos < NUM_POINTS):
        raise ValueError(f"Position {pos} out of range [0..24]")
    return divmod(pos, BOARD_SIZE)


def coord_to_pos(r: int, c: int) -> int:
    """Convert 2D coordinate (row, col) to flat 1D position [0..24]."""
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        raise ValueError(f"Coordinates ({r}, {c}) out of bounds [0..4]x[0..4]")
    return r * BOARD_SIZE + c


def is_diagonal_active(r: int, c: int) -> bool:
    """
    Check if a vertex at (r, c) participates in diagonal line connections.
    In Bagh-Chal topology, diagonal lines exist if and only if (r + c) % 2 == 0.
    """
    return (r + c) % 2 == 0


def _compute_adjacencies() -> Dict[int, List[int]]:
    """
    Precompute adjacency list for all 25 vertices on the Bagh-Chal board.
    Orthogonal connections exist between any adjacent grid points.
    Diagonal connections exist if and only if (r + c) is even.
    """
    adj: Dict[int, List[int]] = {p: [] for p in range(NUM_POINTS)}
    # 8 potential direction vectors: 4 orthogonal, 4 diagonal
    orthogonal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for pos in range(NUM_POINTS):
        r, c = pos_to_coord(pos)
        # Add orthogonal neighbors
        for dr, dc in orthogonal_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                adj[pos].append(coord_to_pos(nr, nc))

        # Add diagonal neighbors only if vertex has even coordinate sum
        if is_diagonal_active(r, c):
            for dr, dc in diagonal_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    adj[pos].append(coord_to_pos(nr, nc))

        adj[pos].sort()

    return adj


# Precomputed adjacency lookup: pos -> list of connected neighbor positions
ADJACENCY_LIST: Dict[int, List[int]] = _compute_adjacencies()
ADJACENCY_SETS: Dict[int, Set[int]] = {p: set(ADJACENCY_LIST[p]) for p in range(NUM_POINTS)}


def _compute_jump_table() -> Dict[Tuple[int, int], int]:
    """
    Precompute valid jump paths for tigers.
    Returns mapping: (from_pos, mid_pos) -> to_pos
    where:
      - mid_pos is an adjacent connected neighbor of from_pos
      - to_pos is two steps in the exact straight direction from from_pos through mid_pos
      - from_pos -> mid_pos -> to_pos lies on a valid board line
    """
    jump_table: Dict[Tuple[int, int], int] = {}

    for from_pos in range(NUM_POINTS):
        r1, c1 = pos_to_coord(from_pos)
        for mid_pos in ADJACENCY_LIST[from_pos]:
            r2, c2 = pos_to_coord(mid_pos)
            dr = r2 - r1
            dc = c2 - c1

            r3 = r2 + dr
            c3 = c2 + dc

            if 0 <= r3 < BOARD_SIZE and 0 <= c3 < BOARD_SIZE:
                to_pos = coord_to_pos(r3, c3)
                # Verify that mid_pos -> to_pos is also an active board connection
                if to_pos in ADJACENCY_SETS[mid_pos]:
                    jump_table[(from_pos, mid_pos)] = to_pos

    return jump_table


JUMP_TABLE: Dict[Tuple[int, int], int] = _compute_jump_table()


def _compute_jumps_from_position() -> Dict[int, List[Tuple[int, int]]]:
    """
    Precompute mapping: from_pos -> list of (mid_pos, to_pos)
    """
    res: Dict[int, List[Tuple[int, int]]] = {p: [] for p in range(NUM_POINTS)}
    for (from_pos, mid_pos), to_pos in JUMP_TABLE.items():
        res[from_pos].append((mid_pos, to_pos))
    return res


JUMPS_BY_SOURCE: Dict[int, List[Tuple[int, int]]] = _compute_jumps_from_position()


def get_neighbors(pos: int) -> List[int]:
    """Get list of valid adjacent neighbor positions for a given board position."""
    return ADJACENCY_LIST[pos]


def get_jump_destinations(from_pos: int) -> List[Tuple[int, int]]:
    """Get list of (mid_pos, to_pos) jump opportunities originating from from_pos."""
    return JUMPS_BY_SOURCE[from_pos]


def _compute_d4_symmetries() -> List[List[int]]:
    """
    Precompute the 8 dihedral symmetries of the 5x5 board as index permutations.
    Each symmetry is a list P of length 25 where P[old_pos] = new_pos.
    The 8 symmetries are:
      0: Identity
      1: Rotate 90 deg clockwise
      2: Rotate 180 deg
      3: Rotate 270 deg clockwise
      4: Horizontal reflection (flip vertically, across middle horizontal line)
      5: Vertical reflection (flip horizontally, across middle vertical line)
      6: Main diagonal reflection (transpose, r <-> c)
      7: Anti-diagonal reflection (r <-> 4-c, c <-> 4-r)
    """
    symmetries: List[List[int]] = []

    transformations = [
        lambda r, c: (r, c),                      # 0: Identity
        lambda r, c: (c, BOARD_SIZE - 1 - r),      # 1: Rot 90 CW
        lambda r, c: (BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c),  # 2: Rot 180
        lambda r, c: (BOARD_SIZE - 1 - c, r),      # 3: Rot 270 CW
        lambda r, c: (BOARD_SIZE - 1 - r, c),      # 4: Flip V (across row 2)
        lambda r, c: (r, BOARD_SIZE - 1 - c),      # 5: Flip H (across col 2)
        lambda r, c: (c, r),                      # 6: Main diagonal
        lambda r, c: (BOARD_SIZE - 1 - c, BOARD_SIZE - 1 - r),  # 7: Anti-diagonal
    ]

    for trans in transformations:
        perm = [0] * NUM_POINTS
        for pos in range(NUM_POINTS):
            r, c = pos_to_coord(pos)
            nr, nc = trans(r, c)
            perm[pos] = coord_to_pos(nr, nc)
        symmetries.append(perm)

    return symmetries


SYMMETRIES: List[List[int]] = _compute_d4_symmetries()


def get_graph_edges() -> List[Tuple[int, int]]:
    """
    Returns list of undirected edges (u, v) with u < v for GNN construction.
    """
    edges: List[Tuple[int, int]] = []
    for u in range(NUM_POINTS):
        for v in ADJACENCY_LIST[u]:
            if u < v:
                edges.append((u, v))
    return edges

