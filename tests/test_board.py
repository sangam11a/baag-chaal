"""
Unit tests for Bagh-Chal board topology, coordinates, adjacency, jump table, and symmetries.
"""

import pytest
from src.environment.board import (
    BOARD_SIZE,
    NUM_POINTS,
    pos_to_coord,
    coord_to_pos,
    is_diagonal_active,
    get_neighbors,
    get_jump_destinations,
    ADJACENCY_LIST,
    ADJACENCY_SETS,
    JUMP_TABLE,
    SYMMETRIES,
    get_graph_edges,
)


def test_board_constants():
    assert BOARD_SIZE == 5
    assert NUM_POINTS == 25


def test_pos_coord_bijective_mapping():
    for pos in range(NUM_POINTS):
        r, c = pos_to_coord(pos)
        assert 0 <= r < BOARD_SIZE
        assert 0 <= c < BOARD_SIZE
        assert coord_to_pos(r, c) == pos


def test_pos_out_of_bounds():
    with pytest.raises(ValueError):
        pos_to_coord(-1)
    with pytest.raises(ValueError):
        pos_to_coord(25)
    with pytest.raises(ValueError):
        coord_to_pos(-1, 0)
    with pytest.raises(ValueError):
        coord_to_pos(0, 5)


def test_diagonal_parity_rule():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            expected = ((r + c) % 2 == 0)
            assert is_diagonal_active(r, c) == expected


def test_corner_degrees():
    # 4 corners have (r+c) even: (0,0), (0,4), (4,0), (4,4)
    # Each corner has 2 orthogonal neighbors + 1 diagonal neighbor = degree 3
    corners = [0, 4, 20, 24]
    for p in corners:
        neighbors = get_neighbors(p)
        assert len(neighbors) == 3, f"Corner {p} expected degree 3, got {len(neighbors)}"


def test_center_degree():
    # Center (2, 2) is pos 12 (even sum 4): 4 orthogonal + 4 diagonal = 8 neighbors
    center = 12
    neighbors = get_neighbors(center)
    assert len(neighbors) == 8


def test_diamond_nodes_degrees():
    # Diamond nodes: (1,1), (1,3), (3,1), (3,3) -> positions 6, 8, 16, 18
    # Each has (r+c) even and is an interior node: 4 orthogonal + 4 diagonal = 8 neighbors
    diamond_nodes = [6, 8, 16, 18]
    for p in diamond_nodes:
        assert len(get_neighbors(p)) == 8, f"Diamond node {p} expected degree 8, got {len(get_neighbors(p))}"


def test_odd_parity_nodes_have_no_diagonals():
    # Any node with (r+c) odd must have NO diagonal connections
    for pos in range(NUM_POINTS):
        r, c = pos_to_coord(pos)
        if not is_diagonal_active(r, c):
            for neighbor in get_neighbors(pos):
                nr, nc = pos_to_coord(neighbor)
                # Must be strictly orthogonal: |dr| + |dc| == 1
                assert abs(nr - r) + abs(nc - c) == 1, (
                    f"Odd parity node ({r},{c}) illegally connected diagonally to ({nr},{nc})"
                )


def test_graph_symmetry_undirected():
    # If v in neighbors(u), then u must be in neighbors(v)
    for u in range(NUM_POINTS):
        for v in get_neighbors(u):
            assert u in get_neighbors(v), f"Asymmetric edge between {u} and {v}"


def test_jump_table_properties():
    # JUMP_TABLE: (from_pos, mid_pos) -> to_pos
    # 1. from_pos and mid_pos must be neighbors
    # 2. mid_pos and to_pos must be neighbors
    # 3. from_pos, mid_pos, to_pos must be collinear with equal step sizes
    for (from_pos, mid_pos), to_pos in JUMP_TABLE.items():
        assert mid_pos in ADJACENCY_SETS[from_pos]
        assert to_pos in ADJACENCY_SETS[mid_pos]

        r1, c1 = pos_to_coord(from_pos)
        r2, c2 = pos_to_coord(mid_pos)
        r3, c3 = pos_to_coord(to_pos)

        assert (r2 - r1) == (r3 - r2), f"Non-collinear jump: {r1},{c1} -> {r2},{c2} -> {r3},{c3}"
        assert (c2 - c1) == (c3 - c2), f"Non-collinear jump: {r1},{c1} -> {r2},{c2} -> {r3},{c3}"


def test_jump_destinations_consistency():
    for pos in range(NUM_POINTS):
        jumps = get_jump_destinations(pos)
        for mid_pos, to_pos in jumps:
            assert (pos, mid_pos) in JUMP_TABLE
            assert JUMP_TABLE[(pos, mid_pos)] == to_pos


def test_symmetries_group_properties():
    assert len(SYMMETRIES) == 8

    # 1. Identity is index 0
    assert SYMMETRIES[0] == list(range(NUM_POINTS))

    # 2. Every symmetry is a bijection (permutation of 0..24)
    for i, perm in enumerate(SYMMETRIES):
        assert len(perm) == NUM_POINTS
        assert set(perm) == set(range(NUM_POINTS)), f"Symmetry {i} is not a valid permutation"

    # 3. Every symmetry must preserve graph adjacency (Graph Automorphism)
    for i, perm in enumerate(SYMMETRIES):
        for u in range(NUM_POINTS):
            pu = perm[u]
            u_neighbors = set(get_neighbors(u))
            pu_neighbors = set(get_neighbors(pu))
            permuted_u_neighbors = {perm[v] for v in u_neighbors}
            assert permuted_u_neighbors == pu_neighbors, (
                f"Symmetry {i} violated graph automorphism at vertex {u}"
            )


def test_graph_edges():
    edges = get_graph_edges()
    # Undirected edges should have u < v
    for u, v in edges:
        assert u < v
        assert v in ADJACENCY_SETS[u]
    # Total directed edges = 2 * len(edges)
    total_directed = sum(len(ADJACENCY_LIST[u]) for u in range(NUM_POINTS))
    assert 2 * len(edges) == total_directed

