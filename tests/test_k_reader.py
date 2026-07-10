"""Tests for :mod:`k_reader`.

The round-trip tests drive the real :func:`dyna_writer.write_k` (pure numpy,
no gmsh) with tiny hand-built meshes and read the result back, so the reader
is checked against genuine writer output. The snippet tests cover LONG=Y,
triangle shells, unknown keywords and *INCLUDE via hand-written decks.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dyna_writer  # noqa: E402
import k_reader  # noqa: E402


def test_roundtrip_tet4_solid(tmp_path):
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    elems = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    path = tmp_path / "tet4.k"
    dyna_writer.write_k(
        str(path), coords, elems, element_kind="solid", pid=1,
        mat={"e": 2.0e11, "pr": 0.3, "ro": 7800.0},
        part_ids=np.array([1, 2]), part_titles={1: "partA", 2: "partB"},
        face_sets=(
            {"kind": "node", "title": "grip", "nodes": [1, 2, 3]},
            {"kind": "segment", "title": "top", "segments": [[1, 2, 3, 3]]},
        ),
    )

    m = k_reader.read_k(str(path))
    assert m.title == "k_mesher mesh"
    assert m.long_format is False
    assert m.node_count == 5
    assert len(m.solids) == 2
    assert len(m.shells) == 0
    assert m.element_count == 2

    # each tet must collapse the degenerate hex back to 4 unique nodes
    for eid, pid, nodes in m.solids:
        assert len(nodes) == 4
    assert m.solids[0] == (1, 1, (1, 2, 3, 4))
    assert m.solids[1] == (2, 2, (2, 3, 4, 5))

    # two parts, section, elastic material
    assert set(m.parts) == {1, 2}
    assert m.parts[1]["title"] == "partA"
    assert m.parts[2]["title"] == "partB"
    assert m.parts[1]["secid"] == 1 and m.parts[1]["mid"] == 1
    assert m.sections[1]["type"] == "*SECTION_SOLID"
    assert m.sections[1]["elform"] == 10
    assert m.materials[1]["type"] == "*MAT_ELASTIC"
    assert m.materials[1]["params"]["e"] == pytest.approx(2.0e11)
    assert m.materials[1]["params"]["pr"] == pytest.approx(0.3)
    assert m.materials[1]["params"]["ro"] == pytest.approx(7800.0)

    # node set and segment set contents survive the round trip
    assert list(m.node_sets.values())[0] == [1, 2, 3]
    seg = list(m.segment_sets.values())[0]
    assert seg == [(1, 2, 3, 3)]

    # node coordinates read back correctly
    assert m.nodes[2] == pytest.approx((1.0, 0.0, 0.0))
    assert m.nodes[5] == pytest.approx((1.0, 1.0, 1.0))


def test_roundtrip_tet10_two_line(tmp_path):
    coords = np.arange(30, dtype=float).reshape(10, 3)
    elems = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    path = tmp_path / "tet10.k"
    dyna_writer.write_k(str(path), coords, elems, element_kind="solid", pid=7)

    m = k_reader.read_k(str(path))
    assert m.node_count == 10
    assert len(m.solids) == 1
    eid, pid, nodes = m.solids[0]
    assert eid == 1 and pid == 7
    assert nodes == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert len(nodes) == 10


def test_roundtrip_shell_triangle_and_quad(tmp_path):
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    elems = np.array([[1, 2, 3, 3], [1, 2, 4, 3]])  # triangle then quad
    path = tmp_path / "shell.k"
    dyna_writer.write_k(
        str(path), coords, elems, element_kind="shell", pid=3, thickness=2.5)

    m = k_reader.read_k(str(path))
    assert m.node_count == 4
    assert len(m.shells) == 2
    assert len(m.solids) == 0
    assert m.shells[0][2] == (1, 2, 3)      # triangle collapsed to 3 nodes
    assert m.shells[1][2] == (1, 2, 4, 3)   # quad keeps 4
    assert m.sections[3]["type"] == "*SECTION_SHELL"
    assert m.sections[3]["thickness"] == pytest.approx(2.5)


def test_roundtrip_rigid_material(tmp_path):
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    elems = np.array([[1, 2, 3, 4]])
    path = tmp_path / "rigid.k"
    dyna_writer.write_k(
        str(path), coords, elems, element_kind="solid", pid=1,
        mat={"e": 2.0e11, "pr": 0.3, "ro": 7800.0, "rigid": True})

    m = k_reader.read_k(str(path))
    assert m.materials[1]["type"] == "*MAT_RIGID"
    assert m.materials[1]["params"]["e"] == pytest.approx(2.0e11)


def test_to_arrays_roundtrips_into_writer(tmp_path):
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    elems = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    path = tmp_path / "a.k"
    dyna_writer.write_k(str(path), coords, elems, element_kind="solid")

    m = k_reader.read_k(str(path))
    out_coords, out_elems = m.to_arrays("solid")
    assert out_coords.shape == (5, 3)
    assert out_elems.shape == (2, 4)
    np.testing.assert_allclose(out_coords, coords)
    np.testing.assert_array_equal(out_elems, elems)

    # feed straight back into the writer -> reads identically
    path2 = tmp_path / "b.k"
    dyna_writer.write_k(str(path2), out_coords, out_elems, element_kind="solid")
    m2 = k_reader.read_k(str(path2))
    assert m2.solids == m.solids


def test_long_format_snippet(tmp_path):
    deck = (
        "*KEYWORD LONG=Y\n"
        "*TITLE\n"
        "long deck\n"
        "$ a comment\n"
        "*NODE\n"
        "$#   nid               x               y               z\n"
        "                   1     0.000000000e+00     0.000000000e+00"
        "     0.000000000e+00\n"
        "                   2     1.000000000e+00     0.000000000e+00"
        "     0.000000000e+00\n"
        "                   3     0.000000000e+00     1.000000000e+00"
        "     0.000000000e+00\n"
        "*ELEMENT_SHELL\n"
        "$#   eid     pid      n1      n2      n3      n4\n"
        "                   1                   9                   1"
        "                   2                   3                   3\n"
        "*END\n"
    )
    path = tmp_path / "long.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.long_format is True
    assert m.title == "long deck"
    assert m.node_count == 3
    assert m.nodes[2] == pytest.approx((1.0, 0.0, 0.0))
    # triangle: n4 == n3 collapses to 3 nodes
    assert m.shells[0] == (1, 9, (1, 2, 3))


def test_unknown_keyword_and_mixed_case(tmp_path):
    deck = (
        "*keyword\n"           # mixed / lower case keyword
        "*TITLE\n"
        "mixed\n"
        "*DEFINE_CURVE_TITLE\n"
        "some curve\n"
        "$#    lcid\n"
        "         1         0       1.0       1.0\n"
        "       0.0       0.0\n"
        "       1.0       1.0\n"
        "*NODE\n"
        "       1 0.0 0.0 0.0\n"
        "       2 1.0 2.0 3.0\n"
        "*END\n"
    )
    path = tmp_path / "u.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert "*DEFINE_CURVE_TITLE" in m.unknown_keywords
    assert m.node_count == 2          # nodes after the unknown block still read
    assert m.nodes[2] == pytest.approx((1.0, 2.0, 3.0))


def test_include_recorded_and_resolved(tmp_path):
    frag = (
        "*KEYWORD\n"
        "*NODE\n"
        "      10 5.0 6.0 7.0\n"
        "*END\n"
    )
    (tmp_path / "frag.k").write_text(frag)
    master = (
        "*KEYWORD\n"
        "*TITLE\n"
        "master\n"
        "*INCLUDE\n"
        "frag.k\n"
        "*NODE\n"
        "       1 0.0 0.0 0.0\n"
        "*END\n"
    )
    path = tmp_path / "master.k"
    path.write_text(master)

    # without resolution the include is just recorded
    m = k_reader.read_k(str(path))
    assert m.includes == ["frag.k"]
    assert m.node_count == 1

    # with resolution the fragment's nodes are merged in (global numbering)
    m2 = k_reader.read_k(str(path), resolve_includes=True)
    assert m2.node_count == 2
    assert m2.nodes[10] == pytest.approx((5.0, 6.0, 7.0))


def test_node_set_continuation_rows(tmp_path):
    deck = (
        "*KEYWORD\n"
        "*SET_NODE_LIST_TITLE\n"
        "big set\n"
        "$#     sid       da1       da2       da3       da4    solver\n"
        "         1       0.0       0.0       0.0       0.0      MECH\n"
        "         1         2         3         4         5         6"
        "         7         8\n"
        "         9        10\n"
        "*END\n"
    )
    path = tmp_path / "ns.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.node_sets[1] == list(range(1, 11))
