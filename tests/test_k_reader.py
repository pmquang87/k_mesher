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

from k_mesher import dyna_writer  # noqa: E402
from k_mesher import k_reader  # noqa: E402


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


# ---------------------------------------------------------------------------
# HEX8 solids and the newer keyword cards (hand-written decks: the writer's
# hex support is being added concurrently, so fixtures don't depend on it).
# ---------------------------------------------------------------------------

def _hex_deck_nodes() -> str:
    """12 nodes: two stacked unit cubes sharing the z=1 face (ids 5..8)."""
    rows = []
    nid = 0
    for z in (0.0, 1.0, 2.0):
        for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            rows.append(f"{nid:8d} {float(x):.1f} {float(y):.1f} {z:.1f}\n")
    return "".join(rows)


HEX_DECK = (
    "*KEYWORD\n"
    "*TITLE\n"
    "hex deck\n"
    "*NODE\n"
    "$#   nid               x               y               z\n"
    + _hex_deck_nodes() +
    "*ELEMENT_SOLID\n"
    "$#   eid     pid      n1      n2      n3      n4      n5      n6"
    "      n7      n8\n"
    "       1       1       1       2       3       4       5       6"
    "       7       8\n"
    "       2       1       5       6       7       8       9      10"
    "      11      12\n"
    "*PART\n"
    "hex part\n"
    "       1       1       1\n"
    "*SECTION_SOLID\n"
    "$#   secid    elform       aet\n"
    "       1       1       0\n"
    "*MAT_ELASTIC\n"
    "       1  7800.0    2.0e11       0.3\n"
    "*END\n"
)


def test_hex8_deck(tmp_path):
    path = tmp_path / "hex.k"
    path.write_text(HEX_DECK)

    m = k_reader.read_k(str(path))
    assert m.node_count == 12
    assert len(m.solids) == 2
    assert m.solids[0] == (1, 1, (1, 2, 3, 4, 5, 6, 7, 8))
    assert m.solids[1] == (2, 1, (5, 6, 7, 8, 9, 10, 11, 12))
    assert m.solid_widths == [8]
    assert m.sections[1]["elform"] == 1

    coords, elems = m.to_arrays("solid")
    assert coords.shape == (12, 3)
    assert elems.shape == (2, 8)
    np.testing.assert_array_equal(elems[0], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(elems[1], [5, 6, 7, 8, 9, 10, 11, 12])


def test_mixed_tet_hex_to_arrays_raises(tmp_path):
    deck = (
        "*KEYWORD\n"
        "*NODE\n"
        + _hex_deck_nodes() +
        "*ELEMENT_SOLID\n"
        "       1       1       1       2       3       4       5       6"
        "       7       8\n"
        "       2       1       9      10      11      12      12      12"
        "      12      12\n"
        "*END\n"
    )
    path = tmp_path / "mixed.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.solid_widths == [4, 8]
    with pytest.raises(ValueError, match=r"\[4, 8\]"):
        m.to_arrays("solid")


def test_degenerate_tet_row_still_collapses(tmp_path):
    deck = (
        "*KEYWORD\n"
        "*NODE\n"
        "       1 0.0 0.0 0.0\n"
        "       2 1.0 0.0 0.0\n"
        "       3 0.0 1.0 0.0\n"
        "       4 0.0 0.0 1.0\n"
        "*ELEMENT_SOLID\n"
        "       7       2       1       2       3       4       4       4"
        "       4       4\n"
        "*END\n"
    )
    path = tmp_path / "degen.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.solids == [(7, 2, (1, 2, 3, 4))]
    _, elems = m.to_arrays("solid")
    assert elems.shape == (1, 4)


def test_wedge_and_pyramid_rows_collapse(tmp_path):
    deck = (
        "*KEYWORD\n"
        "*NODE\n"
        "       1 0.0 0.0 0.0\n"
        "       2 1.0 0.0 0.0\n"
        "       3 1.0 1.0 0.0\n"
        "       4 0.0 1.0 0.0\n"
        "       5 0.0 0.0 1.0\n"
        "       6 1.0 0.0 1.0\n"
        "*ELEMENT_SOLID\n"
        "$ wedge: n5 == n6 and n7 == n8\n"
        "       1       1       1       2       3       4       5       5"
        "       6       6\n"
        "$ pyramid: n5 == n6 == n7 == n8\n"
        "       2       1       1       2       3       4       5       5"
        "       5       5\n"
        "*END\n"
    )
    path = tmp_path / "wedge.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.solids[0] == (1, 1, (1, 2, 3, 4, 5, 6))   # wedge -> 6 unique
    assert m.solids[1] == (2, 1, (1, 2, 3, 4, 5))      # pyramid -> 5 unique
    assert m.solid_widths == [5, 6]


def test_include_transform_recorded_not_resolved(tmp_path):
    frag = (
        "*KEYWORD\n"
        "*NODE\n"
        "      99 9.0 9.0 9.0\n"
        "*END\n"
    )
    (tmp_path / "frag.k").write_text(frag)
    deck = (
        "*KEYWORD\n"
        "*INCLUDE_TRANSFORM\n"
        "frag.k\n"
        "$#  idnoff    ideoff    idpoff    idmoff    idsoff    idfoff    iddoff\n"
        "      1000      2000       100       100       100         0         0\n"
        "*NODE\n"
        "       1 0.0 0.0 0.0\n"
        "*END\n"
    )
    path = tmp_path / "master_tf.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path), resolve_includes=True)
    assert m.includes == []                        # NOT a plain *INCLUDE
    assert "*INCLUDE_TRANSFORM" not in m.unknown_keywords
    assert m.include_transforms == [
        {"file": "frag.k", "offsets": [1000, 2000, 100, 100, 100, 0, 0]}]
    assert m.node_count == 1                       # fragment NOT merged in


def test_set_part_list_title(tmp_path):
    deck = (
        "*KEYWORD\n"
        "*SET_PART_LIST_TITLE\n"
        "impact parts\n"
        "$#     sid       da1       da2       da3       da4    solver\n"
        "         5       0.0       0.0       0.0       0.0      MECH\n"
        "         1         2         3\n"
        "*SET_NODE_LIST\n"
        "         9\n"
        "         7         8\n"
        "*END\n"
    )
    path = tmp_path / "psets.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.part_sets == {5: [1, 2, 3]}
    assert "*SET_PART_LIST_TITLE" not in m.unknown_keywords
    assert m.node_sets == {9: [7, 8]}              # node sets untouched


def test_new_writer_keywords_are_tolerated(tmp_path):
    """The cards the writer is gaining must skip cleanly around real data."""
    deck = (
        "*KEYWORD\n"
        "*CONSTRAINED_NODAL_RIGID_BODY\n"
        "         1         0         1\n"
        "*ELEMENT_MASS\n"
        "       1       1     0.5\n"
        "*ELEMENT_DISCRETE\n"
        "       1       1       1       2       0     0.0         0     0.0\n"
        "*SECTION_DISCRETE\n"
        "         2         0       0.0       0.0       0.0       0.0\n"
        "*MAT_SPRING_ELASTIC\n"
        "         3     100.0\n"
        "*MAT_DAMPER_VISCOUS\n"
        "         4      10.0\n"
        "*DAMPING_GLOBAL\n"
        "         0      0.10\n"
        "*RIGIDWALL_GEOMETRIC_CYLINDER\n"
        "         1         0         0         0\n"
        "       0.0       0.0       0.0       0.0       0.0       1.0\n"
        "       0.5       2.0\n"
        "*RIGIDWALL_PLANAR\n"
        "         2         0         0         0\n"
        "       0.0       0.0       0.0       0.0       0.0       1.0\n"
        "*DATABASE_CROSS_SECTION_PLANE\n"
        "         1\n"
        "       0.0       0.0       0.0       1.0       0.0       0.0\n"
        "*DATABASE_HISTORY_NODE\n"
        "         1         2\n"
        "*CONSTRAINED_SPOTWELD\n"
        "         1         2\n"
        "*INITIAL_VELOCITY_GENERATION\n"
        "         1         2      10.0\n"
        "*BOUNDARY_PRESCRIBED_MOTION_SET\n"
        "         1         1         0         1       1.0\n"
        "*DEFINE_CURVE\n"
        "         1\n"
        "       0.0       0.0\n"
        "       1.0       1.0\n"
        "*NODE\n"
        "       1 0.0 0.0 0.0\n"
        "       2 1.0 2.0 3.0\n"
        "*END\n"
    )
    path = tmp_path / "cards.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.node_count == 2                       # data after the cards reads
    assert m.nodes[2] == pytest.approx((1.0, 2.0, 3.0))
    for kw in ("*CONSTRAINED_NODAL_RIGID_BODY", "*ELEMENT_MASS",
               "*ELEMENT_DISCRETE", "*SECTION_DISCRETE", "*DAMPING_GLOBAL",
               "*RIGIDWALL_GEOMETRIC_CYLINDER", "*RIGIDWALL_PLANAR",
               "*DATABASE_CROSS_SECTION_PLANE", "*DATABASE_HISTORY_NODE",
               "*CONSTRAINED_SPOTWELD", "*INITIAL_VELOCITY_GENERATION",
               "*BOUNDARY_PRESCRIBED_MOTION_SET", "*DEFINE_CURVE"):
        assert kw in m.unknown_keywords, kw
    # the discrete spring/damper materials hit the generic *MAT_ fallback
    assert m.materials[3]["type"] == "*MAT_SPRING_ELASTIC"
    assert m.materials[4]["type"] == "*MAT_DAMPER_VISCOUS"


def test_long_format_hex_row(tmp_path):
    def wide(*vals):
        return "".join(f"{v:>20}" for v in vals) + "\n"

    deck = (
        "*KEYWORD LONG=Y\n"
        "*NODE\n"
        + "".join(wide(i, f"{x:.6e}", f"{y:.6e}", f"{z:.6e}")
                  for i, (x, y, z) in enumerate(
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], start=1))
        + "*ELEMENT_SOLID\n"
        + wide(1, 1, 1, 2, 3, 4, 5, 6, 7, 8)
        + "*END\n"
    )
    path = tmp_path / "long_hex.k"
    path.write_text(deck)

    m = k_reader.read_k(str(path))
    assert m.long_format is True
    assert m.node_count == 8
    assert m.solids == [(1, 1, (1, 2, 3, 4, 5, 6, 7, 8))]
    _, elems = m.to_arrays("solid")
    assert elems.shape == (1, 8)


def test_writer_hex_roundtrip_if_supported(tmp_path):
    """Optional round-trip through dyna_writer once it grows (M, 8) support."""
    coords = np.array(
        [[x, y, z] for z in (0.0, 1.0, 2.0)
         for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))], float)
    elems = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                      [5, 6, 7, 8, 9, 10, 11, 12]])
    path = tmp_path / "hex_rt.k"
    try:
        dyna_writer.write_k(str(path), coords, elems, element_kind="solid",
                            pid=1)
    except (TypeError, ValueError) as exc:
        pytest.skip(f"dyna_writer does not support (M, 8) solids yet: {exc}")

    m = k_reader.read_k(str(path))
    assert m.solid_widths == [8]
    out_coords, out_elems = m.to_arrays("solid")
    np.testing.assert_allclose(out_coords, coords)
    np.testing.assert_array_equal(out_elems, elems)
