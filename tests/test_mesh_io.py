"""Tests for the meshio FE-format bridge (mesh_io).

meshio is installed in the test environment, so these really round-trip
through on-disk formats (Abaqus .inp and VTU) as well as the in-memory
``to_meshio``/``from_meshio`` conversions.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mesh_io  # noqa: E402

meshio = pytest.importorskip("meshio", reason="meshio not installed")


# --------------------------------------------------------------------------
# Hand-made tiny meshes (coords, elems) in k_mesher convention:
# coords (N,3) float, elems 1-based LS-DYNA ordering.
# --------------------------------------------------------------------------
def _tet4():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    elems = np.array([[1, 2, 3, 4]])  # 1-based
    return coords, elems, "solid"


def _tet10():
    corners = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # LS-DYNA mid-edge order: (1,2)(2,3)(3,1)(1,4)(2,4)(3,4) in 1-based corners
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    mids = np.array([(corners[a] + corners[b]) / 2 for a, b in edges])
    coords = np.vstack((corners, mids))
    elems = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # 1-based
    return coords, elems, "solid"


def _tri3():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    # degenerate quad: n4 == n3
    elems = np.array([[1, 2, 3, 3]])
    return coords, elems, "shell"


def _quad4():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    elems = np.array([[1, 2, 3, 4]])
    return coords, elems, "shell"


ALL_MESHES = {
    "tet4": _tet4,
    "tet10": _tet10,
    "tri3": _tri3,
    "quad4": _quad4,
}


def _elem_coord_sets(coords, elems, element_kind):
    """Represent each element as a frozenset of rounded node-coordinate
    tuples, so comparisons are robust to node renumbering and (for triangles)
    the degenerate-quad padding."""
    sets = []
    for row in elems:
        nodes = np.unique(row) - 1  # 1-based -> 0-based; unique drops n4==n3
        pts = frozenset(tuple(np.round(coords[n], 9)) for n in nodes)
        sets.append(pts)
    return sorted(sets, key=lambda s: sorted(s))


# --------------------------------------------------------------------------
# Round-trip through on-disk formats.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_MESHES))
@pytest.mark.parametrize("ext,fmt", [(".inp", "abaqus"), (".vtu", None)])
def test_disk_roundtrip(tmp_path, name, ext, fmt):
    coords, elems, kind = ALL_MESHES[name]()
    path = str(tmp_path / f"{name}{ext}")

    mesh_io.export_mesh(path, coords, elems, kind, file_format=fmt)
    rc, re, rkind = mesh_io.import_mesh(path, file_format=fmt)

    assert rkind == kind
    # connectivity compared as sets of node coordinates per element
    assert _elem_coord_sets(coords, elems, kind) == _elem_coord_sets(
        rc, re, rkind
    )
    # every original node coordinate survives the trip
    for p in coords:
        assert np.any(np.all(np.isclose(rc, p), axis=1)), p


# --------------------------------------------------------------------------
# Direct to_meshio / from_meshio round-trip.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_MESHES))
def test_inmemory_roundtrip(name):
    coords, elems, kind = ALL_MESHES[name]()
    mesh = mesh_io.to_meshio(coords, elems, kind)
    rc, re, rkind = mesh_io.from_meshio(mesh)

    assert rkind == kind
    assert re.shape[0] == elems.shape[0]  # element count preserved
    assert re.shape[1] == elems.shape[1]  # width preserved (tri stays width 4)
    assert _elem_coord_sets(coords, elems, kind) == _elem_coord_sets(
        rc, re, rkind
    )


def test_tet10_keeps_ten_nodes_and_midpoints():
    coords, elems, kind = _tet10()
    mesh = mesh_io.to_meshio(coords, elems, kind)

    # meshio side: one tetra10 cell block with 10 columns
    assert "tetra10" in mesh.cells_dict
    m_conn = mesh.cells_dict["tetra10"]
    assert m_conn.shape == (1, 10)

    # mid nodes are still at the edge midpoints under meshio's edge ordering
    meshio_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    row = m_conn[0]
    for i, (a, b) in enumerate(meshio_edges):
        mid = mesh.points[row[4 + i]]
        exp = (mesh.points[row[a]] + mesh.points[row[b]]) / 2
        assert np.allclose(mid, exp), (i, a, b)

    # round-trip back and confirm mid nodes remain at LS-DYNA edge midpoints
    rc, re, _ = mesh_io.from_meshio(mesh)
    assert re.shape == (1, 10)
    dyna_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    r = re[0] - 1  # 0-based
    for i, (a, b) in enumerate(dyna_edges):
        mid = rc[r[4 + i]]
        exp = (rc[r[a]] + rc[r[b]]) / 2
        assert np.allclose(mid, exp), (i, a, b)


def test_tet10_permutation_is_self_inverse():
    fwd = mesh_io._DYNA_TO_MESHIO_TET10
    inv = mesh_io._MESHIO_TO_DYNA_TET10
    assert [inv[fwd[i]] for i in range(10)] == list(range(10))


# --------------------------------------------------------------------------
# Cross-check: a mesh exported here reads back into the expected meshio
# cell types.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected_types",
    [
        ("tet4", {"tetra"}),
        ("tet10", {"tetra10"}),
        ("tri3", {"triangle"}),
        ("quad4", {"quad"}),
    ],
)
def test_exported_cell_types(tmp_path, name, expected_types):
    coords, elems, kind = ALL_MESHES[name]()
    path = str(tmp_path / f"{name}.vtu")
    mesh_io.export_mesh(path, coords, elems, kind)
    m = meshio.read(path)
    assert set(m.cells_dict) == expected_types


def test_mixed_shell_block_splits_triangle_and_quad():
    # one triangle (degenerate quad) + one quad sharing a node set
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    elems = np.array(
        [
            [1, 2, 3, 4],  # quad
            [2, 5, 3, 3],  # triangle (n4 == n3)
        ]
    )
    mesh = mesh_io.to_meshio(coords, elems, "shell")
    assert set(mesh.cells_dict) == {"triangle", "quad"}
    assert mesh.cells_dict["triangle"].shape == (1, 3)
    assert mesh.cells_dict["quad"].shape == (1, 4)

    rc, re, kind = mesh_io.from_meshio(mesh)
    assert kind == "shell"
    assert re.shape == (2, 4)
    # both triangles come back padded to a degenerate quad
    tri_rows = re[re[:, 3] == re[:, 2]]
    assert tri_rows.shape[0] == 1


def test_solids_win_when_mixed_with_shells():
    coords, elems, _ = _tet4()
    solid = mesh_io.to_meshio(coords, elems, "solid")
    # append a surface triangle cell block to the same mesh
    combined = meshio.Mesh(
        solid.points,
        list(solid.cells) + [("triangle", np.array([[0, 1, 2]]))],
    )
    rc, re, kind = mesh_io.from_meshio(combined)
    assert kind == "solid"
    assert re.shape == (1, 4)


# --------------------------------------------------------------------------
# Error handling.
# --------------------------------------------------------------------------
def test_empty_elems_raises():
    coords, _, _ = _tet4()
    with pytest.raises(ValueError, match="empty"):
        mesh_io.to_meshio(coords, np.zeros((0, 4), dtype=int), "solid")


def test_unsupported_solid_width_raises():
    coords = np.zeros((6, 3))
    elems = np.arange(1, 7).reshape(1, 6)  # width 6 -> not tet4/tet10
    with pytest.raises(ValueError, match="unsupported solid"):
        mesh_io.to_meshio(coords, elems, "solid")


def test_bad_element_kind_raises():
    coords, elems, _ = _tet4()
    with pytest.raises(ValueError, match="element_kind"):
        mesh_io.to_meshio(coords, elems, "beam")


def test_from_meshio_no_supported_types_raises():
    m = meshio.Mesh(
        np.zeros((2, 3)), [("line", np.array([[0, 1]]))]
    )
    with pytest.raises(ValueError, match="no supported cell types"):
        mesh_io.from_meshio(m)


def test_missing_meshio_friendly_error(monkeypatch):
    """Simulate meshio being absent and assert the friendly ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "meshio":
            raise ImportError("no meshio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install meshio"):
        mesh_io._require_meshio()
