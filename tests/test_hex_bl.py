"""HEX8 (transfinite) solid meshing and boundary-layer (near-wall) sizing.

Builds box/cylinder STEP files with gmsh/OCC in-test, meshes the box with
HEX8 (checking node ordering, positive volumes, sizing and stats), verifies
that non-boxlike geometry and unsupported option combinations are rejected
with clear errors, and that the boundary_layer setting produces graded
near-wall refinement for TET4 meshes.

Run: python -m pytest tests/test_hex_bl.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gmsh
from k_mesher import mesher

OUT = os.path.join(ROOT, "tests", "out")

# hex edges / 6-tet decomposition in LS-DYNA/VTK ordering (independent copy,
# so the test checks the convention rather than mesher's own constants)
HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
HEX2TET6 = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def QUIET(_msg):
    pass


def _make_box_step(path: str, dx: float, dy: float, dz: float) -> None:
    gmsh.initialize()
    try:
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _make_cylinder_step(path: str, r: float, h: float) -> None:
    gmsh.initialize()
    try:
        gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, h, r)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


@pytest.fixture(scope="module")
def box_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "hex_box.step")
    _make_box_step(path, 40.0, 20.0, 10.0)
    return path


@pytest.fixture(scope="module")
def bl_box_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "bl_box.step")
    _make_box_step(path, 20.0, 10.0, 10.0)
    return path


@pytest.fixture(scope="module")
def cylinder_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "hex_cyl.step")
    _make_cylinder_step(path, 8.0, 20.0)
    return path


def hex_volumes(coords: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Signed hex volumes in the LS-DYNA/VTK node ordering."""
    p = coords[hexes - 1]
    v = np.zeros(len(p))
    for a, b, c, d in HEX2TET6:
        v += np.einsum("ij,ij->i",
                       np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]),
                       p[:, d] - p[:, a])
    return v / 6.0


def tet_mean_edges(coords: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Mean edge length per TET4 element (corner nodes)."""
    p = coords[elems[:, :4] - 1]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=1)
                     for a, b in pairs], axis=1).mean(axis=1)


def wall_distance(cent: np.ndarray, dx: float, dy: float, dz: float):
    """Distance of points to the nearest wall of the [0,d] box."""
    return np.minimum.reduce([
        cent[:, 0], dx - cent[:, 0], cent[:, 1], dy - cent[:, 1],
        cent[:, 2], dz - cent[:, 2]])


# --------------------------------------------------------------------------
# HEX8
# --------------------------------------------------------------------------

def test_hex8_box(box_step):
    logmsgs = []
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=box_step, element_type="HEX8", size_max=5.0),
        log=lambda m: logmsgs.append(str(m)))

    # (M, 8) connectivity, 1-based, one part
    assert res.elems.ndim == 2 and res.elems.shape[1] == 8
    assert res.elems.min() >= 1 and res.elems.max() <= len(res.coords)
    assert len(res.part_names) == 1
    assert set(np.unique(res.elem_parts)) == {0}

    # every hex positive-volume in LS-DYNA ordering; total = 40*20*10
    v = hex_volumes(res.coords, res.elems)
    assert (v > 0).all(), "all hexes must have positive volume"
    assert abs(v.sum() - 8000.0) / 8000.0 < 0.01
    assert res.stats["measure_label"] == "volume"
    assert abs(res.stats["measure"] - 8000.0) / 8000.0 < 0.01

    # element/node counts plausible for size_max=5 (ideal 8x4x2 = 64
    # divisions; gmsh may round division counts up for recombination)
    n = res.stats["n_elems"]
    assert n == len(res.elems)
    assert 64 <= n <= 512, f"implausible hex count {n} for size 5"
    assert n < res.stats["n_nodes"] <= 8 * n
    # sizing honored: no hex edge longer than size_max
    p = res.coords[res.elems - 1]
    el = np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=1)
                   for a, b in HEX_EDGES], axis=1)
    assert el.max() <= 5.0 + 1e-6, "transfinite spacing must honor size_max"

    # stats: quality, criteria, characteristic length, mass properties
    assert res.stats["quality_min"] > 0.0
    assert sum(res.stats["quality_hist"]) == n
    names = [c["name"] for c in res.stats["criteria"]]
    assert "aspect ratio" in names and "SICN" in names
    assert res.stats["char_length"] > 0
    assert res.stats["duplicate_nodes"] == 0
    assert np.allclose(res.stats["cog"], [20.0, 10.0, 5.0], atol=0.1)
    assert any("HEX8" in m for m in logmsgs)


def test_hex8_nonboxlike_rejected(cylinder_step):
    with pytest.raises(RuntimeError, match="(?i)hex"):
        mesher.mesh_step(mesher.MeshSettings(
            step_file=cylinder_step, element_type="HEX8", size_max=4.0),
            log=QUIET)


def test_hex8_symmetry_rejected(box_step):
    with pytest.raises(RuntimeError, match="symmetry"):
        mesher.mesh_step(mesher.MeshSettings(
            step_file=box_step, element_type="HEX8", size_max=5.0,
            symmetry=[mesher.SymmetryPlane("x", 20.0, "+")]), log=QUIET)


def test_hex8_refinement_and_bl_rejected(box_step):
    with pytest.raises(RuntimeError, match="refinement"):
        mesher.mesh_step(mesher.MeshSettings(
            step_file=box_step, element_type="HEX8", size_max=5.0,
            refinements=[{"kind": "sphere", "params": [0, 0, 0, 5],
                          "size": 1.0}]), log=QUIET)
    with pytest.raises(RuntimeError, match="boundary layer"):
        mesher.mesh_step(mesher.MeshSettings(
            step_file=box_step, element_type="HEX8", size_max=5.0,
            boundary_layer={"faces": "all", "thickness": 2.0}), log=QUIET)
    with pytest.raises(RuntimeError, match="defeaturing"):
        mesher.mesh_step(mesher.MeshSettings(
            step_file=box_step, element_type="HEX8", size_max=5.0,
            defeature_faces=[1]), log=QUIET)


# --------------------------------------------------------------------------
# boundary layer (graded near-wall sizing)
# --------------------------------------------------------------------------

def test_boundary_layer_tet4(bl_box_step):
    base = mesher.mesh_step(mesher.MeshSettings(
        step_file=bl_box_step, size_max=6.0), log=QUIET)
    logmsgs = []
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=bl_box_step, size_max=6.0,
        boundary_layer={"faces": "all", "thickness": 2.5, "ratio": 1.2,
                        "size_wall": 0.6}),
        log=lambda m: logmsgs.append(str(m)))

    assert any("Boundary layer" in m for m in logmsgs), \
        "BL application must be logged"
    assert res.stats["n_elems"] > 2 * base.stats["n_elems"], \
        "near-wall grading must add elements over the no-BL baseline"

    # near-wall elements far finer than the no-BL baseline's near-wall
    # elements (the box is only 10 thick, so gmsh's gradation keeps even the
    # interior fairly fine - spatial selectivity is covered by the
    # face-list test below)
    def near_wall_size(r):
        cent = r.coords[r.elems[:, :4] - 1].mean(axis=1)
        d = wall_distance(cent, 20.0, 10.0, 10.0)
        el = tet_mean_edges(r.coords, r.elems)
        assert (d < 1.0).sum() > 0
        return float(np.median(el[d < 1.0]))

    assert near_wall_size(res) < 0.5 * near_wall_size(base)
    # wall elements at ~size_wall scale, far below size_max
    assert near_wall_size(res) < 1.0
    # volume must still be right (20*10*10)
    assert abs(res.stats["measure"] - 2000.0) / 2000.0 < 0.02


def test_boundary_layer_face_list(bl_box_step):
    faces = mesher.list_faces(
        mesher.MeshSettings(step_file=bl_box_step), log=QUIET)
    assert len(faces) == 6
    x0 = next(f["tag"] for f in faces if abs(f["centroid"][0]) < 1e-6)
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=bl_box_step, size_max=6.0,
        boundary_layer={"faces": [x0], "thickness": 2.5, "size_wall": 1.2}),
        log=QUIET)

    cent = res.coords[res.elems[:, :4] - 1].mean(axis=1)
    el = tet_mean_edges(res.coords, res.elems)
    near_x0, near_x1 = cent[:, 0] < 1.0, cent[:, 0] > 19.0
    assert near_x0.sum() > 0 and near_x1.sum() > 0
    # refined at the listed face only, coarse at the opposite wall
    assert np.median(el[near_x0]) < 0.6 * np.median(el[near_x1])


def test_bl_first_layer_derivation():
    # explicit wall size wins
    assert mesher._bl_first_layer(
        {"thickness": 2.0, "size_wall": 0.5, "ratio": 1.5}) == 0.5
    # geometric progression: h * (r^n - 1) / (r - 1) = thickness
    h = mesher._bl_first_layer(
        {"thickness": 2.0, "ratio": 1.2, "nb_layers": 4})
    assert abs(h * (1.2 ** 4 - 1.0) / 0.2 - 2.0) < 1e-12
    # ratio 1: uniform layers
    assert mesher._bl_first_layer(
        {"thickness": 2.0, "ratio": 1.0, "nb_layers": 4}) == 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
