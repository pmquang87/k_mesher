"""Midsurface (shell) extraction for thin plate-like solids.

Builds thin-plate and near-cube STEP files with gmsh/OCC, extracts a
midsurface shell from the plate, checks the detected thickness and geometry,
verifies a non-plate solid is rejected, and round-trips the thickness through
dyna_writer.write_k -> *SECTION_SHELL.

Run: python -m pytest tests/test_midsurface.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gmsh
from k_mesher import dyna_writer
from k_mesher import mesher

OUT = os.path.join(ROOT, "tests", "out")


def QUIET(_msg):
    pass


def _make_box_step(path: str, dx: float, dy: float, dz: float) -> None:
    """Write a single-solid axis-aligned box to `path` as STEP."""
    gmsh.initialize()
    try:
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _make_half_cylinder_step(path: str, r_out: float, r_in: float,
                             height: float) -> None:
    """Write a half-cylinder shell (a curved, constant-thickness sheet) to
    `path` as STEP: an angle-pi wedge of the annulus between r_in and r_out,
    extruded `height` in z. Its two dominant faces (outer/inner cylinders) are
    NOT globally parallel, exercising the curved-shell path."""
    gmsh.initialize()
    try:
        occ = gmsh.model.occ
        outer = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, height, r_out,
                                angle=math.pi)
        inner = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, height, r_in,
                                angle=math.pi)
        occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


@pytest.fixture(scope="module")
def plate_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "midsurf_plate.step")
    _make_box_step(path, 100.0, 60.0, 2.0)   # thin plate, thickness 2 in z
    return path


@pytest.fixture(scope="module")
def cube_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "midsurf_cube.step")
    _make_box_step(path, 10.0, 10.0, 10.0)   # near cube -> not a plate
    return path


@pytest.fixture(scope="module")
def half_cyl_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "midsurf_halfcyl.step")
    # curved thin shell: wall thickness 2, mid radius 39, height 60
    _make_half_cylinder_step(path, 40.0, 38.0, 60.0)
    return path


def test_midsurface_plate_shell(plate_step):
    res = mesher.midsurface_shell(
        mesher.MeshSettings(step_file=plate_step, size_max=15.0), log=QUIET)

    # a shell mesh: (M, 4) connectivity, area measure, some elements
    assert res.elems.ndim == 2 and res.elems.shape[1] == 4
    assert res.stats["measure_label"] == "area"
    assert res.stats["n_elems"] > 0

    # detected wall thickness ~ 2.0
    th = res.stats["midsurface_thickness"]
    assert abs(th - 2.0) < 1e-6
    assert res.stats["midsurface_thicknesses"] == pytest.approx([2.0], abs=1e-6)

    # shell lies in a plane inside the plate z-range [0, 2] (mid-plane at z=1)
    z = res.coords[:, 2]
    assert z.min() >= -1e-6 and z.max() <= 2.0 + 1e-6
    assert (z.max() - z.min()) < 1e-6, "midsurface shell must be planar"

    # in-plane area ~ 100 * 60
    assert abs(res.stats["measure"] - 6000.0) / 6000.0 < 0.02


def test_midsurface_cube_rejected(cube_step):
    with pytest.raises(RuntimeError):
        mesher.midsurface_shell(
            mesher.MeshSettings(step_file=cube_step, size_max=5.0), log=QUIET)


def test_midsurface_thickness_roundtrip(plate_step):
    res = mesher.midsurface_shell(
        mesher.MeshSettings(step_file=plate_step, size_max=15.0), log=QUIET)
    th = res.stats["midsurface_thickness"]

    k = os.path.join(OUT, "midsurf_plate.k")
    dyna_writer.write_k(k, res.coords, res.elems, element_kind="shell",
                        thickness=th)
    lines = open(k).read().splitlines()

    assert "*SECTION_SHELL" in lines
    assert "*ELEMENT_SHELL" in lines
    assert "*SECTION_SOLID" not in lines

    # *SECTION_SHELL layout: keyword, comment, secid line, comment, thickness
    idx = lines.index("*SECTION_SHELL")
    t1 = float(lines[idx + 4][0:10])
    assert abs(t1 - th) < 1e-3
    assert abs(t1 - 2.0) < 1e-3


def test_midsurface_curved_shell(half_cyl_step):
    # a curved (half-cylinder) thin shell must be accepted, not rejected as a
    # non-plate solid, and yield a shell mesh with thickness ~ the wall gap
    res = mesher.midsurface_shell(
        mesher.MeshSettings(step_file=half_cyl_step, size_max=20.0), log=QUIET)

    assert res.elems.ndim == 2 and res.elems.shape[1] == 4
    assert res.stats["measure_label"] == "area"
    assert res.stats["n_elems"] > 0

    # detected wall thickness ~ 2.0
    th = res.stats["midsurface_thickness"]
    assert abs(th - 2.0) < 0.05
    assert res.stats["midsurface_thicknesses"] == pytest.approx([2.0], abs=0.05)

    # one curved body reported in the per-body stats
    tstats = res.stats["midsurface_thickness_stats"]
    assert len(tstats) == 1 and tstats[0]["curved"] is True

    # nodes lie on the mid-surface: radius ~ (40 + 38) / 2 = 39
    r = np.hypot(res.coords[:, 0], res.coords[:, 1])
    assert abs(float(r.mean()) - 39.0) < 0.2
    assert r.min() > 38.5 and r.max() < 39.5

    # curved shell is genuinely non-planar (spans a z-height and an arc)
    assert (res.coords[:, 2].max() - res.coords[:, 2].min()) > 50.0


def test_midsurface_curved_thickness_roundtrip(half_cyl_step):
    res = mesher.midsurface_shell(
        mesher.MeshSettings(step_file=half_cyl_step, size_max=20.0), log=QUIET)
    th = res.stats["midsurface_thickness"]

    k = os.path.join(OUT, "midsurf_halfcyl.k")
    dyna_writer.write_k(k, res.coords, res.elems, element_kind="shell",
                        thickness=th)
    lines = open(k).read().splitlines()

    assert "*SECTION_SHELL" in lines
    assert "*ELEMENT_SHELL" in lines
    assert "*SECTION_SOLID" not in lines

    idx = lines.index("*SECTION_SHELL")
    t1 = float(lines[idx + 4][0:10])
    assert abs(t1 - th) < 1e-3
    assert abs(t1 - 2.0) < 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
