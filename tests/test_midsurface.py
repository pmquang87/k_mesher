"""Midsurface (shell) extraction for thin plate-like solids.

Builds thin-plate and near-cube STEP files with gmsh/OCC, extracts a
midsurface shell from the plate, checks the detected thickness and geometry,
verifies a non-plate solid is rejected, and round-trips the thickness through
dyna_writer.write_k -> *SECTION_SHELL.

Run: python -m pytest tests/test_midsurface.py -q
"""
import os
import sys

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
