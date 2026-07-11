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


# --------------------------------------------------------------------------
# multi-region (stepped) midsurfacing + per-part shell thickness in write_k
# --------------------------------------------------------------------------

def _make_stepped_step(path: str) -> None:
    """Write a stepped plate to `path` as STEP: two boxes of different
    thickness fused side by side - 60x40x2 (y in [0, 40]) and 60x40x4
    (y in [40, 80]) - sharing the 60-long edge. Yields TWO thin regions
    (mid-planes at z=1 and z=2)."""
    gmsh.initialize()
    try:
        occ = gmsh.model.occ
        b1 = occ.addBox(0.0, 0.0, 0.0, 60.0, 40.0, 2.0)
        b2 = occ.addBox(0.0, 40.0, 0.0, 60.0, 40.0, 4.0)
        occ.fuse([(3, b1)], [(3, b2)])
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


@pytest.fixture(scope="module")
def stepped_step():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "midsurf_stepped.step")
    _make_stepped_step(path)
    return path


@pytest.fixture(scope="module")
def stepped_result(stepped_step):
    logs = []
    res = mesher.midsurface_shell(
        mesher.MeshSettings(step_file=stepped_step, size_max=8.0),
        log=logs.append)
    return res, logs


def test_midsurface_stepped_plate(stepped_result):
    res, logs = stepped_result

    # accepted as a shell mesh
    assert res.elems.ndim == 2 and res.elems.shape[1] == 4
    assert res.stats["measure_label"] == "area"
    assert res.stats["n_elems"] > 0

    # TWO regions with per-region thicknesses ~ {2, 4}
    ths = res.stats["midsurface_thicknesses"]
    assert sorted(ths) == pytest.approx([2.0, 4.0], abs=1e-6)
    assert len(res.stats["midsurface_thickness_stats"]) == 2
    assert all(s["curved"] is False
               for s in res.stats["midsurface_thickness_stats"])

    # backward-compatible mean thickness: area-weighted (equal areas -> 3.0)
    assert res.stats["midsurface_thickness"] == pytest.approx(3.0, abs=1e-6)

    # one part per region with plausible element counts and region names
    assert set(np.unique(res.elem_parts).tolist()) == {0, 1}
    assert set(res.part_names) == {"body1_region1", "body1_region2"}
    counts = np.bincount(res.elem_parts)
    assert counts.min() >= 10

    # each region's shell area ~ 60*40, total ~ 2 * 2400
    assert res.stats["part_measures"] == pytest.approx([2400.0, 2400.0],
                                                       rel=0.02)
    assert abs(res.stats["measure"] - 4800.0) / 4800.0 < 0.02

    # each region sits on ITS mid-plane: z = 1 (2-thick) and z = 2 (4-thick)
    z = np.unique(np.round(res.coords[:, 2], 6))
    assert set(z.tolist()) == {1.0, 2.0}

    # honesty NOTE: the regions are meshed independently, not stitched
    assert any("unconnected at the steps" in m for m in logs)


def test_midsurface_stepped_part_thickness_writer(stepped_result):
    res, _ = stepped_result
    ths = res.stats["midsurface_thicknesses"]
    part_ids = res.elem_parts + 1                # region i -> pid i + 1
    pt = {i + 1: float(t) for i, t in enumerate(ths)}

    k = os.path.join(OUT, "midsurf_stepped.k")
    dyna_writer.write_k(k, res.coords, res.elems, element_kind="shell",
                        part_ids=part_ids, part_thickness=pt,
                        thickness=res.stats["midsurface_thickness"])
    lines = open(k).read().splitlines()

    # one *SECTION_SHELL per part, each with its region's thickness
    sec_idx = [i for i, ln in enumerate(lines) if ln == "*SECTION_SHELL"]
    assert len(sec_idx) == 2
    secs = {int(lines[i + 2][0:10]): float(lines[i + 4][0:10])
            for i in sec_idx}
    assert secs == pytest.approx(pt, abs=1e-3)
    assert sorted(round(v, 3) for v in secs.values()) == [2.0, 4.0]

    # each *PART references its own SECID (= its pid)
    part_idx = [i for i, ln in enumerate(lines) if ln == "*PART"]
    assert len(part_idx) == 2
    for i in part_idx:
        card = lines[i + 3]
        assert int(card[10:20]) == int(card[0:10])


def test_write_k_default_section_unchanged(plate_step):
    # golden-line check: without part_thickness the single shared section and
    # the *PART SECID are byte-identical to the historical output
    res = mesher.midsurface_shell(
        mesher.MeshSettings(step_file=plate_step, size_max=15.0), log=QUIET)
    th = res.stats["midsurface_thickness"]

    k = os.path.join(OUT, "midsurf_plate_default.k")
    dyna_writer.write_k(k, res.coords, res.elems, element_kind="shell",
                        thickness=th)
    lines = open(k).read().splitlines()

    assert lines.count("*SECTION_SHELL") == 1
    i = lines.index("*SECTION_SHELL")
    assert lines[i:i + 5] == [
        "*SECTION_SHELL",
        "$#   secid    elform      shrf       nip     propt"
        "   qr/irid     icomp     setyp",
        f"{1:10d}{2:10d}{0.8333:10.4f}{5:10d}{1.0:10.1f}"
        f"{0:10d}{0:10d}{1:10d}",
        "$#      t1        t2        t3        t4      nloc"
        "     marea      idof    edgset",
        f"{th:10.4g}" * 4,
    ]
    j = lines.index("*PART")
    card = lines[j + 3]
    assert int(card[0:10]) == 1 and int(card[10:20]) == 1


def test_write_k_split_part_thickness(stepped_result):
    # write_k_split forwards part_thickness via **kw: each split file's single
    # part gets its own section thickness
    res, _ = stepped_result
    ths = res.stats["midsurface_thicknesses"]
    pt = {i + 1: float(t) for i, t in enumerate(ths)}

    base = os.path.join(OUT, "midsurf_stepped_split.k")
    written = dyna_writer.write_k_split(
        base, res.coords, res.elems, part_ids=res.elem_parts + 1,
        element_kind="shell", part_thickness=pt, thickness=9.9)
    assert sorted(p for p, _ in written) == [1, 2]
    for p, path in written:
        lines = open(path).read().splitlines()
        assert lines.count("*SECTION_SHELL") == 1
        i = lines.index("*SECTION_SHELL")
        assert int(lines[i + 2][0:10]) == p
        assert float(lines[i + 4][0:10]) == pytest.approx(pt[p], abs=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
