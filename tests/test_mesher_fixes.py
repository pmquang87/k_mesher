"""Regression tests for the mesher correctness/robustness fixes:

1. gmsh concurrency guard (module-level lock around initialize..finalize)
2. stats exceptions are logged, not silently swallowed
3. STL/tessellation unit-scaling is surfaced (warning), not ignored silently
4. TET10 gmsh->LS-DYNA node ordering (midside nodes) pinned by a regression

Self-contained: reuses the geometry-generation helpers from examples/, mirroring
tests/test_headless.py, and caches the meshes in-process.
"""
import logging
import os
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from k_mesher import mesher  # noqa: E402
from make_test_step import make, make_formats  # noqa: E402

EX = os.path.join(ROOT, "examples")
OUT_DIR = os.path.join(ROOT, "tests", "out")
STEP = os.path.join(EX, "test_part.step")
FORMAT_BASE = os.path.join(EX, "test_part")
STL = FORMAT_BASE + ".stl"


def QUIET(_msg):
    pass


def _ensure_step():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(STEP):
        make(STEP)


def _ensure_stl():
    _ensure_step()
    if not os.path.isfile(STL):
        make_formats(FORMAT_BASE)


# --------------------------------------------------------------------------
# 1. gmsh concurrency guard
# --------------------------------------------------------------------------

def test_gmsh_concurrency_guard():
    """list_faces and mesh_step run from two threads must both complete without
    error - the module lock serializes their initialize..finalize blocks so the
    gmsh global singleton is never re-initialized while in use."""
    _ensure_step()
    # the lock must exist and be a real lock
    assert isinstance(mesher._GMSH_LOCK, type(threading.Lock()))

    results, errors = {}, {}

    def run_faces():
        try:
            results["faces"] = mesher.list_faces(
                mesher.MeshSettings(step_file=STEP), log=QUIET)
        except Exception as e:  # noqa: BLE001
            errors["faces"] = e

    def run_mesh():
        try:
            results["mesh"] = mesher.mesh_step(
                mesher.MeshSettings(step_file=STEP, size_max=8.0, size_min=2.0),
                log=QUIET)
        except Exception as e:  # noqa: BLE001
            errors["mesh"] = e

    threads = [threading.Thread(target=run_faces),
               threading.Thread(target=run_mesh)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    for t in threads:
        assert not t.is_alive(), "a worker thread deadlocked / never finished"

    assert not errors, f"threaded runs raised: {errors}"
    assert len(results["faces"]) >= 6
    assert results["mesh"].stats["n_elems"] > 100
    print(f"OK: concurrent list_faces ({len(results['faces'])} faces) + "
          f"mesh_step ({results['mesh'].stats['n_elems']} tets) both completed")


# --------------------------------------------------------------------------
# 2. stats exceptions are logged, not silently swallowed
# --------------------------------------------------------------------------

def test_quality_failure_is_logged(monkeypatch, caplog):
    """When the SICN quality computation fails, the failure must be surfaced
    via logging (not swallowed) - otherwise quality_min goes missing and
    defaults to a perfect 1.0 downstream, silently disabling auto-refine and
    the timestep estimate."""
    _ensure_step()

    def boom(*_a, **_k):
        raise RuntimeError("simulated gmsh quality failure")

    # TET4 (order 1) calls getElementQualities only inside _collect_stats
    monkeypatch.setattr(mesher.gmsh.model.mesh, "getElementQualities", boom)

    with caplog.at_level(logging.WARNING, logger="mesher"):
        res = mesher.mesh_step(
            mesher.MeshSettings(step_file=STEP, size_max=8.0, size_min=2.0),
            log=QUIET)

    # the mesh itself must still be produced
    assert res.stats["n_elems"] > 100
    # the failure must have been logged...
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("quality" in m.lower() for m in msgs), \
        f"quality failure was not surfaced via logging; records: {msgs}"
    # ...and quality_min must be absent rather than a silent fake 1.0
    assert "quality_min" not in res.stats
    # downstream default proves why the silent path was dangerous
    assert res.stats.get("quality_min", 1.0) == 1.0
    print(f"OK: quality failure logged ({len(msgs)} warning record(s)), "
          f"quality_min correctly absent")


# --------------------------------------------------------------------------
# 3. STL/tessellation unit scaling surfaced
# --------------------------------------------------------------------------

def test_tessellation_unit_conversion_warns():
    """Requesting a non-default occ_unit for an STL input must surface a clear
    warning (tessellations carry no unit metadata, so no conversion happens)
    instead of silently ignoring the request."""
    _ensure_stl()
    logs = []
    res = mesher.mesh_step(
        mesher.MeshSettings(step_file=STL, element_type="TRI3", occ_unit="M"),
        log=logs.append)
    assert res.elems.shape[1] == 4  # shells produced
    hit = [m for m in logs
           if "unit conversion" in m.lower() and "not applied" in m.lower()]
    assert hit, f"no unit-conversion warning surfaced; logs: {logs}"
    assert "M" in hit[0]

    # the default (no occ_unit) must NOT emit the warning
    logs2 = []
    mesher.mesh_step(
        mesher.MeshSettings(step_file=STL, element_type="TRI3"),
        log=logs2.append)
    assert not [m for m in logs2 if "unit conversion" in m.lower()], \
        "warning must only fire when a unit conversion was actually requested"
    print(f"OK: STL occ_unit warning surfaced ({hit[0].strip()[:60]}...), "
          f"quiet by default")


# --------------------------------------------------------------------------
# 4. TET10 gmsh -> LS-DYNA midside-node ordering regression
# --------------------------------------------------------------------------

def test_tet10_midside_node_ordering():
    """Pin the gmsh->LS-DYNA TET10 permutation (GMSH2DYNA_TET10 / NEGFIX[10]):
    each midside node must lie at the midpoint of the two corner nodes its
    LS-DYNA position implies. A wrong permutation displaces a midside node to
    a different edge, which this catches."""
    _ensure_step()
    res = mesher.mesh_step(
        mesher.MeshSettings(step_file=STEP, element_type="TET10",
                            size_max=8.0, size_min=2.0), log=QUIET)
    assert res.elems.shape[1] == 10, "TET10 must have 10 nodes per element"

    # LS-DYNA ordering (0-based columns): corners 0..3, then
    #   n5 = mid(1,2)  n6 = mid(2,3)  n7 = mid(3,1)
    #   n8 = mid(1,4)  n9 = mid(2,4)  n10 = mid(3,4)
    pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    p = res.coords[res.elems - 1]                 # (M, 10, 3)

    # sample a spread of elements (task: a sample), but at least a few dozen
    m = len(p)
    step = max(1, m // 200)
    sample = p[::step]
    assert len(sample) >= 10

    for mid_col, (a, b) in zip(range(4, 10), pairs):
        midpoint = 0.5 * (sample[:, a] + sample[:, b])
        dev = np.linalg.norm(sample[:, mid_col] - midpoint, axis=1)
        edge = np.linalg.norm(sample[:, a] - sample[:, b], axis=1)
        # straight-sided tets: dev ~ 0; curved boundary edges deviate a little.
        # A mis-permuted node would land on a different edge -> dev ~ edge len.
        assert (dev <= 0.35 * edge).all(), (
            f"midside column {mid_col} (LS-DYNA n{mid_col + 1}) is not the "
            f"midpoint of corners {a + 1},{b + 1}: max dev/edge = "
            f"{float((dev / np.maximum(edge, 1e-30)).max()):.3f}")

    # positive volume is enforced (NEGFIX applied on the same ordering)
    corners = res.coords[res.elems[:, :4] - 1]
    v6 = np.einsum("ij,ij->i",
                   np.cross(corners[:, 1] - corners[:, 0],
                            corners[:, 2] - corners[:, 0]),
                   corners[:, 3] - corners[:, 0])
    assert (v6 > 0).all(), "all TET10 must have positive volume"
    print(f"OK: {m} TET10 elements, all 6 midside positions verified, "
          f"positive volumes")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            # skip the pytest-fixture test in bare mode
            if "monkeypatch" in fn.__code__.co_varnames:
                continue
            fn()
    print("\nDONE")
