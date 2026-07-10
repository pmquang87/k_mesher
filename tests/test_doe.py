"""Tests for the mesh-convergence / DOE driver (doe.py).

Kept fast on purpose: coarse element sizes so the meshes are tiny. Run with
    pytest tests/test_doe.py -q
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from k_mesher import doe
from k_mesher import mesher
from make_test_step import make

OUT_DIR = os.path.join(ROOT, "tests", "out")
STEP = os.path.join(ROOT, "examples", "test_part.step")

STEEL = {"e": 210000.0, "pr": 0.3, "ro": 7.85e-9}

# coarse -> fine; the test part is 100 x 50 x 25, so these give tiny meshes.
# 12 -> 6 straddles the geometry's sizing floor so the count clearly grows.
SIZES = (12.0, 6.0)


def QUIET(_msg):
    pass


def _ensure():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(STEP):
        make(STEP)


def _base_settings(**kw):
    # curvature refinement off so size_max actually drives the element count
    # (otherwise the hole's curvature dominates and coarse sizes tie)
    kw.setdefault("curvature_refine", False)
    return mesher.MeshSettings(step_file=STEP, size_max=25.0, **kw)


# a module-level cache so the (few) meshes are built once and shared
_cache: dict = {}


def _steel_sweep():
    if "steel" not in _cache:
        _ensure()
        _cache["steel"] = doe.size_sweep(_base_settings(), SIZES, mat=STEEL,
                                         log=QUIET)
    return _cache["steel"]


def test_size_sweep_basic():
    recs = _steel_sweep()
    assert len(recs) == 2
    # every expected field is present on each record
    for r in recs:
        assert r.field == "size_max"
        assert r.error is None
        assert r.n_nodes and r.n_nodes > 0
        assert r.n_elems and r.n_elems > 0
        assert r.quality_min is not None
        assert r.char_length and r.char_length > 0
        assert r.seconds is not None and r.seconds >= 0.0
    # finer size (second, 6.0) yields more elements than the coarser (12.0)
    assert recs[0].value == 12.0 and recs[1].value == 6.0
    assert recs[1].n_elems > recs[0].n_elems


def test_size_sweep_material_dt():
    recs = _steel_sweep()
    for r in recs:
        assert r.critical_timestep is not None
        assert r.critical_timestep > 0.0


def test_size_sweep_without_material():
    _ensure()
    recs = doe.size_sweep(_base_settings(), (25.0,), log=QUIET)
    assert len(recs) == 1
    assert recs[0].critical_timestep is None  # no material -> no dt
    assert recs[0].n_elems > 0


def test_param_sweep_generic():
    _ensure()
    recs = doe.param_sweep(_base_settings(), "size_max", SIZES, log=QUIET)
    assert len(recs) == 2
    assert [r.field for r in recs] == ["size_max", "size_max"]
    assert recs[1].n_elems > recs[0].n_elems


def test_param_sweep_bad_field():
    try:
        doe.param_sweep(_base_settings(), "not_a_field", (1.0,), log=QUIET)
        raise AssertionError("expected ValueError for unknown field")
    except ValueError:
        pass


def test_convergence_series():
    recs = _steel_sweep()
    conv = doe.convergence(recs, metric="n_elems")
    assert conv.metric == "n_elems"
    assert len(conv.values) == 2
    assert len(conv.rel_change) == 1
    assert isinstance(conv.last_change, float)
    assert conv.last_change >= 0.0
    # char_length also convergeable
    conv2 = doe.convergence(recs, metric="char_length")
    assert len(conv2.values) == 2


def test_convergence_skips_errors():
    good = doe.SweepResult(field="size_max", value=10.0, n_elems=100,
                           seconds=0.1)
    bad = doe.SweepResult(field="size_max", value=5.0,
                          error="boom", seconds=0.1)
    conv = doe.convergence([good, bad], metric="n_elems")
    assert conv.values == [100.0]
    assert conv.rel_change == []


def test_to_csv_and_json(tmp_path):
    recs = _steel_sweep()
    csv_path = str(tmp_path / "sweep.csv")
    json_path = str(tmp_path / "sweep.json")
    doe.to_csv(recs, csv_path)
    doe.to_json(recs, json_path)

    import csv as _csv
    with open(csv_path) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 2
    assert "n_elems" in rows[0] and "critical_timestep" in rows[0]

    with open(json_path) as fh:
        data = json.load(fh)
    assert len(data) == 2
    assert data[0]["field"] == "size_max"
    assert set(data[0]) >= {"value", "n_nodes", "n_elems", "quality_min",
                            "char_length", "critical_timestep", "seconds",
                            "error"}


def test_failing_size_recorded(monkeypatch):
    _ensure()
    orig = mesher.mesh_step_auto
    calls = {"n": 0}

    def flaky(settings, log=print, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated meshing failure")
        return orig(settings, log=log, **kw)

    monkeypatch.setattr(mesher, "mesh_step_auto", flaky)
    recs = doe.size_sweep(_base_settings(), SIZES, log=QUIET)
    # the whole sweep still completes: one error record, one good record
    assert len(recs) == 2
    assert recs[0].error is not None and "simulated" in recs[0].error
    assert recs[0].n_elems is None
    assert recs[0].seconds is not None      # wall-clock recorded even on failure
    assert recs[1].error is None and recs[1].n_elems > 0
    # convergence ignores the failed run
    conv = doe.convergence(recs, metric="n_elems")
    assert len(conv.values) == 1


def test_plot_convergence_optional(tmp_path):
    recs = _steel_sweep()
    png = str(tmp_path / "conv.png")
    wrote = doe.plot_convergence(recs, "n_elems", png)
    # matplotlib may or may not be installed; either way the call must be safe
    if wrote:
        assert os.path.isfile(png) and os.path.getsize(png) > 0
    else:
        assert not os.path.exists(png)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name}")
            try:
                fn()
            except TypeError:
                pass  # fixtures (tmp_path/monkeypatch) need pytest
    print("done")
