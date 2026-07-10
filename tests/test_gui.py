"""Headless GUI tests: widget wiring, input validation, settings round-trip.

Runs wherever tkinter and a display are available (CI runs them under Xvfb
on Linux; GitHub's Windows runners have a desktop session). Skips cleanly
otherwise. No meshing happens here - these tests only exercise the tkinter
layer above job_runner.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

tk = pytest.importorskip("tkinter", reason="tkinter not available")

import gui  # noqa: E402
from make_test_step import make, make_two_bodies  # noqa: E402

STEP = os.path.join(ROOT, "examples", "test_part.step")
STEP2 = os.path.join(ROOT, "examples", "test_two_bodies.step")


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Message boxes are modal and would hang a headless run - record them
    instead of showing them."""
    shown = []
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(gui.messagebox, name,
                            lambda *a, _n=name, **k: shown.append(_n))
    yield shown


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    application = gui.KMesherGUI(root)
    # decouple from any settings file lying around
    application.var_step.set(STEP)
    application.var_out.set("")
    yield application
    root.destroy()


def _ensure_geometry():
    if not os.path.isfile(STEP):
        make(STEP)
    if not os.path.isfile(STEP2):
        make_two_bodies(STEP2)


def test_collect_inputs_defaults(app):
    _ensure_geometry()
    settings, out, kopts = app._collect_inputs()
    assert settings.step_file == STEP
    assert out.endswith(".k")
    assert kopts["element_kind"] in ("solid", "shell")
    assert kopts["split_mode"] in (None, "parts", "include")


def test_output_options_wiring(app):
    _ensure_geometry()
    app.var_mesh_only.set(True)
    app.var_split_mode.set("*INCLUDE fragments + master deck")
    app.var_contact.set(True)
    app.var_contact_fs.set("0.15")
    app.var_ctrl_dt.set(True)
    app.var_tssfac.set("0.9")
    app.var_grav.set(True)
    app.var_grav_axis.set("Z")
    app.var_grav_a.set("9810")
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["mesh_only"] is True
    assert kopts["split_mode"] == "include"
    assert kopts["contact_fs"] == 0.15
    assert kopts["tssfac"] == 0.9
    assert kopts["gravity"] == ("z", 9810.0)


def test_tssfac_validation(app):
    _ensure_geometry()
    app.var_ctrl_dt.set(True)
    app.var_tssfac.set("1.5")
    with pytest.raises(ValueError):
        app._collect_inputs(need_out=False)


def test_part_mats_and_rigid(app):
    _ensure_geometry()
    app.var_pm_body.set("2")
    app.var_pm_e.set("70000")
    app.var_pm_nu.set("0.33")
    app.var_pm_ro.set("2.7e-9")
    app.var_pm_rigid.set(False)
    app._add_part_mat()
    app.var_pm_body.set("3")
    app.var_pm_rigid.set(True)
    app._add_part_mat()
    assert len(app.part_mats) == 2
    assert app.part_mats[1].get("rigid") is True
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["part_mats"][2] == {"e": 70000.0, "pr": 0.33, "ro": 2.7e-9}
    assert kopts["part_mats"][3].get("rigid") is True


def test_coord_sets(app):
    _ensure_geometry()
    app.var_cs_kind.set("Plane")
    app.var_cs_params.set("z, 0")
    app.var_cs_role.set("Fix (SPC)")
    app.var_cs_dofs.set("123")
    app._add_coord_set()
    app.var_cs_kind.set("Sphere")
    app.var_cs_params.set("0, 0, 0, 5")
    app.var_cs_role.set("Force Z")
    app.var_cs_val.set("-500")
    app._add_coord_set()
    assert len(app.coord_sets) == 2
    _, _, kopts = app._collect_inputs(need_out=False)
    cs1, cs2 = kopts["coord_sets"]
    assert cs1 == {"kind": "plane", "params": ("z", 0.0), "role": "spc",
                   "dofs": "123"}
    assert cs2["kind"] == "sphere" and cs2["role"] == "force"
    assert cs2["axis"] == "z" and cs2["value"] == -500.0
    # bad params are rejected without appending
    app.var_cs_kind.set("Box")
    app.var_cs_params.set("1, 2, 3")   # needs 6 values
    app._add_coord_set()
    assert len(app.coord_sets) == 2


def test_settings_round_trip(app):
    _ensure_geometry()
    app.var_split_mode.set("Standalone .k per body")
    app.var_batch_par.set(True)
    app.var_batch_workers.set("3")
    app.coord_sets = [{"kind": "plane", "params": ("x", 1.0), "role": "set"}]
    app.part_mats = [{"body": 2, "e": 1.0, "pr": 0.3, "ro": 1.0, "rigid": True}]
    data = app._collect_settings_data()
    app.coord_sets, app.part_mats = [], []
    app.var_split_mode.set(next(iter(gui.SPLIT_MODES)))
    app._apply_settings_data(data)
    assert app.var_split_mode.get() == "Standalone .k per body"
    assert app.var_batch_par.get() is True and app.var_batch_workers.get() == "3"
    assert app.coord_sets and app.coord_sets[0]["kind"] == "plane"
    assert app.part_mats and app.part_mats[0].get("rigid") is True


def test_progress_bar_toggles(app):
    app._set_busy(True)
    assert app.progress.winfo_manager() == "pack"
    app._set_busy(False)
    assert app.progress.winfo_manager() == ""
