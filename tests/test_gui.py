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


def test_control_output_defaults_off(app):
    _ensure_geometry()
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["endtim"] is None
    assert kopts["mass_scale"] is None
    assert kopts["hourglass"] is None
    assert kopts["control_energy"] is False
    assert kopts["databases"] is None
    assert kopts["initial_velocity"] is None
    assert kopts["contacts"] == ()


def test_control_output_wiring(app):
    _ensure_geometry()
    app.var_endtim.set("0.01")
    app.var_mass_scale.set("-1e-6")
    app.var_hourglass.set(True)
    app.var_hg_ihq.set("5")
    app.var_hg_qm.set("0.15")
    app.var_ctrl_energy.set(True)
    app.var_db.set(True)
    app.var_db_dt.set("0.001")
    app.var_db_glstat.set(True)
    app.var_db_rcforc.set(True)
    app.var_ivel_vx.set("100")
    app.var_ivel_vz.set("-5")
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["endtim"] == 0.01
    assert kopts["mass_scale"] == -1e-6
    assert kopts["hourglass"] == {"ihq": 5, "qm": 0.15}
    assert kopts["control_energy"] is True
    assert kopts["databases"] == {"d3plot_dt": 0.001,
                                  "ascii": {"GLSTAT": 0.001, "RCFORC": 0.001}}
    assert kopts["initial_velocity"] == {"vx": 100.0, "vy": 0.0, "vz": -5.0}


def test_contact_type_wiring(app):
    _ensure_geometry()
    app.var_contact.set(True)
    app.var_contact_fs.set("0.2")
    # the default single-surface kind keeps contact_fs and leaves contacts empty
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["contact_fs"] == 0.2
    assert kopts["contacts"] == ()
    # a non-default kind moves the friction into a *CONTACT entry instead
    app.var_contact_type.set("Tied surface to surface")
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["contact_fs"] is None
    assert kopts["contacts"] == ({"type": "tied_surface_to_surface",
                                  "fs": 0.2},)


def test_db_dt_validation(app):
    _ensure_geometry()
    app.var_db.set(True)
    app.var_db_dt.set("0")
    with pytest.raises(ValueError):
        app._collect_inputs(need_out=False)


def test_control_output_settings_round_trip(app):
    _ensure_geometry()
    app.var_endtim.set("0.02")
    app.var_hourglass.set(True)
    app.var_db.set(True)
    app.var_contact_type.set("Automatic surface to surface")
    data = app._collect_settings_data()
    for key in ("endtim", "mass_scale", "hourglass", "hg_ihq", "hg_qm",
                "ctrl_energy", "db", "db_dt", "db_glstat", "db_matsum",
                "db_rcforc", "db_spcforc", "ivel_vx", "ivel_vy", "ivel_vz",
                "contact_type"):
        assert key in data
    app.var_endtim.set("")
    app.var_hourglass.set(False)
    app.var_db.set(False)
    app.var_contact_type.set(next(iter(gui.CONTACT_TYPES)))
    app._apply_settings_data(data)
    assert app.var_endtim.get() == "0.02"
    assert app.var_hourglass.get() is True
    assert app.var_db.get() is True
    assert app.var_contact_type.get() == "Automatic surface to surface"


def test_connections_defaults_off(app):
    _ensure_geometry()
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["midsurface"] is False
    assert kopts["auto_connect"] is None


def test_midsurface_wiring(app):
    _ensure_geometry()
    app.var_midsurface.set(True)
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["midsurface"] is True


def test_auto_connect_spotweld(app):
    _ensure_geometry()
    app.var_connect_mode.set("Spotweld (*CONSTRAINED_SPOTWELD)")
    app.var_connect_tol.set("0.5")
    app.var_connect_spacing.set("10")
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["auto_connect"] == {"mode": "spotweld", "tol": 0.5,
                                     "spacing": 10.0, "fs": 0.0}


def test_auto_connect_tied_blank_numerics(app):
    _ensure_geometry()
    app.var_connect_mode.set("Tied contact")
    app.var_connect_tol.set("")       # blank -> None
    app.var_connect_spacing.set("")   # blank -> None
    app.var_connect_fs.set("0.15")
    _, _, kopts = app._collect_inputs(need_out=False)
    assert kopts["auto_connect"] == {"mode": "tied", "tol": None,
                                     "spacing": None, "fs": 0.15}


def test_connect_tol_validation(app):
    _ensure_geometry()
    app.var_connect_mode.set("Spotweld (*CONSTRAINED_SPOTWELD)")
    app.var_connect_tol.set("-1")
    with pytest.raises(ValueError):
        app._collect_inputs(need_out=False)


def test_connections_settings_round_trip(app):
    _ensure_geometry()
    app.var_midsurface.set(True)
    app.var_connect_mode.set("Tied contact")
    app.var_connect_tol.set("0.25")
    app.var_connect_fs.set("0.2")
    data = app._collect_settings_data()
    for key in ("midsurface", "connect_mode", "connect_tol",
                "connect_spacing", "connect_fs"):
        assert key in data
    app.var_midsurface.set(False)
    app.var_connect_mode.set(next(iter(gui.CONNECT_MODES)))
    app.var_connect_tol.set("")
    app.var_connect_fs.set("0.0")
    app._apply_settings_data(data)
    assert app.var_midsurface.get() is True
    assert app.var_connect_mode.get() == "Tied contact"
    assert app.var_connect_tol.get() == "0.25"
    assert app.var_connect_fs.get() == "0.2"
