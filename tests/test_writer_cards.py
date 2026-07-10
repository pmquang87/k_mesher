"""Unit tests for the new keyword cards added to dyna_writer.write_k.

These exercise the optional keyword-only parameters (control/energy/hourglass
cards, databases, curves, initial velocity, prescribed motion, rigidwalls,
spotwelds, generic materials and contacts) plus the three correctness fixes
(elform default, large-id *NODE integers, empty *SET_* omission). They use a
tiny hand-built tetra/shell mesh so they run instantly without meshing.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from k_mesher import dyna_writer

OUT_DIR = os.path.join(ROOT, "tests", "out")

STEEL = {"e": 210000.0, "pr": 0.3, "ro": 7.85e-9}

# a single positive-volume tet (TET4) and its TET10 extension
TET4_COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TET4_ELEMS = np.array([[1, 2, 3, 4]], dtype=np.int64)

# two triangles forming a quad patch (shell)
SHELL_COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
SHELL_ELEMS = np.array([[1, 2, 3, 3], [1, 3, 4, 4]], dtype=np.int64)


def _out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def _write(name, coords, elems, **kw):
    path = _out(name)
    dyna_writer.write_k(path, coords, elems, **kw)
    return path, open(path).read()


def _keywords(text):
    return [ln.strip().upper() for ln in text.splitlines()
            if ln.startswith("*")]


def _fields(line, width=10):
    return [line[i:i + width] for i in range(0, len(line.rstrip()), width)]


def _card_value_line(text, keyword, offset=2):
    """First value line `offset` lines below `keyword` (skipping $# headers)."""
    lines = text.splitlines()
    i = next(j for j, ln in enumerate(lines)
             if ln.strip().upper() == keyword.upper())
    return lines, i, lines[i + offset]


# --------------------------------------------------------------------------
# 1. correctness fixes
# --------------------------------------------------------------------------

def test_tet10_default_elform():
    # TET10 with elform omitted -> ELFORM 16 in *SECTION_SOLID
    coords = np.vstack([TET4_COORDS, [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
                                      [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
                                      [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]])
    elems = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
    _, txt = _write("cards_tet10_elform.k", coords, elems)
    _, _, vals = _card_value_line(txt, "*SECTION_SOLID")
    secid, elform, _aet = (int(v) for v in _fields(vals)[:3])
    assert elform == 16, f"TET10 default ELFORM must be 16, got {elform}"


def test_tet4_and_shell_default_elform():
    _, txt = _write("cards_tet4_elform.k", TET4_COORDS, TET4_ELEMS)
    _, _, vals = _card_value_line(txt, "*SECTION_SOLID")
    assert int(_fields(vals)[1]) == 10, "TET4 default ELFORM must be 10"
    _, txt = _write("cards_shell_elform.k", SHELL_COORDS, SHELL_ELEMS,
                    element_kind="shell")
    _, _, vals = _card_value_line(txt, "*SECTION_SHELL")
    assert int(_fields(vals)[1]) == 2, "shell default ELFORM must be 2"


def test_large_start_nid_long_format():
    # a node id beyond 2^53 must print exactly (no float promotion)
    big = 2 ** 53 + 12345
    path, txt = _write("cards_bignid.k", TET4_COORDS, TET4_ELEMS,
                       start_nid=big, long_format=True)
    lines = txt.splitlines()
    i = lines.index("*NODE")
    first_id = int(lines[i + 2][0:20])
    assert first_id == big, f"node id must be exact: {first_id} != {big}"


def test_empty_sets_omitted():
    # empty node/segment/element sets must not emit a header with no rows
    path, txt = _write(
        "cards_empty_sets.k", TET4_COORDS, TET4_ELEMS,
        face_sets=(
            {"kind": "node", "title": "EMPTY_N", "nodes": np.array([], int)},
            {"kind": "node", "title": "REAL_N", "nodes": np.array([1, 2])},
            {"kind": "segment", "title": "EMPTY_S",
             "segments": np.zeros((0, 4), int)},
        ),
        elem_sets=({"title": "EMPTY_E", "eids": np.array([], int)},),
        sym_sets=({"axis": "x", "offset": 0.0, "nodes": np.array([], int)},),
    )
    assert "EMPTY_N" not in txt and "EMPTY_S" not in txt and "EMPTY_E" not in txt
    assert "REAL_N" in txt
    # exactly one node-list set survives
    assert _keywords(txt).count("*SET_NODE_LIST_TITLE") == 1


# --------------------------------------------------------------------------
# 2-5. control cards
# --------------------------------------------------------------------------

def test_endtim_control_termination():
    _, txt = _write("cards_endtim.k", TET4_COORDS, TET4_ELEMS, endtim=0.005)
    assert "*CONTROL_TERMINATION" in _keywords(txt)
    _, _, vals = _card_value_line(txt, "*CONTROL_TERMINATION")
    assert abs(float(vals[0:10]) - 0.005) < 1e-9


def test_endtim_skipped_with_implicit():
    # implicit deck writes its own termination -> only one card
    _, txt = _write("cards_endtim_impl.k", TET4_COORDS, TET4_ELEMS,
                    endtim=0.005, implicit_cards=True)
    assert _keywords(txt).count("*CONTROL_TERMINATION") == 1


def test_mass_scale_timestep():
    # mass_scale alone must still emit *CONTROL_TIMESTEP (tssfac -> 0.9)
    _, txt = _write("cards_dt2ms.k", TET4_COORDS, TET4_ELEMS, mass_scale=-1.0e-6)
    assert "*CONTROL_TIMESTEP" in _keywords(txt)
    _, _, vals = _card_value_line(txt, "*CONTROL_TIMESTEP")
    fld = _fields(vals)
    assert abs(float(fld[1]) - 0.9) < 1e-9, "default TSSFAC 0.9"
    assert abs(float(fld[4]) - (-1.0e-6)) < 1e-12, "DT2MS in field 5"


def test_control_energy():
    _, txt = _write("cards_energy.k", TET4_COORDS, TET4_ELEMS,
                    control_energy=True)
    assert "*CONTROL_ENERGY" in _keywords(txt)
    _, _, vals = _card_value_line(txt, "*CONTROL_ENERGY")
    assert [int(v) for v in _fields(vals)[:4]] == [2, 2, 2, 2]


# --------------------------------------------------------------------------
# 4. hourglass
# --------------------------------------------------------------------------

def test_hourglass_hgid_matches_part():
    _, txt = _write("cards_hg.k", TET4_COORDS, TET4_ELEMS,
                    hourglass={"ihq": 4, "qm": 0.1})
    assert "*HOURGLASS" in _keywords(txt)
    _, _, part_vals = _card_value_line(txt, "*PART", offset=3)
    part_hgid = int(_fields(part_vals)[4])   # pid secid mid eosid hgid ...
    _, _, hg_vals = _card_value_line(txt, "*HOURGLASS")
    hg_id = int(_fields(hg_vals)[0])
    assert part_hgid == hg_id and hg_id != 0, "PART hgid must match *HOURGLASS id"


# --------------------------------------------------------------------------
# 6. databases
# --------------------------------------------------------------------------

def test_databases():
    _, txt = _write("cards_db.k", TET4_COORDS, TET4_ELEMS,
                    databases={"d3plot_dt": 0.001,
                               "ascii": {"glstat": 0.0005, "MATSUM": 0.0005}})
    kws = _keywords(txt)
    assert "*DATABASE_BINARY_D3PLOT" in kws
    assert "*DATABASE_GLSTAT" in kws, "ascii names accepted case-insensitively"
    assert "*DATABASE_MATSUM" in kws
    _, _, vals = _card_value_line(txt, "*DATABASE_GLSTAT")
    assert int(_fields(vals)[1]) == 1, "ascii database binary flag = 1"


# --------------------------------------------------------------------------
# 7. define_curves
# --------------------------------------------------------------------------

def test_define_curve_roundtrip():
    pts = [(0.0, 0.0), (0.5, 100.0), (1.0, 250.0)]
    _, txt = _write("cards_curve.k", TET4_COORDS, TET4_ELEMS,
                    define_curves=({"lcid": 42, "points": pts,
                                    "title": "load ramp"},))
    lines = txt.splitlines()
    i = next(j for j, ln in enumerate(lines)
             if ln.startswith("*DEFINE_CURVE_TITLE"))
    assert "load ramp" in lines[i + 1]
    assert int(lines[i + 3][0:10]) == 42, "lcid on the curve header"
    got = []
    for ln in lines[i + 4:]:
        if ln.startswith("*") or ln.startswith("$"):
            break
        got.append((float(ln[0:20]), float(ln[20:40])))
    assert got == pts, f"curve points must round-trip: {got}"


# --------------------------------------------------------------------------
# 8-11. initial velocity / prescribed motion / rigidwall / spotweld
# --------------------------------------------------------------------------

def test_initial_velocity():
    _, txt = _write("cards_iv.k", TET4_COORDS, TET4_ELEMS,
                    initial_velocity={"vx": 10.0, "vy": 0.0, "vz": -5.0})
    assert "*INITIAL_VELOCITY_GENERATION" in _keywords(txt)
    _, _, vals = _card_value_line(txt, "*INITIAL_VELOCITY_GENERATION")
    fld = _fields(vals)
    assert int(fld[0]) == 0, "nsid 0 = all nodes"
    assert abs(float(fld[3]) - 10.0) < 1e-6 and abs(float(fld[5]) + 5.0) < 1e-6


def test_prescribed_motion():
    _, txt = _write("cards_pm.k", TET4_COORDS, TET4_ELEMS,
                    prescribed_motion=({"nsid": 5, "dof": 3, "vad": 2,
                                        "lcid": 7, "sf": 2.5},))
    assert "*BOUNDARY_PRESCRIBED_MOTION_SET" in _keywords(txt)
    _, _, vals = _card_value_line(txt, "*BOUNDARY_PRESCRIBED_MOTION_SET")
    fld = _fields(vals)
    assert [int(fld[0]), int(fld[1]), int(fld[2]), int(fld[3])] == [5, 3, 2, 7]
    assert abs(float(fld[4]) - 2.5) < 1e-6


def test_rigidwall():
    _, txt = _write("cards_rw.k", TET4_COORDS, TET4_ELEMS,
                    rigidwalls=({"tail": (0.0, 0.0, 0.0),
                                 "head": (0.0, 0.0, 1.0), "fric": 0.2},))
    assert "*RIGIDWALL_PLANAR" in _keywords(txt)
    _, _, vals = _card_value_line(txt, "*RIGIDWALL_PLANAR", offset=4)
    fld = _fields(vals)
    assert abs(float(fld[5]) - 1.0) < 1e-6, "head z on card 2"
    assert abs(float(fld[6]) - 0.2) < 1e-6, "friction on card 2"


def test_spotwelds_count():
    pairs = ((1, 2), (2, 3), (3, 4))
    _, txt = _write("cards_sw.k", TET4_COORDS, TET4_ELEMS, spotwelds=pairs)
    assert _keywords(txt).count("*CONSTRAINED_SPOTWELD") == 3
    _, _, vals = _card_value_line(txt, "*CONSTRAINED_SPOTWELD")
    assert [int(v) for v in _fields(vals)[:2]] == [1, 2]


# --------------------------------------------------------------------------
# 12. generic materials + tied/general contacts
# --------------------------------------------------------------------------

def test_material_passthrough():
    raw = {
        "keyword": "*MAT_PIECEWISE_LINEAR_PLASTICITY",
        "cards": [
            "{mid:>10d}  7.85e-09    210000       0.3       400        10",
            "         0         0",
        ],
    }
    _, txt = _write("cards_mat_raw.k", TET4_COORDS, TET4_ELEMS, mat=raw, pid=7)
    assert "*MAT_PIECEWISE_LINEAR_PLASTICITY" in _keywords(txt)
    assert "*MAT_ELASTIC" not in _keywords(txt)
    # the {mid} placeholder must be substituted with the real MID (pid 7)
    lines = txt.splitlines()
    i = next(j for j, ln in enumerate(lines)
             if ln.strip() == "*MAT_PIECEWISE_LINEAR_PLASTICITY")
    assert int(lines[i + 1][0:10]) == 7, "MID placeholder substituted"
    assert "210000" in lines[i + 1], "raw card content preserved verbatim"


def test_general_contacts():
    _, txt = _write("cards_contacts.k", TET4_COORDS, TET4_ELEMS, contacts=(
        {"type": "tied_surface_to_surface", "fs": 0.0},
        {"type": "automatic_surface_to_surface", "fs": 0.15},
    ))
    kws = _keywords(txt)
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" in kws
    assert "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE" in kws
    assert "0.1500" in txt, "friction on the s2s card"


def test_contact_fs_still_single_surface():
    _, txt = _write("cards_contact_fs.k", TET4_COORDS, TET4_ELEMS,
                    contact_fs=0.3)
    assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" in _keywords(txt)
    assert "0.3000" in txt
