"""Unit tests for the second batch of keyword cards added to dyna_writer:

  * pairwise scoped contacts via *SET_PART_LIST (contact_pairs)
  * cross-section force output (*DATABASE_CROSS_SECTION_PLANE + _SECFORC)
  * time-history output (*DATABASE_HISTORY_NODE/_SOLID/_SHELL)
  * the named-material convenience builders (mat_* helpers)

They run on a tiny hand-built tetra mesh so they execute instantly.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from k_mesher import dyna_writer

OUT_DIR = os.path.join(ROOT, "tests", "out")

# two tets sharing a face -> a two-part mesh (parts 1 and 2)
COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
ELEMS = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int64)
PART_IDS = np.array([1, 2], dtype=np.int64)

TET4_COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TET4_ELEMS = np.array([[1, 2, 3, 4]], dtype=np.int64)


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


def _find_all(lines, keyword):
    return [j for j, ln in enumerate(lines)
            if ln.strip().upper() == keyword.upper()]


# --------------------------------------------------------------------------
# 1. pairwise contacts via *SET_PART_LIST
# --------------------------------------------------------------------------

def test_contact_pairs_scoped():
    _, txt = _write(
        "cards2_contact_pairs.k", COORDS, ELEMS, part_ids=PART_IDS,
        contact_pairs=(
            {"slave_parts": [1], "master_parts": [2],
             "type": "automatic_surface_to_surface", "fs": 0.1,
             "title": "tool to blank"},
            {"slave_parts": [2], "master_parts": [1],
             "type": "tied_surface_to_surface", "fs": 0.0},
        ),
    )
    lines = txt.splitlines()
    kws = _keywords(txt)
    # two part-set lists per pair -> four *SET_PART_LIST cards
    assert kws.count("*SET_PART_LIST_TITLE") == 4
    assert "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE" in kws
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" in kws

    # set ids consumed in order 1,2 (pair 1) then 3,4 (pair 2)
    set_hdrs = _find_all(lines, "*SET_PART_LIST_TITLE")
    set_ids = [int(_fields(lines[i + 3])[0]) for i in set_hdrs]
    assert set_ids == [1, 2, 3, 4], set_ids
    # first list holds slave pid 1, second holds master pid 2
    assert int(_fields(lines[set_hdrs[0] + 5])[0]) == 1
    assert int(_fields(lines[set_hdrs[1] + 5])[0]) == 2

    # the automatic (first) contact card: SSID=1 MSID=2 SSTYP=MSTYP=2, fs=0.1
    i = _find_all(lines, "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE")[0]
    fld = _fields(lines[i + 2])
    assert [int(fld[0]), int(fld[1]), int(fld[2]), int(fld[3])] == [1, 2, 2, 2]
    fsline = _fields(lines[i + 4])
    assert abs(float(fsline[0]) - 0.1) < 1e-9, "friction on card 2"

    # the tied (second) contact card references sets 3 and 4
    j = _find_all(lines, "*CONTACT_TIED_SURFACE_TO_SURFACE")[0]
    tfld = _fields(lines[j + 2])
    assert [int(tfld[0]), int(tfld[1]), int(tfld[2]), int(tfld[3])] == [3, 4, 2, 2]


def test_contact_pairs_skipped_in_mesh_only():
    _, txt = _write(
        "cards2_contact_pairs_meshonly.k", COORDS, ELEMS, part_ids=PART_IDS,
        mesh_only=True,
        contact_pairs=({"slave_parts": [1], "master_parts": [2],
                        "type": "automatic_surface_to_surface", "fs": 0.1},))
    kws = _keywords(txt)
    assert "*SET_PART_LIST_TITLE" not in kws
    assert "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE" not in kws


# --------------------------------------------------------------------------
# 2. cross-section force output
# --------------------------------------------------------------------------

def test_cross_sections_auto_secforc():
    _, txt = _write(
        "cards2_xsec.k", COORDS, ELEMS, part_ids=PART_IDS, endtim=0.01,
        cross_sections=(
            {"point": (0.5, 0.0, 0.0), "normal": (1.0, 0.0, 0.0),
             "title": "mid plane"},
            {"point": (0.0, 0.5, 0.0), "normal": (0.0, 1.0, 0.0)},
        ),
    )
    kws = _keywords(txt)
    assert kws.count("*DATABASE_CROSS_SECTION_PLANE") == 2
    assert "*DATABASE_SECFORC" in kws, "SECFORC auto-added for the cross sections"

    lines = txt.splitlines()
    i = _find_all(lines, "*DATABASE_CROSS_SECTION_PLANE")[0]
    # card 1: PSID(0), then XCT/YCT/ZCT = the cut point
    fld = _fields(lines[i + 2])
    assert int(fld[0]) == 0, "PSID 0 = whole model"
    assert abs(float(fld[1]) - 0.5) < 1e-6, "XCT = point.x"
    # the head is one unit along the normal -> XCH = 1.5
    assert abs(float(fld[4]) - 1.5) < 1e-6, "XCH = point + normal"


def test_cross_sections_reuse_user_secforc():
    # if the user already asked for SECFORC, do not add a second one
    _, txt = _write(
        "cards2_xsec_user_secforc.k", COORDS, ELEMS, part_ids=PART_IDS,
        databases={"d3plot_dt": 0.001, "ascii": {"SECFORC": 0.0005}},
        cross_sections=({"point": (0.5, 0.0, 0.0),
                         "normal": (1.0, 0.0, 0.0)},),
    )
    assert _keywords(txt).count("*DATABASE_SECFORC") == 1
    assert "*DATABASE_CROSS_SECTION_PLANE" in _keywords(txt)


# --------------------------------------------------------------------------
# 3. time-history output
# --------------------------------------------------------------------------

def test_history_categories():
    _, txt = _write(
        "cards2_history.k", COORDS, ELEMS, part_ids=PART_IDS,
        history={"nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 "solids": [1, 2], "shells": []},
    )
    kws = _keywords(txt)
    assert "*DATABASE_HISTORY_NODE" in kws
    assert "*DATABASE_HISTORY_SOLID" in kws
    assert "*DATABASE_HISTORY_SHELL" not in kws, "empty category omitted"

    lines = txt.splitlines()
    i = _find_all(lines, "*DATABASE_HISTORY_NODE")[0]
    # 9 node ids -> two lines (8 + 1)
    first = [int(v) for v in _fields(lines[i + 2])]
    second = [int(v) for v in _fields(lines[i + 3])]
    assert first == [1, 2, 3, 4, 5, 6, 7, 8]
    assert second == [9]


# --------------------------------------------------------------------------
# 4. named-material convenience builders
# --------------------------------------------------------------------------

def _mat_block(txt, keyword):
    """Return the card lines following a raw-passthrough material keyword."""
    lines = txt.splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.strip() == keyword)
    out = []
    for ln in lines[i + 1:]:
        if ln.startswith("*") or ln.startswith("$"):
            break
        out.append(ln)
    return i, out


def test_mat_piecewise_linear_plasticity_helper():
    m = dyna_writer.mat_piecewise_linear_plasticity(
        210000.0, 0.3, 7.85e-9, 400.0, etan=1000.0)
    assert m["keyword"] == "*MAT_PIECEWISE_LINEAR_PLASTICITY"
    assert m["e"] == 210000.0 and m["pr"] == 0.3 and m["ro"] == 7.85e-9
    _, txt = _write("cards2_mat024.k", TET4_COORDS, TET4_ELEMS, mat=m, pid=7)
    assert "*MAT_PIECEWISE_LINEAR_PLASTICITY" in _keywords(txt)
    assert "*MAT_ELASTIC" not in _keywords(txt)
    i, cards = _mat_block(txt, "*MAT_PIECEWISE_LINEAR_PLASTICITY")
    fld = _fields(cards[0])
    assert int(fld[0]) == 7, "MID placeholder substituted"
    assert abs(float(fld[1]) - 7.85e-9) < 1e-14, "ro column"
    assert abs(float(fld[2]) - 210000.0) < 1e-3, "E column"
    assert abs(float(fld[3]) - 0.3) < 1e-9, "pr column"
    assert abs(float(fld[4]) - 400.0) < 1e-6, "sigy column"
    assert abs(float(fld[5]) - 1000.0) < 1e-6, "etan column"


def test_mat_plastic_kinematic_helper():
    m = dyna_writer.mat_plastic_kinematic(
        210000.0, 0.3, 7.85e-9, 350.0, etan=500.0, beta=0.5)
    assert m["keyword"] == "*MAT_PLASTIC_KINEMATIC"
    _, txt = _write("cards2_mat003.k", TET4_COORDS, TET4_ELEMS, mat=m, pid=3)
    i, cards = _mat_block(txt, "*MAT_PLASTIC_KINEMATIC")
    fld = _fields(cards[0])
    assert int(fld[0]) == 3, "MID substituted"
    assert abs(float(fld[4]) - 350.0) < 1e-6, "sigy column"
    assert abs(float(fld[5]) - 500.0) < 1e-6, "etan column"
    assert abs(float(fld[6]) - 0.5) < 1e-9, "beta column"


def test_mat_johnson_cook_helper():
    m = dyna_writer.mat_johnson_cook(
        210000.0, 0.3, 7.85e-9, a=350.0, b=275.0, n=0.36, c=0.022, m=1.0,
        tm=1793.0, tr=293.0, epso=1.0, cp=452.0)
    assert m["keyword"] == "*MAT_JOHNSON_COOK"
    _, txt = _write("cards2_mat015.k", TET4_COORDS, TET4_ELEMS, mat=m, pid=5)
    i, cards = _mat_block(txt, "*MAT_JOHNSON_COOK")
    # card 1: MID RO G E PR
    fld1 = _fields(cards[0])
    assert int(fld1[0]) == 5, "MID substituted"
    g = 210000.0 / (2.0 * 1.3)
    assert abs(float(fld1[2]) - g) < 1e-1, "shear modulus derived from E, pr"
    assert abs(float(fld1[3]) - 210000.0) < 1e-3, "E column"
    # card 2: A B N C M TM TR EPSO
    fld2 = _fields(cards[1])
    assert abs(float(fld2[0]) - 350.0) < 1e-6, "A column"
    assert abs(float(fld2[1]) - 275.0) < 1e-6, "B column"
    assert abs(float(fld2[2]) - 0.36) < 1e-6, "n column"


def test_mat_null_helper():
    m = dyna_writer.mat_null(1.0e-9, mu=0.1)
    assert m["keyword"] == "*MAT_NULL"
    assert m["ro"] == 1.0e-9
    _, txt = _write("cards2_mat009.k", TET4_COORDS, TET4_ELEMS, mat=m, pid=9)
    i, cards = _mat_block(txt, "*MAT_NULL")
    fld = _fields(cards[0])
    assert int(fld[0]) == 9, "MID substituted"
    assert abs(float(fld[1]) - 1.0e-9) < 1e-16, "ro column"
    assert abs(float(fld[3]) - 0.1) < 1e-9, "mu column"
