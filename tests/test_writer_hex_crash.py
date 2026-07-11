"""Unit tests for HEX8 solid element output and the crash-adjacent cards
added to dyna_writer.write_k (nodal rigid bodies, point masses, discrete
springs/dampers, global damping, *INCLUDE_TRANSFORM, geometric rigidwalls).

They run on tiny hand-built meshes so they execute instantly.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from k_mesher import dyna_writer

OUT_DIR = os.path.join(ROOT, "tests", "out")

STEEL = {"e": 210000.0, "pr": 0.3, "ro": 7.85e-9}

# two unit cubes stacked in z sharing a face -> 12 nodes, 2 HEX8 elements
# (LS-DYNA / VTK_HEXAHEDRON node order: n1-n4 bottom face CCW, n5-n8 above)
HEX_COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                       [0.0, 0.0, 2.0], [1.0, 0.0, 2.0],
                       [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
HEX_ELEMS = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                      [5, 6, 7, 8, 9, 10, 11, 12]], dtype=np.int64)

TET4_COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TET4_ELEMS = np.array([[1, 2, 3, 4]], dtype=np.int64)

TET10_COORDS = np.vstack([TET4_COORDS, [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
                                        [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
                                        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]])
TET10_ELEMS = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)


def _out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def _write(name, coords, elems, **kw):
    path = _out(name)
    dyna_writer.write_k(path, coords, elems, **kw)
    return path, open(path).read()


def _fields(line, width=10):
    return [line[i:i + width] for i in range(0, len(line.rstrip()), width)]


def _find(lines, keyword):
    return next(j for j, ln in enumerate(lines)
                if ln.strip().upper() == keyword.upper())


def _find_all(lines, keyword):
    return [j for j, ln in enumerate(lines)
            if ln.strip().upper() == keyword.upper()]


def _block_rows(text, keyword):
    """Value rows of the block starting at `keyword` ($ comments skipped)."""
    lines = text.splitlines()
    i = _find(lines, keyword)
    rows = []
    for ln in lines[i + 1:]:
        if ln.startswith("*"):
            break
        if not ln.startswith("$"):
            rows.append(ln)
    return rows


# --------------------------------------------------------------------------
# 1. HEX8 solid elements
# --------------------------------------------------------------------------

def test_hex8_element_solid_one_line():
    _, txt = _write("hex_two_cubes.k", HEX_COORDS, HEX_ELEMS, mat=STEEL)
    rows = _block_rows(txt, "*ELEMENT_SOLID")
    assert len(rows) == 2, "two hexes -> two one-line rows"
    for r, row in enumerate(rows):
        vals = [int(v) for v in _fields(row, 8)]
        assert len(vals) == 10, f"HEX8 row must have 10 int fields: {row!r}"
        eid, pid, nodes = vals[0], vals[1], vals[2:]
        assert eid == r + 1 and pid == 1
        assert len(set(nodes)) == 8, f"HEX8 node ids must be distinct: {nodes}"
        assert nodes == list(HEX_ELEMS[r])


def test_hex8_default_elform_is_1():
    # constant-stress solid; elform=2 (fully integrated) is the alternative
    _, txt = _write("hex_elform_default.k", HEX_COORDS, HEX_ELEMS, mat=STEEL)
    vals = _block_rows(txt, "*SECTION_SOLID")[0]
    assert int(_fields(vals)[1]) == 1, "HEX8 default ELFORM must be 1"


def test_hex8_explicit_elform_honored():
    _, txt = _write("hex_elform2.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    elform=2)
    vals = _block_rows(txt, "*SECTION_SOLID")[0]
    assert int(_fields(vals)[1]) == 2


def test_hex8_long_format():
    _, txt = _write("hex_long.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    long_format=True)
    assert txt.splitlines()[0] == "*KEYWORD LONG=Y"
    rows = _block_rows(txt, "*ELEMENT_SOLID")
    for row in rows:
        vals = [int(v) for v in _fields(row, 20)]
        assert len(vals) == 10
        assert len(set(vals[2:])) == 8
    assert [int(v) for v in _fields(rows[1], 20)][2:] == list(HEX_ELEMS[1])


def test_hex8_start_ids_offset():
    _, txt = _write("hex_offsets.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    start_nid=101, start_eid=501)
    rows = _block_rows(txt, "*ELEMENT_SOLID")
    vals = [int(v) for v in _fields(rows[0], 8)]
    assert vals[0] == 501
    assert vals[2:] == [n - 1 + 101 for n in HEX_ELEMS[0]]


def test_tet4_and_tet10_output_unchanged():
    # golden-line spot checks: hex support must not disturb the tet paths
    _, txt = _write("hex_regress_tet4.k", TET4_COORDS, TET4_ELEMS, mat=STEEL)
    row = _block_rows(txt, "*ELEMENT_SOLID")[0]
    assert row == "".join(f"{v:8d}" for v in (1, 1, 1, 2, 3, 4, 4, 4, 4, 4))
    vals = _block_rows(txt, "*SECTION_SOLID")[0]
    assert int(_fields(vals)[1]) == 10
    _, txt = _write("hex_regress_tet10.k", TET10_COORDS, TET10_ELEMS,
                    mat=STEEL)
    rows = _block_rows(txt, "*ELEMENT_SOLID")
    assert rows[0] == f"{1:8d}{1:8d}"
    assert rows[1] == "".join(f"{v:8d}" for v in range(1, 11))
    vals = _block_rows(txt, "*SECTION_SOLID")[0]
    assert int(_fields(vals)[1]) == 16


def test_unsupported_solid_nn_raises():
    elems = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)  # wedge: unsupported
    with pytest.raises(ValueError, match="6 nodes"):
        dyna_writer.write_k(_out("hex_bad_nn.k"), HEX_COORDS, elems)


def test_hex8_write_k_include_fragments():
    part_ids = np.array([1, 2], dtype=np.int64)
    files = dyna_writer.write_k_include(
        _out("hex_asm.k"), HEX_COORDS, HEX_ELEMS, part_ids=part_ids,
        mat=STEEL)
    assert [p for p, _ in files] == [1, 2]
    master = open(_out("hex_asm.k")).read()
    assert master.count("*INCLUDE\n") == 2
    for p, fp in files:
        frag = open(fp).read()
        row = _block_rows(frag, "*ELEMENT_SOLID")[0]
        vals = [int(v) for v in _fields(row, 8)]
        assert vals[:2] == [p, p]           # global eid == row+1 == pid here
        assert vals[2:] == list(HEX_ELEMS[p - 1])   # global node ids kept


def test_hex8_write_k_split_renumbered():
    part_ids = np.array([1, 2], dtype=np.int64)
    files = dyna_writer.write_k_split(
        _out("hex_split.k"), HEX_COORDS, HEX_ELEMS, part_ids=part_ids,
        mat=STEEL)
    for p, fp in files:
        txt = open(fp).read()
        row = _block_rows(txt, "*ELEMENT_SOLID")[0]
        vals = [int(v) for v in _fields(row, 8)]
        assert vals == [1, p, 1, 2, 3, 4, 5, 6, 7, 8]   # compact renumbering
        assert len(_block_rows(txt, "*NODE")) == 8


# --------------------------------------------------------------------------
# 2. crash-adjacent cards
# --------------------------------------------------------------------------

def test_nodal_rigid_body_cards():
    # a face set first, so the NRB's node set must take the NEXT set id
    face = {"kind": "node", "title": "load face", "nodes": [9, 10, 11, 12]}
    _, txt = _write("hex_nrb.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    face_sets=(face,),
                    nodal_rigid_bodies=({"nodes": [1, 2, 3, 4], "pid": 90,
                                         "title": "base NRB"},))
    lines = txt.splitlines()
    sets = _find_all(lines, "*SET_NODE_LIST_TITLE")
    assert len(sets) == 2
    assert lines[sets[1] + 1] == "base NRB"
    nrb_sid = int(_fields(lines[sets[1] + 3])[0])
    face_sid = int(_fields(lines[sets[0] + 3])[0])
    assert (face_sid, nrb_sid) == (1, 2), "shared set-id counter, no collision"
    assert [int(v) for v in _fields(lines[sets[1] + 4], 10)] == [1, 2, 3, 4]
    vals = _block_rows(txt, "*CONSTRAINED_NODAL_RIGID_BODY")[0]
    pid, cid, nsid = (int(v) for v in _fields(vals)[:3])
    assert (pid, cid, nsid) == (90, 0, nrb_sid)


def test_nodal_rigid_body_requires_pid():
    with pytest.raises(ValueError, match="pid"):
        dyna_writer.write_k(_out("hex_nrb_nopid.k"), HEX_COORDS, HEX_ELEMS,
                            nodal_rigid_bodies=({"nodes": [1, 2, 3]},))


def test_nodal_rigid_body_skipped_in_mesh_only():
    _, txt = _write("hex_nrb_mo.k", HEX_COORDS, HEX_ELEMS, mesh_only=True,
                    nodal_rigid_bodies=({"nodes": [1, 2], "pid": 90},))
    assert "*CONSTRAINED_NODAL_RIGID_BODY" not in txt


def test_point_mass_auto_and_explicit_eids():
    # auto ids start at start_eid + len(elems) = 1 + 2 = 3; explicit "eid"
    # is used verbatim and bumps the counter past it
    _, txt = _write("hex_mass.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    point_masses=({"nid": 1, "mass": 0.5},
                                  {"nid": 2, "mass": 0.25, "eid": 100},
                                  {"nid": 3, "mass": 0.125}))
    lines = txt.splitlines()
    rows = [lines[i + 2] for i in _find_all(lines, "*ELEMENT_MASS")]
    got = [( int(r[0:8]), int(r[8:16]), float(r[16:32]) ) for r in rows]
    assert got[0] == (3, 1, 0.5)
    assert got[1] == (100, 2, 0.25)
    assert got[2] == (101, 3, 0.125)


def test_discretes_property_bookkeeping():
    # two springs sharing k -> ONE part/section/mat; one damper -> another.
    # auto pids number consecutively above max(mesh pids, explicit pids) = 1
    _, txt = _write("hex_disc.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    discretes=({"n1": 1, "n2": 5, "k": 50.0},
                               {"n1": 2, "n2": 6, "k": 50.0},
                               {"n1": 3, "n2": 7, "c": 0.4}))
    lines = txt.splitlines()
    assert len(_find_all(lines, "*SECTION_DISCRETE")) == 2
    spring = _block_rows(txt, "*MAT_SPRING_ELASTIC")[0]
    assert (int(_fields(spring)[0]), float(_fields(spring)[1])) == (2, 50.0)
    damper = _block_rows(txt, "*MAT_DAMPER_VISCOUS")[0]
    assert (int(_fields(damper)[0]), float(_fields(damper)[1])) == (3, 0.4)
    rows = [lines[i + 2] for i in _find_all(lines, "*ELEMENT_DISCRETE")]
    got = [[int(v) for v in _fields(r, 8)[:4]] for r in rows]
    # eids continue the shared counter: start_eid + len(elems) = 3, 4, 5
    assert got == [[3, 2, 1, 5], [4, 2, 2, 6], [5, 3, 3, 7]]


def test_discretes_explicit_pid_and_eid_counter_after_masses():
    _, txt = _write("hex_disc_pid.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    point_masses=({"nid": 1, "mass": 1.0},),
                    discretes=({"n1": 1, "n2": 5, "k": 9.0, "pid": 77},))
    lines = txt.splitlines()
    row = lines[_find(lines, "*ELEMENT_DISCRETE") + 2]
    vals = [int(v) for v in _fields(row, 8)[:4]]
    # the point mass consumed auto eid 3, so the discrete element gets 4
    assert vals == [4, 77, 1, 5]
    spring = _block_rows(txt, "*MAT_SPRING_ELASTIC")[0]
    assert int(_fields(spring)[0]) == 77


def test_discretes_require_exactly_one_of_k_c():
    for bad in ({"n1": 1, "n2": 2}, {"n1": 1, "n2": 2, "k": 1.0, "c": 1.0}):
        with pytest.raises(ValueError, match="exactly one"):
            dyna_writer.write_k(_out("hex_disc_bad.k"), HEX_COORDS,
                                HEX_ELEMS, discretes=(bad,))


def test_discretes_skipped_in_mesh_only():
    _, txt = _write("hex_disc_mo.k", HEX_COORDS, HEX_ELEMS, mesh_only=True,
                    discretes=({"n1": 1, "n2": 5, "k": 1.0},))
    assert "*ELEMENT_DISCRETE" not in txt
    assert "*SECTION_DISCRETE" not in txt


def test_damping_global():
    _, txt = _write("hex_damp.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    damping={"valdmp": 0.05})
    vals = _block_rows(txt, "*DAMPING_GLOBAL")[0]
    assert int(_fields(vals)[0]) == 0
    assert float(_fields(vals)[1]) == 0.05
    _, txt = _write("hex_damp_lcid.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    damping={"lcid": 7, "valdmp": 0.1})
    vals = _block_rows(txt, "*DAMPING_GLOBAL")[0]
    assert int(_fields(vals)[0]) == 7
    # a control-level card: skipped in mesh_only
    _, txt = _write("hex_damp_mo.k", HEX_COORDS, HEX_ELEMS, mesh_only=True,
                    damping={"valdmp": 0.05})
    assert "*DAMPING_GLOBAL" not in txt


def test_include_transform_layout():
    _, txt = _write("hex_inctf.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    include_transforms=({"file": "barrier.k",
                                         "idnoff": 100000, "ideoff": 200000,
                                         "idpoff": 10, "idroff": 3,
                                         "tranid": 5},))
    lines = txt.splitlines()
    i = _find(lines, "*INCLUDE_TRANSFORM")
    assert lines[i + 1] == "barrier.k"
    rows = [ln for ln in lines[i + 2:i + 11] if not ln.startswith("$")]
    offs = [int(v) for v in _fields(rows[0])]
    assert offs == [100000, 200000, 10, 0, 0, 0, 0]   # idnoff..iddoff
    assert int(_fields(rows[1])[0]) == 3              # idroff card
    scale = _fields(rows[2])
    assert [float(v) for v in scale[:4]] == [1.0, 1.0, 1.0, 1.0]
    assert int(scale[4]) == 0                          # incout1
    assert int(_fields(rows[3])[0]) == 5               # tranid card
    # written next to *INCLUDE: present in mesh_only master decks too
    _, txt = _write("hex_inctf_mo.k", HEX_COORDS, HEX_ELEMS, mesh_only=True,
                    include_transforms=({"file": "barrier.k"},))
    assert "*INCLUDE_TRANSFORM" in txt


def test_rigidwall_geometric_shapes():
    _, txt = _write("hex_rw_geom.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    rigidwalls=({"tail": (0.0, 0.0, -1.0),
                                 "head": (0.0, 0.0, 0.0), "fric": 0.2,
                                 "shape": "cylinder", "radius": 0.5,
                                 "length": 3.0},
                                {"tail": (5.0, 0.0, 0.0),
                                 "head": (6.0, 0.0, 0.0),
                                 "shape": "sphere", "radius": 2.0}))
    cyl = _block_rows(txt, "*RIGIDWALL_GEOMETRIC_CYLINDER")
    geo = [float(v) for v in _fields(cyl[1])]
    assert geo[:6] == [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    assert geo[6] == 0.2
    assert [float(v) for v in _fields(cyl[2])] == [0.5, 3.0]
    sph = _block_rows(txt, "*RIGIDWALL_GEOMETRIC_SPHERE")
    assert float(_fields(sph[2])[0]) == 2.0


def test_rigidwall_planar_unchanged_and_bad_shape():
    _, txt = _write("hex_rw_planar.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                    rigidwalls=({"tail": (0, 0, -1), "head": (0, 0, 0),
                                 "fric": 0.3},))
    rows = _block_rows(txt, "*RIGIDWALL_PLANAR")
    assert "*RIGIDWALL_GEOMETRIC" not in txt
    # golden lines: planar output must stay byte-identical
    assert rows[0] == (f"{0:10d}{0:10d}{0:10d}{0.0:10.1f}{0.0:10.1f}"
                       f"{1.0e28:10.4g}{1.0:10.1f}")
    assert rows[1] == (f"{0.0:10.4g}{0.0:10.4g}{-1.0:10.4g}{0.0:10.4g}"
                       f"{0.0:10.4g}{0.0:10.4g}{0.3:10.4g}{0.0:10.1f}")
    with pytest.raises(ValueError, match="rigidwall shape"):
        dyna_writer.write_k(_out("hex_rw_bad.k"), HEX_COORDS, HEX_ELEMS,
                            rigidwalls=({"tail": (0, 0, 0),
                                         "head": (0, 0, 1),
                                         "shape": "cone"},))


def test_new_params_default_noop():
    # defaults must leave the output byte-identical (bar the timestamp line)
    def strip_ts(t):
        return "\n".join(ln for ln in t.splitlines()
                         if not ln.startswith("$ Written by"))
    _, base = _write("hex_noop_base.k", HEX_COORDS, HEX_ELEMS, mat=STEEL)
    _, same = _write("hex_noop_same.k", HEX_COORDS, HEX_ELEMS, mat=STEEL,
                     nodal_rigid_bodies=(), point_masses=(), discretes=(),
                     damping=None, include_transforms=())
    assert strip_ts(base) == strip_ts(same)
