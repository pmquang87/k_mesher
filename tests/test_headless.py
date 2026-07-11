"""Headless end-to-end tests: STEP/IGES/BREP/STL -> TET4/TET10/TRI3/QUAD4 -> .k

Run with pytest (pytest tests/test_headless.py -v) or directly
(python tests/test_headless.py). Expensive meshes are cached in-process and
shared between test functions, so the file stays fast either way.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from k_mesher import dyna_writer
from k_mesher import mesh_cli
from k_mesher import mesher
from make_test_step import make, make_two_bodies, make_shell, make_formats

EX = os.path.join(ROOT, "examples")
OUT_DIR = os.path.join(ROOT, "tests", "out")
STEP = os.path.join(EX, "test_part.step")
STEP2 = os.path.join(EX, "test_two_bodies.step")
STEP_SH = os.path.join(EX, "test_shell.step")
FORMAT_BASE = os.path.join(EX, "test_part")

STEEL = {"e": 210000.0, "pr": 0.3, "ro": 7.85e-9}
ALU = {"e": 70000.0, "pr": 0.33, "ro": 2.7e-9}

_cache: dict = {}


def QUIET(_msg):
    pass


# --------------------------------------------------------------------------
# shared fixtures (plain functions with an in-process cache, so the file
# also runs without pytest)
# --------------------------------------------------------------------------

def _ensure_geometry():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(STEP):
        make(STEP)
    if not os.path.isfile(STEP2):
        make_two_bodies(STEP2)
    if not os.path.isfile(STEP_SH):
        make_shell(STEP_SH)


def _cached(key, factory):
    if key not in _cache:
        _ensure_geometry()
        _cache[key] = factory()
    return _cache[key]


def _mesh(key, **kw):
    return _cached(key, lambda: mesher.mesh_step(
        mesher.MeshSettings(**kw), log=QUIET))


def res_full():
    return _mesh("full", step_file=STEP, size_max=8.0, size_min=2.0)


def res_half_x():
    return _mesh("half_x", step_file=STEP, size_max=8.0, size_min=2.0,
                 symmetry=[mesher.SymmetryPlane("x", 0.0, "+")])


def res_coarse():
    return _mesh("coarse", step_file=STEP, size_max=8.0)


def res_shell_tri3():
    return _mesh("shell_tri3", step_file=STEP_SH, element_type="TRI3",
                 size_max=5.0)


def res_two_glued():
    return _mesh("two_glued", step_file=STEP2, size_max=6.0, glue=True)


def faces_step():
    return _cached("faces", lambda: mesher.list_faces(
        mesher.MeshSettings(step_file=STEP), log=QUIET))


def xmax_tag():
    """The x = +50 face of the test part (area 1250)."""
    return next(f["tag"] for f in faces_step()
                if abs(f["centroid"][0] - 50.0) < 1e-3
                and abs(f["area"] - 1250.0) < 1.0)


def res_xmax_face():
    return _cached("xmax_face", lambda: mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP, size_max=8.0, size_min=2.0,
        collect_faces=[xmax_tag()]), log=QUIET))


# --------------------------------------------------------------------------
# .k parsing helpers
# --------------------------------------------------------------------------

def spc_dofs(text):
    """Return the 6 DOF flags [dofx..dofrz] of the first *BOUNDARY_SPC_SET."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("*BOUNDARY_SPC_SET"):
            data = lines[i + 2]   # keyword, $# header, then the value line
            vals = [int(data[j:j + 10]) for j in range(0, 80, 10)]
            return vals[2:8]
    return None


def parse_k(path):
    """Minimal .k reader: *NODE, *ELEMENT_SOLID (one- and two-line formats)
    and *ELEMENT_SHELL. Returns (nodes {nid: xyz}, elems, shells, keywords).
    Standard (8/16-char) format only - long-format tests parse manually."""
    nodes, elems, shells, keywords = {}, [], [], []
    section = None
    pending = None  # (eid, pid) awaiting the node line (two-line format)
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("$"):
                continue
            if line.startswith("*"):
                section = line.strip().upper()
                keywords.append(section)
                pending = None
                continue
            if not line.strip():
                continue
            if section == "*NODE":
                nid = int(line[0:8])
                nodes[nid] = [float(line[8:24]), float(line[24:40]),
                              float(line[40:56])]
            elif section == "*ELEMENT_SOLID":
                if pending is not None:
                    nds = [int(line[i:i + 8]) for i in range(0, 80, 8)]
                    elems.append(list(pending) + nds)
                    pending = None
                elif len(line.rstrip()) <= 16:
                    pending = (int(line[0:8]), int(line[8:16]))
                else:
                    elems.append([int(line[i:i + 8]) for i in range(0, 80, 8)])
            elif section == "*ELEMENT_SHELL":
                shells.append([int(line[i:i + 8]) for i in range(0, 48, 8)])
    return nodes, elems, shells, keywords


def part_cards(path):
    """[(pid, secid, mid)] from the *PART cards of a standard-format file."""
    lines = open(path).read().splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "*PART":
            vals = lines[i + 3]
            out.append((int(vals[0:10]), int(vals[10:20]), int(vals[20:30])))
    return out


def mat_mids(path):
    """MIDs of all *MAT_ELASTIC cards in a standard-format file."""
    lines = open(path).read().splitlines()
    return sorted(int(lines[i + 2][0:10]) for i, ln in enumerate(lines)
                  if ln.strip().upper() == "*MAT_ELASTIC")


def tet_volumes(nodes, elems):
    v = []
    for e in elems:
        p = np.array([nodes[e[2]], nodes[e[3]], nodes[e[4]], nodes[e[5]]])
        v.append(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]) / 6.0)
    return np.array(v)


def shell_areas(nodes, shells):
    a = []
    for e in shells:
        p = np.array([nodes[n] for n in e[2:6]])
        a.append(0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) +
                 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0])))
    return np.array(a)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_full_tet4():
    print("=== full model (TET4) ===")
    res = res_full()
    assert res.stats["n_elems"] > 100
    # part volume: 100*50*25 - pi*10^2*25 = 117146
    assert abs(res.stats["measure"] - 117146.0) / 117146.0 < 0.02
    assert sum(res.stats["quality_hist"]) == res.stats["n_elems"]
    # quality criteria + mass properties present
    names = [c["name"] for c in res.stats["criteria"]]
    assert "aspect ratio" in names and "SICN" in names
    assert "failed_elems" in res.stats
    assert res.stats["duplicate_nodes"] == 0
    cog = res.stats["cog"]
    assert all(abs(v) < 0.5 for v in cog), f"COG should be ~origin: {cog}"
    inertia = np.array(res.stats["inertia_unit_density"])
    assert (np.diag(inertia) > 0).all() and np.allclose(inertia, inertia.T)
    # characteristic length statistics for the timestep estimate
    assert 0 < res.stats["char_length"] <= res.stats["char_length_pctiles"]["p1"]
    pct = res.stats["char_length_pctiles"]
    assert pct["p1"] <= pct["p10"] <= pct["p50"]

    k_full = os.path.join(OUT_DIR, "full.k")
    dyna_writer.write_k(k_full, res.coords, res.elems)
    nodes, elems, _, keywords = parse_k(k_full)
    assert len(nodes) == res.stats["n_nodes"]
    assert len(elems) == res.stats["n_elems"]
    assert all(e[5] == e[6] == e[7] == e[8] == e[9] for e in elems)
    assert (tet_volumes(nodes, elems) > 0).all()
    print(f"OK: {len(nodes)} nodes, {len(elems)} tets")


def test_half_model_symmetry_sets_ids():
    print("=== half model (X symmetry, keep +) ===")
    res = res_half_x()
    assert res.coords[:, 0].min() >= -1e-6
    assert abs(res.stats["measure"] - 117146.0 / 2) / (117146.0 / 2) < 0.02
    n_sym = len(res.sym_nodes["x"])
    assert n_sym > 10

    k_half = os.path.join(OUT_DIR, "half_x.k")
    sym_sets = ({"axis": "x", "offset": 0.0, "nodes": res.sym_nodes["x"],
                 "spc": True},)
    dyna_writer.write_k(k_half, res.coords, res.elems, sym_sets=sym_sets,
                        start_nid=1000, start_eid=5000, mat=dict(STEEL))
    nodes, elems, _, keywords = parse_k(k_half)
    assert min(nodes) == 1000 and elems[0][0] == 5000
    assert "*SET_NODE_LIST_TITLE" in keywords
    assert "*BOUNDARY_SPC_SET" in keywords
    assert "*MAT_ELASTIC" in keywords
    assert (tet_volumes(nodes, elems) > 0).all()
    sym_nids = [n - 1 + 1000 for n in res.sym_nodes["x"]]
    assert all(abs(nodes[nid][0]) < 1e-6 for nid in sym_nids)
    print(f"OK: {len(nodes)} nodes, {n_sym} in SYM_X set, MAT card present")


def test_quarter_model():
    print("=== quarter model (X keep +, Y keep -) ===")
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP, size_max=8.0, size_min=2.0,
        symmetry=[mesher.SymmetryPlane("x", 0.0, "+"),
                  mesher.SymmetryPlane("y", 0.0, "-")]), log=QUIET)
    assert res.coords[:, 0].min() >= -1e-6
    assert res.coords[:, 1].max() <= 1e-6
    assert abs(res.stats["measure"] - 117146.0 / 4) / (117146.0 / 4) < 0.02
    print(f"OK: volume {res.stats['measure']:.0f}")


def test_tet10():
    print("=== TET10 half model ===")
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP, element_type="TET10", size_max=8.0, size_min=2.0,
        symmetry=[mesher.SymmetryPlane("x", 0.0, "+")]), log=QUIET)
    assert res.elems.shape[1] == 10
    assert abs(res.stats["measure"] - 117146.0 / 2) / (117146.0 / 2) < 0.02
    # mid-edge nodes must sit near the midpoint of their corner pair
    # (LS-DYNA ordering: n5..n10 = mid of (1,2),(2,3),(3,1),(1,4),(2,4),(3,4))
    pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    p = res.coords[res.elems - 1]
    for mid_col, (a, b) in zip(range(4, 10), pairs):
        dev = np.linalg.norm(p[:, mid_col] - 0.5 * (p[:, a] + p[:, b]), axis=1)
        edge = np.linalg.norm(p[:, a] - p[:, b], axis=1)
        assert (dev <= 0.35 * edge).all()
    sym_ids = set(res.sym_nodes["x"].tolist())
    on_plane = np.flatnonzero(np.abs(res.coords[:, 0]) < 1e-6) + 1
    assert set(on_plane.tolist()) == sym_ids

    k_t10 = os.path.join(OUT_DIR, "half_x_tet10.k")
    dyna_writer.write_k(k_t10, res.coords, res.elems, elform=16)
    nodes, elems, _, keywords = parse_k(k_t10)
    assert len(elems) == res.stats["n_elems"]
    assert all(len(e) == 12 for e in elems)
    assert (tet_volumes(nodes, elems) > 0).all()
    print(f"OK: {res.stats['n_elems']} TET10, two-line format, mid nodes verified")


def test_two_bodies_glue_pids():
    print("=== two bodies, glue, per-body PIDs ===")
    res = res_two_glued()
    assert len(res.part_names) == 2
    assert set(np.unique(res.elem_parts)) == {0, 1}
    assert abs(res.stats["measure"] - 16000.0) / 16000.0 < 0.01
    # per-part measures back the per-part mass reporting
    pm = res.stats["part_measures"]
    assert len(pm) == 2 and abs(sum(pm) - res.stats["measure"]) < 1e-6
    assert all(abs(v - 8000.0) / 8000.0 < 0.01 for v in pm)
    iface = res.coords[np.abs(res.coords[:, 0] - 20.0) < 1e-6]
    assert len(iface) == len(np.unique(np.round(iface, 6), axis=0))

    k_two = os.path.join(OUT_DIR, "two_bodies.k")
    dyna_writer.write_k(k_two, res.coords, res.elems, pid=10,
                        part_ids=10 + res.elem_parts,
                        part_titles={10: "left box", 11: "right box"})
    nodes, elems, _, keywords = parse_k(k_two)
    assert keywords.count("*PART") == 2
    assert set(e[1] for e in elems) == {10, 11}
    print("OK: 2 PIDs, shared interface nodes, per-part measures")


def test_face_sets():
    print("=== face sets (node + segment) ===")
    assert len(faces_step()) >= 7
    tag = xmax_tag()
    res = res_xmax_face()
    fn, fs = res.face_nodes[tag], res.face_segs[tag]
    assert len(fn) > 5 and len(fs) > 5 and fs.shape[1] == 4
    assert np.allclose(res.coords[fn - 1][:, 0], 50.0, atol=1e-6)
    assert set(fs.ravel().tolist()) <= set(fn.tolist())

    k_face = os.path.join(OUT_DIR, "face_sets.k")
    dyna_writer.write_k(k_face, res.coords, res.elems, face_sets=(
        {"kind": "node", "title": f"FACE_{tag}", "nodes": fn},
        {"kind": "segment", "title": f"FACE_{tag}", "segments": fs},
    ))
    _, _, _, keywords = parse_k(k_face)
    assert "*SET_NODE_LIST_TITLE" in keywords
    assert "*SET_SEGMENT_TITLE" in keywords
    print(f"OK: face {tag}: {len(fn)} nodes, {len(fs)} segments")


def test_refinement_sphere():
    print("=== local refinement (sphere) ===")
    res_ref = mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP, size_max=8.0,
        refinements=[{"kind": "sphere", "params": [40.0, 0.0, 0.0, 15.0],
                      "size": 2.0}]), log=QUIET)

    def nodes_in_sphere(res):
        d = np.linalg.norm(res.coords - np.array([40.0, 0.0, 0.0]), axis=1)
        return int((d < 15.0).sum())

    n_base, n_ref = nodes_in_sphere(res_coarse()), nodes_in_sphere(res_ref)
    assert n_ref > 3 * n_base
    print(f"OK: nodes inside sphere {n_base} -> {n_ref}")


def test_tri3_shells():
    print("=== TRI3 shells (surface-only STEP) ===")
    res = res_shell_tri3()
    assert res.elems.shape[1] == 4
    assert (res.elems[:, 2] == res.elems[:, 3]).all(), "TRI3 must be degenerate quads"
    assert res.stats["measure_label"] == "area"
    assert abs(res.stats["measure"] - 6200.0) / 6200.0 < 0.01  # box area
    assert res.stats["free_edges"] == 0, "closed box shell must be watertight"
    assert res.stats["nonmanifold_edges"] == 0

    k_tri = os.path.join(OUT_DIR, "shell_tri3.k")
    dyna_writer.write_k(k_tri, res.coords, res.elems, element_kind="shell",
                        elform=4, thickness=1.5)
    nodes, elems, shells, keywords = parse_k(k_tri)
    assert "*ELEMENT_SHELL" in keywords and "*SECTION_SHELL" in keywords
    assert "*ELEMENT_SOLID" not in keywords
    assert len(shells) == res.stats["n_elems"]
    assert abs(shell_areas(nodes, shells).sum() - 6200.0) < 65
    with open(k_tri) as f:
        content = f.read()
    assert "       1.5       1.5       1.5       1.5" in content, "thickness card"
    print(f"OK: {len(shells)} TRI3 shells, area 6200, thickness card written")


def test_tri3_shells_symmetry():
    print("=== TRI3 shells with X symmetry ===")
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP_SH, element_type="TRI3", size_max=5.0,
        symmetry=[mesher.SymmetryPlane("x", 25.0, "-")]), log=QUIET)
    assert res.coords[:, 0].max() <= 25.0 + 1e-6
    # remaining area: x=0 face 600 + halves of the four side faces = 3100
    assert abs(res.stats["measure"] - 3100.0) / 3100.0 < 0.02
    assert len(res.sym_nodes["x"]) > 5
    assert res.stats["free_edges"] > 0, "cut shell must have free edges"
    print(f"OK: area {res.stats['measure']:.0f}, "
          f"{len(res.sym_nodes['x'])} nodes on the cut plane")


def test_quad4_shells():
    print("=== QUAD4 shells (quad-dominant, solid boundary) ===")
    res = mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP, element_type="QUAD4", size_max=6.0), log=QUIET)
    n_quads = int((res.elems[:, 2] != res.elems[:, 3]).sum())
    frac = n_quads / len(res.elems)
    assert frac > 0.5, f"only {frac:.0%} quads"
    # boundary area of box-with-hole: 17500 - 2*pi*100 + 2*pi*10*25 = 18442.5
    assert abs(res.stats["measure"] - 18442.5) / 18442.5 < 0.01

    k_quad = os.path.join(OUT_DIR, "shell_quad4.k")
    dyna_writer.write_k(k_quad, res.coords, res.elems, element_kind="shell",
                        elform=16, thickness=2.0)
    nodes, _, shells, keywords = parse_k(k_quad)
    assert len(shells) == res.stats["n_elems"]
    assert abs(shell_areas(nodes, shells).sum() - 18442.5) / 18442.5 < 0.01
    print(f"OK: {len(shells)} shells, {frac:.0%} quads")


def test_defeature():
    print("=== defeature (remove cylindrical hole) ===")
    cyl = [f["tag"] for f in faces_step() if f["type"] == "Cylinder"]
    assert len(cyl) == 1
    settings = mesher.MeshSettings(step_file=STEP, size_max=8.0,
                                   defeature_faces=cyl)
    res = mesher.mesh_step(settings, log=QUIET)
    assert abs(res.stats["measure"] - 125000.0) / 125000.0 < 0.005, \
        f"hole not removed: volume {res.stats['measure']}"
    # rescan shows post-defeature faces
    faces2 = mesher.list_faces(settings, log=QUIET)
    assert len(faces2) == 6
    print(f"OK: volume {res.stats['measure']:.0f} (solid block), "
          f"{len(faces2)} faces after defeature")


def test_bc_loads_qa_implicit():
    print("=== BC/load cards, QA element set, implicit template ===")
    tag = xmax_tag()
    res = res_xmax_face()
    fn, fs = res.face_nodes[tag], res.face_segs[tag]
    k_bc = os.path.join(OUT_DIR, "bc_loads.k")
    dyna_writer.write_k(
        k_bc, res.coords, res.elems, implicit_cards=True, mat=dict(STEEL),
        face_sets=(
            {"kind": "node", "title": "SPC_FACE", "nodes": fn, "spc_dofs": "123"},
            {"kind": "segment", "title": "PRES_FACE", "segments": fs,
             "pressure": 0.5},
            {"kind": "node", "title": "FORCE_FACE", "nodes": fn,
             "force": ("z", -500.0)},
        ),
        elem_sets=({"title": "QA failures", "eids": np.array([1, 2, 3])},),
    )
    _, _, _, keywords = parse_k(k_bc)
    for kw in ("*DEFINE_CURVE_TITLE", "*BOUNDARY_SPC_SET", "*LOAD_SEGMENT_SET",
               "*LOAD_NODE_SET", "*SET_SOLID_TITLE", "*CONTROL_TERMINATION",
               "*CONTROL_IMPLICIT_GENERAL", "*CONTROL_IMPLICIT_AUTO",
               "*CONTROL_IMPLICIT_SOLUTION", "*DATABASE_BINARY_D3PLOT"):
        assert kw in keywords, f"missing {kw}"
    with open(k_bc) as f:
        content = f.read()
    assert "$ total force -500" in content
    print("OK: all BC/load/implicit/QA keywords present")


def test_face_size():
    print("=== local mesh size on a face ===")
    tag = xmax_tag()
    res_fs = mesher.mesh_step(mesher.MeshSettings(
        step_file=STEP, size_max=8.0, face_sizes={tag: 2.0}), log=QUIET)

    def nodes_on_xmax(res):
        return int((np.abs(res.coords[:, 0] - 50.0) < 1e-6).sum())

    n0, n1 = nodes_on_xmax(res_coarse()), nodes_on_xmax(res_fs)
    assert n1 > 2 * n0, f"face size had no effect: {n0} -> {n1}"
    print(f"OK: nodes on the sized face {n0} -> {n1}")


def test_auto_refine_wrapper():
    print("=== auto-refine wrapper (returns the best mesh) ===")
    settings = mesher.MeshSettings(step_file=STEP2, size_max=6.0,
                                   auto_refine=True, auto_refine_rounds=1,
                                   auto_refine_threshold=0.99)
    res = mesher.mesh_step_auto(settings, log=QUIET)
    assert res.stats["n_elems"] > 10
    print(f"OK: returned {res.stats['n_elems']} elements, "
          f"min quality {res.stats['quality_min']:.3f}")


def test_error_empty_cut():
    print("=== error handling (cut outside part) ===")
    settings = mesher.MeshSettings(
        step_file=STEP, size_max=8.0,
        symmetry=[mesher.SymmetryPlane("x", 1000.0, "+")])
    try:
        mesher.mesh_step(settings, log=QUIET)
        raise AssertionError("expected RuntimeError for empty cut")
    except RuntimeError as e:
        print(f"OK: got expected error: {e}")


def test_symmetry_bc_options():
    print("=== symmetry BC options (node set / SPC / anti-sym / custom) ===")
    res = res_half_x()
    nodes_x = res.sym_nodes["x"]

    # (a) node set only, no boundary condition
    k = os.path.join(OUT_DIR, "sym_nodeset_only.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": False},))
    _, _, _, kws = parse_k(k)
    assert "*SET_NODE_LIST_TITLE" in kws, "node set must be written"
    assert "*BOUNDARY_SPC_SET" not in kws, "no SPC when spc=False"

    # (b) default symmetric -> constrain X translation + in-plane rotations
    k = os.path.join(OUT_DIR, "sym_symmetric.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True},))
    assert spc_dofs(open(k).read()) == [1, 0, 0, 0, 1, 1]

    # (c) anti-symmetric -> the complement
    k = os.path.join(OUT_DIR, "sym_antisym.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "constraint": "antisymmetric"},))
    txt = open(k).read()
    assert spc_dofs(txt) == [0, 1, 1, 1, 0, 0]
    assert "(antisymmetric)" in txt, "title should note the constraint kind"

    # (d) fixed -> all six DOFs
    k = os.path.join(OUT_DIR, "sym_fixed.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "constraint": "fixed"},))
    assert spc_dofs(open(k).read()) == [1, 1, 1, 1, 1, 1]

    # (e) custom DOFs "13" -> dofx, dofz
    k = os.path.join(OUT_DIR, "sym_custom.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "dofs": "13"},))
    assert spc_dofs(open(k).read()) == [1, 0, 1, 0, 0, 0]

    # (f) start_sid offsets every set id (merge-friendly; parallels start_nid)
    k = os.path.join(OUT_DIR, "sym_startsid.k")
    dyna_writer.write_k(k, res.coords, res.elems, start_sid=100, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True},))
    lines = open(k).read().splitlines()
    i = next(j for j, ln in enumerate(lines)
             if ln.startswith("*SET_NODE_LIST_TITLE"))
    assert int(lines[i + 3][0:10]) == 100, "node set SID must start at start_sid"
    s = next(j for j, ln in enumerate(lines)
             if ln.startswith("*BOUNDARY_SPC_SET"))
    assert int(lines[s + 2][0:10]) == 100, "SPC must reference the offset set id"

    # (g) symmetry-plane segment set: boundary faces on the plane, all corners
    # on the plane, written as *SET_SEGMENT
    seg = res.sym_segs["x"]
    on_plane = set(nodes_x.tolist())
    assert len(seg) > 5 and seg.shape[1] == 4
    assert all(int(c) in on_plane for c in seg[:, :3].ravel())
    assert (seg[:, 2] == seg[:, 3]).all(), "tets -> triangle segments padded to quad"
    k = os.path.join(OUT_DIR, "sym_segset.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "segset": seg},))
    _, _, _, kws = parse_k(k)
    assert "*SET_SEGMENT_TITLE" in kws, "segment set must be written"
    print("OK: node-set-only, symmetric, anti-symmetric, fixed, custom, "
          "start_sid, plane segset verified")


def test_multi_format_input():
    print("=== multi-format input (BREP / IGES / STL) ===")
    _ensure_geometry()
    if not all(os.path.isfile(FORMAT_BASE + e)
               for e in (".iges", ".brep", ".stl")):
        make_formats(FORMAT_BASE)

    # BREP solid reproduces the STEP volume exactly (same OCC kernel)
    rb = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".brep", size_max=8.0, size_min=2.0), log=QUIET)
    assert abs(rb.stats["measure"] - 117146.0) / 117146.0 < 0.02

    # IGES imports as surfaces and is sewn into a solid
    ri = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".iges", size_max=8.0, size_min=2.0), log=QUIET)
    assert abs(ri.stats["measure"] - 117146.0) / 117146.0 < 0.02

    # IGES rejects the OCC target-unit override -> import must fall back to
    # file units instead of crashing
    riu = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".iges", size_max=8.0, size_min=2.0,
        occ_unit="MM"), log=QUIET)
    assert abs(riu.stats["measure"] - 117146.0) / 117146.0 < 0.02

    # IGES shell: sewing makes it watertight
    ris = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".iges", element_type="TRI3",
        size_max=8.0), log=QUIET)
    assert ris.stats["free_edges"] == 0, \
        f"IGES shell not watertight: {ris.stats['free_edges']} free edges"

    # STL shell: tessellation used as-is, coincident nodes welded -> watertight
    rs = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".stl", element_type="TRI3"), log=QUIET)
    assert rs.elems.shape[1] == 4 and (rs.elems[:, 2] == rs.elems[:, 3]).all()
    assert rs.stats["free_edges"] == 0, \
        f"STL shell not watertight: {rs.stats['free_edges']} free edges"

    # STL -> solid: reconstruct the volume from the watertight surface (TET4)
    rss = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".stl", element_type="TET4",
        size_max=8.0), log=QUIET)
    assert rss.elems.shape[1] == 4
    assert abs(rss.stats["measure"] - 117146.0) / 117146.0 < 0.02, \
        f"STL solid volume off: {rss.stats['measure']}"
    k_stl_solid = os.path.join(OUT_DIR, "stl_solid.k")
    dyna_writer.write_k(k_stl_solid, rss.coords, rss.elems)
    nodes, elems, _, _ = parse_k(k_stl_solid)
    assert (tet_volumes(nodes, elems) > 0).all(), "STL tets must be positive-volume"
    # STL -> TET10 solid
    rst = mesher.mesh_step(mesher.MeshSettings(
        step_file=FORMAT_BASE + ".stl", element_type="TET10",
        size_max=8.0), log=QUIET)
    assert rst.elems.shape[1] == 10

    # STL guards: symmetry, defeature and face scanning still rejected (no B-rep)
    for bad in (dict(element_type="TRI3",
                     symmetry=[mesher.SymmetryPlane("x", 0.0, "+")]),
                dict(element_type="TRI3", defeature_faces=[1])):
        try:
            mesher.mesh_step(mesher.MeshSettings(
                step_file=FORMAT_BASE + ".stl", **bad), log=QUIET)
            raise AssertionError(f"STL should reject {bad}")
        except RuntimeError:
            pass
    try:
        mesher.list_faces(mesher.MeshSettings(step_file=FORMAT_BASE + ".stl"),
                          log=QUIET)
        raise AssertionError("list_faces should reject STL")
    except RuntimeError:
        pass

    # an open (non-watertight) tessellation cannot be tetrahedralized
    open_stl = os.path.join(OUT_DIR, "open.stl")
    with open(open_stl, "w") as fh:
        fh.write("solid t\nfacet normal 0 0 0\n outer loop\n"
                 "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
                 " endloop\nendfacet\nendsolid t\n")
    try:
        mesher.mesh_step(mesher.MeshSettings(step_file=open_stl,
                                             element_type="TET4"), log=QUIET)
        raise AssertionError("open STL should be rejected for solid meshing")
    except RuntimeError:
        pass
    print("OK: BREP/IGES/STL import, sewing, STL solids, guards fire")


def test_auto_refine_threshold_preview_resilience():
    print("=== auto-refine threshold / best-mesh preview / failure resilience ===")
    logmsgs = []
    pv = os.path.join(OUT_DIR, "autoref_preview.msh")
    settings = mesher.MeshSettings(step_file=STEP, size_max=8.0, size_min=2.0,
                                   auto_refine=True, auto_refine_rounds=1,
                                   auto_refine_threshold=0.99)
    res = mesher.mesh_step_auto(settings, log=lambda m: logmsgs.append(str(m)),
                                preview_path=pv)
    retries = [m for m in logmsgs if m.startswith("Auto-refine: min quality")]
    assert retries, "threshold 0.99 must trigger an auto-refine retry"
    assert os.path.isfile(pv), "preview of the kept mesh must exist"
    assert not [p for p in os.listdir(OUT_DIR) if "autoref_preview.round" in p], \
        "per-round preview files must be cleaned up"
    # the preview must show the KEPT mesh, not simply the last round's
    import gmsh as _gmsh
    _gmsh.initialize(interruptible=False)
    _gmsh.open(pv)
    n_prev = len(_gmsh.model.mesh.getElementsByType(4)[0])
    _gmsh.finalize()
    assert n_prev == res.stats["n_elems"], \
        f"preview mesh ({n_prev}) != kept mesh ({res.stats['n_elems']})"

    # a failing refinement round must not throw away the good first mesh
    orig_mesh_step, calls = mesher.mesh_step, {"n": 0}

    def flaky(s, log=print, preview_path=None, export_paths=()):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("simulated PLC error")
        return orig_mesh_step(s, log=log, preview_path=preview_path,
                              export_paths=export_paths)

    mesher.mesh_step = flaky
    try:
        res2 = mesher.mesh_step_auto(settings, log=QUIET)
    finally:
        mesher.mesh_step = orig_mesh_step
    assert calls["n"] == 2 and res2.stats["n_elems"] > 100, \
        "round-1 failure must return the round-0 mesh"
    print(f"OK: retry fired, preview matches kept mesh ({n_prev} tets), "
          f"failed round kept {res2.stats['n_elems']} elements")


def test_mesh_only_output():
    print("=== mesh-only output (for *INCLUDE) ===")
    res = res_half_x()
    k_inc = os.path.join(OUT_DIR, "mesh_only.k")
    dyna_writer.write_k(
        k_inc, res.coords, res.elems, mesh_only=True, implicit_cards=True,
        mat=dict(STEEL), contact_fs=0.1, tssfac=0.9,
        sym_sets=({"axis": "x", "offset": 0.0, "nodes": res.sym_nodes["x"],
                   "spc": True},))
    nodes, elems, _, kws = parse_k(k_inc)
    for kw in ("*PART", "*SECTION_SOLID", "*MAT_ELASTIC",
               "*CONTROL_TERMINATION", "*CONTROL_IMPLICIT_GENERAL",
               "*CONTROL_TIMESTEP", "*CONTACT_AUTOMATIC_SINGLE_SURFACE"):
        assert kw not in kws, f"mesh-only file must not contain {kw}"
    for kw in ("*NODE", "*ELEMENT_SOLID", "*SET_NODE_LIST_TITLE",
               "*BOUNDARY_SPC_SET", "*END"):
        assert kw in kws, f"mesh-only file must contain {kw}"
    assert len(nodes) == res.stats["n_nodes"] and len(elems) == res.stats["n_elems"]
    print(f"OK: {len(nodes)} nodes / {len(elems)} elements, cards suppressed")


def test_critical_timestep():
    print("=== explicit critical timestep estimate ===")
    res = res_half_x()
    lc = res.stats["char_length"]
    assert lc > 0, "solid stats must expose the characteristic length"
    dt = mesher.critical_timestep(res.stats, "TET4", STEEL)
    c_solid = (210000.0 * 0.7 / (1.3 * 0.4 * 7.85e-9)) ** 0.5
    assert abs(dt - lc / c_solid) < 1e-12 * dt, "solid dt must be Lc/c (bulk c)"
    # shells: plane-stress wave speed on the LS-DYNA characteristic length
    res_sh = res_shell_tri3()
    dt_sh = mesher.critical_timestep(res_sh.stats, "TRI3", STEEL)
    c_shell = (210000.0 / (7.85e-9 * (1 - 0.09))) ** 0.5
    assert abs(dt_sh - res_sh.stats["char_length"] / c_shell) < 1e-12 * dt_sh
    # shell char length = (1+beta) * area / longest edge; for the triangles
    # here it must be below the min edge length times 2/sqrt(3) (equilateral)
    assert res_sh.stats["char_length"] > 0
    # invalid input -> None, not a crash
    assert mesher.critical_timestep(res.stats, "TET4", None) is None
    assert mesher.critical_timestep({}, "TET4", STEEL) is None
    assert mesher.critical_timestep(
        res.stats, "TET4", {"e": 210000.0, "pr": 0.5, "ro": 7.85e-9}) is None
    print(f"OK: solid dt {dt:.4g} s, shell dt {dt_sh:.4g} s, guards fire")


def test_mass_and_timestep_reporting():
    print("=== per-part mass + dt reporting helper ===")
    res = res_two_glued()
    logmsgs = []
    mass, masses, dt = mesher.mass_and_timestep(
        res, "TET4", STEEL, {11: ALU}, 10, log=lambda m: logmsgs.append(str(m)))
    # 8000 mm^3 per box: steel 6.28e-5 t, alu 2.16e-5 t
    assert abs(masses[10] - 8000 * 7.85e-9) / (8000 * 7.85e-9) < 0.01
    assert abs(masses[11] - 8000 * 2.7e-9) / (8000 * 2.7e-9) < 0.01
    assert abs(mass - masses[10] - masses[11]) < 1e-12
    assert dt and dt > 0
    # the estimate must use the stiffest (fastest) material -> min dt
    dt_steel = mesher.critical_timestep(res.stats, "TET4", STEEL)
    dt_alu = mesher.critical_timestep(res.stats, "TET4", ALU)
    assert abs(dt - min(dt_steel, dt_alu)) < 1e-15
    assert any("PID 10 mass" in m for m in logmsgs)
    assert any("dt distribution" in m for m in logmsgs)
    # a part without any material -> unknown total
    mass2, masses2, _ = mesher.mass_and_timestep(
        res, "TET4", None, {11: ALU}, 10, log=QUIET)
    assert mass2 is None and list(masses2) == [11]
    print(f"OK: total {mass:.4g} t, per-part masses, dt {dt:.4g} s")


def test_part_mats_and_contact_cards():
    print("=== per-part *MAT cards + single-surface contact ===")
    res = res_two_glued()
    k = os.path.join(OUT_DIR, "part_mats.k")
    dyna_writer.write_k(k, res.coords, res.elems, pid=10,
                        part_ids=10 + res.elem_parts, mat=dict(STEEL),
                        part_mats={11: dict(ALU)}, contact_fs=0.15)
    _, elems, _, kws = parse_k(k)
    assert kws.count("*MAT_ELASTIC") == 2
    assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" in kws
    assert part_cards(k) == [(10, 10, 10), (11, 10, 11)], \
        "part 11 must reference its own MID, part 10 the global one"
    assert mat_mids(k) == [10, 11]
    txt = open(k).read()
    assert "0.1500" in txt, "friction coefficient must be on the contact card"
    assert "7e+04" in txt, "the alu E-modulus must be in the file"
    # no NOTE about missing materials - everything is defined
    assert "$ NOTE: define" not in txt
    # missing materials are pointed out per MID
    k2 = os.path.join(OUT_DIR, "part_mats_missing.k")
    dyna_writer.write_k(k2, res.coords, res.elems, pid=10,
                        part_ids=10 + res.elem_parts, part_mats={11: dict(ALU)})
    assert "MID = 10" in open(k2).read()
    print("OK: per-part MIDs/MAT cards, contact card, missing-MAT note")


def test_split_parts():
    print("=== per-part .k files (write_k_split) ===")
    res = res_two_glued()
    k = os.path.join(OUT_DIR, "split.k")
    all_nodes = np.arange(1, res.stats["n_nodes"] + 1)
    files = dyna_writer.write_k_split(
        k, res.coords, res.elems, part_ids=10 + res.elem_parts,
        part_titles={10: "left box", 11: "right box"},
        mat=dict(STEEL), part_mats={11: dict(ALU)}, contact_fs=0.5,
        face_sets=({"kind": "node", "title": "ALL", "nodes": all_nodes},),
        elem_sets=({"title": "QA", "eids": np.arange(1, len(res.elems) + 1)},))
    assert [p for p, _ in files] == [10, 11]
    paths = dict(files)
    assert paths[10].endswith("split_p10_left_box.k")
    assert paths[11].endswith("split_p11_right_box.k")

    n_nodes_sum = n_elems_sum = vol_sum = 0.0
    for p, fpath in files:
        nodes, elems, _, kws = parse_k(fpath)
        assert part_cards(fpath) == [(p, p, p)], "standalone PART/SECID/MID"
        assert kws.count("*PART") == 1 and kws.count("*MAT_ELASTIC") == 1
        assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" not in kws, \
            "contact acts between parts and must be skipped"
        assert mat_mids(fpath) == [p]
        v = tet_volumes(nodes, elems)
        assert (v > 0).all()
        # node ids must be compact 1..N
        assert min(nodes) == 1 and max(nodes) == len(nodes)
        # the filtered ALL-nodes set must match this file's node count
        lines = open(fpath).read().splitlines()
        i = next(j for j, ln in enumerate(lines)
                 if ln.startswith("*SET_NODE_LIST_TITLE"))
        set_ids = []
        for ln in lines[i + 4:]:
            if ln.startswith("*") or ln.startswith("$"):
                break
            set_ids += [int(ln[c:c + 10]) for c in range(0, len(ln.rstrip()), 10)]
        assert len(set_ids) == len(nodes)
        n_nodes_sum += len(nodes)
        n_elems_sum += len(elems)
        vol_sum += v.sum()
    # every element exactly once; interface nodes duplicated across the files
    assert n_elems_sum == res.stats["n_elems"]
    assert n_nodes_sum > res.stats["n_nodes"]
    assert abs(vol_sum - res.stats["measure"]) / res.stats["measure"] < 1e-9
    # material routing: alu E in the p11 file only
    assert "7e+04" in open(paths[11]).read()
    assert "7e+04" not in open(paths[10]).read()
    print(f"OK: {len(files)} standalone files, volumes/elements add up")


def test_select_nodes():
    print("=== coordinate node selection (plane / box / sphere) ===")
    res = res_full()
    ids = mesher.select_nodes(res.coords, "plane", ("x", 50.0))
    assert len(ids) > 5
    assert np.allclose(res.coords[ids - 1][:, 0], 50.0, atol=1e-6)
    expected = np.flatnonzero(np.abs(res.coords[:, 0] - 50.0) <= 1e-4) + 1
    assert set(ids.tolist()) == set(expected.tolist())
    ids_box = mesher.select_nodes(res.coords, "box",
                                  [0.0, -25.0, -12.5, 50.0, 25.0, 12.5])
    assert (res.coords[ids_box - 1][:, 0] >= -1e-6).all()
    assert 0 < len(ids_box) < res.stats["n_nodes"]
    ids_sph = mesher.select_nodes(res.coords, "sphere", [0.0, 0.0, 0.0, 15.0])
    d = np.linalg.norm(res.coords[ids_sph - 1], axis=1)
    assert (d <= 15.0 + 1e-3).all() and len(ids_sph) > 0
    try:
        mesher.select_nodes(res.coords, "cylinder", [0, 0, 0, 1])
        raise AssertionError("unknown kind must raise")
    except ValueError:
        pass
    print(f"OK: plane {len(ids)}, box {len(ids_box)}, sphere {len(ids_sph)} nodes")


def test_mat_rigid():
    print("=== rigid parts (*MAT_RIGID) ===")
    res = res_two_glued()
    rigid = {**STEEL, "rigid": True}
    k = os.path.join(OUT_DIR, "rigid.k")
    dyna_writer.write_k(k, res.coords, res.elems, pid=10,
                        part_ids=10 + res.elem_parts, mat=dict(STEEL),
                        part_mats={11: rigid})
    _, _, _, kws = parse_k(k)
    assert "*MAT_RIGID" in kws and "*MAT_ELASTIC" in kws
    assert part_cards(k) == [(10, 10, 10), (11, 10, 11)]
    # a rigid part must not control the timestep estimate: make the rigid
    # material much stiffer - the dt must still be the elastic steel's
    stiff_rigid = {"e": 2.1e9, "pr": 0.3, "ro": 7.85e-9, "rigid": True}
    _, _, dt = mesher.mass_and_timestep(res, "TET4", STEEL,
                                        {11: stiff_rigid}, 10, log=QUIET)
    dt_steel = mesher.critical_timestep(res.stats, "TET4", STEEL)
    assert abs(dt - dt_steel) < 1e-15, "rigid mat must be excluded from dt"
    print("OK: MAT_RIGID card written, rigid part excluded from dt")


def test_write_k_include():
    print("=== *INCLUDE assembly (write_k_include) ===")
    res = res_two_glued()
    k = os.path.join(OUT_DIR, "incl.k")
    files = dyna_writer.write_k_include(
        k, res.coords, res.elems, pid=10,
        part_ids=10 + res.elem_parts,
        part_titles={10: "left box", 11: "right box"},
        mat=dict(STEEL), part_mats={11: dict(ALU)}, contact_fs=0.1)
    assert [p for p, _ in files] == [10, 11]

    # master: all modeling cards + *INCLUDE lines, but no mesh blocks
    _, _, _, kws = parse_k(k)
    assert kws.count("*INCLUDE") == 2
    assert kws.count("*PART") == 2 and kws.count("*MAT_ELASTIC") == 2
    assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" in kws
    assert "*NODE" not in kws and "*ELEMENT_SOLID" not in kws
    master = open(k).read()
    for _, fpath in files:
        assert os.path.basename(fpath) in master, "*INCLUDE must name the fragment"

    # fragments: global numbering, disjoint node ownership, full coverage
    all_nodes, all_elems = {}, []
    per_frag_nodes = []
    for p, fpath in files:
        nodes, elems, _, fkws = parse_k(fpath)
        assert "*PART" not in fkws, "fragments are mesh-only"
        assert all(e[1] == p for e in elems), "fragment carries its own PID"
        per_frag_nodes.append(set(nodes))
        all_nodes.update(nodes)
        all_elems += elems
    assert not (per_frag_nodes[0] & per_frag_nodes[1]), \
        "interface nodes must be defined in exactly one fragment"
    assert len(all_nodes) == res.stats["n_nodes"], "all nodes covered once"
    assert len(all_elems) == res.stats["n_elems"]
    assert sorted(e[0] for e in all_elems) == list(range(1, len(all_elems) + 1)), \
        "global element ids"
    # the combined fragments rebuild the full mesh
    v = tet_volumes(all_nodes, all_elems)
    assert (v > 0).all()
    assert abs(v.sum() - res.stats["measure"]) / res.stats["measure"] < 1e-9
    print(f"OK: master + {len(files)} fragments rebuild the assembly exactly")


def test_export_paths():
    print("=== mesh export (.vtk) + auto-refine round handling ===")
    vtk = os.path.join(OUT_DIR, "export.vtk")
    settings = mesher.MeshSettings(step_file=STEP, size_max=8.0, size_min=2.0,
                                   auto_refine=True, auto_refine_rounds=1,
                                   auto_refine_threshold=0.99)
    mesher.mesh_step_auto(settings, log=QUIET, export_paths=(vtk,))
    assert os.path.isfile(vtk) and os.path.getsize(vtk) > 1000
    assert not [p for p in os.listdir(OUT_DIR) if "export.round" in p], \
        "per-round export files must be cleaned up"
    print("OK: VTK exported, round files cleaned")


def test_job_runner():
    print("=== job_runner (GUI-independent job execution) ===")
    from k_mesher import job_runner
    _ensure_geometry()
    kopts = {
        "pid": 10, "elform": 10, "element_kind": "solid", "thickness": 1.0,
        "start_nid": 1, "start_eid": 1, "start_sid": 1,
        "sym_nodeset": True, "sym_spc": True, "sym_constraint": "symmetric",
        "sym_dofs": "", "sym_segset": False,
        "mat": dict(STEEL), "part_mats": {11: dict(ALU)},
        "face_nodesets": False, "face_segsets": False, "face_roles": [],
        "plain_faces": [], "face_titles": {},
        "coord_sets": [{"kind": "plane", "params": ("x", 0.0), "role": "spc",
                        "dofs": "123"}],
        "implicit_cards": False, "qa_sets": True, "mesh_only": False,
        "split_mode": "include", "contact_fs": 0.1, "tssfac": 0.9,
        "gravity": ("z", 9810.0),
    }
    settings = mesher.MeshSettings(step_file=STEP2, size_max=6.0, glue=True)
    out = os.path.join(OUT_DIR, "runner.k")
    logmsgs = []
    preview = job_runner.run_job(settings, out, kopts,
                                 log=lambda m: logmsgs.append(str(m)))
    assert preview.endswith("_preview.msh") and os.path.isfile(preview)
    _, _, _, kws = parse_k(out)
    assert kws.count("*INCLUDE") == 2, "split_mode=include -> master deck"
    assert "*CONTROL_TIMESTEP" in kws and "*LOAD_BODY_Z" in kws
    assert "*SET_NODE_LIST_TITLE" in kws and "*BOUNDARY_SPC_SET" in kws
    assert any("Coordinate set" in m for m in logmsgs)
    # captured variant for the parallel batch queue
    label, ok_flag, text, prev2 = job_runner.run_job_captured(
        ("job-1", settings, out, {**kopts, "split_mode": None}))
    assert label == "job-1" and ok_flag and "Done in" in text and prev2
    bad = mesher.MeshSettings(step_file="missing.step")
    _, ok_flag, text, _ = job_runner.run_job_captured(("job-2", bad, out, kopts))
    assert not ok_flag and "ERROR" in text
    print("OK: run_job + captured variant, include mode, coord set applied")


def test_parallel_batch_workers():
    print("=== parallel batch (worker processes) ===")
    import multiprocessing

    from k_mesher import job_runner
    from concurrent.futures import ProcessPoolExecutor, as_completed
    _ensure_geometry()
    kopts = {
        "pid": 1, "elform": 10, "element_kind": "solid", "thickness": 1.0,
        "start_nid": 1, "start_eid": 1, "start_sid": 1,
        "sym_nodeset": False, "sym_spc": False, "sym_constraint": "symmetric",
        "sym_dofs": "", "sym_segset": False,
        "mat": None, "part_mats": {},
        "face_nodesets": False, "face_segsets": False, "face_roles": [],
        "plain_faces": [], "face_titles": {}, "coord_sets": [],
        "implicit_cards": False, "qa_sets": False, "mesh_only": False,
        "split_mode": None, "contact_fs": None, "tssfac": None,
        "gravity": None,
    }
    jobs = [("par-1", mesher.MeshSettings(step_file=STEP, size_max=10.0),
             os.path.join(OUT_DIR, "par1.k"), kopts),
            ("par-2", mesher.MeshSettings(step_file=STEP2, size_max=8.0),
             os.path.join(OUT_DIR, "par2.k"), kopts)]
    results = {}
    # spawn, not fork: forked children of a gmsh/OpenMP parent deadlock
    with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(job_runner.run_job_captured, j) for j in jobs]
        for fut in as_completed(futures):
            label, ok_flag, text, preview = fut.result()
            results[label] = (ok_flag, text)
    assert set(results) == {"par-1", "par-2"}
    for label, (ok_flag, text) in results.items():
        assert ok_flag, f"{label} failed:\n{text}"
        assert "Done in" in text
    assert os.path.isfile(os.path.join(OUT_DIR, "par1.k"))
    assert os.path.isfile(os.path.join(OUT_DIR, "par2.k"))
    print("OK: two jobs meshed in parallel worker processes")


def test_cli_nset_stl_and_export():
    print("=== CLI: --nset on STL input, --export, --target-dt ===")
    _ensure_geometry()
    if not os.path.isfile(FORMAT_BASE + ".stl"):
        make_formats(FORMAT_BASE)
    k_cli = os.path.join(OUT_DIR, "cli_stl_nset.k")
    vtk = os.path.join(OUT_DIR, "cli_export.vtk")
    rc = mesh_cli.main([FORMAT_BASE + ".stl", "--etype", "tet4",
                        "-o", k_cli, "--size-max", "8", "--mat",
                        "--nset", "plane,z,12.5,spc=123,title=TOP_FIX",
                        "--nset", "sphere,0,0,0,15,force=z:-500",
                        "--nset", "box,999,999,999,1000,1000,1000",
                        "--export", vtk, "--target-dt", "1"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*SET_NODE_LIST_TITLE" in kws
    assert "*BOUNDARY_SPC_SET" in kws, "nset spc= role must write the SPC"
    assert "*LOAD_NODE_SET" in kws, "nset force= role must write the load"
    txt = open(k_cli).read()
    assert "TOP_FIX" in txt, "nset title= must name the set"
    assert os.path.isfile(vtk), "--export must write the VTK file"
    print("OK: BCs/loads on STL via coordinate sets, VTK exported")


def test_cli_split_include():
    print("=== CLI: --split-include (+ mutex with --split-parts) ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_include.k")
    js_cli = os.path.join(OUT_DIR, "cli_include.json")
    rc = mesh_cli.main([STEP2, "-o", k_cli, "--size-max", "6", "--glue",
                        "--mat", "--part-rigid", "2", "--split-include",
                        "--stats-json", js_cli])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert kws.count("*INCLUDE") == 2 and "*NODE" not in kws
    assert "*MAT_RIGID" in kws, "--part-rigid must write the rigid card"
    with open(js_cli) as f:
        payload = json.load(f)
    assert len(payload["split_files"]) == 2
    for fpath in payload["split_files"].values():
        assert os.path.isfile(fpath)
    # the two split modes are mutually exclusive
    rc2 = mesh_cli.main([STEP2, "-o", k_cli, "--split-parts", "--split-include"])
    assert rc2 == 2
    print("OK: master + fragments via CLI, rigid card, mutex enforced")


def test_control_timestep_and_gravity_cards():
    print("=== *CONTROL_TIMESTEP + *LOAD_BODY (gravity) ===")
    res = res_full()
    k = os.path.join(OUT_DIR, "ctrl_grav.k")
    dyna_writer.write_k(k, res.coords, res.elems, tssfac=0.9,
                        body_load=("z", 9810.0))
    _, _, _, kws = parse_k(k)
    assert "*CONTROL_TIMESTEP" in kws
    assert "*LOAD_BODY_Z" in kws
    assert "*DEFINE_CURVE_TITLE" in kws, "gravity alone must emit the ramp curve"
    lines = open(k).read().splitlines()
    i = next(j for j, ln in enumerate(lines)
             if ln.startswith("*CONTROL_TIMESTEP"))
    assert abs(float(lines[i + 2][10:20]) - 0.9) < 1e-12, "TSSFAC on the card"
    i = next(j for j, ln in enumerate(lines) if ln.startswith("*LOAD_BODY_Z"))
    assert int(lines[i + 2][0:10]) == 1, "body load must reference the curve"
    assert abs(float(lines[i + 2][10:20]) - 9810.0) < 1e-9
    print("OK: control-timestep and body-load cards written")


def test_long_format():
    print("=== LONG=Y format (id overflow) ===")
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    elems = np.array([[1, 2, 3, 4]], dtype=np.int64)

    # ids beyond 8 characters -> automatic LONG=Y
    k = os.path.join(OUT_DIR, "long_auto.k")
    dyna_writer.write_k(k, coords, elems, start_nid=99_999_998)
    lines = open(k).read().splitlines()
    assert lines[0] == "*KEYWORD LONG=Y"
    assert any("LONG=Y format enabled automatically" in ln for ln in lines)
    i = lines.index("*NODE")
    node_line = lines[i + 2]
    assert int(node_line[0:20]) == 99_999_998, "node id in a 20-char field"
    assert len(node_line) == 20 + 3 * 20, "20-char coordinate fields"
    j = lines.index("*ELEMENT_SOLID")
    el = lines[j + 2]
    assert int(el[0:20]) == 1 and int(el[40:60]) == 99_999_998

    # explicit opt-in without large ids
    k2 = os.path.join(OUT_DIR, "long_forced.k")
    dyna_writer.write_k(k2, coords, elems, long_format=True)
    with open(k2) as fh:
        assert fh.readline().strip() == "*KEYWORD LONG=Y"

    # and the default stays the standard format
    k3 = os.path.join(OUT_DIR, "std_fmt.k")
    dyna_writer.write_k(k3, coords, elems)
    with open(k3) as fh:
        assert fh.readline().strip() == "*KEYWORD"
    print("OK: automatic + forced LONG=Y, standard by default")


def test_cli_output_options():
    print("=== CLI: --mesh-only, --title, --stats-json ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_meshonly.k")
    js_cli = os.path.join(OUT_DIR, "cli_stats.json")
    rc = mesh_cli.main([STEP, "-o", k_cli, "--size-max", "8", "--size-min", "2",
                        "--mesh-only", "--title", "CLI TITLE TEST",
                        "--stats-json", js_cli, "--mat"])
    assert rc == 0
    _, elems, _, kws = parse_k(k_cli)
    assert "*PART" not in kws and "*SECTION_SOLID" not in kws \
        and "*MAT_ELASTIC" not in kws, "mesh-only CLI file must skip cards"
    with open(k_cli) as f:
        head = f.read(400)
    assert "CLI TITLE TEST" in head, "--title must set the deck title"
    with open(js_cli) as f:
        payload = json.load(f)
    assert payload["element_type"] == "TET4"
    assert payload["stats"]["n_elems"] == len(elems)
    assert payload["mass"] > 0 and payload["critical_timestep"] > 0
    assert payload["stats"]["quality_min"] > 0
    assert isinstance(payload["stats"]["failed_elems"], list)
    print(f"OK: title + mesh-only written, stats JSON has "
          f"{payload['stats']['n_elems']} elements")


def test_cli_assembly_options():
    print("=== CLI: --part-mat, --contact, --tssfac, --gravity ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_assembly.k")
    js_cli = os.path.join(OUT_DIR, "cli_assembly.json")
    rc = mesh_cli.main([STEP2, "-o", k_cli, "--size-max", "6", "--glue",
                        "--mat", "--part-mat", "2:70000:0.33:2.7e-9",
                        "--contact", "0.15", "--tssfac", "0.9",
                        "--gravity", "z:9810", "--split-parts",
                        "--stats-json", js_cli])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" in kws
    assert "*CONTROL_TIMESTEP" in kws
    assert "*LOAD_BODY_Z" in kws
    assert kws.count("*MAT_ELASTIC") == 2, "steel default + alu override"
    assert part_cards(k_cli) == [(1, 1, 1), (2, 1, 2)]
    with open(js_cli) as f:
        payload = json.load(f)
    assert len(payload["part_masses"]) == 2
    assert payload["mass"] > 0 and payload["critical_timestep"] > 0
    # --split-parts: one standalone file per body, listed in the JSON
    assert len(payload["split_files"]) == 2
    for p, fpath in payload["split_files"].items():
        assert os.path.isfile(fpath)
        _, _, _, pkws = parse_k(fpath)
        assert pkws.count("*PART") == 1
        assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" not in pkws
        assert "*CONTROL_TIMESTEP" in pkws, "control cards carry over"
        assert part_cards(fpath) == [(int(p),) * 3]
    print("OK: assembly cards, per-part masses, split files written")


def test_cli_version():
    print("=== CLI: --version flag / _version import ===")
    from k_mesher import _version
    assert _version.__version__, "package must expose a version string"
    try:
        mesh_cli.build_parser().parse_args(["x", "--version"])
        raise AssertionError("--version must exit")
    except SystemExit as e:
        assert e.code == 0, "--version must exit cleanly"
    print(f"OK: --version exits, version {_version.__version__}")


def test_cli_dyna_control_cards():
    print("=== CLI: LS-DYNA control / load flags ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_control.k")
    rc = mesh_cli.main([
        STEP, "-o", k_cli, "--size-max", "10", "--mat",
        "--endtim", "0.01", "--mass-scale=-1e-6",
        "--hourglass", "5:0.05", "--control-energy",
        "--d3plot-dt", "1e-4", "--database", "GLSTAT:1e-5",
        "--init-velocity", "0:0:-5000",
        "--contact", "0.1", "--contact-type", "tied_surface_to_surface",
        "--rigidwall", "0:0:-50:0:0:1:0.2", "--spotweld", "1:2",
        "--define-curve", "99:0,0;1,1",
        "--prescribed-motion", "1:3:0:99:1.0"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    for kw in ("*CONTROL_TERMINATION", "*CONTROL_TIMESTEP", "*CONTROL_ENERGY",
               "*HOURGLASS", "*DATABASE_BINARY_D3PLOT", "*DATABASE_GLSTAT",
               "*INITIAL_VELOCITY_GENERATION", "*CONTACT_TIED_SURFACE_TO_SURFACE",
               "*RIGIDWALL_PLANAR", "*CONSTRAINED_SPOTWELD",
               "*BOUNDARY_PRESCRIBED_MOTION_SET"):
        assert kw in kws, f"missing {kw}"
    # --contact-type routes contact through the general list, not single-surface
    assert "*CONTACT_AUTOMATIC_SINGLE_SURFACE" not in kws
    txt = open(k_cli).read()
    assert "-1e-06" in txt, "DT2MS mass-scale value must be on *CONTROL_TIMESTEP"
    print("OK: all new control/load cards emitted via the CLI")


def test_job_runner_control_cards():
    print("=== job_runner: new LS-DYNA control/load kopts ===")
    from k_mesher import job_runner
    _ensure_geometry()
    kopts = {
        "pid": 1, "elform": 10, "element_kind": "solid", "thickness": 1.0,
        "start_nid": 1, "start_eid": 1, "start_sid": 1,
        "sym_nodeset": False, "sym_spc": False, "sym_constraint": "symmetric",
        "sym_dofs": "", "sym_segset": False,
        "mat": dict(STEEL), "part_mats": {},
        "face_nodesets": False, "face_segsets": False, "face_roles": [],
        "plain_faces": [], "face_titles": {}, "coord_sets": [],
        "implicit_cards": False, "qa_sets": False, "mesh_only": False,
        "split_mode": None, "contact_fs": None, "tssfac": 0.9,
        "gravity": None,
        # the GUI-shared contract subset
        "endtim": 0.02, "mass_scale": -1e-6,
        "hourglass": {"ihq": 5, "qm": 0.1}, "control_energy": True,
        "databases": {"d3plot_dt": 1e-4, "ascii": {"GLSTAT": 1e-5}},
        "initial_velocity": {"vx": 0.0, "vy": 0.0, "vz": -1000.0},
        "contacts": ({"type": "automatic_surface_to_surface", "fs": 0.2},),
    }
    settings = mesher.MeshSettings(step_file=STEP, size_max=10.0)
    out = os.path.join(OUT_DIR, "runner_control.k")
    job_runner.run_job(settings, out, kopts, log=QUIET)
    _, _, _, kws = parse_k(out)
    for kw in ("*CONTROL_TERMINATION", "*CONTROL_TIMESTEP", "*CONTROL_ENERGY",
               "*HOURGLASS", "*DATABASE_BINARY_D3PLOT", "*DATABASE_GLSTAT",
               "*INITIAL_VELOCITY_GENERATION",
               "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE"):
        assert kw in kws, f"missing {kw}"
    assert "-1e-06" in open(out).read(), "DT2MS mass-scale value must appear"
    print("OK: control/load cards land through run_job kopts")


# --------------------------------------------------------------------------
# midsurface + automatic connection detection
# --------------------------------------------------------------------------

PLATE = os.path.join(EX, "test_plate.step")


def _make_plate(path):
    """A thin, constant-thickness plate solid (80 x 60 x 3) for midsurface
    extraction: two dominant 80x60 faces 3 apart -> wall thickness 3."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.model.occ.addBox(0, 0, 0, 80, 60, 3)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _ensure_plate():
    _ensure_geometry()
    if not os.path.isfile(PLATE):
        _make_plate(PLATE)


def section_shell_thickness(path):
    """The T1 thickness on the first *SECTION_SHELL of a standard-format file."""
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "*SECTION_SHELL":
            for j in range(i + 1, len(lines)):
                if lines[j].lstrip().startswith("$#      t1"):
                    return float(lines[j + 1][0:10])
    return None


def test_cli_midsurface():
    print("=== CLI: --midsurface (thin plate -> shell) ===")
    _ensure_plate()
    k_cli = os.path.join(OUT_DIR, "cli_midsurface.k")
    rc = mesh_cli.main([PLATE, "-o", k_cli, "--size-max", "10", "--midsurface",
                        "--mat"])
    assert rc == 0
    _, _, shells, kws = parse_k(k_cli)
    assert "*ELEMENT_SHELL" in kws and "*SECTION_SHELL" in kws
    assert "*ELEMENT_SOLID" not in kws, "midsurface must not write solids"
    assert len(shells) > 5
    t = section_shell_thickness(k_cli)
    assert t is not None and abs(t - 3.0) < 1e-3, \
        f"*SECTION_SHELL thickness must match the 3.0 plate wall: {t}"
    # a tessellation input must be rejected for midsurface
    _ensure_geometry()
    if not os.path.isfile(FORMAT_BASE + ".stl"):
        make_formats(FORMAT_BASE)
    rc2 = mesh_cli.main([FORMAT_BASE + ".stl", "--midsurface",
                         "-o", os.path.join(OUT_DIR, "cli_mid_bad.k")])
    assert rc2 == 2, "midsurface on an STL tessellation must error"
    # an explicit --thickness overrides the detected value
    k_cli2 = os.path.join(OUT_DIR, "cli_midsurface_t.k")
    rc3 = mesh_cli.main([PLATE, "-o", k_cli2, "--size-max", "10",
                         "--midsurface", "--thickness", "1.25"])
    assert rc3 == 0
    assert abs(section_shell_thickness(k_cli2) - 1.25) < 1e-3, \
        "explicit --thickness must win over the detected thickness"
    print(f"OK: {len(shells)} shells, detected thickness {t:.4g}, guards fire")


def test_cli_auto_spotweld():
    print("=== CLI: --auto-spotweld (unglued two-body interface) ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_spotweld.k")
    # STEP2 is two boxes touching at x=20; WITHOUT --glue they keep two
    # coincident node layers -> a detectable interface
    rc = mesh_cli.main([STEP2, "-o", k_cli, "--size-max", "8",
                        "--auto-spotweld"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*CONSTRAINED_SPOTWELD" in kws, "auto-spotweld must weld the interface"
    n_welds = kws.count("*CONSTRAINED_SPOTWELD")
    assert n_welds > 3
    # SPACING thins the pattern to fewer welds
    k_cli2 = os.path.join(OUT_DIR, "cli_spotweld_sp.k")
    rc2 = mesh_cli.main([STEP2, "-o", k_cli2, "--size-max", "8",
                         "--auto-spotweld", "15"])
    assert rc2 == 0
    _, _, _, kws2 = parse_k(k_cli2)
    assert 0 < kws2.count("*CONSTRAINED_SPOTWELD") < n_welds, \
        "spacing must thin the weld pattern"
    print(f"OK: {n_welds} welds, thinned to "
          f"{kws2.count('*CONSTRAINED_SPOTWELD')} with spacing")


def test_cli_tied_contact():
    print("=== CLI: --tied-contact (unglued two-body interface) ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_tied.k")
    rc = mesh_cli.main([STEP2, "-o", k_cli, "--size-max", "8",
                        "--tied-contact"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" in kws, "tied contact card"
    assert "*SET_SEGMENT_TITLE" in kws, "interface segment sets"
    # a single-body model has no interface -> warn and continue (rc 0)
    rc2 = mesh_cli.main([STEP, "-o", os.path.join(OUT_DIR, "cli_tied_single.k"),
                         "--size-max", "10", "--tied-contact"])
    assert rc2 == 0
    _, _, _, kws1 = parse_k(os.path.join(OUT_DIR, "cli_tied_single.k"))
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" not in kws1, \
        "no interface on a single body -> no tied contact"
    print("OK: tied contact + segment sets written, single-body warns")


def test_job_runner_midsurface_and_connect():
    print("=== job_runner: midsurface + auto_connect kopts ===")
    from k_mesher import job_runner
    _ensure_plate()
    base = {
        "pid": 1, "elform": None, "element_kind": "solid", "thickness": 1.0,
        "start_nid": 1, "start_eid": 1, "start_sid": 1,
        "sym_nodeset": False, "sym_spc": False, "sym_constraint": "symmetric",
        "sym_dofs": "", "sym_segset": False,
        "mat": dict(STEEL), "part_mats": {},
        "face_nodesets": False, "face_segsets": False, "face_roles": [],
        "plain_faces": [], "face_titles": {}, "coord_sets": [],
        "implicit_cards": False, "qa_sets": False, "mesh_only": False,
        "split_mode": None, "contact_fs": None, "tssfac": None,
        "gravity": None,
    }
    # (a) midsurface: shell section + thickness from the detected wall gap
    kopts_mid = {**base, "elform": 4, "midsurface": True}
    out_mid = os.path.join(OUT_DIR, "runner_midsurface.k")
    logmsgs = []
    job_runner.run_job(mesher.MeshSettings(step_file=PLATE, size_max=10.0),
                       out_mid, kopts_mid, log=lambda m: logmsgs.append(str(m)))
    _, _, shells, kws = parse_k(out_mid)
    assert "*SECTION_SHELL" in kws and "*ELEMENT_SHELL" in kws
    assert "*ELEMENT_SOLID" not in kws
    assert abs(section_shell_thickness(out_mid) - 3.0) < 1e-3
    assert any("Midsurface thickness" in m for m in logmsgs)

    # (b) auto_connect spotweld on the unglued two-body model
    kopts_sw = {**base, "auto_connect": {"mode": "spotweld", "tol": None,
                                         "spacing": None, "fs": 0.0}}
    out_sw = os.path.join(OUT_DIR, "runner_spotweld.k")
    job_runner.run_job(mesher.MeshSettings(step_file=STEP2, size_max=8.0),
                       out_sw, kopts_sw, log=QUIET)
    _, _, _, kws_sw = parse_k(out_sw)
    assert "*CONSTRAINED_SPOTWELD" in kws_sw

    # (c) auto_connect tied on the unglued two-body model
    kopts_tie = {**base, "auto_connect": {"mode": "tied", "tol": None,
                                          "spacing": None, "fs": 0.0}}
    out_tie = os.path.join(OUT_DIR, "runner_tied.k")
    job_runner.run_job(mesher.MeshSettings(step_file=STEP2, size_max=8.0),
                       out_tie, kopts_tie, log=QUIET)
    _, _, _, kws_tie = parse_k(out_tie)
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" in kws_tie
    assert "*SET_SEGMENT_TITLE" in kws_tie
    print("OK: midsurface shell + spotweld + tied contact via run_job kopts")


def test_cli_auto_contact():
    print("=== CLI: --auto-contact (per-pair scoped contact) ===")
    _ensure_geometry()
    # STEP2 is two boxes touching at x=20; UNglued -> two bodies with a
    # detectable interface -> one scoped contact for the touching pair
    k_cli = os.path.join(OUT_DIR, "cli_auto_contact.k")
    rc = mesh_cli.main([STEP2, "-o", k_cli, "--size-max", "8",
                        "--auto-contact"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE" in kws, \
        "auto-contact must scope a surface-to-surface contact"
    assert "*SET_PART_LIST_TITLE" in kws, "scoped contact uses part-list sets"
    # an explicit TYPE picks the contact keyword
    k_cli2 = os.path.join(OUT_DIR, "cli_auto_contact_tied.k")
    rc2 = mesh_cli.main([STEP2, "-o", k_cli2, "--size-max", "8",
                         "--auto-contact", "tied_surface_to_surface"])
    assert rc2 == 0
    _, _, _, kws2 = parse_k(k_cli2)
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" in kws2
    # single body -> warn, no contact card
    k_cli3 = os.path.join(OUT_DIR, "cli_auto_contact_single.k")
    rc3 = mesh_cli.main([STEP, "-o", k_cli3, "--size-max", "10",
                         "--auto-contact"])
    assert rc3 == 0
    _, _, _, kws3 = parse_k(k_cli3)
    assert not any(k.startswith("*CONTACT_") for k in kws3), \
        "single-body model must not get a contact card"
    print("OK: scoped per-pair contact, type choice, single-body warns")


def test_cli_cross_section_and_history():
    print("=== CLI: --cross-section + --history-node ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_cross_hist.k")
    rc = mesh_cli.main([STEP, "-o", k_cli, "--size-max", "10", "--mat",
                        "--cross-section", "0:0:0:1:0:0:MIDCUT",
                        "--history-node", "1,2,3", "--d3plot-dt", "1e-4"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*DATABASE_CROSS_SECTION_PLANE" in kws, "cross-section plane card"
    assert "*DATABASE_SECFORC" in kws, "SECFORC must be added automatically"
    assert "*DATABASE_HISTORY_NODE" in kws, "history node card"
    txt = open(k_cli).read()
    assert "MIDCUT" in txt, "cross-section title must appear"
    print("OK: cross-section plane + auto SECFORC + history node")


def test_cli_mat_model():
    print("=== CLI: --mat-model (plasticity in place of --mat) ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_mat_model.k")
    rc = mesh_cli.main([
        STEP, "-o", k_cli, "--size-max", "10",
        "--mat-model", "plastic_kinematic:210000:0.3:7.85e-9:1000:200"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    assert "*MAT_PLASTIC_KINEMATIC" in kws, "kinematic plasticity card"
    assert "*MAT_ELASTIC" not in kws, "mat-model replaces the elastic card"
    k_cli2 = os.path.join(OUT_DIR, "cli_mat_model_pw.k")
    rc2 = mesh_cli.main([
        STEP, "-o", k_cli2, "--size-max", "10",
        "--mat-model", "piecewise:210000:0.3:7.85e-9:1000"])
    assert rc2 == 0
    _, _, _, kws2 = parse_k(k_cli2)
    assert "*MAT_PIECEWISE_LINEAR_PLASTICITY" in kws2
    print("OK: plastic_kinematic + piecewise via --mat-model")


def test_job_runner_auto_contact():
    print("=== job_runner: auto_connect mode 'contact' ===")
    from k_mesher import job_runner
    _ensure_geometry()
    base = {
        "pid": 1, "elform": 10, "element_kind": "solid", "thickness": 1.0,
        "start_nid": 1, "start_eid": 1, "start_sid": 1,
        "sym_nodeset": False, "sym_spc": False, "sym_constraint": "symmetric",
        "sym_dofs": "", "sym_segset": False,
        "mat": dict(STEEL), "part_mats": {},
        "face_nodesets": False, "face_segsets": False, "face_roles": [],
        "plain_faces": [], "face_titles": {}, "coord_sets": [],
        "implicit_cards": False, "qa_sets": False, "mesh_only": False,
        "split_mode": None, "contact_fs": None, "tssfac": None,
        "gravity": None,
    }
    kopts = {**base, "auto_connect": {"mode": "contact", "tol": None,
                                      "ctype": "automatic_surface_to_surface",
                                      "fs": 0.0}}
    out = os.path.join(OUT_DIR, "runner_contact.k")
    job_runner.run_job(mesher.MeshSettings(step_file=STEP2, size_max=8.0),
                       out, kopts, log=QUIET)
    _, _, _, kws = parse_k(out)
    assert "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE" in kws
    assert "*SET_PART_LIST_TITLE" in kws
    print("OK: scoped per-pair contact via run_job auto_connect 'contact'")


# --------------------------------------------------------------------------
# HEX8 / boundary layer / crash cards
# --------------------------------------------------------------------------

def section_solid_elform(path):
    """The ELFORM on the first *SECTION_SOLID of a standard-format file."""
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "*SECTION_SOLID":
            return int(lines[i + 2][10:20])
    return None


def test_cli_hex8():
    print("=== CLI: --etype hex8 (transfinite box) ===")
    _ensure_plate()
    k_cli = os.path.join(OUT_DIR, "cli_hex8.k")
    rc = mesh_cli.main([PLATE, "-o", k_cli, "--size-max", "10",
                        "--etype", "hex8", "--mat"])
    assert rc == 0
    nodes, elems, _, kws = parse_k(k_cli)
    assert "*ELEMENT_SOLID" in kws and "*ELEMENT_SHELL" not in kws
    assert "*SECTION_SOLID" in kws
    assert len(elems) > 5 and len(nodes) > 5
    # every solid row is eid, pid + 8 DISTINCT node ids (no degenerate tets)
    assert all(len(set(e[2:10])) == 8 for e in elems), \
        "hexes must have 8 distinct nodes per element"
    assert section_solid_elform(k_cli) == 1, "HEX8 default ELFORM must be 1"
    # an explicit --elform 2 (fully integrated S/R) is honoured
    k_cli2 = os.path.join(OUT_DIR, "cli_hex8_ef2.k")
    rc2 = mesh_cli.main([PLATE, "-o", k_cli2, "--size-max", "10",
                         "--etype", "hex8", "--elform", "2"])
    assert rc2 == 0
    assert section_solid_elform(k_cli2) == 2, "--elform 2 must land on the card"
    # the holed test part is not box-like -> clear mesher error, exit 1
    rc3 = mesh_cli.main([STEP, "-o", os.path.join(OUT_DIR, "cli_hex8_bad.k"),
                         "--size-max", "10", "--etype", "hex8"])
    assert rc3 == 1, "hex8 on a non-boxlike part must exit 1"
    print(f"OK: {len(elems)} hexes, elform 1 default / 2 explicit, guard fires")


def test_cli_boundary_layer():
    print("=== CLI: --boundary-layer (near-wall grading) ===")
    _ensure_geometry()
    k_base = os.path.join(OUT_DIR, "cli_bl_base.k")
    k_bl = os.path.join(OUT_DIR, "cli_bl.k")
    rc = mesh_cli.main([STEP, "-o", k_base, "--size-max", "10"])
    assert rc == 0
    rc2 = mesh_cli.main([STEP, "-o", k_bl, "--size-max", "10",
                         "--boundary-layer", "5:1.2:3:3"])
    assert rc2 == 0
    _, elems_base, _, _ = parse_k(k_base)
    _, elems_bl, _, _ = parse_k(k_bl)
    assert len(elems_bl) > len(elems_base), \
        (f"boundary layer must refine near the walls: "
         f"{len(elems_base)} -> {len(elems_bl)}")
    print(f"OK: {len(elems_base)} -> {len(elems_bl)} tets with the layer")


def test_cli_crash_cards():
    print("=== CLI: masses / springs / rigid bodies / damping / walls ===")
    _ensure_geometry()
    k_cli = os.path.join(OUT_DIR, "cli_crash.k")
    rc = mesh_cli.main([STEP, "-o", k_cli, "--size-max", "10", "--mat",
                        "--point-mass", "1:0.5", "--spring", "1:2:1000",
                        "--damper", "3:4:0.2", "--damping", "0.1",
                        "--nodal-rigid-body", "99:1,2,3",
                        "--rigidwall-sphere", "0:0:-50:0:0:1:25",
                        "--rigidwall-cylinder", "0:0:-50:1:0:0:10:100:0.2"])
    assert rc == 0
    _, _, _, kws = parse_k(k_cli)
    for kw in ("*ELEMENT_MASS", "*ELEMENT_DISCRETE", "*SECTION_DISCRETE",
               "*MAT_SPRING_ELASTIC", "*MAT_DAMPER_VISCOUS", "*DAMPING_GLOBAL",
               "*CONSTRAINED_NODAL_RIGID_BODY", "*RIGIDWALL_GEOMETRIC_SPHERE",
               "*RIGIDWALL_GEOMETRIC_CYLINDER"):
        assert kw in kws, f"missing {kw}"
    # one spring + one damper -> two discrete property blocks and elements
    assert kws.count("*ELEMENT_DISCRETE") == 2
    assert kws.count("*SECTION_DISCRETE") == 2
    txt = open(k_cli).read()
    assert "0.1" in txt and "1000" in txt
    print("OK: all crash cards emitted via the CLI")


def test_job_runner_crash_cards():
    print("=== job_runner: point_masses / discretes / NRB / damping kopts ===")
    from k_mesher import job_runner
    _ensure_geometry()
    kopts = {
        "pid": 1, "elform": 10, "element_kind": "solid", "thickness": 1.0,
        "start_nid": 1, "start_eid": 1, "start_sid": 1,
        "sym_nodeset": False, "sym_spc": False, "sym_constraint": "symmetric",
        "sym_dofs": "", "sym_segset": False,
        "mat": dict(STEEL), "part_mats": {},
        "face_nodesets": False, "face_segsets": False, "face_roles": [],
        "plain_faces": [], "face_titles": {}, "coord_sets": [],
        "implicit_cards": False, "qa_sets": False, "mesh_only": False,
        "split_mode": None, "contact_fs": None, "tssfac": None,
        "gravity": None,
        # the GUI-shared contract keys
        "point_masses": ({"nid": 1, "mass": 0.5},),
        "discretes": ({"n1": 1, "n2": 2, "k": 1000.0},
                      {"n1": 3, "n2": 4, "c": 0.2}),
        "nodal_rigid_bodies": ({"nodes": [1, 2, 3], "pid": 99},),
        "damping": {"valdmp": 0.1},
    }
    settings = mesher.MeshSettings(step_file=STEP, size_max=10.0)
    out = os.path.join(OUT_DIR, "runner_crash.k")
    job_runner.run_job(settings, out, kopts, log=QUIET)
    _, _, _, kws = parse_k(out)
    for kw in ("*ELEMENT_MASS", "*ELEMENT_DISCRETE", "*SECTION_DISCRETE",
               "*MAT_SPRING_ELASTIC", "*MAT_DAMPER_VISCOUS", "*DAMPING_GLOBAL",
               "*CONSTRAINED_NODAL_RIGID_BODY"):
        assert kw in kws, f"missing {kw}"
    print("OK: crash cards land through run_job kopts")


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for _name, fn in tests:
        fn()
    print(f"\nALL TESTS PASSED ({len(tests)} test functions)")


if __name__ == "__main__":
    main()
