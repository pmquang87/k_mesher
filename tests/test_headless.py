"""Headless end-to-end test: STEP -> TET4/TET10/TRI3/QUAD4 -> .k

Run:  python tests/test_headless.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import dyna_writer
import mesher
from make_test_step import make, make_two_bodies, make_shell, make_formats

QUIET = lambda m: None


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
    and *ELEMENT_SHELL. Returns (nodes {nid: xyz}, elems, shells, keywords)."""
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


def main():
    ex = os.path.join(ROOT, "examples")
    step = os.path.join(ex, "test_part.step")
    step2 = os.path.join(ex, "test_two_bodies.step")
    step_sh = os.path.join(ex, "test_shell.step")
    if not os.path.isfile(step):
        make(step)
    if not os.path.isfile(step2):
        make_two_bodies(step2)
    if not os.path.isfile(step_sh):
        make_shell(step_sh)

    out_dir = os.path.join(ROOT, "tests", "out")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. full model, TET4 ------------------------------------------------
    print("=== full model (TET4) ===")
    settings = mesher.MeshSettings(step_file=step, size_max=8.0, size_min=2.0)
    res = mesher.mesh_step(settings, log=QUIET)
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

    k_full = os.path.join(out_dir, "full.k")
    dyna_writer.write_k(k_full, res.coords, res.elems)
    nodes, elems, _, keywords = parse_k(k_full)
    assert len(nodes) == res.stats["n_nodes"]
    assert len(elems) == res.stats["n_elems"]
    assert all(e[5] == e[6] == e[7] == e[8] == e[9] for e in elems)
    assert (tet_volumes(nodes, elems) > 0).all()
    print(f"OK: {len(nodes)} nodes, {len(elems)} tets")

    # ---- 2. half model, X symmetry, with sets and start IDs -----------------
    print("=== half model (X symmetry, keep +) ===")
    settings = mesher.MeshSettings(
        step_file=step, size_max=8.0, size_min=2.0,
        symmetry=[mesher.SymmetryPlane(axis="x", offset=0.0, keep="+")])
    res = mesher.mesh_step(settings, log=QUIET)
    assert res.coords[:, 0].min() >= -1e-6
    assert abs(res.stats["measure"] - 117146.0 / 2) / (117146.0 / 2) < 0.02
    n_sym = len(res.sym_nodes["x"])
    assert n_sym > 10

    k_half = os.path.join(out_dir, "half_x.k")
    sym_sets = ({"axis": "x", "offset": 0.0, "nodes": res.sym_nodes["x"], "spc": True},)
    dyna_writer.write_k(k_half, res.coords, res.elems, sym_sets=sym_sets,
                        start_nid=1000, start_eid=5000,
                        mat={"e": 210000.0, "pr": 0.3, "ro": 7.85e-9})
    nodes, elems, _, keywords = parse_k(k_half)
    assert min(nodes) == 1000 and elems[0][0] == 5000
    assert "*SET_NODE_LIST_TITLE" in keywords
    assert "*BOUNDARY_SPC_SET" in keywords
    assert "*MAT_ELASTIC" in keywords
    assert (tet_volumes(nodes, elems) > 0).all()
    sym_nids = [n - 1 + 1000 for n in res.sym_nodes["x"]]
    assert all(abs(nodes[nid][0]) < 1e-6 for nid in sym_nids)
    print(f"OK: {len(nodes)} nodes, {n_sym} in SYM_X set, MAT card present")

    # ---- 3. quarter model ----------------------------------------------------
    print("=== quarter model (X keep +, Y keep -) ===")
    settings = mesher.MeshSettings(
        step_file=step, size_max=8.0, size_min=2.0,
        symmetry=[mesher.SymmetryPlane("x", 0.0, "+"),
                  mesher.SymmetryPlane("y", 0.0, "-")])
    res = mesher.mesh_step(settings, log=QUIET)
    assert res.coords[:, 0].min() >= -1e-6
    assert res.coords[:, 1].max() <= 1e-6
    assert abs(res.stats["measure"] - 117146.0 / 4) / (117146.0 / 4) < 0.02
    print(f"OK: volume {res.stats['measure']:.0f}")

    # ---- 4. TET10 -------------------------------------------------------------
    print("=== TET10 half model ===")
    settings = mesher.MeshSettings(
        step_file=step, element_type="TET10", size_max=8.0, size_min=2.0,
        symmetry=[mesher.SymmetryPlane("x", 0.0, "+")])
    res = mesher.mesh_step(settings, log=QUIET)
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

    k_t10 = os.path.join(out_dir, "half_x_tet10.k")
    dyna_writer.write_k(k_t10, res.coords, res.elems, elform=16)
    nodes, elems, _, keywords = parse_k(k_t10)
    assert len(elems) == res.stats["n_elems"]
    assert all(len(e) == 12 for e in elems)
    assert (tet_volumes(nodes, elems) > 0).all()
    print(f"OK: {res.stats['n_elems']} TET10, two-line format, mid nodes verified")

    # ---- 5. two bodies, glued -> two PIDs -------------------------------------
    print("=== two bodies, glue, per-body PIDs ===")
    settings = mesher.MeshSettings(step_file=step2, size_max=6.0, glue=True)
    res = mesher.mesh_step(settings, log=QUIET)
    assert len(res.part_names) == 2
    assert set(np.unique(res.elem_parts)) == {0, 1}
    assert abs(res.stats["measure"] - 16000.0) / 16000.0 < 0.01
    iface = res.coords[np.abs(res.coords[:, 0] - 20.0) < 1e-6]
    assert len(iface) == len(np.unique(np.round(iface, 6), axis=0))

    k_two = os.path.join(out_dir, "two_bodies.k")
    dyna_writer.write_k(k_two, res.coords, res.elems, pid=10,
                        part_ids=10 + res.elem_parts,
                        part_titles={10: "left box", 11: "right box"})
    nodes, elems, _, keywords = parse_k(k_two)
    assert keywords.count("*PART") == 2
    assert set(e[1] for e in elems) == {10, 11}
    print("OK: 2 PIDs, shared interface nodes")

    # ---- 6. face sets ----------------------------------------------------------
    print("=== face sets (node + segment) ===")
    faces = mesher.list_faces(mesher.MeshSettings(step_file=step), log=QUIET)
    assert len(faces) >= 7
    xmax_face = next(f for f in faces
                     if abs(f["centroid"][0] - 50.0) < 1e-3
                     and abs(f["area"] - 1250.0) < 1.0)
    tag = xmax_face["tag"]
    settings = mesher.MeshSettings(step_file=step, size_max=8.0, size_min=2.0,
                                   collect_faces=[tag])
    res = mesher.mesh_step(settings, log=QUIET)
    fn, fs = res.face_nodes[tag], res.face_segs[tag]
    assert len(fn) > 5 and len(fs) > 5 and fs.shape[1] == 4
    assert np.allclose(res.coords[fn - 1][:, 0], 50.0, atol=1e-6)
    assert set(fs.ravel().tolist()) <= set(fn.tolist())

    k_face = os.path.join(out_dir, "face_sets.k")
    dyna_writer.write_k(k_face, res.coords, res.elems, face_sets=(
        {"kind": "node", "title": f"FACE_{tag}", "nodes": fn},
        {"kind": "segment", "title": f"FACE_{tag}", "segments": fs},
    ))
    _, _, _, keywords = parse_k(k_face)
    assert "*SET_NODE_LIST_TITLE" in keywords
    assert "*SET_SEGMENT_TITLE" in keywords
    print(f"OK: face {tag}: {len(fn)} nodes, {len(fs)} segments")

    # ---- 7. local refinement ---------------------------------------------------
    print("=== local refinement (sphere) ===")
    res_base = mesher.mesh_step(
        mesher.MeshSettings(step_file=step, size_max=8.0), log=QUIET)
    res_ref = mesher.mesh_step(mesher.MeshSettings(
        step_file=step, size_max=8.0,
        refinements=[{"kind": "sphere", "params": [40.0, 0.0, 0.0, 15.0],
                      "size": 2.0}]), log=QUIET)

    def nodes_in_sphere(res):
        d = np.linalg.norm(res.coords - np.array([40.0, 0.0, 0.0]), axis=1)
        return int((d < 15.0).sum())

    n_base, n_ref = nodes_in_sphere(res_base), nodes_in_sphere(res_ref)
    assert n_ref > 3 * n_base
    print(f"OK: nodes inside sphere {n_base} -> {n_ref}")

    # ---- 8. TRI3 shells on a surface-only model --------------------------------
    print("=== TRI3 shells (surface-only STEP) ===")
    settings = mesher.MeshSettings(step_file=step_sh, element_type="TRI3",
                                   size_max=5.0)
    res = mesher.mesh_step(settings, log=QUIET)
    assert res.elems.shape[1] == 4
    assert (res.elems[:, 2] == res.elems[:, 3]).all(), "TRI3 must be degenerate quads"
    assert res.stats["measure_label"] == "area"
    assert abs(res.stats["measure"] - 6200.0) / 6200.0 < 0.01  # box area
    assert res.stats["free_edges"] == 0, "closed box shell must be watertight"
    assert res.stats["nonmanifold_edges"] == 0

    k_tri = os.path.join(out_dir, "shell_tri3.k")
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

    # ---- 9. TRI3 shells with symmetry -------------------------------------------
    print("=== TRI3 shells with X symmetry ===")
    settings = mesher.MeshSettings(
        step_file=step_sh, element_type="TRI3", size_max=5.0,
        symmetry=[mesher.SymmetryPlane("x", 25.0, "-")])
    res = mesher.mesh_step(settings, log=QUIET)
    assert res.coords[:, 0].max() <= 25.0 + 1e-6
    # remaining area: x=0 face 600 + halves of the four side faces = 3100
    assert abs(res.stats["measure"] - 3100.0) / 3100.0 < 0.02
    assert len(res.sym_nodes["x"]) > 5
    assert res.stats["free_edges"] > 0, "cut shell must have free edges"
    print(f"OK: area {res.stats['measure']:.0f}, "
          f"{len(res.sym_nodes['x'])} nodes on the cut plane, "
          f"{res.stats['free_edges']} free edges")

    # ---- 10. QUAD4 shells on a solid's boundary ---------------------------------
    print("=== QUAD4 shells (quad-dominant, solid boundary) ===")
    settings = mesher.MeshSettings(step_file=step, element_type="QUAD4",
                                   size_max=6.0)
    res = mesher.mesh_step(settings, log=QUIET)
    n_quads = int((res.elems[:, 2] != res.elems[:, 3]).sum())
    frac = n_quads / len(res.elems)
    assert frac > 0.5, f"only {frac:.0%} quads"
    # boundary area of box-with-hole: 17500 - 2*pi*100 + 2*pi*10*25 = 18442.5
    assert abs(res.stats["measure"] - 18442.5) / 18442.5 < 0.01

    k_quad = os.path.join(out_dir, "shell_quad4.k")
    dyna_writer.write_k(k_quad, res.coords, res.elems, element_kind="shell",
                        elform=16, thickness=2.0)
    nodes, _, shells, keywords = parse_k(k_quad)
    assert len(shells) == res.stats["n_elems"]
    assert abs(shell_areas(nodes, shells).sum() - 18442.5) / 18442.5 < 0.01
    print(f"OK: {len(shells)} shells, {frac:.0%} quads, area "
          f"{res.stats['measure']:.0f}")

    # ---- 11. defeaturing: remove the hole ---------------------------------------
    print("=== defeature (remove cylindrical hole) ===")
    faces = mesher.list_faces(mesher.MeshSettings(step_file=step), log=QUIET)
    cyl = [f["tag"] for f in faces if f["type"] == "Cylinder"]
    assert len(cyl) == 1
    settings = mesher.MeshSettings(step_file=step, size_max=8.0,
                                   defeature_faces=cyl)
    res = mesher.mesh_step(settings, log=QUIET)
    assert abs(res.stats["measure"] - 125000.0) / 125000.0 < 0.005, \
        f"hole not removed: volume {res.stats['measure']}"
    # rescan shows post-defeature faces
    faces2 = mesher.list_faces(settings, log=QUIET)
    assert len(faces2) == 6
    print(f"OK: volume {res.stats['measure']:.0f} (solid block), "
          f"{len(faces2)} faces after defeature")

    # ---- 12. loads, SPC, QA sets, implicit cards ---------------------------------
    print("=== BC/load cards, QA element set, implicit template ===")
    xmax = next(f["tag"] for f in faces
                if abs(f["centroid"][0] - 50.0) < 1e-3
                and abs(f["area"] - 1250.0) < 1.0)
    settings = mesher.MeshSettings(step_file=step, size_max=8.0, size_min=2.0,
                                   collect_faces=[xmax])
    res = mesher.mesh_step(settings, log=QUIET)
    fn, fs = res.face_nodes[xmax], res.face_segs[xmax]
    k_bc = os.path.join(out_dir, "bc_loads.k")
    dyna_writer.write_k(
        k_bc, res.coords, res.elems, implicit_cards=True,
        mat={"e": 210000.0, "pr": 0.3, "ro": 7.85e-9},
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

    # ---- 13. per-face mesh size ---------------------------------------------------
    print("=== local mesh size on a face ===")
    res_base = mesher.mesh_step(
        mesher.MeshSettings(step_file=step, size_max=8.0), log=QUIET)
    res_fs = mesher.mesh_step(
        mesher.MeshSettings(step_file=step, size_max=8.0,
                            face_sizes={xmax: 2.0}), log=QUIET)

    def nodes_on_xmax(res):
        return int((np.abs(res.coords[:, 0] - 50.0) < 1e-6).sum())

    n0, n1 = nodes_on_xmax(res_base), nodes_on_xmax(res_fs)
    assert n1 > 2 * n0, f"face size had no effect: {n0} -> {n1}"
    print(f"OK: nodes on the sized face {n0} -> {n1}")

    # ---- 14. auto-refine wrapper ---------------------------------------------------
    print("=== auto-refine wrapper (clean geometry: no retry needed) ===")
    settings = mesher.MeshSettings(step_file=step2, size_max=6.0,
                                   auto_refine=True, auto_refine_rounds=1,
                                   auto_refine_threshold=0.99)
    res = mesher.mesh_step_auto(settings, log=QUIET)
    assert res.stats["n_elems"] > 10
    print(f"OK: returned {res.stats['n_elems']} elements, "
          f"min quality {res.stats['quality_min']:.3f}")

    # ---- 15. error case -----------------------------------------------------------
    print("=== error handling (cut outside part) ===")
    settings = mesher.MeshSettings(
        step_file=step, size_max=8.0,
        symmetry=[mesher.SymmetryPlane("x", 1000.0, "+")])
    try:
        mesher.mesh_step(settings, log=QUIET)
        raise AssertionError("expected RuntimeError for empty cut")
    except RuntimeError as e:
        print(f"OK: got expected error: {e}")

    # ---- 16. symmetry BC options: node-set-only, anti-sym, fixed, custom -------
    print("=== symmetry BC options (node set / SPC / anti-sym / custom) ===")
    settings = mesher.MeshSettings(
        step_file=step, size_max=8.0, size_min=2.0,
        symmetry=[mesher.SymmetryPlane("x", 0.0, "+")])
    res = mesher.mesh_step(settings, log=QUIET)
    nodes_x = res.sym_nodes["x"]

    # (a) node set only, no boundary condition
    k = os.path.join(out_dir, "sym_nodeset_only.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": False},))
    _, _, _, kws = parse_k(k)
    assert "*SET_NODE_LIST_TITLE" in kws, "node set must be written"
    assert "*BOUNDARY_SPC_SET" not in kws, "no SPC when spc=False"

    # (b) default symmetric -> constrain X translation + in-plane rotations
    k = os.path.join(out_dir, "sym_symmetric.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True},))
    assert spc_dofs(open(k).read()) == [1, 0, 0, 0, 1, 1]

    # (c) anti-symmetric -> the complement
    k = os.path.join(out_dir, "sym_antisym.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "constraint": "antisymmetric"},))
    txt = open(k).read()
    assert spc_dofs(txt) == [0, 1, 1, 1, 0, 0]
    assert "(antisymmetric)" in txt, "title should note the constraint kind"

    # (d) fixed -> all six DOFs
    k = os.path.join(out_dir, "sym_fixed.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "constraint": "fixed"},))
    assert spc_dofs(open(k).read()) == [1, 1, 1, 1, 1, 1]

    # (e) custom DOFs "13" -> dofx, dofz
    k = os.path.join(out_dir, "sym_custom.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "dofs": "13"},))
    assert spc_dofs(open(k).read()) == [1, 0, 1, 0, 0, 0]

    # (f) start_sid offsets every set id (merge-friendly; parallels start_nid)
    k = os.path.join(out_dir, "sym_startsid.k")
    dyna_writer.write_k(k, res.coords, res.elems, start_sid=100, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True},))
    lines = open(k).read().splitlines()
    i = next(j for j, l in enumerate(lines) if l.startswith("*SET_NODE_LIST_TITLE"))
    assert int(lines[i + 3][0:10]) == 100, "node set SID must start at start_sid"
    s = next(j for j, l in enumerate(lines) if l.startswith("*BOUNDARY_SPC_SET"))
    assert int(lines[s + 2][0:10]) == 100, "SPC must reference the offset set id"

    # (g) symmetry-plane segment set: boundary faces on the plane, all corners
    # on the plane, written as *SET_SEGMENT
    seg = res.sym_segs["x"]
    on_plane = set(nodes_x.tolist())
    assert len(seg) > 5 and seg.shape[1] == 4
    assert all(int(c) in on_plane for c in seg[:, :3].ravel())
    assert (seg[:, 2] == seg[:, 3]).all(), "tets -> triangle segments padded to quad"
    k = os.path.join(out_dir, "sym_segset.k")
    dyna_writer.write_k(k, res.coords, res.elems, sym_sets=(
        {"axis": "x", "offset": 0.0, "nodes": nodes_x, "spc": True,
         "segset": seg},))
    _, _, _, kws = parse_k(k)
    assert "*SET_SEGMENT_TITLE" in kws, "segment set must be written"
    print("OK: node-set-only, symmetric, anti-symmetric, fixed, custom, "
          "start_sid, plane segset verified")

    # ---- 17. multi-format input: BREP / IGES / STL ----------------------------
    print("=== multi-format input (BREP / IGES / STL) ===")
    base = os.path.join(ex, "test_part")
    if not all(os.path.isfile(base + e) for e in (".iges", ".brep", ".stl")):
        make_formats(base)

    # BREP solid reproduces the STEP volume exactly (same OCC kernel)
    rb = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".brep", size_max=8.0, size_min=2.0), log=QUIET)
    assert abs(rb.stats["measure"] - 117146.0) / 117146.0 < 0.02

    # IGES imports as surfaces and is sewn into a solid
    ri = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".iges", size_max=8.0, size_min=2.0), log=QUIET)
    assert abs(ri.stats["measure"] - 117146.0) / 117146.0 < 0.02

    # IGES rejects the OCC target-unit override -> import must fall back to
    # file units instead of crashing
    riu = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".iges", size_max=8.0, size_min=2.0,
        occ_unit="MM"), log=QUIET)
    assert abs(riu.stats["measure"] - 117146.0) / 117146.0 < 0.02

    # IGES shell: sewing makes it watertight
    ris = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".iges", element_type="TRI3", size_max=8.0), log=QUIET)
    assert ris.stats["free_edges"] == 0, \
        f"IGES shell not watertight: {ris.stats['free_edges']} free edges"

    # STL shell: tessellation used as-is, coincident nodes welded -> watertight
    rs = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".stl", element_type="TRI3"), log=QUIET)
    assert rs.elems.shape[1] == 4 and (rs.elems[:, 2] == rs.elems[:, 3]).all()
    assert rs.stats["free_edges"] == 0, \
        f"STL shell not watertight: {rs.stats['free_edges']} free edges"

    # STL -> solid: reconstruct the volume from the watertight surface (TET4)
    rss = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".stl", element_type="TET4", size_max=8.0), log=QUIET)
    assert rss.elems.shape[1] == 4
    assert abs(rss.stats["measure"] - 117146.0) / 117146.0 < 0.02, \
        f"STL solid volume off: {rss.stats['measure']}"
    k_stl_solid = os.path.join(out_dir, "stl_solid.k")
    dyna_writer.write_k(k_stl_solid, rss.coords, rss.elems)
    nodes, elems, _, _ = parse_k(k_stl_solid)
    assert (tet_volumes(nodes, elems) > 0).all(), "STL tets must be positive-volume"
    # STL -> TET10 solid
    rst = mesher.mesh_step(mesher.MeshSettings(
        step_file=base + ".stl", element_type="TET10", size_max=8.0), log=QUIET)
    assert rst.elems.shape[1] == 10

    # STL guards: symmetry, defeature and face scanning still rejected (no B-rep)
    for bad in (dict(element_type="TRI3",
                     symmetry=[mesher.SymmetryPlane("x", 0.0, "+")]),
                dict(element_type="TRI3", defeature_faces=[1])):
        try:
            mesher.mesh_step(mesher.MeshSettings(step_file=base + ".stl", **bad),
                             log=QUIET)
            raise AssertionError(f"STL should reject {bad}")
        except RuntimeError:
            pass
    try:
        mesher.list_faces(mesher.MeshSettings(step_file=base + ".stl"), log=QUIET)
        raise AssertionError("list_faces should reject STL")
    except RuntimeError:
        pass

    # an open (non-watertight) tessellation cannot be tetrahedralized
    open_stl = os.path.join(out_dir, "open.stl")
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
    print(f"OK: BREP vol {rb.stats['measure']:.0f}, IGES vol "
          f"{ri.stats['measure']:.0f}, IGES/STL shells watertight, STL solid "
          f"vol {rss.stats['measure']:.0f}, guards fire")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
