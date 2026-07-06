"""LS-DYNA keyword (.k) file writer for TET4/TET10 solid and TRI3/QUAD4
shell meshes.

Formats used (standard fixed-width small format):
  *NODE            nid(I8)  x,y,z(E16.9)
  *ELEMENT_SOLID   TET4:  eid,pid,n1..n8 (10I8), tet as n1 n2 n3 n4 n4 n4 n4 n4
                   TET10: two-line format - eid,pid (2I8) / n1..n10 (10I8)
  *ELEMENT_SHELL   eid,pid,n1..n4 (6I8), triangles as n1 n2 n3 n3
  *SET_NODE_LIST / *SET_SEGMENT   I10 fields
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

# Symmetry SPC DOFs per plane: constrain the translation normal to the plane
# and the two in-plane rotations (needed for shells; irrelevant but harmless
# for solids). This is the standard "symmetric" boundary condition.
#            dofx dofy dofz dofrx dofry dofrz
_SYM_DOFS = {
    "x": (1, 0, 0, 0, 1, 1),
    "y": (0, 1, 0, 1, 0, 1),
    "z": (0, 0, 1, 1, 1, 0),
}

# Anti-symmetric boundary condition (the complement): constrain the two
# in-plane translations and the rotation about the plane normal. Use this on a
# symmetry plane when the loading is anti-symmetric about it.
_ANTISYM_DOFS = {
    "x": (0, 1, 1, 1, 0, 0),
    "y": (1, 0, 1, 0, 1, 0),
    "z": (1, 1, 0, 0, 0, 1),
}

# fully fixed (encastre)
_FIXED_DOFS = (1, 1, 1, 1, 1, 1)

# named constraint kinds usable on a symmetry plane
SYM_CONSTRAINTS = ("symmetric", "antisymmetric", "fixed", "custom")


def _resolve_sym_dofs(item: dict) -> tuple[int, ...]:
    """Return the 6 DOF flags for a symmetry set item.

    Honours an explicit ``dofs`` key (a 6-sequence of 0/1, or a digit string
    such as ``"16"`` meaning DOFs 1 and 6), otherwise falls back to the named
    ``constraint`` (symmetric / antisymmetric / fixed), defaulting to the
    symmetric pattern for the item's axis.
    """
    dofs = item.get("dofs")
    if dofs is not None:
        if isinstance(dofs, str):
            return tuple(1 if str(d) in dofs else 0 for d in range(1, 7))
        return tuple(int(bool(v)) for v in dofs)
    constraint = (item.get("constraint") or "symmetric").lower()
    if constraint == "antisymmetric":
        return _ANTISYM_DOFS[item["axis"]]
    if constraint == "fixed":
        return _FIXED_DOFS
    return _SYM_DOFS[item["axis"]]


def write_k(
    path: str,
    coords: np.ndarray,        # (N, 3)
    elems: np.ndarray,         # solids: (M, 4|10), shells: (M, 4); 1-based,
                               # LS-DYNA node order, shell tris with n4 = n3
    *,
    element_kind: str = "solid",       # "solid" | "shell"
    pid: int = 1,              # base part id; also used as SECID and MID
    elform: int = 10,
    thickness: float = 1.0,    # shell thickness (shells only)
    start_nid: int = 1,
    start_eid: int = 1,
    start_sid: int = 1,        # first set id (node/segment/SPC/element sets)
    title: str = "k_mesher mesh",
    comments: tuple[str, ...] = (),
    sym_sets: tuple[dict, ...] = (),   # {"axis","offset","nodes"(1-based),
                                       #  "spc":bool - write *BOUNDARY_SPC_SET,
                                       #  "constraint": symmetric|antisymmetric|
                                       #                fixed (default symmetric),
                                       #  "dofs": optional 6-seq/digit-str override,
                                       #  "segset": optional (S,4) plane segments}
    mat: dict | None = None,           # {"e": E, "pr": nu, "ro": density} -> *MAT_ELASTIC
    part_ids: np.ndarray | None = None,   # (M,) per-element part id; default all = pid
    part_titles: dict[int, str] | None = None,
    face_sets: tuple[dict, ...] = (),  # {"kind":"node"|"segment","title",
                                       #  "nodes"|(S,4)"segments"} (1-based);
                                       # optional roles: "spc_dofs" ("123456"),
                                       # "pressure" (float), "force" (dof, total)
    elem_sets: tuple[dict, ...] = (),  # {"title","eids"(1-based rows)} -> failed
    implicit_cards: bool = False,      # write basic implicit static control deck
    curve_id: int = 1,                 # id of the generated unit ramp curve
) -> None:
    n_nodes = len(coords)
    nn = elems.shape[1]
    nids = np.arange(start_nid, start_nid + n_nodes, dtype=np.int64)
    eids = np.arange(start_eid, start_eid + len(elems), dtype=np.int64)
    elem_nids = elems - 1 + start_nid
    if part_ids is None:
        part_ids = np.full(len(elems), pid, dtype=np.int64)
    unique_pids = list(dict.fromkeys(int(p) for p in part_ids))
    part_titles = part_titles or {}

    with open(path, "w", newline="\n") as f:
        f.write("*KEYWORD\n")
        f.write("*TITLE\n")
        f.write(f"{title[:80]}\n")
        f.write(f"$ Written by k_mesher on {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for c in comments:
            f.write(f"$ {c}\n")

        if implicit_cards:
            _write_implicit_cards(f)

        # --- PART / SECTION / MAT -------------------------------------------
        for p in unique_pids:
            f.write("*PART\n")
            f.write(f"{part_titles.get(p, f'part {p}')[:70]}\n")
            f.write("$#     pid     secid       mid     eosid      hgid      grav"
                    "    adpopt      tmid\n")
            f.write(f"{p:10d}{pid:10d}{pid:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}\n")
        if mat is None:
            f.write(f"$ NOTE: define a *MAT_ card with MID = {pid} "
                    f"before running LS-DYNA.\n")

        if element_kind == "solid":
            f.write("*SECTION_SOLID\n")
            f.write("$#   secid    elform       aet\n")
            f.write(f"{pid:10d}{elform:10d}{0:10d}\n")
        else:
            f.write("*SECTION_SHELL\n")
            f.write("$#   secid    elform      shrf       nip     propt"
                    "   qr/irid     icomp     setyp\n")
            f.write(f"{pid:10d}{elform:10d}{0.8333:10.4f}{5:10d}{1.0:10.1f}"
                    f"{0:10d}{0:10d}{1:10d}\n")
            f.write("$#      t1        t2        t3        t4      nloc"
                    "     marea      idof    edgset\n")
            f.write(f"{thickness:10.4g}{thickness:10.4g}{thickness:10.4g}"
                    f"{thickness:10.4g}\n")

        if mat is not None:
            f.write("*MAT_ELASTIC\n")
            f.write("$#     mid        ro         e        pr        da        db\n")
            f.write(f"{pid:10d}{mat['ro']:10.3e}{mat['e']:10.4g}{mat['pr']:10.4f}"
                    f"{0.0:10.1f}{0.0:10.1f}\n")

        # --- NODES ----------------------------------------------------------
        f.write("*NODE\n")
        f.write("$#   nid               x               y               z\n")
        node_block = np.column_stack((nids.astype(float), coords))
        np.savetxt(f, node_block, fmt="%8d%16.9e%16.9e%16.9e")

        # --- ELEMENTS ---------------------------------------------------------
        if element_kind == "shell":
            f.write("*ELEMENT_SHELL\n")
            f.write("$#   eid     pid      n1      n2      n3      n4\n")
            elem_block = np.column_stack((eids, part_ids, elem_nids))
            np.savetxt(f, elem_block, fmt="%8d" * 6)
        elif nn == 4:
            f.write("*ELEMENT_SOLID\n")
            f.write("$#   eid     pid      n1      n2      n3      n4      n5"
                    "      n6      n7      n8\n")
            elem_block = np.column_stack(
                (eids, part_ids, elem_nids,
                 elem_nids[:, 3], elem_nids[:, 3], elem_nids[:, 3], elem_nids[:, 3])
            )
            np.savetxt(f, elem_block, fmt="%8d" * 10)
        else:  # TET10, two-line format
            f.write("*ELEMENT_SOLID\n")
            f.write("$#   eid     pid\n")
            f.write("$#    n1      n2      n3      n4      n5      n6      n7"
                    "      n8      n9     n10\n")
            elem_block = np.column_stack((eids, part_ids, elem_nids))
            np.savetxt(f, elem_block, fmt="%8d%8d\n" + "%8d" * 10)

        # --- SETS (symmetry, then picked faces) -------------------------------
        need_curve = any(s.get("pressure") is not None
                         or s.get("force") is not None for s in face_sets)
        if need_curve:
            f.write("*DEFINE_CURVE_TITLE\n")
            f.write("k_mesher unit ramp (loads are scaled via SF)\n")
            f.write("$#    lcid      sidr       sfa       sfo      offa      offo"
                    "    dattyp     lcint\n")
            f.write(f"{curve_id:10d}{0:10d}{1.0:10.1f}{1.0:10.1f}"
                    f"{0.0:10.1f}{0.0:10.1f}{0:10d}{0:10d}\n")
            f.write(f"{0.0:20.10g}{0.0:20.10g}\n")
            f.write(f"{1.0:20.10g}{1.0:20.10g}\n")

        sid = start_sid - 1
        for s in sym_sets:
            sid += 1
            set_nids = np.asarray(s["nodes"], dtype=np.int64) - 1 + start_nid
            constraint = (s.get("constraint") or "symmetric").lower()
            suffix = "" if constraint == "symmetric" else f" ({constraint})"
            _write_node_set(f, sid, f"SYM_{s['axis'].upper()} plane at "
                                    f"{s['offset']:g}{suffix}", set_nids)
            if s.get("spc", True):
                _write_spc(f, sid, _resolve_sym_dofs(s))
            if s.get("segset") is not None and len(s["segset"]):
                sid += 1
                segs = np.asarray(s["segset"], dtype=np.int64) - 1 + start_nid
                _write_segment_set(f, sid, f"SYM_{s['axis'].upper()} plane at "
                                           f"{s['offset']:g} segments", segs)

        for s in face_sets:
            sid += 1
            if s["kind"] == "node":
                set_nids = np.asarray(s["nodes"], dtype=np.int64) - 1 + start_nid
                _write_node_set(f, sid, s["title"], set_nids)
                if s.get("spc_dofs"):
                    dofs = tuple(1 if str(d) in s["spc_dofs"] else 0
                                 for d in range(1, 7))
                    _write_spc(f, sid, dofs)
                if s.get("force") is not None:
                    dof_name, total = s["force"]
                    dof = {"x": 1, "y": 2, "z": 3}[dof_name.lower()]
                    sf = float(total) / max(len(set_nids), 1)
                    f.write(f"$ total force {total:g} in {dof_name.upper()} "
                            f"over {len(set_nids)} nodes -> {sf:g} per node\n")
                    f.write("*LOAD_NODE_SET\n")
                    f.write("$#    nsid       dof      lcid        sf\n")
                    f.write(f"{sid:10d}{dof:10d}{curve_id:10d}{sf:10.4g}\n")
            else:
                segs = np.asarray(s["segments"], dtype=np.int64) - 1 + start_nid
                _write_segment_set(f, sid, s["title"], segs)
                if s.get("pressure") is not None:
                    f.write("*LOAD_SEGMENT_SET\n")
                    f.write("$#    ssid      lcid        sf        at\n")
                    f.write(f"{sid:10d}{curve_id:10d}"
                            f"{float(s['pressure']):10.4g}{0.0:10.1f}\n")

        # --- ELEMENT SETS (quality check failures etc.) -----------------------
        for s in elem_sets:
            sid += 1
            set_eids = np.asarray(s["eids"], dtype=np.int64) - 1 + start_eid
            if element_kind == "solid":
                f.write("*SET_SOLID_TITLE\n")
                f.write(f"{s['title'][:70]}\n")
                f.write("$#     sid    solver\n")
                f.write(f"{sid:10d}{'MECH':<10s}\n")
            else:
                f.write("*SET_SHELL_LIST_TITLE\n")
                f.write(f"{s['title'][:70]}\n")
                f.write("$#     sid       da1       da2       da3       da4"
                        "    solver\n")
                f.write(f"{sid:10d}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}"
                        f"{0.0:10.1f}{'MECH':>10s}\n")
            for row in range(0, len(set_eids), 8):
                chunk = set_eids[row:row + 8]
                f.write("".join(f"{e:10d}" for e in chunk) + "\n")

        f.write("*END\n")


def _write_spc(f, sid: int, dofs) -> None:
    f.write("*BOUNDARY_SPC_SET\n")
    f.write("$#    nsid       cid      dofx      dofy      dofz"
            "     dofrx     dofry     dofrz\n")
    f.write(f"{sid:10d}{0:10d}" + "".join(f"{d:10d}" for d in dofs) + "\n")


def _write_implicit_cards(f) -> None:
    """Minimal implicit static setup: 1.0 s of pseudo-time, auto stepping,
    nonlinear solver, d3plot output. Review before production use."""
    f.write("$ --- basic implicit static setup (review before production) ---\n")
    f.write("*CONTROL_TERMINATION\n")
    f.write("$#  endtim    endcyc     dtmin    endeng    endmas\n")
    f.write(f"{1.0:10.1f}\n")
    f.write("*CONTROL_IMPLICIT_GENERAL\n")
    f.write("$#  imflag       dt0    imform      nsbs       igs     cnstn      form\n")
    f.write(f"{1:10d}{0.1:10.2f}\n")
    f.write("*CONTROL_IMPLICIT_AUTO\n")
    f.write("$#   iauto    iteopt    itewin     dtmin     dtmax\n")
    f.write(f"{1:10d}\n")
    f.write("*CONTROL_IMPLICIT_SOLUTION\n")
    f.write("$#  nsolvr    ilimit    maxref     dctol     ectol\n")
    f.write(f"{12:10d}\n")
    f.write("*DATABASE_BINARY_D3PLOT\n")
    f.write("$#      dt\n")
    f.write(f"{0.1:10.2f}\n")
    f.write("*DATABASE_GLSTAT\n")
    f.write("$#      dt    binary\n")
    f.write(f"{0.01:10.3f}\n")


def _write_node_set(f, sid: int, title: str, nids: np.ndarray) -> None:
    f.write("*SET_NODE_LIST_TITLE\n")
    f.write(f"{title[:70]}\n")
    f.write("$#     sid       da1       da2       da3       da4    solver\n")
    f.write(f"{sid:10d}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{'MECH':>10s}\n")
    for row in range(0, len(nids), 8):
        chunk = nids[row:row + 8]
        f.write("".join(f"{n:10d}" for n in chunk) + "\n")


def _write_segment_set(f, sid: int, title: str, segs: np.ndarray) -> None:
    """segs: (S, 4) segment corner node ids (already offset); triangles have
    the 4th node repeated."""
    f.write("*SET_SEGMENT_TITLE\n")
    f.write(f"{title[:70]}\n")
    f.write("$#     sid       da1       da2       da3       da4    solver\n")
    f.write(f"{sid:10d}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{'MECH':>10s}\n")
    f.write("$#      n1        n2        n3        n4\n")
    np.savetxt(f, segs, fmt="%10d" * 4)
