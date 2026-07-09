"""LS-DYNA keyword (.k) file writer for TET4/TET10 solid and TRI3/QUAD4
shell meshes.

Formats used (standard fixed-width small format):
  *NODE            nid(I8)  x,y,z(E16.9)
  *ELEMENT_SOLID   TET4:  eid,pid,n1..n8 (10I8), tet as n1 n2 n3 n4 n4 n4 n4 n4
                   TET10: two-line format - eid,pid (2I8) / n1..n10 (10I8)
  *ELEMENT_SHELL   eid,pid,n1..n4 (6I8), triangles as n1 n2 n3 n3
  *SET_NODE_LIST / *SET_SEGMENT   I10 fields

When a node or element id would overflow the 8-character standard fields the
file is automatically written as ``*KEYWORD LONG=Y`` with 20-character fields
throughout (also selectable explicitly via ``long_format=True``).
"""
from __future__ import annotations

import os
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

# largest id that fits the 8-character *NODE/*ELEMENT fields of the standard
# keyword format; beyond it LONG=Y (20-character fields) is required
MAX_STD_ID = 99_999_999


class _Widths:
    """Field widths for the standard or the LONG=Y keyword format."""

    def __init__(self, long_format: bool):
        self.long = long_format
        self.f = 20 if long_format else 10   # card / set fields
        self.n = 20 if long_format else 8    # node & element id fields
        self.c = 20 if long_format else 16   # node coordinate fields

    def ints(self, *vals) -> str:
        return "".join(f"{int(v):{self.f}d}" for v in vals)


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
    mesh_only: bool = False,           # *INCLUDE-friendly file: skip PART/
                                       # SECTION/MAT and control cards
    part_mats: dict[int, dict] | None = None,  # pid -> {"e","pr","ro"}: per-part
                                       # *MAT_ELASTIC (overrides `mat` for that pid)
    contact_fs: float | None = None,   # write *CONTACT_AUTOMATIC_SINGLE_SURFACE
                                       # over all parts with this friction coeff.
    tssfac: float | None = None,       # write *CONTROL_TIMESTEP with this TSSFAC
    body_load: tuple[str, float] | None = None,  # ("x"|"y"|"z", accel) ->
                                       # *LOAD_BODY_ with the unit ramp curve
    long_format: bool = False,         # force LONG=Y 20-char fields (otherwise
                                       # auto-enabled when ids overflow I8)
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
    part_mats = part_mats or {}

    max_id = max(start_nid + n_nodes - 1, start_eid + len(elems) - 1)
    auto_long = max_id > MAX_STD_ID and not long_format
    w = _Widths(long_format or max_id > MAX_STD_ID)

    with open(path, "w", newline="\n") as f:
        f.write("*KEYWORD LONG=Y\n" if w.long else "*KEYWORD\n")
        f.write("*TITLE\n")
        f.write(f"{title[:80]}\n")
        f.write(f"$ Written by k_mesher on {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for c in comments:
            f.write(f"$ {c}\n")
        if auto_long:
            f.write("$ LONG=Y format enabled automatically: ids exceed the "
                    "8-character standard fields\n")

        if mesh_only:
            f.write(f"$ mesh-only file (for *INCLUDE): define *PART {pid}"
                    f"{'' if len(unique_pids) == 1 else f'..{max(unique_pids)}'}"
                    f", *SECTION and *MAT in the master deck.\n")
        else:
            if implicit_cards:
                _write_implicit_cards(f, w)
            if tssfac is not None:
                _write_control_timestep(f, w, tssfac)

            # --- PART / SECTION / MAT ---------------------------------------
            for p in unique_pids:
                mid = p if p in part_mats else pid
                f.write("*PART\n")
                f.write(f"{part_titles.get(p, f'part {p}')[:70]}\n")
                f.write("$#     pid     secid       mid     eosid      hgid"
                        "      grav    adpopt      tmid\n")
                f.write(w.ints(p, pid, mid, 0, 0, 0, 0, 0) + "\n")

            # one material card per referenced MID; per-part materials
            # override the global one for their pid
            mats_by_mid = {}
            if mat is not None:
                mats_by_mid[pid] = mat
            mats_by_mid.update({p: m for p, m in part_mats.items() if m})
            mids_used = {p if p in part_mats else pid for p in unique_pids}
            missing = sorted(mids_used - set(mats_by_mid))
            if missing:
                f.write(f"$ NOTE: define *MAT_ card(s) with MID = "
                        f"{', '.join(str(m) for m in missing)} "
                        f"before running LS-DYNA.\n")

            if element_kind == "solid":
                f.write("*SECTION_SOLID\n")
                f.write("$#   secid    elform       aet\n")
                f.write(w.ints(pid, elform, 0) + "\n")
            else:
                f.write("*SECTION_SHELL\n")
                f.write("$#   secid    elform      shrf       nip     propt"
                        "   qr/irid     icomp     setyp\n")
                f.write(f"{pid:{w.f}d}{elform:{w.f}d}{0.8333:{w.f}.4f}"
                        f"{5:{w.f}d}{1.0:{w.f}.1f}{0:{w.f}d}{0:{w.f}d}"
                        f"{1:{w.f}d}\n")
                f.write("$#      t1        t2        t3        t4      nloc"
                        "     marea      idof    edgset\n")
                f.write(f"{thickness:{w.f}.4g}" * 4 + "\n")

            for mid in sorted(mats_by_mid):
                m = mats_by_mid[mid]
                f.write("*MAT_ELASTIC\n")
                f.write("$#     mid        ro         e        pr        da"
                        "        db\n")
                f.write(f"{mid:{w.f}d}{m['ro']:{w.f}.3e}{m['e']:{w.f}.4g}"
                        f"{m['pr']:{w.f}.4f}{0.0:{w.f}.1f}{0.0:{w.f}.1f}\n")

            if contact_fs is not None:
                _write_contact(f, w, contact_fs)

        # --- NODES ----------------------------------------------------------
        f.write("*NODE\n")
        f.write("$#   nid               x               y               z\n")
        node_block = np.column_stack((nids.astype(float), coords))
        np.savetxt(f, node_block, fmt=f"%{w.n}d" + f"%{w.c}.9e" * 3)

        # --- ELEMENTS ---------------------------------------------------------
        if element_kind == "shell":
            f.write("*ELEMENT_SHELL\n")
            f.write("$#   eid     pid      n1      n2      n3      n4\n")
            elem_block = np.column_stack((eids, part_ids, elem_nids))
            np.savetxt(f, elem_block, fmt=f"%{w.n}d" * 6)
        elif nn == 4:
            f.write("*ELEMENT_SOLID\n")
            f.write("$#   eid     pid      n1      n2      n3      n4      n5"
                    "      n6      n7      n8\n")
            elem_block = np.column_stack(
                (eids, part_ids, elem_nids,
                 elem_nids[:, 3], elem_nids[:, 3], elem_nids[:, 3], elem_nids[:, 3])
            )
            np.savetxt(f, elem_block, fmt=f"%{w.n}d" * 10)
        else:  # TET10, two-line format
            f.write("*ELEMENT_SOLID\n")
            f.write("$#   eid     pid\n")
            f.write("$#    n1      n2      n3      n4      n5      n6      n7"
                    "      n8      n9     n10\n")
            elem_block = np.column_stack((eids, part_ids, elem_nids))
            np.savetxt(f, elem_block, fmt=f"%{w.n}d" * 2 + "\n" + f"%{w.n}d" * 10)

        # --- SETS (symmetry, then picked faces) -------------------------------
        need_curve = (body_load is not None
                      or any(s.get("pressure") is not None
                             or s.get("force") is not None for s in face_sets))
        if need_curve:
            f.write("*DEFINE_CURVE_TITLE\n")
            f.write("k_mesher unit ramp (loads are scaled via SF)\n")
            f.write("$#    lcid      sidr       sfa       sfo      offa      offo"
                    "    dattyp     lcint\n")
            f.write(f"{curve_id:{w.f}d}{0:{w.f}d}{1.0:{w.f}.1f}{1.0:{w.f}.1f}"
                    f"{0.0:{w.f}.1f}{0.0:{w.f}.1f}{0:{w.f}d}{0:{w.f}d}\n")
            f.write(f"{0.0:{2 * w.f}.10g}{0.0:{2 * w.f}.10g}\n")
            f.write(f"{1.0:{2 * w.f}.10g}{1.0:{2 * w.f}.10g}\n")

        if body_load is not None:
            axis, accel = body_load
            f.write(f"*LOAD_BODY_{axis.upper()}\n")
            f.write("$#    lcid        sf    lciddr        xc        yc"
                    "        zc       cid\n")
            f.write(f"{curve_id:{w.f}d}{float(accel):{w.f}.4g}\n")

        sid = start_sid - 1
        for s in sym_sets:
            sid += 1
            set_nids = np.asarray(s["nodes"], dtype=np.int64) - 1 + start_nid
            constraint = (s.get("constraint") or "symmetric").lower()
            suffix = "" if constraint == "symmetric" else f" ({constraint})"
            _write_node_set(f, w, sid, f"SYM_{s['axis'].upper()} plane at "
                                       f"{s['offset']:g}{suffix}", set_nids)
            if s.get("spc", True):
                _write_spc(f, w, sid, _resolve_sym_dofs(s))
            if s.get("segset") is not None and len(s["segset"]):
                sid += 1
                segs = np.asarray(s["segset"], dtype=np.int64) - 1 + start_nid
                _write_segment_set(f, w, sid, f"SYM_{s['axis'].upper()} plane at "
                                              f"{s['offset']:g} segments", segs)

        for s in face_sets:
            sid += 1
            if s["kind"] == "node":
                set_nids = np.asarray(s["nodes"], dtype=np.int64) - 1 + start_nid
                _write_node_set(f, w, sid, s["title"], set_nids)
                if s.get("spc_dofs"):
                    dofs = tuple(1 if str(d) in s["spc_dofs"] else 0
                                 for d in range(1, 7))
                    _write_spc(f, w, sid, dofs)
                if s.get("force") is not None:
                    dof_name, total = s["force"]
                    dof = {"x": 1, "y": 2, "z": 3}[dof_name.lower()]
                    sf = float(total) / max(len(set_nids), 1)
                    f.write(f"$ total force {total:g} in {dof_name.upper()} "
                            f"over {len(set_nids)} nodes -> {sf:g} per node\n")
                    f.write("*LOAD_NODE_SET\n")
                    f.write("$#    nsid       dof      lcid        sf\n")
                    f.write(f"{sid:{w.f}d}{dof:{w.f}d}{curve_id:{w.f}d}"
                            f"{sf:{w.f}.4g}\n")
            else:
                segs = np.asarray(s["segments"], dtype=np.int64) - 1 + start_nid
                _write_segment_set(f, w, sid, s["title"], segs)
                if s.get("pressure") is not None:
                    f.write("*LOAD_SEGMENT_SET\n")
                    f.write("$#    ssid      lcid        sf        at\n")
                    f.write(f"{sid:{w.f}d}{curve_id:{w.f}d}"
                            f"{float(s['pressure']):{w.f}.4g}{0.0:{w.f}.1f}\n")

        # --- ELEMENT SETS (quality check failures etc.) -----------------------
        for s in elem_sets:
            sid += 1
            set_eids = np.asarray(s["eids"], dtype=np.int64) - 1 + start_eid
            if element_kind == "solid":
                f.write("*SET_SOLID_TITLE\n")
                f.write(f"{s['title'][:70]}\n")
                f.write("$#     sid    solver\n")
                f.write(f"{sid:{w.f}d}{'MECH':<{w.f}s}\n")
            else:
                f.write("*SET_SHELL_LIST_TITLE\n")
                f.write(f"{s['title'][:70]}\n")
                f.write("$#     sid       da1       da2       da3       da4"
                        "    solver\n")
                f.write(f"{sid:{w.f}d}" + f"{0.0:{w.f}.1f}" * 4
                        + f"{'MECH':>{w.f}s}\n")
            for row in range(0, len(set_eids), 8):
                chunk = set_eids[row:row + 8]
                f.write("".join(f"{e:{w.f}d}" for e in chunk) + "\n")

        f.write("*END\n")


def write_k_split(
    path: str,
    coords: np.ndarray,
    elems: np.ndarray,
    *,
    part_ids: np.ndarray,              # (M,) per-element part id
    part_titles: dict[int, str] | None = None,
    sym_sets: tuple[dict, ...] = (),
    face_sets: tuple[dict, ...] = (),
    elem_sets: tuple[dict, ...] = (),
    part_mats: dict[int, dict] | None = None,
    contact_fs: float | None = None,   # accepted for signature parity but
                                       # skipped - contact acts BETWEEN parts
    title: str = "k_mesher mesh",
    **kw,                              # remaining write_k keyword arguments
) -> list[tuple[int, str]]:
    """Write one standalone .k file per part: ``<path stem>_p<PID>[_<name>].k``.

    Each file contains only that part's elements with the nodes it uses,
    compactly renumbered from ``start_nid`` - the files are self-contained
    (they do NOT combine into an assembly; use the single-file output with
    start IDs for that). Node/segment/element sets are filtered to the part
    and renumbered; sets that end up empty are omitted. The part's own
    material (``part_mats`` falling back to ``mat``) is written with
    SECID = MID = PID. Contact cards are skipped - contact acts between
    parts. Returns [(pid, file path), ...] in part order.
    """
    part_ids = np.asarray(part_ids, dtype=np.int64)
    part_titles = part_titles or {}
    part_mats = part_mats or {}
    base, ext = os.path.splitext(path)
    written = []
    for p in dict.fromkeys(int(v) for v in part_ids):
        emask = part_ids == p
        pelems = elems[emask]

        # compact node renumbering: old 1-based id -> new 1-based id (0 = absent)
        used = np.zeros(len(coords) + 1, dtype=bool)
        used[pelems.ravel()] = True
        remap = np.zeros(len(coords) + 1, dtype=np.int64)
        remap[used] = np.arange(1, int(used.sum()) + 1)
        pcoords = coords[used[1:]]
        new_elems = remap[pelems]

        def map_nodes(ids):
            mapped = remap[np.asarray(ids, dtype=np.int64)]
            return mapped[mapped > 0]

        def map_segs(segs):
            mapped = remap[np.asarray(segs, dtype=np.int64)]
            return mapped[(mapped > 0).all(axis=1)]

        psym = []
        for s in sym_sets:
            item = dict(s)
            item["nodes"] = map_nodes(s["nodes"])
            if len(item["nodes"]) == 0:
                continue
            if s.get("segset") is not None and len(s["segset"]):
                segs = map_segs(s["segset"])
                item["segset"] = segs if len(segs) else None
            psym.append(item)

        pface = []
        for s in face_sets:
            item = dict(s)
            if s["kind"] == "node":
                item["nodes"] = map_nodes(s["nodes"])
                if len(item["nodes"]) == 0:
                    continue
            else:
                item["segments"] = map_segs(s["segments"])
                if len(item["segments"]) == 0:
                    continue
            pface.append(item)

        # element sets hold 1-based rows into `elems`; renumber to part rows
        row_map = np.zeros(len(elems) + 1, dtype=np.int64)
        old_rows = np.flatnonzero(emask)
        row_map[old_rows + 1] = np.arange(1, len(old_rows) + 1)
        pelem_sets = []
        for s in elem_sets:
            eids = row_map[np.asarray(s["eids"], dtype=np.int64)]
            eids = eids[eids > 0]
            if len(eids):
                pelem_sets.append({**s, "eids": eids})

        name = part_titles.get(p, "")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                       for ch in name).strip("_")[:30]
        ppath = f"{base}_p{p}{'_' + safe if safe else ''}{ext or '.k'}"
        write_k(ppath, pcoords, new_elems, pid=p,
                part_titles={p: name or f"part {p}"},
                title=f"{title} - {name or f'part {p}'}"[:80],
                sym_sets=tuple(psym), face_sets=tuple(pface),
                elem_sets=tuple(pelem_sets),
                part_mats={p: part_mats[p]} if p in part_mats else {},
                **kw)
        written.append((p, ppath))
    return written


def _write_spc(f, w: _Widths, sid: int, dofs) -> None:
    f.write("*BOUNDARY_SPC_SET\n")
    f.write("$#    nsid       cid      dofx      dofy      dofz"
            "     dofrx     dofry     dofrz\n")
    f.write(w.ints(sid, 0, *dofs) + "\n")


def _write_control_timestep(f, w: _Widths, tssfac: float) -> None:
    f.write("*CONTROL_TIMESTEP\n")
    f.write("$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm"
            "     erode     ms1st\n")
    f.write(f"{0.0:{w.f}.1f}{float(tssfac):{w.f}.4f}\n")


def _write_contact(f, w: _Widths, fs: float) -> None:
    """Single-surface contact over all parts (SSID 0). Zeros on cards 2/3
    mean LS-DYNA defaults; only the friction coefficients are set."""
    f.write("*CONTACT_AUTOMATIC_SINGLE_SURFACE\n")
    f.write("$#    ssid      msid     sstyp     mstyp    sboxid    mboxid"
            "       spr       mpr\n")
    f.write(w.ints(0, 0, 2, 0, 0, 0, 0, 0) + "\n")
    f.write("$#      fs        fd        dc        vc       vdc    penchk"
            "        bt        dt\n")
    f.write(f"{float(fs):{w.f}.4f}{float(fs):{w.f}.4f}"
            + f"{0.0:{w.f}.1f}" * 6 + "\n")
    f.write("$#     sfs       sfm       sst       mst      sfst      sfmt"
            "       fsf       vsf\n")
    f.write(f"{0.0:{w.f}.1f}" * 8 + "\n")


def _write_implicit_cards(f, w: _Widths) -> None:
    """Minimal implicit static setup: 1.0 s of pseudo-time, auto stepping,
    nonlinear solver, d3plot output. Review before production use."""
    f.write("$ --- basic implicit static setup (review before production) ---\n")
    f.write("*CONTROL_TERMINATION\n")
    f.write("$#  endtim    endcyc     dtmin    endeng    endmas\n")
    f.write(f"{1.0:{w.f}.1f}\n")
    f.write("*CONTROL_IMPLICIT_GENERAL\n")
    f.write("$#  imflag       dt0    imform      nsbs       igs     cnstn      form\n")
    f.write(f"{1:{w.f}d}{0.1:{w.f}.2f}\n")
    f.write("*CONTROL_IMPLICIT_AUTO\n")
    f.write("$#   iauto    iteopt    itewin     dtmin     dtmax\n")
    f.write(f"{1:{w.f}d}\n")
    f.write("*CONTROL_IMPLICIT_SOLUTION\n")
    f.write("$#  nsolvr    ilimit    maxref     dctol     ectol\n")
    f.write(f"{12:{w.f}d}\n")
    f.write("*DATABASE_BINARY_D3PLOT\n")
    f.write("$#      dt\n")
    f.write(f"{0.1:{w.f}.2f}\n")
    f.write("*DATABASE_GLSTAT\n")
    f.write("$#      dt    binary\n")
    f.write(f"{0.01:{w.f}.3f}\n")


def _write_node_set(f, w: _Widths, sid: int, title: str, nids: np.ndarray) -> None:
    f.write("*SET_NODE_LIST_TITLE\n")
    f.write(f"{title[:70]}\n")
    f.write("$#     sid       da1       da2       da3       da4    solver\n")
    f.write(f"{sid:{w.f}d}" + f"{0.0:{w.f}.1f}" * 4 + f"{'MECH':>{w.f}s}\n")
    for row in range(0, len(nids), 8):
        chunk = nids[row:row + 8]
        f.write("".join(f"{n:{w.f}d}" for n in chunk) + "\n")


def _write_segment_set(f, w: _Widths, sid: int, title: str, segs: np.ndarray) -> None:
    """segs: (S, 4) segment corner node ids (already offset); triangles have
    the 4th node repeated."""
    f.write("*SET_SEGMENT_TITLE\n")
    f.write(f"{title[:70]}\n")
    f.write("$#     sid       da1       da2       da3       da4    solver\n")
    f.write(f"{sid:{w.f}d}" + f"{0.0:{w.f}.1f}" * 4 + f"{'MECH':>{w.f}s}\n")
    f.write("$#      n1        n2        n3        n4\n")
    np.savetxt(f, segs, fmt=f"%{w.f}d" * 4)
