"""Execute one mesh + write job from plain-data settings/options.

This is the meshing/writing core behind the GUI's "Generate mesh" button and
its batch queue. It deliberately does not touch tkinter, so it can run in
multiprocessing worker processes (parallel batch) on every platform and be
tested headlessly.

``kopts`` keys (all plain data, picklable):
    pid, elform, element_kind, thickness, start_nid, start_eid, start_sid,
    sym_nodeset, sym_spc, sym_constraint, sym_dofs, sym_segset,
    mat, part_mats, face_nodesets, face_segsets, face_roles, plain_faces,
    face_titles, coord_sets, implicit_cards, qa_sets, mesh_only, split_mode
    (None | "parts" | "include"), contact_fs, tssfac, gravity

Mesh-time features (all read with a safe default, so a caller that omits them
still works):
    midsurface -> bool. Mesh thin plate solids as a midsurface shell instead of
        mesh_step_auto; forces element_kind="shell" and takes the thickness from
        the detected wall thickness (stats["midsurface_thickness"]).
    auto_connect -> None | {"mode": "spotweld"|"tied", "tol": float|None,
        "spacing": float|None, "fs": float}. After meshing a multi-body model,
        detect the interfaces and inject *CONSTRAINED_SPOTWELD pairs (spotweld)
        or the interface segment sets + *CONTACT_TIED_SURFACE_TO_SURFACE (tied)
        into the write_k / write_k_include calls (never the split files).

Optional LS-DYNA control / load cards (all read with a safe default, so a
caller that omits them still works):
    endtim, mass_scale, hourglass, control_energy, databases,
    initial_velocity, contacts, define_curves, prescribed_motion,
    rigidwalls, spotwelds
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np

import connections
import dyna_writer
import mesher


def run_job(settings, out: str, kopts: dict, log=print) -> str:
    """Mesh + write one job. Returns the preview file path."""
    preview_path = os.path.splitext(out)[0] + "_preview.msh"
    t_start = time.perf_counter()

    if settings.defeature_faces and (kopts["face_roles"]
                                     or settings.face_sizes
                                     or kopts["plain_faces"]):
        log("NOTE: defeaturing changes face tags - verify that the other "
            "face assignments still point at the intended faces "
            "(rescan shows the post-defeature tags).")

    midsurface = kopts.get("midsurface", False)
    if midsurface:
        result = mesher.midsurface_shell(settings, log=log,
                                         preview_path=preview_path)
        element_kind = "shell"
        thickness = result.stats["midsurface_thickness"]
        log(f"Midsurface thickness: {thickness:.4g}")
        # a solid element_type on the settings still needs a shell type for the
        # mass/timestep estimate below (the extracted mesh is shells)
        elem_type = (settings.element_type
                     if mesher.ETYPES[settings.element_type]["family"] == "shell"
                     else "TRI3")
    else:
        result = mesher.mesh_step_auto(settings, log=log,
                                       preview_path=preview_path)
        element_kind = kopts["element_kind"]
        thickness = kopts["thickness"]
        elem_type = settings.element_type

    sym_sets = []
    # segset alone still needs the sym_sets items (the node set it comes
    # with is the carrier the writer requires)
    if kopts["sym_nodeset"] or kopts["sym_spc"] or kopts["sym_segset"]:
        for sp in settings.symmetry:
            item = {"axis": sp.axis, "offset": sp.offset,
                    "nodes": result.sym_nodes[sp.axis],
                    "spc": kopts["sym_spc"],
                    "constraint": kopts["sym_constraint"]}
            if kopts["sym_constraint"] == "custom":
                item["dofs"] = kopts["sym_dofs"]
            if kopts["sym_segset"] and len(result.sym_segs.get(sp.axis, ())) > 0:
                item["segset"] = result.sym_segs[sp.axis]
            sym_sets.append(item)

    face_titles = kopts.get("face_titles") or {}
    face_sets = []
    for tag in kopts["plain_faces"]:
        name = face_titles.get(tag, "")
        base = f"FACE_{tag}" + (f" {name}" if name else "")
        if kopts["face_nodesets"] and tag in result.face_nodes:
            face_sets.append({"kind": "node", "title": base,
                              "nodes": result.face_nodes[tag]})
        if kopts["face_segsets"] and len(result.face_segs.get(tag, ())) > 0:
            face_sets.append({"kind": "segment", "title": base,
                              "segments": result.face_segs[tag]})
    for r in kopts["face_roles"]:
        tag = r["tag"]
        if r["role"] == "Fix (SPC)" and tag in result.face_nodes:
            face_sets.append({"kind": "node", "title": f"SPC_FACE_{tag}",
                              "nodes": result.face_nodes[tag],
                              "spc_dofs": r["dofs"]})
        elif r["role"] == "Pressure" and len(result.face_segs.get(tag, ())) > 0:
            face_sets.append({"kind": "segment", "title": f"PRES_FACE_{tag}",
                              "segments": result.face_segs[tag],
                              "pressure": r["value"]})
        elif r["role"].startswith("Force") and tag in result.face_nodes:
            axis = r["role"][-1].lower()
            face_sets.append({"kind": "node", "title": f"FORCE_FACE_{tag}",
                              "nodes": result.face_nodes[tag],
                              "force": (axis, r["value"])})

    # coordinate-based node sets (work for STL/OBJ/PLY too - no CAD faces)
    for i, cs in enumerate(kopts.get("coord_sets") or (), start=1):
        label = cs.get("title") or f"NSET_{i}_{cs['kind'].upper()}"
        ids = mesher.select_nodes(result.coords, cs["kind"], cs["params"],
                                  tol=cs.get("tol"))
        if len(ids) == 0:
            log(f"WARNING: coordinate set {label} matched no nodes - skipped")
            continue
        item = {"kind": "node", "title": label, "nodes": ids}
        role = cs.get("role") or "set"
        if role == "spc":
            item["spc_dofs"] = cs.get("dofs") or "123456"
        elif role == "force":
            item["force"] = (cs["axis"], cs["value"])
        face_sets.append(item)
        log(f"Coordinate set {label}: {len(ids)} nodes")

    # automatic connection detection (spotwelds / tied contact go to the
    # single-file and *INCLUDE-master output only, never the split files -
    # like contact); tied segment sets are merged into face_sets
    auto_spotwelds = ()
    auto_contacts = ()
    auto = kopts.get("auto_connect")
    if auto:
        tol = auto.get("tol")
        if auto.get("mode") == "spotweld":
            auto_spotwelds = tuple(connections.spotweld_pairs(
                result, tol=tol, spacing=auto.get("spacing")))
            if auto_spotwelds:
                log(f"Auto-spotweld: {len(auto_spotwelds)} weld(s) on the "
                    f"detected interfaces")
            else:
                log("WARNING: auto_connect spotweld found no interfaces "
                    "(single body or no bodies within tolerance)")
        elif auto.get("mode") == "tied":
            tie_sets, auto_contacts = connections.tied_contact(
                result, tol=tol, fs=auto.get("fs", 0.0))
            if tie_sets:
                face_sets.extend(tie_sets)
                log(f"Tied contact: {len(tie_sets)} interface segment set(s)")
            else:
                log("WARNING: auto_connect tied found no interfaces "
                    "(single body or no bodies within tolerance)")

    elem_sets = []
    failed = result.stats.get("failed_elems", ())
    if kopts["qa_sets"] and len(failed):
        elem_sets.append({"title": "QA quality-criteria failures",
                          "eids": (failed + 1)})

    pid0 = kopts["pid"]
    part_ids = pid0 + result.elem_parts
    part_titles = {pid0 + i: (name or f"body {i + 1}")
                   for i, name in enumerate(result.part_names)}
    if len(result.part_names) > 1:
        for p, t in part_titles.items():
            log(f"  PID {p}: {t}")

    thickness_scale = thickness if element_kind == "shell" else 1.0
    mesher.mass_and_timestep(result, elem_type, kopts["mat"],
                             kopts["part_mats"], pid0, thickness_scale,
                             log=log)
    if kopts["mat"] and not kopts["part_mats"]:
        # inertia scaling only well-defined for a homogeneous material
        scale = kopts["mat"]["ro"] * thickness_scale
        inertia = np.array(result.stats["inertia_unit_density"]) * scale
        log(f"Inertia about COG (Ixx, Iyy, Izz): "
            f"{inertia[0, 0]:.6g}, {inertia[1, 1]:.6g}, {inertia[2, 2]:.6g}")

    comments = [f"Source geometry: {settings.step_file}",
                f"Element size: {settings.size_min:g} .. {settings.size_max:g}"]
    for sp in settings.symmetry:
        comments.append(f"Symmetry: {sp.axis.upper()} = {sp.offset:g}, "
                        f"kept '{sp.keep}' side")

    title = os.path.splitext(os.path.basename(out))[0]
    common = dict(
        element_kind=element_kind, elform=kopts["elform"],
        thickness=thickness, start_nid=kopts["start_nid"],
        start_eid=kopts["start_eid"], start_sid=kopts["start_sid"],
        comments=tuple(comments), sym_sets=tuple(sym_sets),
        mat=kopts["mat"], part_mats=kopts["part_mats"],
        face_sets=tuple(face_sets), elem_sets=tuple(elem_sets),
        implicit_cards=kopts["implicit_cards"], mesh_only=kopts["mesh_only"],
        tssfac=kopts["tssfac"], body_load=kopts["gravity"],
        # optional LS-DYNA control / load cards (default: no-op)
        endtim=kopts.get("endtim"), mass_scale=kopts.get("mass_scale"),
        hourglass=kopts.get("hourglass"),
        control_energy=kopts.get("control_energy", False),
        initial_velocity=kopts.get("initial_velocity"),
        define_curves=tuple(kopts.get("define_curves") or ()),
        prescribed_motion=tuple(kopts.get("prescribed_motion") or ()),
        rigidwalls=tuple(kopts.get("rigidwalls") or ()),
    )
    databases = kopts.get("databases")
    if databases:
        common["databases"] = databases
    # spotwelds/contacts stay out of `common` so the split files omit them
    spotwelds = tuple(kopts.get("spotwelds") or ()) + auto_spotwelds
    contacts = tuple(kopts.get("contacts") or ()) + auto_contacts
    split_mode = kopts.get("split_mode")
    log(f"Writing LS-DYNA keyword file: {out}")
    if split_mode == "include":
        files = dyna_writer.write_k_include(
            out, result.coords, result.elems, pid=pid0,
            part_ids=part_ids, part_titles=part_titles, title=title,
            contact_fs=kopts["contact_fs"], contacts=contacts,
            spotwelds=spotwelds, **common)
        for p, fpath in files:
            log(f"Wrote mesh fragment: {fpath} (PID {p})")
        log(f"Master deck with *INCLUDE cards: {out}")
    else:
        dyna_writer.write_k(
            out, result.coords, result.elems, pid=pid0,
            part_ids=part_ids, part_titles=part_titles, title=title,
            contact_fs=kopts["contact_fs"], contacts=contacts,
            spotwelds=spotwelds, **common)
        if split_mode == "parts":
            files = dyna_writer.write_k_split(
                out, result.coords, result.elems,
                part_ids=part_ids, part_titles=part_titles, title=title,
                **common)
            for p, fpath in files:
                log(f"Wrote per-part file: {fpath} (PID {p})")

    log(f"Done in {time.perf_counter() - t_start:.1f} s. "
        f"{result.stats['n_nodes']} nodes / "
        f"{result.stats['n_elems']} {elem_type} "
        f"elements written.")
    return preview_path


def run_job_captured(args):
    """Worker-process entry for the parallel batch queue: run one job with
    the log captured. args = (label, settings, out, kopts); returns
    (label, ok, log text, preview path or None)."""
    label, settings, out, kopts = args
    lines = []
    try:
        preview = run_job(settings, out, kopts,
                          log=lambda m: lines.append(str(m)))
        return label, True, "\n".join(lines), preview
    except Exception as e:
        lines.append(f"ERROR: {e}")
        lines.append(traceback.format_exc())
        return label, False, "\n".join(lines), None
