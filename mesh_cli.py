"""Command-line interface: mesh a CAD/mesh file to an LS-DYNA .k without the GUI.

Supported inputs: STEP (.step/.stp), IGES (.iges/.igs), BREP (.brep/.brp) CAD,
and STL/OBJ/PLY surface tessellations. A watertight tessellation can also be
tetrahedralized (TET4/TET10) by reconstructing the enclosed volume.

Examples:
    python mesh_cli.py part.stp --size-max 8 --size-min 1
    python mesh_cli.py part.iges -o half.k --sym x --mat 210000:0.3:7.85e-9
    python mesh_cli.py part.brep --tet10 --algo hxt --sym x:0:+ --sym y:5:-
    python mesh_cli.py part.stp --sym x --sym-constraint antisymmetric
    python mesh_cli.py part.stp --sym x --no-sym-spc      # node set only
    python mesh_cli.py part.stl --etype tri3 --thickness 1.2
    python mesh_cli.py part.stp --list-faces
    python mesh_cli.py part.stp --face-nodeset 7 --face-segset 12
    python mesh_cli.py part.stp --refine-sphere 0:0:0:15:1.5 --refine-box 0:0:0:10:10:10:2
    python mesh_cli.py asm.stp --glue --mat --part-mat 2:70000:0.33:2.7e-9
    python mesh_cli.py asm.stp --contact 0.15 --tssfac 0.9 --gravity z:9810
    python mesh_cli.py part.stp --mesh-only --stats-json part_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

import dyna_writer
import mesher


def _json_default(o):
    """json.dump fallback for the numpy scalars/arrays in the mesh stats."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")

ALGO_CHOICES = {
    "delaunay": "Delaunay",
    "frontal": "Frontal",
    "hxt": "HXT (parallel Delaunay)",
}
STEEL_MMTS = {"e": 210000.0, "pr": 0.3, "ro": 7.85e-9}  # steel, mm-t-s units


def parse_sym(text: str) -> mesher.SymmetryPlane:
    """axis[:offset[:side]] e.g. 'x', 'x:10', 'x:10:-'"""
    parts = text.split(":")
    axis = parts[0].strip().lower()
    if axis not in ("x", "y", "z"):
        raise argparse.ArgumentTypeError(f"symmetry axis must be x, y or z: {text!r}")
    try:
        offset = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad symmetry offset in {text!r}") from None
    keep = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "+"
    if keep not in ("+", "-"):
        raise argparse.ArgumentTypeError(f"symmetry side must be + or -: {text!r}")
    return mesher.SymmetryPlane(axis=axis, offset=offset, keep=keep)


def parse_mat(text: str) -> dict:
    """E[:nu[:rho]] e.g. '210000:0.3:7.85e-9'; empty -> steel mm-t-s"""
    if not text:
        return dict(STEEL_MMTS)
    parts = text.split(":")
    try:
        mat = dict(STEEL_MMTS)
        mat["e"] = float(parts[0])
        if len(parts) > 1:
            mat["pr"] = float(parts[1])
        if len(parts) > 2:
            mat["ro"] = float(parts[2])
        return mat
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad material spec {text!r}, "
                                         f"expected E[:nu[:rho]]") from None


def parse_part_mat(text: str) -> tuple[int, dict]:
    """BODY:E[:NU[:RHO]] - per-body *MAT_ELASTIC override; BODY is the
    1-based body index (missing values default to steel mm-t-s)."""
    parts = text.split(":")
    try:
        body = int(parts[0])
        if body < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"part-mat body index must be a positive integer: {text!r}") from None
    return body, parse_mat(":".join(parts[1:]))


def parse_gravity(text: str) -> tuple[str, float]:
    """AXIS:ACCEL e.g. 'z:-9810' (mm/s^2 in mm-t-s units)."""
    parts = text.split(":")
    axis = parts[0].strip().lower()
    if axis not in ("x", "y", "z") or len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"gravity must be AXIS:ACCEL with axis x/y/z: {text!r}")
    try:
        return axis, float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad gravity acceleration in {text!r}") from None


def _parse_floats(text: str, n: int, what: str) -> list[float]:
    try:
        vals = [float(v) for v in text.split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad {what}: {text!r}") from None
    if len(vals) != n:
        raise argparse.ArgumentTypeError(
            f"{what} needs {n} colon-separated numbers, got {len(vals)}: {text!r}")
    return vals


def parse_refine_sphere(text: str) -> dict:
    v = _parse_floats(text, 5, "sphere refinement (cx:cy:cz:r:size)")
    return {"kind": "sphere", "params": v[:4], "size": v[4]}


def parse_refine_box(text: str) -> dict:
    v = _parse_floats(text, 7, "box refinement (x0:y0:z0:x1:y1:z1:size)")
    return {"kind": "box", "params": v[:6], "size": v[6]}


def _parse_tag_value(text: str) -> tuple[int, float]:
    parts = text.split(":")
    try:
        return int(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"expected TAG:VALUE, got {text!r}") from None


def _parse_spc(text: str) -> tuple[int, str]:
    parts = text.split(":")
    try:
        tag = int(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad SPC face tag: {text!r}") from None
    dofs = parts[1] if len(parts) > 1 and parts[1] else "123456"
    if any(ch not in "123456" for ch in dofs):
        raise argparse.ArgumentTypeError(f"SPC DOFs must be digits 1-6: {text!r}")
    return tag, dofs


def _parse_force(text: str) -> tuple[int, str, float]:
    parts = text.split(":")
    try:
        tag, axis, total = int(parts[0]), parts[1].lower(), float(parts[2])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"expected TAG:AXIS:TOTAL, got {text!r}") from None
    if axis not in ("x", "y", "z"):
        raise argparse.ArgumentTypeError(f"force axis must be x, y or z: {text!r}")
    return tag, axis, total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mesh a STEP/IGES/BREP/STL file to TET4/TET10/TRI3/QUAD4 "
                    "and write an LS-DYNA .k file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="CAD/mesh input: STEP/IGES/BREP solid or "
                                 "surface, or an STL/OBJ/PLY tessellation "
                                 "(shells, or tets if watertight)")
    p.add_argument("-o", "--output", help="output .k file (default: input with .k)")
    p.add_argument("--etype", choices=["tet4", "tet10", "tri3", "quad4"],
                   default="tet4",
                   help="element type: solid tets or shells (tri3 / "
                        "quad-dominant)")
    p.add_argument("--tet10", action="store_true",
                   help="deprecated alias for --etype tet10")
    p.add_argument("--thickness", type=float, default=1.0,
                   help="shell thickness (shell element types only)")
    p.add_argument("--size-max", type=float, default=10.0, help="max element size")
    p.add_argument("--size-min", type=float, default=0.0, help="min element size")
    p.add_argument("--curvature", type=int, default=16, metavar="N",
                   help="elements per 2*pi for curvature refinement, 0 = off")
    p.add_argument("--algo", choices=ALGO_CHOICES, default="delaunay",
                   help="3D meshing algorithm")
    p.add_argument("--no-optimize", action="store_true",
                   help="skip mesh quality optimization")
    p.add_argument("--heal", action="store_true", help="heal geometry on import")
    p.add_argument("--glue", action="store_true",
                   help="glue touching bodies (conformal shared nodes)")
    p.add_argument("--unit", choices=["", "MM", "CM", "M", "IN"], default="",
                   help="force geometry unit conversion ('' = keep file units)")
    p.add_argument("--sym", type=parse_sym, action="append", default=[],
                   metavar="AXIS[:OFFSET[:SIDE]]",
                   help="symmetry plane, repeatable, e.g. x, x:10, x:0:-")
    p.add_argument("--no-sym-sets", action="store_true",
                   help="do not write node sets / SPC for symmetry planes")
    p.add_argument("--no-sym-spc", action="store_true",
                   help="write the symmetry-plane node set(s) but NOT the "
                        "*BOUNDARY_SPC_SET boundary condition")
    p.add_argument("--sym-constraint",
                   choices=["symmetric", "antisymmetric", "fixed"],
                   default="symmetric",
                   help="symmetry-plane BC type: symmetric (normal translation "
                        "+ in-plane rotations), antisymmetric (the complement), "
                        "or fixed (123456)")
    p.add_argument("--sym-dofs", metavar="DOFS", default=None,
                   help="custom SPC DOF digits for the symmetry planes "
                        "(e.g. 13), overriding --sym-constraint")
    p.add_argument("--sym-segset", action="store_true",
                   help="also write a *SET_SEGMENT for the symmetry-plane "
                        "faces (solids only)")
    p.add_argument("--refine-sphere", type=parse_refine_sphere, action="append",
                   dest="refinements", default=[], metavar="CX:CY:CZ:R:SIZE",
                   help="spherical refinement region, repeatable")
    p.add_argument("--refine-box", type=parse_refine_box, action="append",
                   dest="refinements", metavar="X0:Y0:Z0:X1:Y1:Z1:SIZE",
                   help="box refinement region, repeatable")
    p.add_argument("--list-faces", action="store_true",
                   help="list the model faces (tag, type, area, centroid) and exit")
    p.add_argument("--face-nodeset", type=int, action="append", default=[],
                   metavar="TAG", help="write *SET_NODE_LIST for this face tag "
                                       "(see --list-faces), repeatable")
    p.add_argument("--face-segset", type=int, action="append", default=[],
                   metavar="TAG", help="write *SET_SEGMENT for this face tag, "
                                       "repeatable")
    p.add_argument("--defeature", type=int, action="append", default=[],
                   metavar="TAG", help="remove this face (hole/fillet) from the "
                                       "solid before meshing, repeatable")
    p.add_argument("--face-size", type=_parse_tag_value, action="append",
                   default=[], metavar="TAG:SIZE",
                   help="local element size on a face, repeatable")
    p.add_argument("--spc", type=_parse_spc, action="append", default=[],
                   metavar="TAG[:DOFS]",
                   help="fix a face: *BOUNDARY_SPC_SET (default DOFs 123456)")
    p.add_argument("--pressure", type=_parse_tag_value, action="append",
                   default=[], metavar="TAG:VALUE",
                   help="pressure on a face via *LOAD_SEGMENT_SET, repeatable")
    p.add_argument("--force", type=_parse_force, action="append", default=[],
                   metavar="TAG:AXIS:TOTAL",
                   help="total force on a face's nodes via *LOAD_NODE_SET, "
                        "e.g. 12:z:-500")
    p.add_argument("--auto-refine", action="store_true",
                   help="remesh with local refinement at bad spots (max 2 rounds)")
    p.add_argument("--implicit-cards", action="store_true",
                   help="write basic implicit static control cards")
    p.add_argument("--mesh-only", action="store_true",
                   help="write an *INCLUDE-friendly file: nodes/elements/sets "
                        "only, no PART/SECTION/MAT or control cards "
                        "(--implicit-cards and --mat cards are suppressed)")
    p.add_argument("--no-qa-sets", action="store_true",
                   help="do not write the quality-failure element set")
    p.add_argument("--title", default=None,
                   help="deck *TITLE (default: output file name)")
    p.add_argument("--stats-json", metavar="PATH",
                   help="write the mesh statistics (counts, quality criteria, "
                        "mass properties, timestep estimate) as JSON")
    p.add_argument("--pid", type=int, default=1,
                   help="base part ID (multiple bodies get PID, PID+1, ...)")
    p.add_argument("--elform", type=int, choices=[2, 4, 10, 13, 16, 17],
                   default=None,
                   help="section formulation (defaults: TET4 10, TET10 16, "
                        "TRI3 4, QUAD4 16)")
    p.add_argument("--start-nid", type=int, default=1, help="first node ID")
    p.add_argument("--start-eid", type=int, default=1, help="first element ID")
    p.add_argument("--start-sid", type=int, default=1,
                   help="first set ID (node/segment/SPC/element sets)")
    p.add_argument("--mat", type=parse_mat, nargs="?", const=dict(STEEL_MMTS),
                   default=None, metavar="E[:NU[:RHO]]",
                   help="write *MAT_ELASTIC (bare --mat = steel in mm-t-s: "
                        "210000:0.3:7.85e-9)")
    p.add_argument("--part-mat", type=parse_part_mat, action="append",
                   default=[], metavar="BODY:E[:NU[:RHO]]",
                   help="per-body *MAT_ELASTIC override for multi-body models "
                        "(BODY is the 1-based body index), repeatable; bodies "
                        "without an override use --mat")
    p.add_argument("--contact", type=float, nargs="?", const=0.0, default=None,
                   metavar="FS",
                   help="write *CONTACT_AUTOMATIC_SINGLE_SURFACE over all "
                        "parts, optionally with friction coefficient FS "
                        "(for bonded bodies use --glue instead)")
    p.add_argument("--tssfac", type=float, default=None, metavar="F",
                   help="write *CONTROL_TIMESTEP with this TSSFAC "
                        "(e.g. 0.9)")
    p.add_argument("--gravity", type=parse_gravity, default=None,
                   metavar="AXIS:ACCEL",
                   help="body load via *LOAD_BODY_ with the unit ramp curve, "
                        "e.g. z:9810 (note: LOAD_BODY acts opposite to the "
                        "axis for positive values)")
    p.add_argument("--long-format", action="store_true",
                   help="write LONG=Y keyword format (20-char fields); "
                        "enabled automatically when ids exceed 8 characters")
    p.add_argument("--preview", action="store_true",
                   help="open the Gmsh viewer on the result")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    if args.tet10 and args.etype != "tet4":
        print(f"warning: --tet10 (deprecated) overrides --etype {args.etype}; "
              f"meshing TET10", file=sys.stderr)
    etype = "TET10" if args.tet10 else args.etype.upper()
    defaults = {"TET4": 10, "TET10": 16, "TRI3": 4, "QUAD4": 16}
    allowed = {"TET4": (10, 13), "TET10": (16, 17),
               "TRI3": (4, 17), "QUAD4": (2, 16)}
    elform = args.elform if args.elform is not None else defaults[etype]
    if elform not in allowed[etype]:
        print(f"error: ELFORM {elform} does not match {etype} "
              f"(allowed: {allowed[etype]})", file=sys.stderr)
        return 2
    if args.sym_dofs and any(ch not in "123456" for ch in args.sym_dofs):
        print(f"error: --sym-dofs must be digits 1-6: {args.sym_dofs!r}",
              file=sys.stderr)
        return 2
    is_shell = mesher.ETYPES[etype]["family"] == "shell"

    face_tags = sorted(set(args.face_nodeset) | set(args.face_segset)
                       | {t for t, _ in args.spc} | {t for t, _ in args.pressure}
                       | {t for t, _, _ in args.force})
    settings = mesher.MeshSettings(
        step_file=args.input,
        element_type=etype,
        size_max=args.size_max,
        size_min=args.size_min,
        curvature_refine=args.curvature > 0,
        curvature_elems=max(args.curvature, 1),
        algorithm3d=ALGO_CHOICES[args.algo],
        optimize=not args.no_optimize,
        heal=args.heal,
        glue=args.glue,
        occ_unit=args.unit,
        symmetry=args.sym,
        refinements=args.refinements,
        face_sizes=dict(args.face_size),
        defeature_faces=args.defeature,
        collect_faces=face_tags,
        auto_refine=args.auto_refine,
    )

    if args.list_faces:
        try:
            faces = mesher.list_faces(settings, log=lambda m: None)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"{'tag':>5}  {'type':<10}  {'area':>12}  {'diag':>10}  "
              f"{'centroid':>32}  name")
        for f in faces:
            c = f["centroid"]
            print(f"{f['tag']:>5}  {f.get('type', ''):<10}  {f['area']:>12.5g}  "
                  f"{f.get('diag', 0):>10.4g}  "
                  f"({c[0]:>9.3f}, {c[1]:>9.3f}, {c[2]:>9.3f})  {f['name']}")
        return 0

    out = args.output or os.path.splitext(args.input)[0] + ".k"
    preview_path = os.path.splitext(out)[0] + "_preview.msh"
    try:
        result = mesher.mesh_step_auto(
            settings, log=print,
            preview_path=preview_path if args.preview else None)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    sym_sets = []
    if not args.no_sym_sets:
        for sp in settings.symmetry:
            item = {"axis": sp.axis, "offset": sp.offset,
                    "nodes": result.sym_nodes[sp.axis],
                    "spc": not args.no_sym_spc,
                    "constraint": args.sym_constraint}
            if args.sym_dofs:
                item["dofs"] = args.sym_dofs
            if args.sym_segset and len(result.sym_segs.get(sp.axis, ())) > 0:
                item["segset"] = result.sym_segs[sp.axis]
            sym_sets.append(item)

    face_sets = []
    for tag in face_tags:
        if tag in args.face_nodeset and tag in result.face_nodes:
            face_sets.append({"kind": "node", "title": f"FACE_{tag}",
                              "nodes": result.face_nodes[tag]})
        if tag in args.face_segset and len(result.face_segs.get(tag, ())) > 0:
            face_sets.append({"kind": "segment", "title": f"FACE_{tag}",
                              "segments": result.face_segs[tag]})
    for tag, dofs in args.spc:
        if tag in result.face_nodes:
            face_sets.append({"kind": "node", "title": f"SPC_FACE_{tag}",
                              "nodes": result.face_nodes[tag], "spc_dofs": dofs})
    for tag, value in args.pressure:
        if len(result.face_segs.get(tag, ())) > 0:
            face_sets.append({"kind": "segment", "title": f"PRES_FACE_{tag}",
                              "segments": result.face_segs[tag], "pressure": value})
    for tag, axis, total in args.force:
        if tag in result.face_nodes:
            face_sets.append({"kind": "node", "title": f"FORCE_FACE_{tag}",
                              "nodes": result.face_nodes[tag],
                              "force": (axis, total)})

    elem_sets = []
    failed = result.stats.get("failed_elems", ())
    if not args.no_qa_sets and len(failed):
        elem_sets.append({"title": "QA quality-criteria failures",
                          "eids": failed + 1})

    n_bodies = len(result.part_names)
    part_mats = {}
    for body, m in args.part_mat:
        if body > n_bodies:
            print(f"warning: --part-mat body {body} ignored - the model has "
                  f"only {n_bodies} body/bodies", file=sys.stderr)
            continue
        part_mats[args.pid + body - 1] = m

    mass, part_masses, dt_est = mesher.mass_and_timestep(
        result, etype, args.mat, part_mats, args.pid,
        args.thickness if is_shell else 1.0, log=print)

    part_ids = args.pid + result.elem_parts
    part_titles = {args.pid + i: (name or f"body {i + 1}")
                   for i, name in enumerate(result.part_names)}

    comments = [f"Source geometry: {args.input}",
                f"Element size: {args.size_min:g} .. {args.size_max:g}"]
    for sp in settings.symmetry:
        comments.append(f"Symmetry: {sp.axis.upper()} = {sp.offset:g}, "
                        f"kept '{sp.keep}' side")

    dyna_writer.write_k(
        out, result.coords, result.elems,
        element_kind="shell" if is_shell else "solid",
        pid=args.pid, elform=elform, thickness=args.thickness,
        start_nid=args.start_nid, start_eid=args.start_eid,
        start_sid=args.start_sid,
        title=args.title or os.path.splitext(os.path.basename(out))[0],
        comments=tuple(comments), sym_sets=tuple(sym_sets), mat=args.mat,
        part_ids=part_ids, part_titles=part_titles, face_sets=tuple(face_sets),
        elem_sets=tuple(elem_sets), implicit_cards=args.implicit_cards,
        mesh_only=args.mesh_only, part_mats=part_mats,
        contact_fs=args.contact, tssfac=args.tssfac, body_load=args.gravity,
        long_format=args.long_format,
    )
    print(f"Wrote {out}")

    if args.stats_json:
        payload = {
            "input": args.input, "output": out,
            "element_type": etype, "elform": elform,
            "n_parts": len(result.part_names), "part_names": result.part_names,
            "mass": mass, "part_masses": part_masses,
            "critical_timestep": dt_est,
            "material": args.mat, "part_materials": part_mats,
            "stats": result.stats,
        }
        with open(args.stats_json, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"Wrote {args.stats_json}")

    if args.preview:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview.py")
        subprocess.Popen([sys.executable, script, preview_path])
    return 0


if __name__ == "__main__":
    sys.exit(main())
