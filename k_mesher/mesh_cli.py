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
    python mesh_cli.py asm.stp --split-include --mat --part-rigid 2
    python mesh_cli.py part.stl --etype tet4 --nset plane,z,0,spc=123 \
                                --nset sphere,0,0,40,15,force=z:-500
    python mesh_cli.py part.stp --export part.vtk --mat --target-dt 5e-7
    python mesh_cli.py asm.stp --auto-contact --contact 0.1
    python mesh_cli.py part.stp --cross-section 0:0:0:1:0:0 --history-node 1,2,3
    python mesh_cli.py part.stp --mat-model plastic_kinematic:210000:0.3:7.85e-9:1000:200
    python mesh_cli.py block.stp --etype hex8 --size-max 5      # box-like solids only
    python mesh_cli.py part.stp --boundary-layer 3:1.2:4 --bl-faces 2,5
    python mesh_cli.py part.stp --mat --point-mass 101:0.5 --spring 101:202:1000 \
                                --damping 0.1 --nodal-rigid-body 99:1,2,3
    python mesh_cli.py part.stp --rigidwall-sphere 0:0:-50:0:0:1:25:0.2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from k_mesher import _version
from k_mesher import connections
from k_mesher import dyna_writer
from k_mesher import mesher

# sentinel: --auto-spotweld given with no SPACING value (welds at every node)
_SPOTWELD_ON = object()


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
CONTACT_TYPES = ("automatic_single_surface", "automatic_surface_to_surface",
                 "tied_surface_to_surface", "tied_nodes_to_surface")


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


def parse_part_rigid(text: str) -> tuple[int, dict]:
    """BODY[:E[:NU[:RHO]]] - make a body rigid (*MAT_RIGID); E/nu/rho are
    used by LS-DYNA for the contact stiffness (defaults: steel mm-t-s)."""
    body, mat = parse_part_mat(text)
    mat["rigid"] = True
    return body, mat


def parse_nset(text: str) -> dict:
    """Coordinate node set: KIND,PARAMS...[,key=value...]. Kinds:
    plane,AXIS,OFFSET / box,X0,Y0,Z0,X1,Y1,Z1 / sphere,CX,CY,CZ,R.
    Optional keys: spc=DOFS, force=AXIS:TOTAL, tol=T, title=NAME."""
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    kind = tokens[0].lower() if tokens else ""
    n_params = {"plane": 2, "box": 6, "sphere": 4}.get(kind)
    if n_params is None:
        raise argparse.ArgumentTypeError(
            f"node-set kind must be plane, box or sphere: {text!r}")
    raw = tokens[1:1 + n_params]
    if len(raw) != n_params or any("=" in t for t in raw):
        raise argparse.ArgumentTypeError(
            f"{kind} needs {n_params} comma-separated parameters: {text!r}")
    try:
        if kind == "plane":
            axis = raw[0].lower()
            if axis not in ("x", "y", "z"):
                raise ValueError
            params = (axis, float(raw[1]))
        else:
            params = [float(t) for t in raw]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad {kind} parameters in {text!r}") from None
    cs = {"kind": kind, "params": params, "role": "set"}
    for kv in tokens[1 + n_params:]:
        key, sep, val = kv.partition("=")
        key = key.strip().lower()
        if not sep:
            raise argparse.ArgumentTypeError(
                f"expected key=value after the parameters: {kv!r}")
        if key == "spc":
            if not val or any(ch not in "123456" for ch in val):
                raise argparse.ArgumentTypeError(
                    f"spc= DOFs must be digits 1-6: {kv!r}")
            cs.update(role="spc", dofs=val)
        elif key == "force":
            axis, _, total = val.partition(":")
            if axis.lower() not in ("x", "y", "z"):
                raise argparse.ArgumentTypeError(
                    f"force= needs AXIS:TOTAL with axis x/y/z: {kv!r}")
            try:
                cs.update(role="force", axis=axis.lower(), value=float(total))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"bad force total in {kv!r}") from None
        elif key == "tol":
            try:
                cs["tol"] = float(val)
            except ValueError:
                raise argparse.ArgumentTypeError(f"bad tol in {kv!r}") from None
        elif key == "title":
            cs["title"] = val[:60]
        else:
            raise argparse.ArgumentTypeError(
                f"unknown node-set option {key!r} (use spc/force/tol/title)")
    return cs


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


def parse_hourglass(text: str) -> dict:
    """IHQ[:QM] e.g. '5' or '5:0.05' -> *HOURGLASS (QM defaults to 0.1)."""
    parts = text.split(":")
    try:
        ihq = int(parts[0])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"hourglass IHQ must be an integer: {text!r}") from None
    try:
        qm = float(parts[1]) if len(parts) > 1 and parts[1] else 0.1
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad hourglass QM in {text!r}") from None
    return {"ihq": ihq, "qm": qm}


def parse_database(text: str) -> tuple[str, float]:
    """NAME:DT e.g. 'GLSTAT:1e-4' -> one *DATABASE_<NAME> ascii request."""
    name, sep, dt = text.partition(":")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"expected NAME:DT, got {text!r}")
    try:
        return name.strip().upper(), float(dt)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad database interval in {text!r}") from None


def parse_init_velocity(text: str) -> dict:
    """VX:VY:VZ[:VXR:VYR:VZR] initial velocity; the optional rotational
    components map to an OMEGA about the axis they define."""
    parts = text.split(":")
    if len(parts) not in (3, 6):
        raise argparse.ArgumentTypeError(
            f"init-velocity needs VX:VY:VZ or VX:VY:VZ:VXR:VYR:VZR: {text!r}")
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad init-velocity numbers in {text!r}") from None
    iv = {"vx": vals[0], "vy": vals[1], "vz": vals[2]}
    if len(vals) == 6:
        iv.update(vxr=vals[3], vyr=vals[4], vzr=vals[5])
    return iv


def parse_rigidwall(text: str) -> dict:
    """TX:TY:TZ:HX:HY:HZ[:FRIC] -> *RIGIDWALL_PLANAR (normal tail -> head)."""
    parts = text.split(":")
    if len(parts) not in (6, 7):
        raise argparse.ArgumentTypeError(
            f"rigidwall needs 6 or 7 colon-separated numbers "
            f"(tail xyz, head xyz[, fric]): {text!r}")
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad rigidwall numbers in {text!r}") from None
    rw = {"tail": tuple(vals[0:3]), "head": tuple(vals[3:6])}
    if len(vals) == 7:
        rw["fric"] = vals[6]
    return rw


def parse_rigidwall_sphere(text: str) -> dict:
    """TX:TY:TZ:HX:HY:HZ:R[:FRIC] -> *RIGIDWALL_GEOMETRIC_SPHERE
    (tail = centre, tail -> head sets the orientation vector)."""
    parts = text.split(":")
    if len(parts) not in (7, 8):
        raise argparse.ArgumentTypeError(
            f"rigidwall-sphere needs TX:TY:TZ:HX:HY:HZ:R[:FRIC]: {text!r}")
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad rigidwall-sphere numbers in {text!r}") from None
    if vals[6] <= 0:
        raise argparse.ArgumentTypeError(
            f"rigidwall-sphere radius must be > 0: {text!r}")
    rw = {"tail": tuple(vals[0:3]), "head": tuple(vals[3:6]),
          "shape": "sphere", "radius": vals[6]}
    if len(vals) == 8:
        rw["fric"] = vals[7]
    return rw


def parse_rigidwall_cylinder(text: str) -> dict:
    """TX:TY:TZ:HX:HY:HZ:R:L[:FRIC] -> *RIGIDWALL_GEOMETRIC_CYLINDER
    (tail = base-cap centre, tail -> head = axis)."""
    parts = text.split(":")
    if len(parts) not in (8, 9):
        raise argparse.ArgumentTypeError(
            f"rigidwall-cylinder needs TX:TY:TZ:HX:HY:HZ:R:L[:FRIC]: {text!r}")
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad rigidwall-cylinder numbers in {text!r}") from None
    if vals[6] <= 0 or vals[7] <= 0:
        raise argparse.ArgumentTypeError(
            f"rigidwall-cylinder radius and length must be > 0: {text!r}")
    rw = {"tail": tuple(vals[0:3]), "head": tuple(vals[3:6]),
          "shape": "cylinder", "radius": vals[6], "length": vals[7]}
    if len(vals) == 9:
        rw["fric"] = vals[8]
    return rw


def parse_boundary_layer(text: str) -> dict:
    """THICKNESS[:RATIO[:NLAYERS[:SIZEWALL]]] -> near-wall grading spec
    (distance-graded sizing over THICKNESS; RATIO defaults to 1.2)."""
    parts = text.split(":")
    if len(parts) > 4:
        raise argparse.ArgumentTypeError(
            f"boundary-layer needs THICKNESS[:RATIO[:NLAYERS[:SIZEWALL]]]: "
            f"{text!r}")
    try:
        thickness = float(parts[0])
        ratio = float(parts[1]) if len(parts) > 1 and parts[1] else 1.2
        nb = int(parts[2]) if len(parts) > 2 and parts[2] else None
        size_wall = float(parts[3]) if len(parts) > 3 and parts[3] else None
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad boundary-layer numbers in {text!r} "
            f"(THICKNESS[:RATIO[:NLAYERS[:SIZEWALL]]])") from None
    if thickness <= 0:
        raise argparse.ArgumentTypeError(
            f"boundary-layer thickness must be > 0: {text!r}")
    if ratio <= 0:
        raise argparse.ArgumentTypeError(
            f"boundary-layer ratio must be > 0: {text!r}")
    if nb is not None and nb < 1:
        raise argparse.ArgumentTypeError(
            f"boundary-layer NLAYERS must be >= 1: {text!r}")
    if size_wall is not None and size_wall <= 0:
        raise argparse.ArgumentTypeError(
            f"boundary-layer SIZEWALL must be > 0: {text!r}")
    bl = {"thickness": thickness, "ratio": ratio}
    if nb is not None:
        bl["nb_layers"] = nb
    if size_wall is not None:
        bl["size_wall"] = size_wall
    return bl


def parse_point_mass(text: str) -> dict:
    """NID:MASS -> one *ELEMENT_MASS at that node."""
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"point-mass needs NID:MASS: {text!r}")
    try:
        return {"nid": int(parts[0]), "mass": float(parts[1])}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad point-mass values in {text!r} (NID:MASS)") from None


def _parse_discrete(text: str, key: str, what: str) -> dict:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{what} needs N1:N2:{key.upper()}: {text!r}")
    try:
        return {"n1": int(parts[0]), "n2": int(parts[1]),
                key: float(parts[2])}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad {what} values in {text!r} (N1:N2:{key.upper()})") from None


def parse_spring(text: str) -> dict:
    """N1:N2:K -> a spring *ELEMENT_DISCRETE (*MAT_SPRING_ELASTIC)."""
    return _parse_discrete(text, "k", "spring")


def parse_damper(text: str) -> dict:
    """N1:N2:C -> a damper *ELEMENT_DISCRETE (*MAT_DAMPER_VISCOUS)."""
    return _parse_discrete(text, "c", "damper")


def parse_nodal_rigid_body(text: str) -> dict:
    """PID:NID1,NID2[,NID3...] -> *CONSTRAINED_NODAL_RIGID_BODY over the
    listed nodes (PID must be free, i.e. above the mesh part ids)."""
    pid_str, sep, nid_list = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"nodal-rigid-body needs PID:NID1,NID2[,...]: {text!r}")
    try:
        pid = int(pid_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"nodal-rigid-body PID must be an integer: {text!r}") from None
    nodes = parse_id_list(nid_list)
    if len(nodes) < 2:
        raise argparse.ArgumentTypeError(
            f"nodal-rigid-body needs at least two node ids: {text!r}")
    return {"pid": pid, "nodes": nodes}


def parse_spotweld(text: str) -> tuple[int, int]:
    """N1:N2 node-id pair -> *CONSTRAINED_SPOTWELD."""
    parts = text.split(":")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"spotweld needs two node ids N1:N2: {text!r}") from None


def parse_define_curve(text: str) -> dict:
    """LCID:x1,y1;x2,y2;... -> *DEFINE_CURVE with the listed points."""
    lcid_str, sep, pts = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"define-curve needs LCID:x1,y1;x2,y2;...: {text!r}")
    try:
        lcid = int(lcid_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"define-curve LCID must be an integer: {text!r}") from None
    points = []
    for pair in pts.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        xy = pair.split(",")
        if len(xy) != 2:
            raise argparse.ArgumentTypeError(
                f"define-curve point must be x,y: {pair!r} in {text!r}")
        try:
            points.append((float(xy[0]), float(xy[1])))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"bad define-curve point {pair!r} in {text!r}") from None
    if not points:
        raise argparse.ArgumentTypeError(
            f"define-curve needs at least one point: {text!r}")
    return {"lcid": lcid, "points": points}


def parse_cross_section(text: str) -> dict:
    """X:Y:Z:NX:NY:NZ[:TITLE] -> a *DATABASE_CROSS_SECTION_PLANE cut through
    the whole model (PSID 0), defined by an in-plane point and a normal."""
    parts = text.split(":")
    if len(parts) not in (6, 7):
        raise argparse.ArgumentTypeError(
            f"cross-section needs X:Y:Z:NX:NY:NZ[:TITLE]: {text!r}")
    try:
        vals = [float(v) for v in parts[:6]]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad cross-section numbers in {text!r}") from None
    cs = {"point": tuple(vals[0:3]), "normal": tuple(vals[3:6])}
    if len(parts) == 7 and parts[6].strip():
        cs["title"] = parts[6][:60]
    return cs


def parse_id_list(text: str) -> list[int]:
    """A comma-separated list of integer ids (e.g. '1,2,3'); the flag is
    repeatable, so multiple lists accumulate."""
    ids = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated integer ids: {text!r}") from None
    if not ids:
        raise argparse.ArgumentTypeError(
            f"id list needs at least one integer: {text!r}")
    return ids


# --mat-model MODEL keywords mapped to the dyna_writer.mat_* helpers
_MAT_MODELS = {
    "plastic_kinematic": dyna_writer.mat_plastic_kinematic,
    "kinematic": dyna_writer.mat_plastic_kinematic,
    "piecewise": dyna_writer.mat_piecewise_linear_plasticity,
    "piecewise_linear_plasticity": dyna_writer.mat_piecewise_linear_plasticity,
}


def parse_mat_model(text: str) -> dict:
    """MODEL:E:NU:RHO:SIGY[:ETAN] -> a material dict from the dyna_writer.mat_*
    helpers (used in place of --mat). MODEL is plastic_kinematic or piecewise
    (*MAT_PLASTIC_KINEMATIC / *MAT_PIECEWISE_LINEAR_PLASTICITY)."""
    model, sep, rest = text.partition(":")
    fn = _MAT_MODELS.get(model.strip().lower())
    if fn is None:
        raise argparse.ArgumentTypeError(
            f"mat-model MODEL must be one of {sorted(_MAT_MODELS)}: {text!r}")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"mat-model needs {model}:E:NU:RHO:SIGY[:ETAN]: {text!r}")
    try:
        nums = [float(v) for v in rest.split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad mat-model numbers in {text!r}") from None
    if len(nums) not in (4, 5):
        raise argparse.ArgumentTypeError(
            f"mat-model needs E:NU:RHO:SIGY[:ETAN]: {text!r}")
    e, pr, ro, sigy = nums[:4]
    etan = nums[4] if len(nums) == 5 else 0.0
    return fn(e, pr, ro, sigy, etan=etan)


def parse_prescribed_motion(text: str) -> dict:
    """NSID:DOF:VAD:LCID[:SF] -> *BOUNDARY_PRESCRIBED_MOTION_SET."""
    parts = text.split(":")
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            f"prescribed-motion needs NSID:DOF:VAD:LCID[:SF]: {text!r}")
    try:
        pm = {"nsid": int(parts[0]), "dof": int(parts[1]),
              "vad": int(parts[2]), "lcid": int(parts[3])}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"prescribed-motion NSID/DOF/VAD/LCID must be integers: "
            f"{text!r}") from None
    if len(parts) == 5:
        try:
            pm["sf"] = float(parts[4])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"bad prescribed-motion SF in {text!r}") from None
    return pm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mesh a STEP/IGES/BREP/STL file to TET4/TET10/HEX8/TRI3/"
                    "QUAD4 and write an LS-DYNA .k file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {_version.__version__}")
    p.add_argument("input", help="CAD/mesh input: STEP/IGES/BREP solid or "
                                 "surface, or an STL/OBJ/PLY tessellation "
                                 "(shells, or tets if watertight)")
    p.add_argument("-o", "--output", help="output .k file (default: input with .k)")
    p.add_argument("--etype", choices=["tet4", "tet10", "hex8", "tri3", "quad4"],
                   default="tet4",
                   help="element type: solid tets, solid hexes (hex8 is "
                        "transfinite/structured and needs box-like or "
                        "sweepable solids) or shells (tri3 / quad-dominant)")
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
    p.add_argument("--boundary-layer", type=parse_boundary_layer, default=None,
                   metavar="THICKNESS[:RATIO[:NLAYERS[:SIZEWALL]]]",
                   help="distance-graded near-wall sizing over THICKNESS from "
                        "the wall faces (RATIO default 1.2, NLAYERS sets the "
                        "first-layer height unless SIZEWALL is explicit); "
                        "TET/shell meshing only, see --bl-faces")
    p.add_argument("--bl-faces", type=parse_id_list, default=None,
                   metavar="TAG[,TAG...]",
                   help="wall face tags for --boundary-layer (see "
                        "--list-faces); default: all boundary faces")
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
    p.add_argument("--elform", type=int, choices=[1, 2, 4, 10, 13, 16, 17],
                   default=None,
                   help="section formulation (defaults: TET4 10, TET10 16, "
                        "HEX8 1, TRI3 4, QUAD4 16)")
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
    p.add_argument("--split-parts", action="store_true",
                   help="additionally write one standalone .k per body "
                        "(<output>_p<PID>[_<name>].k) with that body's "
                        "elements, nodes and material; sets are filtered per "
                        "part, contact cards are skipped, and a face force's "
                        "total is re-spread over each file's own face nodes")
    p.add_argument("--split-include", action="store_true",
                   help="write the output as an *INCLUDE assembly: one mesh "
                        "fragment per body keeping the GLOBAL numbering plus "
                        "a master deck with the PART/MAT/contact/set cards "
                        "and *INCLUDE lines (mutually exclusive with "
                        "--split-parts)")
    p.add_argument("--part-rigid", type=parse_part_rigid, action="append",
                   default=[], metavar="BODY[:E[:NU[:RHO]]]",
                   help="make a body rigid (*MAT_RIGID; E/nu/rho set the "
                        "contact stiffness), repeatable; overrides --part-mat "
                        "for that body")
    p.add_argument("--nset", type=parse_nset, action="append", default=[],
                   metavar="KIND,PARAMS[,key=value]",
                   help="coordinate node set (works for STL/OBJ/PLY too): "
                        "plane,AXIS,OFFSET | box,X0,Y0,Z0,X1,Y1,Z1 | "
                        "sphere,CX,CY,CZ,R; optional spc=DOFS, "
                        "force=AXIS:TOTAL, tol=T, title=NAME; repeatable")
    p.add_argument("--export", action="append", default=[], metavar="FILE",
                   help="also export the mesh to a gmsh-supported format "
                        "chosen by the extension (.msh/.vtk/...), repeatable")
    p.add_argument("--target-dt", type=float, default=None, metavar="DT",
                   help="with --mat: report the characteristic length needed "
                        "to reach this explicit timestep (mass-scaling guide)")
    p.add_argument("--preview", action="store_true",
                   help="open the Gmsh viewer on the result")

    g = p.add_argument_group("LS-DYNA control / loads")
    g.add_argument("--endtim", type=float, default=None, metavar="TIME",
                   help="write *CONTROL_TERMINATION with this end time")
    g.add_argument("--mass-scale", type=float, default=None, metavar="DT2MS",
                   help="mass scaling: *CONTROL_TIMESTEP DT2MS (typically "
                        "negative, e.g. -1e-6); forces the card even without "
                        "--tssfac")
    g.add_argument("--hourglass", type=parse_hourglass, default=None,
                   metavar="IHQ[:QM]",
                   help="write *HOURGLASS (IHQ type, QM coefficient default "
                        "0.1); its HGID is stamped on every *PART")
    g.add_argument("--control-energy", action="store_true",
                   help="write *CONTROL_ENERGY (hourglass / sliding / "
                        "rigidwall / Rayleigh energy tracking)")
    g.add_argument("--d3plot-dt", type=float, default=None, metavar="DT",
                   help="write *DATABASE_BINARY_D3PLOT with this output "
                        "interval")
    g.add_argument("--database", type=parse_database, action="append",
                   default=[], metavar="NAME:DT",
                   help="ASCII database output, e.g. GLSTAT:1e-4 -> one "
                        "*DATABASE_<NAME> card, repeatable")
    g.add_argument("--init-velocity", type=parse_init_velocity, default=None,
                   metavar="VX:VY:VZ[:VXR:VYR:VZR]",
                   help="*INITIAL_VELOCITY_GENERATION over all nodes "
                        "(rotational components -> OMEGA about their axis)")
    g.add_argument("--contact-type", choices=CONTACT_TYPES, default=None,
                   help="contact type used with --contact (default is plain "
                        "single-surface); writes the matching *CONTACT_ card "
                        "with --contact's friction")
    g.add_argument("--rigidwall", type=parse_rigidwall, action="append",
                   default=[], metavar="TX:TY:TZ:HX:HY:HZ[:FRIC]",
                   help="*RIGIDWALL_PLANAR, normal pointing tail -> head, "
                        "repeatable")
    g.add_argument("--spotweld", type=parse_spotweld, action="append",
                   default=[], metavar="N1:N2",
                   help="*CONSTRAINED_SPOTWELD between two node ids, "
                        "repeatable")
    g.add_argument("--define-curve", type=parse_define_curve, action="append",
                   default=[], metavar="LCID:x1,y1;x2,y2;...",
                   help="*DEFINE_CURVE with the given point list, repeatable")
    g.add_argument("--prescribed-motion", type=parse_prescribed_motion,
                   action="append", default=[],
                   metavar="NSID:DOF:VAD:LCID[:SF]",
                   help="*BOUNDARY_PRESCRIBED_MOTION_SET (vad 0 vel / 1 accel "
                        "/ 2 disp), repeatable")

    m = p.add_argument_group("masses / springs / rigid bodies")
    m.add_argument("--point-mass", type=parse_point_mass, action="append",
                   dest="point_masses", default=[], metavar="NID:MASS",
                   help="*ELEMENT_MASS lumped mass at a node, repeatable")
    m.add_argument("--spring", type=parse_spring, action="append",
                   dest="springs", default=[], metavar="N1:N2:K",
                   help="spring *ELEMENT_DISCRETE between two nodes "
                        "(*MAT_SPRING_ELASTIC stiffness K), repeatable")
    m.add_argument("--damper", type=parse_damper, action="append",
                   dest="dampers", default=[], metavar="N1:N2:C",
                   help="damper *ELEMENT_DISCRETE between two nodes "
                        "(*MAT_DAMPER_VISCOUS constant C), repeatable")
    m.add_argument("--nodal-rigid-body", type=parse_nodal_rigid_body,
                   action="append", dest="nodal_rigid_bodies", default=[],
                   metavar="PID:NID1,NID2[,NID3...]",
                   help="*CONSTRAINED_NODAL_RIGID_BODY over the listed nodes; "
                        "PID must be free (above the mesh part ids), "
                        "repeatable")
    m.add_argument("--damping", type=float, default=None, metavar="VALDMP",
                   help="*DAMPING_GLOBAL system damping constant")
    m.add_argument("--rigidwall-sphere", type=parse_rigidwall_sphere,
                   action="append", dest="rigidwall_spheres", default=[],
                   metavar="TX:TY:TZ:HX:HY:HZ:R[:FRIC]",
                   help="*RIGIDWALL_GEOMETRIC_SPHERE (tail = centre, tail -> "
                        "head = orientation), repeatable")
    m.add_argument("--rigidwall-cylinder", type=parse_rigidwall_cylinder,
                   action="append", dest="rigidwall_cylinders", default=[],
                   metavar="TX:TY:TZ:HX:HY:HZ:R:L[:FRIC]",
                   help="*RIGIDWALL_GEOMETRIC_CYLINDER (tail = base-cap "
                        "centre, tail -> head = axis), repeatable")

    c = p.add_argument_group("mesh-time connections / midsurface")
    c.add_argument("--midsurface", action="store_true",
                   help="mesh thin, constant-thickness plate solids as a "
                        "midsurface shell (CAD B-rep only); forces a shell "
                        "element type and takes the *SECTION_SHELL thickness "
                        "from the detected wall thickness (an explicit "
                        "--thickness wins)")
    c.add_argument("--auto-spotweld", type=float, nargs="?",
                   const=_SPOTWELD_ON, default=None, metavar="SPACING",
                   help="detect the interfaces of a multi-body model and weld "
                        "them with *CONSTRAINED_SPOTWELD node pairs; the "
                        "optional SPACING thins the pattern to that minimum "
                        "spot spacing (see --connect-tol)")
    c.add_argument("--tied-contact", action="store_true",
                   help="detect the interfaces of a multi-body model and tie "
                        "them with *CONTACT_TIED_SURFACE_TO_SURFACE over the "
                        "interface segment sets (see --connect-tol)")
    c.add_argument("--connect-tol", type=float, default=None, metavar="TOL",
                   help="node-matching tolerance shared by --auto-spotweld and "
                        "--tied-contact (default: 1e-3 of the bounding-box "
                        "diagonal)")

    a = p.add_argument_group("assembly contacts / output / materials")
    a.add_argument("--auto-contact", nargs="?", choices=CONTACT_TYPES,
                   const="automatic_surface_to_surface", default=None,
                   metavar="TYPE",
                   help="detect the touching part pairs of a multi-body model "
                        "and write one scoped *CONTACT_ (via *SET_PART_LIST) "
                        "per pair; the optional TYPE (default "
                        "automatic_surface_to_surface) picks the contact type, "
                        "the friction comes from --contact (see --connect-tol). "
                        "A single-body model warns and writes no card")
    a.add_argument("--cross-section", type=parse_cross_section, action="append",
                   dest="cross_sections", default=[],
                   metavar="X:Y:Z:NX:NY:NZ[:TITLE]",
                   help="*DATABASE_CROSS_SECTION_PLANE cut (point + normal) "
                        "through the whole model; *DATABASE_SECFORC is added "
                        "automatically for the section force output, repeatable")
    a.add_argument("--history-node", type=parse_id_list, action="append",
                   default=[], metavar="ID[,ID...]",
                   help="node ids for *DATABASE_HISTORY_NODE (comma list, "
                        "repeatable)")
    a.add_argument("--history-solid", type=parse_id_list, action="append",
                   default=[], metavar="ID[,ID...]",
                   help="solid element ids for *DATABASE_HISTORY_SOLID (comma "
                        "list, repeatable)")
    a.add_argument("--history-shell", type=parse_id_list, action="append",
                   default=[], metavar="ID[,ID...]",
                   help="shell element ids for *DATABASE_HISTORY_SHELL (comma "
                        "list, repeatable)")
    a.add_argument("--mat-model", type=parse_mat_model, default=None,
                   metavar="MODEL:E:NU:RHO:SIGY[:ETAN]",
                   help="write a plasticity material in place of --mat: MODEL "
                        "is plastic_kinematic (*MAT_PLASTIC_KINEMATIC) or "
                        "piecewise (*MAT_PIECEWISE_LINEAR_PLASTICITY)")
    return p


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2
    # did the user pass --thickness explicitly? (default 1.0 stays otherwise)
    user_thickness = any(a == "--thickness" or a.startswith("--thickness=")
                         for a in raw_argv)

    if args.tet10 and args.etype != "tet4":
        print(f"warning: --tet10 (deprecated) overrides --etype {args.etype}; "
              f"meshing TET10", file=sys.stderr)
    etype = "TET10" if args.tet10 else args.etype.upper()
    if args.midsurface:
        if mesher.input_kind(args.input) == "mesh":
            print("error: --midsurface needs CAD solids (STEP/IGES/BREP); a "
                  "tessellated mesh (STL/OBJ/PLY) carries no solid geometry",
                  file=sys.stderr)
            return 2
        # midsurface always writes shells: keep a shell --etype, else default
        # to TRI3 (a solid --etype is just the "no shell type given" case)
        shell_etype = etype if mesher.ETYPES[etype]["family"] == "shell" \
            else "TRI3"
        if shell_etype != etype:
            print(f"note: --midsurface writes shells; using {shell_etype} "
                  f"(pass --etype tri3/quad4 to choose)", file=sys.stderr)
        etype = shell_etype
    defaults = {"TET4": 10, "TET10": 16, "HEX8": 1, "TRI3": 4, "QUAD4": 16}
    allowed = {"TET4": (10, 13), "TET10": (16, 17), "HEX8": (1, 2),
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
    if args.split_parts and args.split_include:
        print("error: --split-parts and --split-include are mutually "
              "exclusive (standalone files vs. *INCLUDE fragments)",
              file=sys.stderr)
        return 2
    is_shell = mesher.ETYPES[etype]["family"] == "shell"

    boundary_layer = None
    if args.boundary_layer is not None:
        boundary_layer = dict(args.boundary_layer)
        boundary_layer["faces"] = args.bl_faces if args.bl_faces else "all"
    elif args.bl_faces:
        print("warning: --bl-faces without --boundary-layer has no effect",
              file=sys.stderr)

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
        boundary_layer=boundary_layer,
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
    mesh_fn = mesher.midsurface_shell if args.midsurface else mesher.mesh_step_auto
    try:
        result = mesh_fn(
            settings, log=print,
            preview_path=preview_path if args.preview else None,
            export_paths=tuple(args.export))
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for path in args.export:
        print(f"Wrote {path}")

    # shell thickness: the CLI value, unless midsurface detected one and the
    # user did not pass --thickness explicitly (then the detected value wins)
    thickness = args.thickness
    if args.midsurface:
        detected = result.stats["midsurface_thickness"]
        print(f"Detected midsurface thickness: {detected:.4g}")
        if not user_thickness:
            thickness = detected

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

    # coordinate node sets: selected on the mesh, so they work for any input
    for i, cs in enumerate(args.nset, start=1):
        label = cs.get("title") or f"NSET_{i}_{cs['kind'].upper()}"
        ids = mesher.select_nodes(result.coords, cs["kind"], cs["params"],
                                  tol=cs.get("tol"))
        if len(ids) == 0:
            print(f"warning: coordinate set {label} matched no nodes - "
                  f"skipped", file=sys.stderr)
            continue
        item = {"kind": "node", "title": label, "nodes": ids}
        if cs["role"] == "spc":
            item["spc_dofs"] = cs.get("dofs") or "123456"
        elif cs["role"] == "force":
            item["force"] = (cs["axis"], cs["value"])
        face_sets.append(item)
        print(f"Coordinate set {label}: {len(ids)} nodes")

    # automatic connection detection between touching bodies (spotwelds are
    # written by write_k/write_k_include only, like contact - not the split
    # files); tied-contact segment sets are merged into face_sets
    auto_spotwelds = ()
    auto_contacts = ()
    auto_contact_pairs = ()
    if args.auto_spotweld is not None:
        spacing = (None if args.auto_spotweld is _SPOTWELD_ON
                   else args.auto_spotweld)
        auto_spotwelds = tuple(connections.spotweld_pairs(
            result, tol=args.connect_tol, spacing=spacing))
        if auto_spotwelds:
            print(f"Auto-spotweld: {len(auto_spotwelds)} weld(s) on the "
                  f"detected interfaces")
        else:
            print("warning: --auto-spotweld found no interfaces (single body "
                  "or no bodies within tolerance)", file=sys.stderr)
    if args.tied_contact:
        tie_sets, auto_contacts = connections.tied_contact(
            result, tol=args.connect_tol)
        if tie_sets:
            face_sets.extend(tie_sets)
            print(f"Tied contact: {len(tie_sets)} interface segment set(s)")
        else:
            print("warning: --tied-contact found no interfaces (single body "
                  "or no bodies within tolerance)", file=sys.stderr)
    if args.auto_contact is not None:
        if len(result.part_names) < 2:
            print("warning: --auto-contact needs a multi-body model - no "
                  "contact card written", file=sys.stderr)
        else:
            auto_contact_pairs = tuple(connections.contact_pairs(
                result, base_pid=args.pid, tol=args.connect_tol,
                ctype=args.auto_contact, fs=args.contact or 0.0))
            if auto_contact_pairs:
                print(f"Auto-contact: {len(auto_contact_pairs)} scoped "
                      f"contact pair(s) on the detected interfaces")
            else:
                print("warning: --auto-contact found no touching interfaces",
                      file=sys.stderr)

    elem_sets = []
    failed = result.stats.get("failed_elems", ())
    if not args.no_qa_sets and len(failed):
        elem_sets.append({"title": "QA quality-criteria failures",
                          "eids": failed + 1})

    n_bodies = len(result.part_names)
    part_mats = {}
    for opt, pairs in (("--part-mat", args.part_mat),
                       ("--part-rigid", args.part_rigid)):
        for body, m in pairs:
            if body > n_bodies:
                print(f"warning: {opt} body {body} ignored - the model has "
                      f"only {n_bodies} body/bodies", file=sys.stderr)
                continue
            part_mats[args.pid + body - 1] = m

    # a --mat-model plasticity dict is used in place of a plain --mat elastic
    mat = args.mat_model if args.mat_model is not None else args.mat

    mass, part_masses, dt_est = mesher.mass_and_timestep(
        result, etype, mat, part_mats, args.pid,
        thickness if is_shell else 1.0, log=print)

    if args.target_dt and dt_est:
        lc = result.stats.get("char_length", 0.0)
        pct = result.stats.get("char_length_pctiles") or {}
        lc_needed = args.target_dt * lc / dt_est
        if dt_est >= args.target_dt:
            print(f"Target dt {args.target_dt:g}: met (estimated dt "
                  f"{dt_est:.4g})")
        else:
            below = [k for k in ("p1", "p10", "p50")
                     if pct.get(k, lc) < lc_needed]
            print(f"Target dt {args.target_dt:g}: needs characteristic "
                  f"length >= {lc_needed:.4g} (worst now {lc:.4g}; "
                  f"element percentiles below the target: "
                  f"{', '.join(below) if below else 'none - only outliers'})"
                  f" - refine the sizing, defeature bad spots, or mass-scale")

    part_ids = args.pid + result.elem_parts
    part_titles = {args.pid + i: (name or f"body {i + 1}")
                   for i, name in enumerate(result.part_names)}

    comments = [f"Source geometry: {args.input}",
                f"Element size: {args.size_min:g} .. {args.size_max:g}"]
    for sp in settings.symmetry:
        comments.append(f"Symmetry: {sp.axis.upper()} = {sp.offset:g}, "
                        f"kept '{sp.keep}' side")

    # contact: a plain --contact writes single-surface via contact_fs; adding
    # --contact-type routes it into the general `contacts` list instead (kept
    # out of common so the per-part split files still omit contact)
    contact_fs = args.contact
    contacts = ()
    if args.contact_type is not None:
        fs = args.contact if args.contact is not None else 0.0
        contacts = ({"type": args.contact_type, "fs": fs},)
        contact_fs = None
    # auto tied-contact appends to whatever contact the user requested
    contacts = tuple(contacts) + tuple(auto_contacts)
    spotwelds = tuple(args.spotweld) + tuple(auto_spotwelds)

    databases = {}
    if args.d3plot_dt is not None:
        databases["d3plot_dt"] = args.d3plot_dt
    if args.database:
        databases["ascii"] = dict(args.database)

    # cross sections and history requests (write_k / write_k_include only, like
    # contact - kept out of `common` so the per-part split files omit them)
    cross_sections = tuple(args.cross_sections)
    history = {}
    for key, lists in (("nodes", args.history_node),
                       ("solids", args.history_solid),
                       ("shells", args.history_shell)):
        ids = [i for lst in lists for i in lst]
        if ids:
            history[key] = ids
    history = history or None

    # masses / springs / rigid bodies / global damping (write_k /
    # write_k_include only, like contact - kept out of `common` so the
    # per-part split files omit them)
    point_masses = tuple(args.point_masses)
    discretes = tuple(args.springs) + tuple(args.dampers)
    nodal_rigid_bodies = tuple(args.nodal_rigid_bodies)
    damping = {"valdmp": args.damping} if args.damping is not None else None

    # geometric rigid walls (sphere/cylinder) join the planar ones
    rigidwalls = tuple(args.rigidwall) + tuple(args.rigidwall_spheres) \
        + tuple(args.rigidwall_cylinders)

    common = dict(
        element_kind="shell" if is_shell else "solid",
        elform=elform, thickness=thickness,
        start_nid=args.start_nid, start_eid=args.start_eid,
        start_sid=args.start_sid,
        title=args.title or os.path.splitext(os.path.basename(out))[0],
        comments=tuple(comments), sym_sets=tuple(sym_sets), mat=mat,
        part_mats=part_mats, face_sets=tuple(face_sets),
        elem_sets=tuple(elem_sets), implicit_cards=args.implicit_cards,
        mesh_only=args.mesh_only, tssfac=args.tssfac,
        body_load=args.gravity, long_format=args.long_format,
        endtim=args.endtim, mass_scale=args.mass_scale,
        hourglass=args.hourglass, control_energy=args.control_energy,
        initial_velocity=args.init_velocity,
        define_curves=tuple(args.define_curve),
        prescribed_motion=tuple(args.prescribed_motion),
        rigidwalls=rigidwalls,
    )
    if databases:
        common["databases"] = databases
    if args.midsurface:
        ms_ts = result.stats.get("midsurface_thicknesses") or ()
        if len(ms_ts) > 1:
            # multi-region midsurface: per-region *SECTION_SHELL thicknesses
            # (region i -> PID args.pid + i)
            common["part_thickness"] = {args.pid + i: t
                                        for i, t in enumerate(ms_ts)}
    split_files = []
    if args.split_include:
        split_files = dyna_writer.write_k_include(
            out, result.coords, result.elems, pid=args.pid,
            part_ids=part_ids, part_titles=part_titles,
            contact_fs=contact_fs, contacts=contacts,
            contact_pairs=auto_contact_pairs, cross_sections=cross_sections,
            history=history, spotwelds=spotwelds, point_masses=point_masses,
            discretes=discretes, nodal_rigid_bodies=nodal_rigid_bodies,
            damping=damping, **common)
        for p, f in split_files:
            print(f"Wrote {f} (mesh fragment, PID {p})")
        print(f"Wrote {out} (master deck with *INCLUDE cards)")
    else:
        dyna_writer.write_k(
            out, result.coords, result.elems, pid=args.pid,
            part_ids=part_ids, part_titles=part_titles,
            contact_fs=contact_fs, contacts=contacts,
            contact_pairs=auto_contact_pairs, cross_sections=cross_sections,
            history=history, spotwelds=spotwelds, point_masses=point_masses,
            discretes=discretes, nodal_rigid_bodies=nodal_rigid_bodies,
            damping=damping, **common)
        print(f"Wrote {out}")
        if args.split_parts:
            split_files = dyna_writer.write_k_split(
                out, result.coords, result.elems,
                part_ids=part_ids, part_titles=part_titles, **common)
            for p, f in split_files:
                print(f"Wrote {f} (PID {p})")

    if args.stats_json:
        payload = {
            "input": args.input, "output": out,
            "element_type": etype, "elform": elform,
            "n_parts": len(result.part_names), "part_names": result.part_names,
            "mass": mass, "part_masses": part_masses,
            "critical_timestep": dt_est,
            "material": mat, "part_materials": part_mats,
            "split_files": {p: f for p, f in split_files},
            "stats": result.stats,
        }
        with open(args.stats_json, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"Wrote {args.stats_json}")

    if args.preview:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview.py")
        subprocess.Popen([sys.executable, script, preview_path])
    return 0


def cli_entry() -> None:
    """Console-script entry point (see pyproject.toml)."""
    raise SystemExit(main())


if __name__ == "__main__":
    sys.exit(main())
