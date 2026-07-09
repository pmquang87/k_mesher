"""Core meshing: STEP geometry -> TET4/TET10 solids or TRI3/QUAD4 shells
using the gmsh API.

This module is GUI-agnostic so it can also be used from scripts/tests.
"""
from __future__ import annotations

import collections
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
import gmsh

# gmsh 3D algorithm ids
ALGO3D = {
    "Delaunay": 1,
    "Frontal": 4,
    "HXT (parallel Delaunay)": 10,
}

# Geometry.OCCTargetUnit values (empty string = keep file units)
OCC_UNITS = {
    "File units (no conversion)": "",
    "mm": "MM",
    "cm": "CM",
    "m": "M",
    "in": "IN",
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# ---------------------------------------------------------------------------
# Supported input formats.
#
# CAD (boundary-representation) formats go through the OpenCASCADE kernel via
# occ.importShapes and support the full pipeline (solids, symmetry cuts,
# defeaturing, face scanning, curvature sizing, tet/shell meshing):
#   .step/.stp  STEP    - the recommended solid exchange format
#   .iges/.igs  IGES    - legacy surface format; sewn into solids on import
#   .brep/.brp  BREP    - OpenCASCADE native, exact geometry
#
# Tessellated (mesh) formats carry only a surface triangulation, no CAD
# geometry. They are imported as-is (coincident nodes welded) and meshed with
# shell elements only - no symmetry/defeature/refinement, which need a B-rep:
#   .stl        STL     - triangulated surface mesh (3D printing / CAD export)
#   .obj        OBJ     - Wavefront surface mesh
#   .ply        PLY     - Stanford polygon / scanned-surface mesh
CAD_FORMATS = {".step", ".stp", ".iges", ".igs", ".brep", ".brp"}
MESH_FORMATS = {".stl", ".obj", ".ply"}
INPUT_FORMATS = CAD_FORMATS | MESH_FORMATS

# human-readable format name per extension (for logs / errors)
FORMAT_NAMES = {
    ".step": "STEP", ".stp": "STEP",
    ".iges": "IGES", ".igs": "IGES",
    ".brep": "BREP", ".brp": "BREP",
    ".stl": "STL", ".obj": "OBJ", ".ply": "PLY",
}


def input_kind(path: str) -> str:
    """Return 'mesh' for tessellated inputs (STL), else 'cad' (STEP/IGES/BREP
    or any other extension, which is optimistically handed to OpenCASCADE)."""
    return "mesh" if os.path.splitext(path)[1].lower() in MESH_FORMATS else "cad"


def format_name(path: str) -> str:
    return FORMAT_NAMES.get(os.path.splitext(path)[1].lower(), "CAD")

# supported element types
ETYPES = {
    "TET4":  {"family": "solid", "order": 1},
    "TET10": {"family": "solid", "order": 2},
    "TRI3":  {"family": "shell", "recombine": False},
    "QUAD4": {"family": "shell", "recombine": True},
}

# solid order -> (gmsh tet type id, nodes per tet, gmsh surface tri type id,
#                 nodes per surface tri)
TET_TYPE = {1: (4, 4, 2, 3), 2: (11, 10, 9, 6)}
GMSH_TRI3, GMSH_QUAD4 = 2, 3

# gmsh TET10 -> LS-DYNA TET10 node ordering: corner nodes and the first four
# mid-edge nodes coincide; the mid nodes of edges (2,4) and (3,4) are swapped.
GMSH2DYNA_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]

# node permutation that mirrors an element (fixes negative volume),
# in LS-DYNA ordering
NEGFIX = {4: [0, 2, 1, 3], 10: [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]}

# quality criteria limits (LS-DYNA practice)
QUALITY_LIMITS = {
    "solid": {"aspect ratio": ("max", 8.0), "SICN": ("min", 0.2)},
    "shell": {"aspect ratio": ("max", 5.0), "SICN": ("min", 0.3),
              "warpage [deg]": ("max", 15.0), "min angle [deg]": ("min", 20.0)},
}


@dataclass
class SymmetryPlane:
    """A symmetry plane normal to `axis` located at `offset`.

    keep = '+' keeps the material on the side where coordinate >= offset,
    keep = '-' keeps the side where coordinate <= offset.
    """
    axis: str
    offset: float = 0.0
    keep: str = "+"


@dataclass
class MeshSettings:
    step_file: str                     # input CAD/mesh file: STEP/IGES/BREP/STL
    element_type: str = "TET4"         # TET4 | TET10 | TRI3 | QUAD4
    size_max: float = 10.0
    size_min: float = 0.0
    curvature_refine: bool = True
    curvature_elems: int = 16          # target elements per 2*pi of curvature
    algorithm3d: str = "Delaunay"
    optimize: bool = True
    heal: bool = False                 # OCC import-time shape healing
    glue: bool = False                 # merge coincident faces of touching bodies
    occ_unit: str = ""                 # "", "MM", "CM", "M", "IN"
    symmetry: list[SymmetryPlane] = field(default_factory=list)
    # refinement regions: {"kind": "sphere", "params": [cx,cy,cz,r], "size": s}
    #                     {"kind": "box", "params": [x0,y0,z0,x1,y1,z1], "size": s}
    refinements: list[dict] = field(default_factory=list)
    face_sizes: dict[int, float] = field(default_factory=dict)  # tag -> size
    defeature_faces: list[int] = field(default_factory=list)    # remove these
    collect_faces: list[int] = field(default_factory=list)      # face tags for sets
    auto_refine: bool = False          # remesh with local refinement at bad spots
    auto_refine_rounds: int = 2
    auto_refine_threshold: float = 0.05  # retry while min SICN below this


@dataclass
class MeshResult:
    coords: np.ndarray                 # (N, 3) float
    elems: np.ndarray                  # solids: (M, 4|10), shells: (M, 4)
                                       # 1-based, LS-DYNA node ordering
    elem_parts: np.ndarray             # (M,) 0-based body index per element
    part_names: list[str]              # one entry per body ("" if unnamed)
    sym_nodes: dict[str, np.ndarray]   # axis -> 1-based node indices on plane
    face_nodes: dict[int, np.ndarray]  # face tag -> 1-based node indices
    face_segs: dict[int, np.ndarray]   # face tag -> (S, 4) 1-based segment nodes
    stats: dict
    # axis -> (S, 4) 1-based boundary segments lying on the symmetry plane
    # (solids only; triangles padded to a degenerate quad)
    sym_segs: dict[str, np.ndarray] = field(default_factory=dict)


def mesh_step(settings: MeshSettings, log=print, preview_path: str | None = None) -> MeshResult:
    """Mesh a CAD/mesh file (STEP/IGES/BREP/STL/OBJ/PLY). Optionally save a
    .msh copy for preview."""
    if settings.element_type not in ETYPES:
        raise ValueError(f"Unknown element type: {settings.element_type}")
    family = ETYPES[settings.element_type]["family"]

    # interruptible=False: skips SIGINT handler installation, which would fail
    # when meshing runs in the GUI worker thread (signals need the main thread)
    gmsh.initialize(interruptible=False)
    try:
        bbox, already_meshed = _load_geometry(settings, log)
        if not already_meshed:
            _apply_refinements(settings, log)
            _generate_mesh(settings, log)

        shell_checks = {}
        if family == "solid":
            coords, elems, elem_parts, part_names, tag_map = _extract_tets(
                ETYPES[settings.element_type]["order"], log)
        else:
            coords, elems, elem_parts, part_names, tag_map, shell_checks = \
                _extract_shells(log)

        sym_nodes = _find_symmetry_nodes(coords, settings.symmetry, bbox)
        sym_segs = _find_symmetry_segments(
            coords, settings.symmetry, sym_nodes, tag_map, settings.element_type)
        face_nodes, face_segs = _collect_face_data(
            settings.collect_faces, tag_map, settings.element_type, log)
        stats = _collect_stats(coords, elems, settings.element_type, bbox)
        stats.update(shell_checks)
        stats["duplicate_nodes"] = _count_duplicate_nodes(coords)
        _log_stats(stats, sym_nodes, log)

        if preview_path:
            gmsh.write(preview_path)

        return MeshResult(coords=coords, elems=elems, elem_parts=elem_parts,
                          part_names=part_names, sym_nodes=sym_nodes,
                          face_nodes=face_nodes, face_segs=face_segs, stats=stats,
                          sym_segs=sym_segs)
    finally:
        gmsh.finalize()


def mesh_step_auto(settings: MeshSettings, log=print,
                   preview_path: str | None = None) -> MeshResult:
    """mesh_step with an optional quality-driven retry loop: if the minimum
    element quality is below the threshold, add refinement spheres at the
    worst spots and remesh (BatchMesher-style). Returns the best mesh; the
    preview file (if requested) always matches the mesh that is returned."""
    s = settings
    rounds = settings.auto_refine_rounds if settings.auto_refine else 0
    best, best_preview, round_previews = None, None, []
    for rnd in range(rounds + 1):
        rp = preview_path
        if preview_path and rounds:
            # per-round file so the preview of the KEPT mesh can be restored
            # even when a later (worse) round overwrote a shared path
            # (suffix goes before the extension - gmsh picks the format by it)
            base, ext = os.path.splitext(preview_path)
            rp = f"{base}.round{rnd}{ext}"
            round_previews.append(rp)
        try:
            res = mesh_step(s, log=log, preview_path=rp)
        except Exception as e:
            if best is None:
                raise
            # a failed refinement round must not throw away the good mesh
            log(f"Auto-refine: remeshing failed in round {rnd}/{rounds} ({e}) "
                f"- keeping the best mesh from the earlier rounds")
            res = best
            break
        if (best is None or res.stats.get("quality_min", 1.0)
                > best.stats.get("quality_min", 1.0)):
            best, best_preview = res, rp
        qmin = res.stats.get("quality_min", 1.0)
        worst = [w for w in (res.stats.get("worst_elements") or ())
                 if w[0] < s.auto_refine_threshold]
        if rnd == rounds or qmin >= s.auto_refine_threshold or not worst:
            break
        add = [{"kind": "sphere", "params": [c[0], c[1], c[2], 3.0 * h],
                "size": max(h / 2.5, 1e-9)}
               for _, c, h in worst]
        log(f"Auto-refine: min quality {qmin:.4f} < "
            f"{s.auto_refine_threshold:g} - adding {len(add)} refinement "
            f"sphere(s) and remeshing (round {rnd + 1}/{rounds}) ...")
        s = replace(s, refinements=list(s.refinements) + add)
    if preview_path and rounds:
        if best_preview and os.path.isfile(best_preview):
            os.replace(best_preview, preview_path)
        for p in round_previews:
            if p != best_preview and os.path.isfile(p):
                os.remove(p)
    if best.stats.get("quality_min", 1.0) != res.stats.get("quality_min", 1.0):
        log(f"Auto-refine: keeping the best of all rounds "
            f"(min quality {best.stats.get('quality_min', 1.0):.4f})")
    return best


def list_faces(settings: MeshSettings, log=print) -> list[dict]:
    """Load the geometry (with healing/glue/defeature/symmetry applied, no
    meshing) and return the model faces:
    [{"tag", "name", "type", "area", "diag", "centroid"}]."""
    if input_kind(settings.step_file) == "mesh":
        raise RuntimeError(
            "Face scanning is not available for STL/tessellated input - it "
            "has no CAD faces, only triangles. Use a STEP/IGES/BREP file.")
    gmsh.initialize(interruptible=False)
    try:
        _load_geometry(settings, log)
        faces = []
        for _, tag in sorted(gmsh.model.getEntities(2)):
            try:
                area = gmsh.model.occ.getMass(2, tag)
                com = gmsh.model.occ.getCenterOfMass(2, tag)
            except Exception:
                area, com = 0.0, (0.0, 0.0, 0.0)
            try:
                ftype = gmsh.model.getType(2, tag)
                bb = gmsh.model.getBoundingBox(2, tag)
                diag = float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3])))
            except Exception:
                ftype, diag = "", 0.0
            name = gmsh.model.getEntityName(2, tag)
            faces.append({"tag": tag, "name": name.split("/")[-1] if name else "",
                          "type": ftype, "area": float(area), "diag": diag,
                          "centroid": tuple(float(c) for c in com)})
        log(f"Found {len(faces)} faces")
        return faces
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# geometry loading
# --------------------------------------------------------------------------

def _load_geometry(settings: MeshSettings, log):
    """Import the CAD/mesh file, heal/sew/glue/defeature as configured, apply
    symmetry cuts. Returns (bounding box after all cuts, already_meshed).

    ``already_meshed`` is True for tessellated inputs (STL) whose triangles are
    used directly - the caller then skips refinement and mesh generation."""
    family = ETYPES[settings.element_type]["family"]
    gmsh.option.setNumber("General.Terminal", 0)

    if input_kind(settings.step_file) == "mesh":
        return _load_tessellation(settings, family, log), True

    fmt = format_name(settings.step_file)
    if settings.occ_unit:
        gmsh.option.setString("Geometry.OCCTargetUnit", settings.occ_unit)
    if settings.heal:
        # NOTE: deliberately NOT setting OCCSewFaces - resewing the faces
        # of a valid solid can leave open shells and silently destroy the
        # solid (observed with real-world STEP exports).
        for opt in ("OCCFixDegenerated", "OCCFixSmallEdges", "OCCFixSmallFaces"):
            gmsh.option.setNumber(f"Geometry.{opt}", 1)

    log(f"Importing {fmt}: {settings.step_file}")
    try:
        gmsh.model.occ.importShapes(settings.step_file)
    except Exception as e:
        # Some formats (notably IGES) reject the OpenCASCADE target-unit
        # override; retry once in the file's own units so the import succeeds.
        if settings.occ_unit:
            log(f"  unit conversion to {settings.occ_unit} not supported for "
                f"{fmt}; importing in file units instead")
            gmsh.option.setString("Geometry.OCCTargetUnit", "")
            gmsh.model.occ.importShapes(settings.step_file)
        else:
            raise
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    surfs = gmsh.model.getEntities(2)
    if family == "solid":
        if not vols and surfs:
            # Surface-only model (always the case for IGES, sometimes STEP):
            # sew the faces into closed shells and build solids from them. The
            # sewing tolerance is scaled to the model size, otherwise the tiny
            # gaps typical of IGES exports are never bridged.
            log(f"No solids in the {fmt} file - trying to sew the "
                f"{len(surfs)} surface(s) into a solid ...")
            bb = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3])))
            tol = max(1e-5 * diag, 1e-6)
            gmsh.model.occ.healShapes(sewFaces=True, makeSolids=True,
                                      tolerance=tol)
            gmsh.model.occ.synchronize()
            vols = gmsh.model.getEntities(3)
            if vols:
                log(f"Sewn into {len(vols)} solid(s) (tolerance {tol:.3g})")
        if not vols:
            raise RuntimeError(
                f"The {fmt} file contains no solid volumes and no closed solid "
                "could be built from its surfaces. Only solid bodies can be "
                "meshed with tetrahedra - re-export the model as a solid "
                "(STEP/BREP keep solids best), or mesh it with shell elements "
                "instead."
            )
        log(f"Imported {len(vols)} solid volume(s)")
    else:
        if not surfs:
            raise RuntimeError(f"The {fmt} file contains no surfaces to mesh "
                               "with shell elements.")
        if vols:
            log(f"Imported {len(vols)} solid volume(s) - their boundary "
                f"surfaces will be meshed with shells")
        else:
            log(f"Imported {len(surfs)} surface(s)")
            if len(surfs) > 1:
                # Sew the free-standing faces so neighbouring faces share
                # edges (a conformal shell mesh). IGES exports in particular
                # leave sub-tolerance gaps that removeAllDuplicates alone will
                # not bridge; the tolerance is scaled to the model size. Only
                # sewing is done (no small-edge/face fixing) to preserve tags.
                bb = gmsh.model.getBoundingBox(-1, -1)
                diag = float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3])))
                try:
                    gmsh.model.occ.healShapes(
                        sewFaces=True, makeSolids=False,
                        tolerance=max(1e-5 * diag, 1e-6),
                        fixDegenerated=False, fixSmallEdges=False,
                        fixSmallFaces=False)
                    gmsh.model.occ.synchronize()
                except Exception as e:
                    log(f"  (surface sewing skipped: {e})")

    vols = gmsh.model.getEntities(3)
    surfs = gmsh.model.getEntities(2)
    n_bodies = len(vols) if vols else len(surfs)
    # surface-only models: always merge duplicated border edges, otherwise
    # each face is meshed independently and the shell mesh is not conformal
    force_glue = family == "shell" and not vols and len(surfs) > 1
    if (settings.glue or force_glue) and n_bodies > 1:
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        n = len(gmsh.model.getEntities(3)) or len(gmsh.model.getEntities(2))
        log(f"Glued touching {'faces' if force_glue else 'bodies'} "
            f"(conformal interfaces): {n_bodies} -> {n}")

    if settings.defeature_faces:
        _defeature(settings.defeature_faces, log)

    bbox = gmsh.model.getBoundingBox(-1, -1)
    log(f"Bounding box: x[{bbox[0]:.4g}, {bbox[3]:.4g}] "
        f"y[{bbox[1]:.4g}, {bbox[4]:.4g}] z[{bbox[2]:.4g}, {bbox[5]:.4g}]")

    if settings.symmetry:
        _apply_symmetry_cuts(settings.symmetry, bbox, log)
        bbox = gmsh.model.getBoundingBox(-1, -1)
    return bbox, False


def _load_tessellation(settings: MeshSettings, family: str, log):
    """Import a tessellated surface mesh (STL/OBJ/PLY) and weld coincident
    nodes. Returns the bounding box. Shells use the triangulation as-is; solids
    reconstruct a volume from the (watertight) surface and tetrahedralize the
    interior. CAD-only features are rejected up front with an explanation."""
    fmt = format_name(settings.step_file)
    unsupported = [
        ("symmetry planes", settings.symmetry),
        ("defeaturing", settings.defeature_faces),
        ("refinement regions", settings.refinements),
        ("per-face mesh sizes", settings.face_sizes),
        ("face sets / roles", settings.collect_faces),
    ]
    active = [name for name, val in unsupported if val]
    if active:
        raise RuntimeError(
            f"{fmt} input carries no CAD geometry (only triangles), so these "
            f"are not available: {', '.join(active)}. Use a STEP/IGES/BREP "
            f"file for those features.")

    log(f"Importing {fmt} tessellation: {settings.step_file}")
    gmsh.merge(settings.step_file)
    # STL stores every facet independently; weld coincident vertices so the
    # mesh shares nodes (otherwise every edge is a free/cracked edge).
    n_before = len(gmsh.model.mesh.getNodes()[0])
    gmsh.model.mesh.removeDuplicateNodes()
    n_after = len(gmsh.model.mesh.getNodes()[0])
    n_tris = len(gmsh.model.mesh.getElementsByType(GMSH_TRI3)[0])
    n_quads = len(gmsh.model.mesh.getElementsByType(GMSH_QUAD4)[0])
    if n_tris + n_quads == 0:
        raise RuntimeError(
            f"No surface elements were found in the {fmt} file - it may be "
            "empty or in an unsupported variant.")
    if n_after < n_before:
        log(f"Welded {n_before - n_after} coincident node(s) "
            f"({n_before} -> {n_after})")
    faces = f"{n_tris} triangles" + (f" + {n_quads} quads" if n_quads else "")
    log(f"Imported {faces}, {n_after} nodes")

    if family == "solid":
        _tessellation_to_solid(settings, fmt, n_tris, n_quads, log)
    return gmsh.model.getBoundingBox(-1, -1)


def _tessellation_to_solid(settings: MeshSettings, fmt: str,
                           n_tris: int, n_quads: int, log) -> None:
    """Reconstruct a solid volume from a watertight surface tessellation and
    tetrahedralize the interior, keeping the surface triangulation as the
    boundary (no surface remeshing). The mesh must be a closed triangle
    manifold."""
    order = ETYPES[settings.element_type]["order"]
    if n_quads:
        raise RuntimeError(
            f"{fmt} solid meshing needs an all-triangle surface, but this file "
            f"has {n_quads} quad facet(s). Triangulate it, or mesh it with "
            f"shell elements (TRI3/QUAD4).")

    # A volume can only be built from a closed (watertight) surface; count the
    # free edges of the triangulation and refuse otherwise.
    conn = np.asarray(gmsh.model.mesh.getElementsByType(GMSH_TRI3)[1],
                      dtype=np.int64).reshape(-1, 3)
    edges = np.sort(np.vstack([conn[:, [0, 1]], conn[:, [1, 2]], conn[:, [2, 0]]]),
                    axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_free = int((counts == 1).sum())
    n_nonman = int((counts > 2).sum())
    if n_free:
        raise RuntimeError(
            f"{fmt} solid meshing needs a watertight (closed) surface, but "
            f"{n_free} free edge(s) were found - the surface has holes or gaps. "
            f"Repair the mesh (e.g. MeshLab/netfabb), or mesh it with shell "
            f"elements (TRI3/QUAD4) instead.")
    if n_nonman:
        log(f"WARNING: {n_nonman} non-manifold edge(s) found - the volume "
            f"mesh may fail or be invalid.")

    log("Reconstructing a solid volume from the surface tessellation ...")
    gmsh.model.mesh.createTopology()
    surfs = [t for _, t in gmsh.model.getEntities(2)]
    sl = gmsh.model.geo.addSurfaceLoop(surfs)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # only the interior is meshed; the surface triangulation is fixed, so the
    # size settings just control how coarse the interior tets may be
    gmsh.option.setNumber("Mesh.MeshSizeMax", settings.size_max)
    gmsh.option.setNumber("Mesh.MeshSizeMin", settings.size_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", ALGO3D.get(settings.algorithm3d, 1))
    gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 1)
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    t0 = time.perf_counter()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        raise RuntimeError(
            f"Volume meshing of the reconstructed surface failed: {e}\n"
            "The tessellation must be a clean, closed, non-self-intersecting "
            "manifold. Repair it (MeshLab/netfabb) or mesh it with shells."
        ) from e
    n_tet = len(gmsh.model.mesh.getElementsByType(TET_TYPE[1][0])[0])
    log(f"  volume mesh: {time.perf_counter() - t0:.1f} s ({n_tet} tets)")

    if settings.optimize:
        t0 = time.perf_counter()
        try:
            gmsh.model.mesh.optimize("Netgen")
            log(f"  optimization (Netgen): {time.perf_counter() - t0:.1f} s")
        except Exception as e:
            log(f"  optimization skipped: {e}")

    if order == 2:
        t0 = time.perf_counter()
        gmsh.model.mesh.setOrder(2)
        log(f"  second order (TET10): {time.perf_counter() - t0:.1f} s")


def _defeature(face_tags: list[int], log) -> None:
    """Remove the given faces (holes, fillets, ...) from the solids using
    OpenCASCADE defeaturing."""
    vols = [t for _, t in gmsh.model.getEntities(3)]
    if not vols:
        raise RuntimeError("Defeaturing requires solid volumes "
                           "(not available for surface-only models).")
    n_faces = len(gmsh.model.getEntities(2))
    try:
        gmsh.model.occ.defeature(vols, list(face_tags))
        gmsh.model.occ.synchronize()
    except Exception as e:
        raise RuntimeError(
            f"Defeaturing failed: {e}\n"
            "Not every feature can be removed - the surrounding faces must "
            "be extendable to close the gap. Try removing fewer faces "
            "(e.g. a fillet together with its adjacent faces)."
        ) from e
    log(f"Defeatured {len(face_tags)} face(s): "
        f"{n_faces} -> {len(gmsh.model.getEntities(2))} faces")


def _apply_symmetry_cuts(planes: list[SymmetryPlane], bbox, log) -> None:
    """Boolean-intersect the model with half-space boxes, one per symmetry
    plane. Works on solids if present, otherwise on the surfaces."""
    occ = gmsh.model.occ
    diag = float(np.linalg.norm(np.array(bbox[3:]) - np.array(bbox[:3])))
    margin = 0.5 * diag + 1.0
    cut_dim = 3 if gmsh.model.getEntities(3) else 2

    for sp in planes:
        i = AXIS_INDEX[sp.axis]
        lo = [bbox[0] - margin, bbox[1] - margin, bbox[2] - margin]
        hi = [bbox[3] + margin, bbox[4] + margin, bbox[5] + margin]
        if sp.keep == "+":
            lo[i] = sp.offset
        else:
            hi[i] = sp.offset

        if hi[i] <= lo[i]:
            raise RuntimeError(
                f"Symmetry plane {sp.axis.upper()} at {sp.offset} (keep {sp.keep}) "
                f"leaves no geometry to mesh."
            )

        tool = occ.addBox(lo[0], lo[1], lo[2],
                          hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        objs = gmsh.model.getEntities(cut_dim)
        out, _ = occ.intersect(objs, [(3, tool)], removeObject=True, removeTool=True)
        occ.synchronize()

        if not gmsh.model.getEntities(cut_dim):
            raise RuntimeError(
                f"Symmetry cut {sp.axis.upper()}={sp.offset} (keep {sp.keep}) "
                f"removed all geometry. Check the plane offset and side."
            )
        log(f"Applied symmetry cut: {sp.axis.upper()} = {sp.offset} "
            f"(kept {sp.keep} side, {len(out)} entit(ies) remain)")


# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------

def _apply_refinements(settings: MeshSettings, log) -> None:
    """Create gmsh size fields for refinement regions and per-face sizes."""
    if not settings.refinements and not settings.face_sizes:
        return
    fld = gmsh.model.mesh.field
    ids = []
    for r in settings.refinements:
        size = float(r["size"])
        p = [float(v) for v in r["params"]]
        if r["kind"] == "sphere":
            f = fld.add("Ball")
            fld.setNumber(f, "XCenter", p[0])
            fld.setNumber(f, "YCenter", p[1])
            fld.setNumber(f, "ZCenter", p[2])
            fld.setNumber(f, "Radius", p[3])
            log(f"Refinement sphere at ({p[0]:g}, {p[1]:g}, {p[2]:g}) "
                f"r={p[3]:g}, size {size:g}")
        else:
            f = fld.add("Box")
            fld.setNumber(f, "XMin", p[0])
            fld.setNumber(f, "YMin", p[1])
            fld.setNumber(f, "ZMin", p[2])
            fld.setNumber(f, "XMax", p[3])
            fld.setNumber(f, "YMax", p[4])
            fld.setNumber(f, "ZMax", p[5])
            log(f"Refinement box ({p[0]:g}, {p[1]:g}, {p[2]:g}) .. "
                f"({p[3]:g}, {p[4]:g}, {p[5]:g}), size {size:g}")
        fld.setNumber(f, "VIn", size)
        fld.setNumber(f, "VOut", settings.size_max)
        fld.setNumber(f, "Thickness", 2.0 * size)  # smooth size transition
        ids.append(f)

    existing = {t for _, t in gmsh.model.getEntities(2)}
    for tag, size in settings.face_sizes.items():
        if int(tag) not in existing:
            log(f"WARNING: face {tag} for local size not found - skipped")
            continue
        size = float(size)
        fd = fld.add("Distance")
        fld.setNumbers(fd, "SurfacesList", [int(tag)])
        fld.setNumber(fd, "Sampling", 100)
        ft = fld.add("Threshold")
        fld.setNumber(ft, "InField", fd)
        fld.setNumber(ft, "SizeMin", size)
        fld.setNumber(ft, "SizeMax", settings.size_max)
        fld.setNumber(ft, "DistMin", 0.0)
        fld.setNumber(ft, "DistMax", 4.0 * size)
        ids.append(ft)
        log(f"Local size {size:g} on face {tag}")

    fmin = fld.add("Min")
    fld.setNumbers(fmin, "FieldsList", ids)
    fld.setAsBackgroundMesh(fmin)


def _generate_mesh(settings: MeshSettings, log) -> None:
    etype = ETYPES[settings.element_type]
    family = etype["family"]

    # region/face sizes must not be clamped away by the global minimum size
    eff_min = min([settings.size_min] +
                  [float(r["size"]) for r in settings.refinements] +
                  [float(s) for s in settings.face_sizes.values()])
    gmsh.option.setNumber("Mesh.MeshSizeMax", settings.size_max)
    gmsh.option.setNumber("Mesh.MeshSizeMin", eff_min)
    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        settings.curvature_elems if settings.curvature_refine else 0,
    )
    gmsh.option.setNumber("Mesh.Algorithm3D", ALGO3D.get(settings.algorithm3d, 1))
    gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 1)
    # optimization is run explicitly below so its cost shows up in the log
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    if family == "shell" and etype["recombine"]:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads

    log(f"Meshing {settings.element_type} (element size {settings.size_min:g} "
        f".. {settings.size_max:g}) ...")
    stages = ((1, "curve"), (2, "surface"))
    if family == "solid":
        stages += ((3, "volume"),)
    for dim, name in stages:
        t0 = time.perf_counter()
        try:
            gmsh.model.mesh.generate(dim)
        except Exception as e:
            raise RuntimeError(
                f"Mesh generation failed during {name} meshing: {e}\n"
                "Hints: try disabling 'Heal geometry', a different 3D algorithm "
                "(HXT/Frontal), or smaller elements. 'PLC Error' usually means "
                "the surface mesh self-intersects at a dirty geometry spot "
                "(tangent faces, knife edges, tiny gaps)."
            ) from e
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        log(f"  {name} mesh: {time.perf_counter() - t0:.1f} s "
            f"({n_nodes} nodes so far)")

    if settings.optimize:
        methods = ((("", "standard"), ("Netgen", "Netgen")) if family == "solid"
                   else (("Laplace2D", "Laplace2D"), ("Relocate2D", "Relocate2D")))
        for method, label in methods:
            t0 = time.perf_counter()
            try:
                gmsh.model.mesh.optimize(method)
                log(f"  optimization ({label or 'standard'}): "
                    f"{time.perf_counter() - t0:.1f} s")
            except Exception as e:
                log(f"  optimization ({label}) skipped: {e}")

    if family == "solid" and etype["order"] == 2:
        t0 = time.perf_counter()
        gmsh.model.mesh.setOrder(2)
        log(f"  second order (TET10): {time.perf_counter() - t0:.1f} s")
        try:
            etags, _ = gmsh.model.mesh.getElementsByType(TET_TYPE[2][0])
            q = np.asarray(gmsh.model.mesh.getElementQualities(etags, "minSICN"))
            n_bad = int((q < 0).sum())
            if n_bad:
                log(f"  {n_bad} invalid curved elements - running high-order "
                    f"optimizer ...")
                gmsh.model.mesh.optimize("HighOrderFastCurving")
        except Exception as e:
            log(f"  high-order check skipped: {e}")
    log("Mesh generation finished")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _get_sorted_nodes():
    node_tags, node_xyz, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_xyz = np.asarray(node_xyz, dtype=float).reshape(-1, 3)
    srt = np.argsort(node_tags)
    return node_tags[srt], node_xyz[srt]


def _map_and_prune(node_tags, node_xyz, conn):
    """Renumber connectivity to compact 1-based ids, prune unused nodes.
    Returns (coords, elems 1-based, tag_map)."""
    conn_idx = np.searchsorted(node_tags, conn)  # 0-based indices
    used = np.zeros(len(node_tags), dtype=bool)
    used[conn_idx.ravel()] = True
    new_index = np.cumsum(used) - 1              # old 0-based -> new 0-based
    coords = node_xyz[used]
    elems = new_index[conn_idx] + 1              # 1-based
    new_id_all = np.where(used, new_index + 1, 0)   # 0 = pruned
    return coords, elems, (node_tags, new_id_all)


def _extract_tets(order: int, log):
    """Return (coords, tets (M,nn) 1-based LS-DYNA ordering, elem_parts,
    part_names, tag_map); unused nodes pruned, positive volumes enforced."""
    tet_type, nn = TET_TYPE[order][:2]
    node_tags, node_xyz = _get_sorted_nodes()

    conn_blocks, part_blocks, part_names = [], [], []
    for _, tag in sorted(gmsh.model.getEntities(3)):
        etags, conn = gmsh.model.mesh.getElementsByType(tet_type, tag)
        if len(etags) == 0:
            continue
        conn_blocks.append(np.asarray(conn, dtype=np.int64).reshape(-1, nn))
        part_blocks.append(np.full(len(etags), len(part_names), dtype=np.int64))
        name = gmsh.model.getEntityName(3, tag)
        part_names.append(name.split("/")[-1] if name else "")
    if not conn_blocks:
        raise RuntimeError("No tetrahedra were generated.")
    conn = np.vstack(conn_blocks)
    elem_parts = np.concatenate(part_blocks)
    if len(part_names) > 1:
        log(f"{len(part_names)} bodies -> separate parts")

    coords, tets, tag_map = _map_and_prune(node_tags, node_xyz, conn)

    if order == 2:
        tets = tets[:, GMSH2DYNA_TET10]

    # enforce positive volume (LS-DYNA requires positive jacobian)
    p = coords[tets[:, :4] - 1]
    v6 = np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    )
    neg = v6 < 0
    if neg.any():
        tets[neg] = tets[neg][:, NEGFIX[nn]]
        log(f"Reoriented {int(neg.sum())} inverted tetrahedra")
    zero = v6 == 0
    if zero.any():
        raise RuntimeError(f"{int(zero.sum())} degenerate (zero-volume) tetrahedra found.")

    return coords, tets, elem_parts, part_names, tag_map


def _extract_shells(log):
    """Return (coords, shells (M,4) 1-based - triangles with the 4th node
    repeated, elem_parts, part_names, tag_map, checks). All shells form one
    part; orientation is made consistent, free edges are counted."""
    node_tags, node_xyz = _get_sorted_nodes()

    blocks = []
    for gtype, nn in ((GMSH_TRI3, 3), (GMSH_QUAD4, 4)):
        etags, conn = gmsh.model.mesh.getElementsByType(gtype)
        if len(etags) == 0:
            continue
        conn = np.asarray(conn, dtype=np.int64).reshape(-1, nn)
        if nn == 3:
            conn = np.column_stack((conn, conn[:, 2]))
        blocks.append(conn)
    if not blocks:
        raise RuntimeError("No shell elements were generated.")
    conn = np.vstack(blocks)

    coords, shells, tag_map = _map_and_prune(node_tags, node_xyz, conn)
    shells, checks = _orient_shells(shells, log)
    elem_parts = np.zeros(len(shells), dtype=np.int64)
    return coords, shells, elem_parts, [""], tag_map, checks


def _orient_shells(shells: np.ndarray, log):
    """Make shell normals consistent across shared (manifold) edges by
    flipping elements; count free and non-manifold edges."""
    m = len(shells)
    corners = []
    for row in shells:
        if row[2] == row[3]:
            corners.append([int(row[0]), int(row[1]), int(row[2])])
        else:
            corners.append([int(v) for v in row])

    edge_use: dict = {}
    for e, cs in enumerate(corners):
        k = len(cs)
        for i in range(k):
            a, b = cs[i], cs[(i + 1) % k]
            edge_use.setdefault((min(a, b), max(a, b)), []).append((e, a))
    free = sum(1 for v in edge_use.values() if len(v) == 1)
    nonman = sum(1 for v in edge_use.values() if len(v) > 2)

    adj = [[] for _ in range(m)]
    for v in edge_use.values():
        if len(v) == 2:
            (e1, a1), (e2, a2) = v
            same_dir = a1 == a2   # same start node -> inconsistent orientation
            adj[e1].append((e2, same_dir))
            adj[e2].append((e1, same_dir))

    flipped = np.zeros(m, dtype=bool)
    visited = np.zeros(m, dtype=bool)
    for seed in range(m):
        if visited[seed]:
            continue
        visited[seed] = True
        dq = collections.deque([seed])
        while dq:
            e = dq.popleft()
            for nb, same_dir in adj[e]:
                if not visited[nb]:
                    visited[nb] = True
                    flipped[nb] = bool(same_dir) != bool(flipped[e])
                    dq.append(nb)

    n_flip = int(flipped.sum())
    if n_flip:
        for e in np.flatnonzero(flipped):
            cs = corners[e]
            corners[e] = [cs[0]] + cs[:0:-1]   # cyclic reversal, keep node 1
        log(f"Aligned shell normals: flipped {n_flip} element(s)")

    out = np.empty((m, 4), dtype=np.int64)
    for e, cs in enumerate(corners):
        out[e] = cs if len(cs) == 4 else [cs[0], cs[1], cs[2], cs[2]]
    return out, {"free_edges": free, "nonmanifold_edges": nonman,
                 "orientation_flips": n_flip}


def _collect_face_data(face_tags, tag_map, element_type: str, log):
    """Nodes and segments (surface element corner nodes, (S,4) with triangles
    padded) for the requested face tags, as 1-based node indices."""
    if not face_tags:
        return {}, {}
    tags_sorted, new_id_all = tag_map
    if ETYPES[element_type]["family"] == "solid":
        order = ETYPES[element_type]["order"]
        face_elem_types = [(TET_TYPE[order][2], TET_TYPE[order][3], 3)]
    else:
        face_elem_types = [(GMSH_TRI3, 3, 3), (GMSH_QUAD4, 4, 4)]

    def to_new(arr):
        arr = np.asarray(arr, dtype=np.int64)
        idx = np.minimum(np.searchsorted(tags_sorted, arr), len(tags_sorted) - 1)
        ok = tags_sorted[idx] == arr
        return np.where(ok, new_id_all[idx], 0)

    face_nodes, face_segs = {}, {}
    for tag in face_tags:
        try:
            ntags = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)[0]
        except Exception:
            log(f"WARNING: face {tag} not found in the meshed model - skipped")
            continue
        ids = to_new(ntags)
        ids = np.unique(ids[ids > 0])
        if len(ids) == 0:
            log(f"WARNING: face {tag} has no mesh nodes - skipped")
            continue
        face_nodes[tag] = ids

        segs = []
        for gtype, nn, corner_n in face_elem_types:
            try:
                _, conn = gmsh.model.mesh.getElementsByType(gtype, tag)
            except Exception:
                continue
            if len(conn) == 0:
                continue
            c = to_new(np.asarray(conn, dtype=np.int64).reshape(-1, nn)[:, :corner_n])
            if corner_n == 3:
                c = np.column_stack((c, c[:, 2]))
            segs.append(c[(c > 0).all(axis=1)])
        face_segs[tag] = (np.vstack(segs) if segs
                          else np.zeros((0, 4), dtype=np.int64))
        log(f"Face {tag}: {len(ids)} nodes, {len(face_segs[tag])} segments")
    return face_nodes, face_segs


def _find_symmetry_nodes(coords: np.ndarray, planes: list[SymmetryPlane], bbox) -> dict:
    """1-based node indices lying on each symmetry plane."""
    diag = float(np.linalg.norm(np.array(bbox[3:]) - np.array(bbox[:3])))
    tol = max(1e-6 * diag, 1e-9)
    sym_nodes = {}
    for sp in planes:
        i = AXIS_INDEX[sp.axis]
        mask = np.abs(coords[:, i] - sp.offset) <= tol
        sym_nodes[sp.axis] = np.flatnonzero(mask) + 1
    return sym_nodes


def _find_symmetry_segments(coords, planes, sym_nodes, tag_map,
                            element_type: str) -> dict:
    """Boundary surface segments (as (S, 4) 1-based node ids, triangles padded
    to a degenerate quad) that lie entirely on each symmetry plane.

    Only meaningful for solids - the symmetry cut leaves a flat model face
    whose boundary triangles have all three corners on the plane. For shells
    the plane is a cut edge with no area, so the result is empty."""
    if ETYPES[element_type]["family"] != "solid" or not planes:
        return {}
    order = ETYPES[element_type]["order"]
    gtype, nn = TET_TYPE[order][2], TET_TYPE[order][3]  # surface tri type / nodes
    tags_sorted, new_id_all = tag_map
    try:
        _, conn = gmsh.model.mesh.getElementsByType(gtype)
    except Exception:
        return {}
    if len(conn) == 0:
        return {}
    corners = np.asarray(conn, dtype=np.int64).reshape(-1, nn)[:, :3]
    idx = np.minimum(np.searchsorted(tags_sorted, corners), len(tags_sorted) - 1)
    ok = tags_sorted[idx] == corners
    ids = np.where(ok, new_id_all[idx], 0)          # (T, 3) compact 1-based ids
    ids = ids[(ids > 0).all(axis=1)]

    n = len(coords)
    sym_segs = {}
    for sp in planes:
        on_plane = np.zeros(n + 1, dtype=bool)       # index by 1-based node id
        on_plane[np.asarray(sym_nodes[sp.axis], dtype=np.int64)] = True
        segs = ids[on_plane[ids].all(axis=1)]
        sym_segs[sp.axis] = (np.column_stack((segs, segs[:, 2])) if len(segs)
                             else np.zeros((0, 4), dtype=np.int64))
    return sym_segs


def _count_duplicate_nodes(coords: np.ndarray) -> int:
    _, counts = np.unique(np.round(coords, 8), axis=0, return_counts=True)
    return int((counts > 1).sum())


# --------------------------------------------------------------------------
# statistics, quality criteria, mass properties
# --------------------------------------------------------------------------

def _poly_angles(p: np.ndarray) -> np.ndarray:
    """Interior angles [deg] of planar polygons p (M, k, 3)."""
    k = p.shape[1]
    ang = []
    for i in range(k):
        a = p[:, (i - 1) % k] - p[:, i]
        b = p[:, (i + 1) % k] - p[:, i]
        denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1),
                           1e-300)
        cosv = np.clip(np.einsum("ij,ij->i", a, b) / denom, -1.0, 1.0)
        ang.append(np.degrees(np.arccos(cosv)))
    return np.stack(ang, axis=1)


def _quality_criteria(coords, elems, element_type, sicn):
    """Evaluate LS-DYNA-style quality criteria.
    Returns (criteria list, failed element row indices)."""
    family = ETYPES[element_type]["family"]
    limits = QUALITY_LIMITS[family]
    crit, fail_masks = [], []

    def add(name, values, info_only=False):
        if name in limits and not info_only:
            sense, lim = limits[name]
            mask = values > lim if sense == "max" else values < lim
            worst = float(values.max() if sense == "max" else values.min())
            crit.append({"name": name, "worst": worst,
                         "limit": f"{'<=' if sense == 'max' else '>='} {lim:g}",
                         "n_fail": int(mask.sum())})
            fail_masks.append(mask)
        else:
            worst = float(values.min())
            crit.append({"name": name, "worst": worst, "limit": "info",
                         "n_fail": 0})

    if family == "solid":
        p = coords[elems[:, :4] - 1]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        el = np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=1)
                       for a, b in pairs], axis=1)
        add("aspect ratio", el.max(1) / np.maximum(el.min(1), 1e-300))
        if sicn is not None:
            add("SICN", sicn)
        v = np.abs(np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0])) / 6.0
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        fa = np.stack([0.5 * np.linalg.norm(
            np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]), axis=1)
            for a, b, c in faces], axis=1)
        # min altitude = timestep-critical characteristic length for tets
        add("min altitude (dt)", 3.0 * v / np.maximum(fa.max(1), 1e-300),
            info_only=True)
        add("min edge", el.min(1), info_only=True)
    else:
        p = coords[elems - 1]                     # (M, 4, 3)
        is_tri = elems[:, 2] == elems[:, 3]
        el4 = np.stack([np.linalg.norm(p[:, i] - p[:, (i + 1) % 4], axis=1)
                        for i in range(4)], axis=1)
        # triangles: ignore the degenerate 4th edge pair
        emin = np.where(is_tri, np.sort(el4, axis=1)[:, 1], el4.min(1))
        add("aspect ratio", el4.max(1) / np.maximum(emin, 1e-300))
        if sicn is not None:
            add("SICN", sicn)
        n1 = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n2 = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
        denom = np.maximum(np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1),
                           1e-300)
        warp = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", n1, n2) / denom, -1.0, 1.0)))
        warp[is_tri] = 0.0
        add("warpage [deg]", warp)
        ang4 = _poly_angles(p)
        ang_min = np.where(is_tri, _poly_angles(p[:, :3]).min(1), ang4.min(1))
        add("min angle [deg]", ang_min)
        add("min edge", emin, info_only=True)

    if fail_masks:
        failed = np.flatnonzero(np.logical_or.reduce(fail_masks))
    else:
        failed = np.zeros(0, dtype=np.int64)
    return crit, failed


def _mass_properties(coords, elems, element_type):
    """COG and inertia tensor about the COG, per unit density (solids) or
    per unit density*thickness (shells)."""
    p = coords[elems[:, :4] - 1]
    if ETYPES[element_type]["family"] == "solid":
        v = np.einsum("ij,ij->i",
                      np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                      p[:, 3] - p[:, 0]) / 6.0
        total = v.sum()
        cog = (v[:, None] * p.mean(axis=1)).sum(0) / total
        # 4-point quadrature, exact for the quadratic integrand x x^T
        a, b = 0.5854101966249685, 0.13819660112501053
        c_mat = np.zeros((3, 3))
        for k in range(4):
            lam = np.full(4, b)
            lam[k] = a
            q = np.einsum("j,mjk->mk", lam, p) - cog
            c_mat += np.einsum("m,mi,mj->ij", 0.25 * v, q, q)
    else:
        tris = (p[:, [0, 1, 2]], p[:, [0, 2, 3]])   # padded tris: 2nd is empty
        areas = [0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
            for t in tris]
        total = sum(a.sum() for a in areas)
        cog = sum((a[:, None] * t.mean(axis=1)).sum(0)
                  for a, t in zip(areas, tris)) / total
        c_mat = np.zeros((3, 3))
        for a, t in zip(areas, tris):
            # midpoint rule (3 edge midpoints), exact for quadratic integrands
            for i, j in ((0, 1), (1, 2), (2, 0)):
                q = 0.5 * (t[:, i] + t[:, j]) - cog
                c_mat += np.einsum("m,mi,mj->ij", a / 3.0, q, q)
    inertia = np.trace(c_mat) * np.eye(3) - c_mat
    return cog, inertia


def critical_timestep(stats: dict, element_type: str, mat: dict) -> float | None:
    """Estimated explicit critical timestep dt = Lc / c (no TSSFAC applied).

    Lc is the smallest element characteristic length from the mesh statistics
    (tet minimum altitude for solids, minimum edge for shells - a conservative
    proxy) and c the acoustic wave speed of the elastic material: constrained
    (bulk) for solids, plane-stress for shells. Units follow the model
    (e.g. seconds in mm-t-s). Returns None if data is missing or invalid."""
    lc = stats.get("char_length")
    if not lc or lc <= 0 or not mat:
        return None
    e, nu, ro = float(mat["e"]), float(mat["pr"]), float(mat["ro"])
    if e <= 0 or ro <= 0 or not -1.0 < nu < 0.5:
        return None
    if ETYPES[element_type]["family"] == "solid":
        c = (e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * ro)) ** 0.5
    else:
        c = (e / (ro * (1.0 - nu * nu))) ** 0.5
    return float(lc) / c


def _collect_stats(coords: np.ndarray, elems: np.ndarray, element_type: str,
                   bbox) -> dict:
    family = ETYPES[element_type]["family"]
    p = coords[elems[:, :4] - 1]
    if family == "solid":
        measure = np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        ) / 6.0
        gmsh_types = [TET_TYPE[ETYPES[element_type]["order"]][0]]
        label = "volume"
    else:
        a1 = 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        a2 = 0.5 * np.linalg.norm(
            np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=1)
        measure = a1 + a2
        gmsh_types = [GMSH_TRI3, GMSH_QUAD4]
        label = "area"

    stats = {
        "n_nodes": int(len(coords)),
        "n_elems": int(len(elems)),
        "measure": float(measure.sum()),
        "measure_label": label,
        "bbox": tuple(bbox),
    }

    cog, inertia = _mass_properties(coords, elems, element_type)
    stats["cog"] = tuple(float(x) for x in cog)
    stats["inertia_unit_density"] = inertia.tolist()

    q = None
    try:
        q_all = []
        for t in gmsh_types:
            etags, _ = gmsh.model.mesh.getElementsByType(t)
            if len(etags):
                q_all.append(np.asarray(
                    gmsh.model.mesh.getElementQualities(etags, "minSICN")))
        q = np.concatenate(q_all)
        stats["quality_min"] = float(q.min())
        stats["quality_avg"] = float(q.mean())
        hist, _ = np.histogram(np.clip(q, 0.0, 1.0), bins=10, range=(0.0, 1.0))
        stats["quality_hist"] = hist.tolist()
    except Exception:
        pass

    try:
        crit, failed = _quality_criteria(coords, elems, element_type, q)
        stats["criteria"] = crit
        stats["failed_elems"] = failed        # 0-based element rows
        # timestep-critical characteristic length of the worst element
        lc_name = "min altitude (dt)" if family == "solid" else "min edge"
        for cr in crit:
            if cr["name"] == lc_name:
                stats["char_length"] = cr["worst"]
    except Exception:
        pass

    if family == "solid" and q is not None:
        # locations + local size of the 5 worst elements, whatever their
        # quality (auto-refine filters against its own threshold, the log
        # warning against 0.05); getElementsByType returns elements grouped
        # by volume in tag order, matching the row order of `elems`
        bad = np.argsort(q)[:5]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        stats["worst_elements"] = []
        for i in bad:
            c = p[i].mean(axis=0)
            h = float(np.mean([np.linalg.norm(p[i, a] - p[i, b])
                               for a, b in pairs]))
            stats["worst_elements"].append(
                (float(q[i]), tuple(round(float(x), 2) for x in c), h))
    return stats


def _log_stats(stats: dict, sym_nodes: dict, log) -> None:
    log(f"Nodes: {stats['n_nodes']}   elements: {stats['n_elems']}   "
        f"mesh {stats['measure_label']}: {stats['measure']:.6g}")
    c = stats.get("cog")
    if c:
        log(f"Center of gravity: ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})")
    if stats.get("duplicate_nodes"):
        log(f"WARNING: {stats['duplicate_nodes']} duplicate node locations "
            f"(coincident but unmerged nodes)")
    if "free_edges" in stats:
        log(f"Shell checks: {stats['free_edges']} free edges "
            f"(0 = watertight), {stats['nonmanifold_edges']} non-manifold "
            f"edges, {stats['orientation_flips']} normals flipped")
    if "quality_min" in stats:
        log(f"Element quality (SICN, 1.0 = ideal): min {stats['quality_min']:.3f}, "
            f"avg {stats['quality_avg']:.3f}")
        if stats.get("quality_hist"):
            hist = stats["quality_hist"]
            peak = max(max(hist), 1)
            log("Quality histogram:")
            for i, n in enumerate(hist):
                bar = "#" * round(30 * n / peak)
                log(f"  {i / 10:.1f}-{(i + 1) / 10:.1f} {n:>9d} {bar}")
    if stats.get("criteria"):
        log("Quality criteria:")
        for cr in stats["criteria"]:
            fail = f"FAILS {cr['n_fail']}" if cr["n_fail"] else "ok"
            log(f"  {cr['name']:<18} worst {cr['worst']:>10.4g}   "
                f"limit {cr['limit']:<10} {fail}")
        n_failed = len(stats.get("failed_elems", ()))
        if n_failed:
            log(f"{n_failed} element(s) fail at least one criterion")
    bad_spots = [w for w in stats.get("worst_elements", ()) if w[0] < 0.05]
    if bad_spots:
        log("WARNING: badly shaped elements (SICN < 0.05) at:")
        for qual, c, h in bad_spots:
            log(f"    SICN {qual:.5f} near ({c[0]}, {c[1]}, {c[2]})")
        log("These are usually caused by dirty geometry at those locations "
            "(tangent faces, knife edges, tiny edges/faces). Suppressing the "
            "offending faces (defeature) or fixing the CAD is the best remedy.")
    for axis, ids in sym_nodes.items():
        log(f"Nodes on {axis.upper()}-symmetry plane: {len(ids)}")
