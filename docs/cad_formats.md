# CAD / mesh input formats for k_mesher

This note is the result of a survey of the CAD-interchange landscape and of what
the k_mesher tool-chain (gmsh + the OpenCASCADE geometry kernel) can actually
read. It explains **which formats are supported today**, **which could be added
and at what cost**, and **which are impractical** without commercial licences.

## How k_mesher reads geometry

k_mesher does not parse CAD files itself. It hands the file to
[gmsh](https://gmsh.info), which uses the **OpenCASCADE Technology (OCCT)**
kernel for boundary-representation (B-rep) geometry. Two very different import
paths exist, and the distinction drives everything below:

| Path | gmsh call | What you get | Pipeline features |
|---|---|---|---|
| **B-rep (CAD)** | `occ.importShapes` | exact solids/surfaces (NURBS, analytic faces) | full: solids→tets, symmetry cuts, defeature, per-face sizing, curvature refinement, face sets/BCs/loads |
| **Tessellation (mesh)** | `gmsh.merge` | a fixed surface triangulation | shells as-is; **tets by reconstructing the enclosed volume if the surface is watertight**; no symmetry/defeature/refinement (there is no CAD geometry to cut or resize) |

Everything that makes k_mesher useful for solids (the boolean symmetry cut, the
OpenCASCADE defeaturing, face scanning with areas/centroids, curvature-driven
sizing) requires a **B-rep**. A tessellation is just triangles: it becomes a
shell deck, or — if it is a closed, watertight manifold — a tet mesh whose
boundary *is* the input triangulation (the interior is filled with tets, the
surface is not remeshed).

## Supported today

| Format | Ext. | Kind | Solid meshing | Shell meshing | Notes |
|---|---|---|---|---|---|
| **STEP** | `.step` `.stp` | B-rep | ✅ | ✅ | Recommended. ISO 10303; carries solids reliably. |
| **BREP** | `.brep` `.brp` | B-rep | ✅ | ✅ | OCCT-native, exact. Behaves identically to STEP. |
| **IGES** | `.iges` `.igs` | B-rep | ✅¹ | ✅¹ | Surface-based legacy format (frozen since 1996). |
| **STL** | `.stl` | mesh | ✅² | ✅ | Triangulated surface; the de-facto 3-D-print/CAD export mesh. |
| **OBJ** | `.obj` | mesh | ✅² | ✅ | Wavefront surface mesh (tris and/or quads; quads → shells only). |
| **PLY** | `.ply` | mesh | ✅² | ✅ | Stanford polygon / 3-D-scan surface mesh. |

¹ **IGES specifics.** IGES stores only trimmed surfaces, never a solid body, so
on import you always get loose faces. k_mesher automatically **sews** those
faces with a tolerance scaled to the model size (`1e-5 × bbox diagonal`):

- for **solid** meshing it then tries `makeSolids` to close the shell into a
  volume that can be filled with tets;
- for **shell** meshing it sews the faces so neighbours share edges and the
  shell mesh is conformal (watertight) instead of a pile of independent
  patches.

The default OCCT sewing tolerance (`1e-8`) is far too tight for the sub-tolerance
gaps typical of IGES exports, which is why the size-relative tolerance matters.
IGES round-trips are still less reliable than STEP/BREP — prefer STEP when you
can choose the export.

### How the mesh formats are handled

STL/OBJ/PLY are loaded with `gmsh.merge`, then **coincident nodes are welded**
(`removeDuplicateNodes`). This is essential: STL in particular stores every
facet with its own three vertices, so without welding every edge is a free edge
and LS-DYNA would see a cracked mesh of unconnected triangles. After welding,
the triangles become a proper conformal shell mesh, shell normals are aligned,
and free/non-manifold edges are reported exactly as for CAD shells.

²**Tets from a tessellation.** When a solid element type is chosen for a
tessellated input, k_mesher reconstructs a volume from the surface and fills it
with tetrahedra:

1. weld the nodes, then check the surface is **watertight** — the free-edge
   count of the triangulation must be zero (an open surface can't bound a
   volume, and is rejected with a clear message);
2. `mesh.createTopology` builds the boundary curves/points of the discrete
   surface, then a surface loop + volume are created in the `geo` kernel;
3. `mesh.generate(3)` tetrahedralizes the **interior only** — the input
   triangulation stays as the boundary mesh (it is *not* re-triangulated, which
   is what makes this fast and robust; remeshing a reparametrized STL is slow
   and prone to hanging, so it is deliberately avoided).

`Max element size` then controls how coarse the interior tets may be; TET10 is
supported (`setOrder(2)` after meshing, straight-sided). Requirements: an
all-triangle (STL/scan) surface that is closed and non-self-intersecting. Repair
dirty meshes in MeshLab/netfabb first, or fall back to shell meshing.

Because there is still no CAD geometry behind a tessellation, k_mesher rejects
symmetry planes, defeaturing, refinement regions, per-face sizes and face-set
roles for these inputs with an explanatory message — convert to STEP/IGES/BREP
for those.

## Could be added, but need commercial OCCT components

OCCT can read the formats below only through its **Advanced Data Exchange
Components**, which are a *separately licensed, closed-source* add-on. The
open-source OCCT that gmsh links against does **not** include them, so they are
not available in k_mesher without building a custom gmsh against a licensed
OCCT. For all of these the practical answer is **export to STEP** from the
source CAD system.

| Format | Ext. | Origin | Why not (yet) |
|---|---|---|---|
| Parasolid | `.x_t` `.x_b` | Siemens kernel (NX, SolidWorks, Onshape…) | needs OCCT commercial reader or the Parasolid SDK |
| ACIS SAT/SAB | `.sat` `.sab` | Spatial/Dassault kernel (older CATIA, etc.) | needs OCCT commercial reader |
| JT | `.jt` | Siemens visualization (ISO 14306) | needs OCCT commercial reader; often tessellation-only anyway |
| DXF/DWG | `.dxf` `.dwg` | AutoCAD | 2-D/entity soup; needs commercial reader + non-trivial solid reconstruction |
| IFC | `.ifc` | Building/BIM (ISO 16739) | needs commercial reader; BIM semantics, rarely a clean solid |

## Impractical: native proprietary CAD

Native part files are undocumented, versioned, kernel-specific containers with
no open reader. There is **no realistic path** to importing them directly;
every one of these systems exports STEP (AP203/AP214/AP242).

| System | Ext. | Do this instead |
|---|---|---|
| CATIA V5/V6 | `.CATPart` `.CATProduct` | export STEP AP242 |
| Siemens NX | `.prt` | export STEP |
| PTC Creo/Pro-E | `.prt` `.asm` | export STEP |
| SolidWorks | `.sldprt` `.sldasm` | export STEP |
| Autodesk Inventor | `.ipt` `.iam` | export STEP |
| Fusion 360 | — | export STEP or BREP |

## Other mesh / graphics formats

gmsh's `merge` can also read some additional surface-mesh formats
(**OFF**, **VTK**, **VRML/WRL**, Nastran **BDF**). They work exactly like STL if
you add the extension to `MESH_FORMATS` in `mesher.py`, but they are FE/graphics
formats rather than CAD interchange, so they are left out of the default list.
**glTF/GLB**, **3MF** and **AMF** are *not* readable by the gmsh build here
(only OCCT's core writes glTF; gmsh does not expose it for import).

## Recommendations

1. **Prefer STEP** (AP242 if offered) for solids — most reliable, keeps solids
   and assembly structure.
2. **BREP** is a great lossless option when the source is already OpenCASCADE
   (FreeCAD, gmsh).
3. **IGES** works but is legacy; expect to lean on the auto-sew and to verify
   the result (watch for stray free edges on shells).
4. **STL/OBJ/PLY** — shell decks (thin-walled parts, 3-D-printed or scanned
   surfaces), or **tet solids** when the surface is a clean watertight manifold.
   Still no symmetry/defeature/refinement (those need a B-rep).
5. Anything else: **export STEP from the originating CAD system.**

## Implementation pointers

- Format tables and detection: `CAD_FORMATS`, `MESH_FORMATS`, `INPUT_FORMATS`,
  `input_kind()`, `format_name()` in [`mesher.py`](../mesher.py).
- B-rep import + IGES sewing: `_load_geometry`.
- Tessellation import + node welding + capability guards: `_load_tessellation`.
- Volume reconstruction from a watertight tessellation: `_tessellation_to_solid`.
- Adding another gmsh-mergeable surface format is a one-line change to
  `MESH_FORMATS` (and `FORMAT_NAMES`).

### Sources

- Gmsh reference manual — <https://gmsh.info/doc/texinfo/gmsh.html>
- OpenCASCADE Technology, Data Exchange — <https://dev.opencascade.org/about/data_exchange>
- OCCT Advanced Data Exchange Components — <https://old.opencascade.com/content/advanced-data-exchange-components>
