"""meshio bridge: convert between k_mesher arrays and other FE mesh formats.

k_mesher stores a mesh as two arrays::

    coords  (N, 3) float    node coordinates
    elems   (M, K) int      1-based connectivity in LS-DYNA node ordering

with these element flavours (matching ``dyna_writer.write_k``):

    solids  tet4  -> (M, 4)
            hex8  -> (M, 8)
            tet10 -> (M, 10)
    shells  tri3  -> (M, 4) stored as a degenerate quad (n4 == n3)
            quad4 -> (M, 4)

meshio uses **0-based** connectivity and groups elements into typed cell
blocks ("tetra", "tetra10", "hexahedron", "triangle", "quad").  meshio
deliberately does
*not* support LS-DYNA, so this module lets users import an existing mesh
(Nastran .bdf, Abaqus .inp, VTK/VTU, gmsh .msh, ...) into k_mesher's arrays,
or export k_mesher's arrays to any format meshio can write.

Public API
----------
    to_meshio(coords, elems, element_kind) -> meshio.Mesh
    from_meshio(mesh) -> (coords, elems, element_kind)
    export_mesh(path, coords, elems, element_kind, file_format=None)
    import_mesh(path, file_format=None) -> (coords, elems, element_kind)

TET10 node ordering
-------------------
LS-DYNA (and Nastran CTETRA) order the six mid-edge nodes n5..n10 by the
corner-index pairs::

    n5:(1,2) n6:(2,3) n7:(3,1) n8:(1,4) n9:(2,4) n10:(3,4)   [1-based corners]

which in 0-based corner indices is the edge sequence

    (0,1) (1,2) (2,0) (0,3) (1,3) (2,3).

VTK's ``VTK_QUADRATIC_TETRA`` -- the ordering meshio uses for "tetra10" --
numbers its mid-edge nodes over exactly the same edge sequence, so for
meshio 5.x the LS-DYNA -> meshio tet10 permutation is the identity.  Rather
than hard-code that, the permutation is *computed* from the two edge tables
below (``_DYNA_TET10_EDGES`` / ``_MESHIO_TET10_EDGES``); if a future meshio
ever reorders its edges, only the ``_MESHIO_TET10_EDGES`` table needs
updating and everything else follows.

HEX8 node ordering
------------------
LS-DYNA ``*ELEMENT_SOLID`` hex8 orders nodes n1..n4 around the bottom quad
and n5..n8 around the top quad directly above -- exactly the
``VTK_HEXAHEDRON`` ordering that meshio's "hexahedron" cell type uses, so
the LS-DYNA <-> meshio permutation is the identity.  This was verified
empirically (unit-cube hex round-tripped through VTU and Abaqus keeps every
node in place and its volume positive) and is encoded explicitly in
``_DYNA_TO_MESHIO_HEX8`` / ``_MESHIO_TO_DYNA_HEX8`` below so any future
reordering only needs those tables changed.
"""
from __future__ import annotations

import numpy as np

# Mid-edge node ordering, expressed as (corner_a, corner_b) 0-based index
# pairs, for the six second-order nodes n5..n10 of a 10-node tetrahedron.
_DYNA_TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
_MESHIO_TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def _tet10_permutation(src_edges, dst_edges):
    """Return an index list ``perm`` of length 10 such that
    ``dst_nodes = src_nodes[perm]`` reorders the mid-edge nodes from the
    ``src_edges`` convention to the ``dst_edges`` convention.  Corner nodes
    (0..3) are always mapped identically."""
    perm = [0, 1, 2, 3]
    src_lookup = {frozenset(e): i for i, e in enumerate(src_edges)}
    for edge in dst_edges:
        perm.append(4 + src_lookup[frozenset(edge)])
    return perm


# LS-DYNA <-> meshio are mutual inverses; compute both from the edge tables.
_DYNA_TO_MESHIO_TET10 = _tet10_permutation(_DYNA_TET10_EDGES, _MESHIO_TET10_EDGES)
_MESHIO_TO_DYNA_TET10 = _tet10_permutation(_MESHIO_TET10_EDGES, _DYNA_TET10_EDGES)

# LS-DYNA hex8 and meshio's "hexahedron" both use VTK_HEXAHEDRON ordering
# (n1-n4 bottom quad, n5-n8 top quad), so the permutation is the identity.
# Kept as explicit tables (like the tet10 mapping) so a convention change
# only ever touches these two lines.
_DYNA_TO_MESHIO_HEX8 = [0, 1, 2, 3, 4, 5, 6, 7]
_MESHIO_TO_DYNA_HEX8 = [0, 1, 2, 3, 4, 5, 6, 7]


def _require_meshio():
    """Import meshio lazily with a friendly error if it is not installed."""
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "meshio is required for FE-format conversion; install it with "
            "'pip install meshio'"
        ) from exc
    return meshio


def to_meshio(coords, elems, element_kind="solid"):
    """Convert k_mesher arrays to a :class:`meshio.Mesh`.

    Parameters
    ----------
    coords : (N, 3) array_like
        Node coordinates.
    elems : (M, K) array_like
        1-based connectivity in LS-DYNA ordering.  ``K`` selects the element
        type: solids use 4 (tet4), 8 (hex8) or 10 (tet10); shells use 4
        (quad4, or a degenerate quad with ``n4 == n3`` for a triangle).
    element_kind : {"solid", "shell"}
        Which family ``elems`` describes.

    Returns
    -------
    meshio.Mesh
        0-based connectivity with the correct meshio cell type(s).  A shell
        block that mixes triangles and quads is split into a "triangle" and a
        "quad" cell block.
    """
    meshio = _require_meshio()

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3); got shape {coords.shape}")

    elems = np.asarray(elems)
    if elems.size == 0:
        raise ValueError("elems is empty; nothing to convert")
    if elems.ndim != 2:
        raise ValueError(f"elems must be 2-D (M, K); got shape {elems.shape}")

    if element_kind not in ("solid", "shell"):
        raise ValueError(
            f"element_kind must be 'solid' or 'shell'; got {element_kind!r}"
        )

    conn = elems.astype(np.int64) - 1  # LS-DYNA is 1-based; meshio is 0-based
    cells = []

    if element_kind == "solid":
        k = conn.shape[1]
        if k == 4:
            cells.append(("tetra", conn))
        elif k == 8:
            cells.append(("hexahedron", conn[:, _DYNA_TO_MESHIO_HEX8]))
        elif k == 10:
            cells.append(("tetra10", conn[:, _DYNA_TO_MESHIO_TET10]))
        else:
            raise ValueError(
                f"unsupported solid connectivity width {k}; expected 4 (tet4), "
                "8 (hex8) or 10 (tet10)"
            )
    else:  # shell
        k = conn.shape[1]
        if k != 4:
            raise ValueError(
                f"unsupported shell connectivity width {k}; expected 4 "
                "(quad4, or degenerate quad tri3 with n4 == n3)"
            )
        is_tri = conn[:, 3] == conn[:, 2]  # degenerate quad -> triangle
        tris = conn[is_tri]
        quads = conn[~is_tri]
        if tris.shape[0]:
            cells.append(("triangle", tris[:, :3]))
        if quads.shape[0]:
            cells.append(("quad", quads))

    return meshio.Mesh(coords, cells)


# meshio cell type -> (k_mesher family, connectivity builder)
_SOLID_TYPES = ("tetra", "tetra10", "hexahedron")
_SHELL_TYPES = ("triangle", "quad")
# Quadratic hexes exist in meshio but have no k_mesher representation yet.
_UNSUPPORTED_HEX_TYPES = ("hexahedron20", "hexahedron27")


def from_meshio(mesh):
    """Convert a :class:`meshio.Mesh` to k_mesher arrays.

    Combines "tetra"/"tetra10"/"hexahedron" blocks into a solid mesh and
    "triangle"/"quad" blocks into a shell mesh.  Triangles are padded to a
    degenerate quad (``n4 = n3``) so all shells share one (M, 4) array.
    Solid meshes must be uniform (all-tet or all-hex); quadratic hexes
    ("hexahedron20"/"hexahedron27") raise :class:`NotImplementedError`.

    If the mesh contains *both* solids and shells (e.g. a solid volume with
    its bounding surface), the **solids win**: the shells are dropped and
    only the solid family is returned.  This mirrors k_mesher's model where a
    single mesh is one family, and a volume mesh's surface facets are
    redundant.  Callers that want the shells should filter the meshio mesh
    first.

    Returns
    -------
    (coords, elems, element_kind)
        ``coords`` is (N, 3) float; ``elems`` is (M, K) int, 1-based, in
        LS-DYNA node ordering; ``element_kind`` is "solid" or "shell".
    """
    _require_meshio()

    coords = np.asarray(mesh.points, dtype=float)
    if coords.shape[1] == 2:  # 2-D mesh -> pad z with zeros
        coords = np.column_stack((coords, np.zeros(len(coords))))

    blocks = {ct: [] for ct in (*_SOLID_TYPES, *_SHELL_TYPES)}
    seen = set()
    for cell_block in mesh.cells:
        ct = cell_block.type
        if ct in _UNSUPPORTED_HEX_TYPES:
            raise NotImplementedError(
                f"meshio cell type {ct!r} (quadratic hexahedron) is not "
                "supported; only linear 'hexahedron' (hex8) solids are"
            )
        if ct in blocks:
            blocks[ct].append(np.asarray(cell_block.data, dtype=np.int64))
        else:
            seen.add(ct)

    have_solid = any(blocks[ct] for ct in _SOLID_TYPES)
    have_shell = any(blocks[ct] for ct in _SHELL_TYPES)

    if not have_solid and not have_shell:
        supported = ", ".join((*_SOLID_TYPES, *_SHELL_TYPES))
        found = ", ".join(sorted(seen)) or "none"
        raise ValueError(
            f"meshio mesh has no supported cell types (found: {found}); "
            f"expected one of: {supported}"
        )

    if have_solid:
        # k_mesher meshes are uniform: one solid type per (M, K) array.
        have_tet = bool(blocks["tetra"] or blocks["tetra10"])
        if have_tet and blocks["hexahedron"]:
            raise ValueError(
                "mesh mixes tetrahedra and hexahedra solids; k_mesher meshes "
                "are uniform -- convert them separately"
            )
        if blocks["tetra"] and blocks["tetra10"]:
            raise ValueError(
                "mesh mixes tet4 and tet10 solids; convert them separately"
            )
        if blocks["hexahedron"]:
            data = np.vstack(blocks["hexahedron"])
            elems = data[:, _MESHIO_TO_DYNA_HEX8] + 1
        elif blocks["tetra10"]:
            data = np.vstack(blocks["tetra10"])
            elems = data[:, _MESHIO_TO_DYNA_TET10] + 1
        else:
            elems = np.vstack(blocks["tetra"]) + 1
        return coords, elems.astype(np.int64), "solid"

    # shells only
    parts = []
    for tri in blocks["triangle"]:
        # pad triangle to degenerate quad: n4 = n3
        parts.append(np.column_stack((tri, tri[:, 2])))
    for quad in blocks["quad"]:
        parts.append(quad)
    elems = np.vstack(parts) + 1
    return coords, elems.astype(np.int64), "shell"


def export_mesh(path, coords, elems, element_kind="solid", file_format=None):
    """Write k_mesher arrays to *path* in any format meshio can write.

    The format is inferred from the file extension unless *file_format* is
    given explicitly (e.g. "nastran", "abaqus", "vtu", "gmsh").
    """
    meshio = _require_meshio()
    mesh = to_meshio(coords, elems, element_kind)
    meshio.write(path, mesh, file_format=file_format)
    return mesh


def import_mesh(path, file_format=None):
    """Read a mesh file with meshio and return k_mesher arrays.

    The format is inferred from the file extension unless *file_format* is
    given explicitly.  Returns ``(coords, elems, element_kind)``.
    """
    meshio = _require_meshio()
    mesh = meshio.read(path, file_format=file_format)
    return from_meshio(mesh)


def main(argv=None) -> int:
    """Console entry (``k-mesher-convert``): convert a mesh between formats.

    Reads any meshio-supported mesh and writes it out again. If the output
    ends in ``.k`` the LS-DYNA writer is used (meshio cannot write LS-DYNA).
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="k-mesher-convert",
        description="Convert a mesh file between FE formats (via meshio); "
                    "an .k output is written with k_mesher's LS-DYNA writer.")
    p.add_argument("input", help="input mesh (meshio-readable: .inp/.bdf/.vtu/...)")
    p.add_argument("output", help="output mesh (.k for LS-DYNA, else any meshio format)")
    p.add_argument("--thickness", type=float, default=1.0,
                   help="shell thickness when writing .k shells")
    args = p.parse_args(argv)

    coords, elems, element_kind = import_mesh(args.input)
    if args.output.lower().endswith(".k"):
        from k_mesher import dyna_writer
        dyna_writer.write_k(args.output, coords, elems,
                            element_kind=element_kind, thickness=args.thickness)
    else:
        export_mesh(args.output, coords, elems, element_kind)
    print(f"Wrote {args.output} ({len(coords)} nodes, {len(elems)} {element_kind} elements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
