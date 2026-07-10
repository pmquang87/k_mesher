"""Automatic connection detection for meshed multi-body assemblies.

Given a :class:`mesher.MeshResult` (CAD meshed to LS-DYNA solids/shells) this
module finds the interfaces where two bodies *touch or nearly touch* and turns
them into connection data ready for :func:`dyna_writer.write_k`:

  * :func:`detect_interfaces` - the raw interface geometry (matched node pairs
    and the boundary facets on each side) for every unordered body pair.
  * :func:`spotweld_pairs`    - node-id pairs for ``write_k(spotwelds=...)``
    (-> ``*CONSTRAINED_SPOTWELD``), optionally thinned to a spot pattern.
  * :func:`tied_segment_sets` - ``face_sets``-style segment sets for the
    interface facets, for ``write_k(face_sets=...)`` (-> ``*SET_SEGMENT``).
  * :func:`tied_contact`      - convenience returning both the segment sets and
    a ``contacts`` tuple (-> ``*CONTACT_TIED_SURFACE_TO_SURFACE``).
  * :func:`interface_pairs`   - just the touching ``(body_i, body_j)`` index
    pairs (0-based), a lower-level helper over :func:`detect_interfaces`.
  * :func:`contact_pairs`     - one scoped, per-pair contact dict per touching
    body pair for ``write_k(contact_pairs=...)`` (part-to-part contact).

Indexing convention
-------------------
Every node id produced or consumed here is **1-based into ``result.coords``**,
exactly like ``result.elems`` (LS-DYNA node numbering with ``start_nid == 1``).
``write_k`` applies its own ``start_nid`` offset, so pass these ids straight
through - with the default ``start_nid == 1`` they match the ids in the .k file.

Tolerance
---------
Distances are compared against ``tol``. When ``tol`` is ``None`` it defaults to
:data:`DEFAULT_TOL_FRAC` (1e-3) times the model bounding-box diagonal - small
enough not to bridge genuinely separate bodies, large enough to catch the two
near-coincident node layers of an unglued shared interface (and small physical
gaps of "nearly touching" parts). Pass an explicit ``tol`` to widen/tighten.

This module is pure numpy: it never calls gmsh. A KD-tree (scipy) is used when
available and falls back to a spatial-hash grid otherwise, so scipy is optional.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

try:  # scipy is optional - a numpy grid fallback is used when it is absent
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only where scipy is missing
    cKDTree = None
    _HAVE_SCIPY = False

# default tolerance as a fraction of the model bounding-box diagonal
DEFAULT_TOL_FRAC = 1e-3

# the four triangular faces of a tet, as column indices into the corner nodes
# (n1..n4); orientation is irrelevant here since faces are matched by node set
_TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


@dataclass
class _Boundary:
    """The boundary of one body: its surface facets and the nodes on them."""
    facets: np.ndarray    # (F, 4) 1-based corner node ids; tris padded n4 = n3
    node_ids: np.ndarray  # (K,) 1-based node ids appearing on the facets


def _is_shell(result) -> bool:
    """True if the mesh is shells (area) rather than solids (volume)."""
    return result.stats.get("measure_label") == "area"


def _diag(result) -> float:
    """Model bounding-box diagonal length (from stats if present, else coords)."""
    bbox = result.stats.get("bbox")
    if bbox is not None:
        lo, hi = np.asarray(bbox[:3], float), np.asarray(bbox[3:], float)
        return float(np.linalg.norm(hi - lo))
    c = np.asarray(result.coords, float)
    if len(c) == 0:
        return 0.0
    return float(np.linalg.norm(c.max(0) - c.min(0)))


def default_tol(result) -> float:
    """The default matching tolerance for ``result`` (see module docstring):
    :data:`DEFAULT_TOL_FRAC` of the bounding-box diagonal, floored at 1e-9."""
    return max(DEFAULT_TOL_FRAC * _diag(result), 1e-9)


def _body_boundary(result, body: int) -> _Boundary:
    """Boundary facets and node ids of a single body (0-based ``elem_parts``).

    Solids (tet4/tet10): a facet lies on the body surface iff it is used by
    exactly one of the body's tets; tet10 uses its corner nodes only. Shells:
    every element is itself a boundary facet. Returns 1-based ids throughout.
    """
    elems = np.asarray(result.elems, dtype=np.int64)
    parts = np.asarray(result.elem_parts, dtype=np.int64)
    mine = elems[parts == body]

    if _is_shell(result):
        facets = mine[:, :4].copy()                    # already padded tris
        node_ids = np.unique(facets)
        return _Boundary(facets, node_ids)

    corners = mine[:, :4]                               # tet corner nodes
    faces = corners[:, _TET_FACES].reshape(-1, 3)       # (4*M, 3) all tet faces
    keys = np.sort(faces, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                               return_counts=True)
    on_surface = counts[inv.ravel()] == 1               # face used by one tet
    tris = faces[on_surface]
    facets = np.column_stack((tris, tris[:, 2])) if len(tris) \
        else np.zeros((0, 4), dtype=np.int64)
    node_ids = np.unique(tris) if len(tris) else np.zeros(0, dtype=np.int64)
    return _Boundary(facets, node_ids)


def _match_nodes(src: np.ndarray, dst: np.ndarray, tol: float,
                 use_kdtree: bool = True):
    """For every point in ``src`` return (index of the nearest ``dst`` point
    within ``tol``, distance); index is -1 (distance inf) where none is within
    ``tol``. Uses a scipy KD-tree when available, else a numpy grid bucket."""
    n = len(src)
    idx = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf)
    if n == 0 or len(dst) == 0:
        return idx, dist
    if use_kdtree and _HAVE_SCIPY:
        d, j = cKDTree(dst).query(src, k=1)
        d = np.atleast_1d(d).astype(float)
        j = np.atleast_1d(j).astype(np.int64)
        ok = d <= tol
        idx[ok] = j[ok]
        dist[ok] = d[ok]
        return idx, dist
    return _match_nodes_grid(src, dst, tol)


def _match_nodes_grid(src: np.ndarray, dst: np.ndarray, tol: float):
    """Numpy fallback for :func:`_match_nodes`: bucket ``dst`` into tol-sized
    cells and scan the 3x3x3 neighbourhood of each ``src`` point. Any point
    within ``tol`` necessarily lies in that neighbourhood."""
    n = len(src)
    idx = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf)
    origin = dst.min(0)
    inv = 1.0 / tol
    dkeys = np.floor((dst - origin) * inv).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for j, key in enumerate(map(tuple, dkeys)):
        buckets.setdefault(key, []).append(j)
    skeys = np.floor((src - origin) * inv).astype(np.int64)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
               for c in (-1, 0, 1)]
    for i in range(n):
        bx, by, bz = int(skeys[i, 0]), int(skeys[i, 1]), int(skeys[i, 2])
        best_d, best_j = tol, -1
        for ox, oy, oz in offsets:
            for j in buckets.get((bx + ox, by + oy, bz + oz), ()):
                d = float(np.linalg.norm(src[i] - dst[j]))
                if d <= best_d:
                    best_d, best_j = d, j
        if best_j >= 0:
            idx[i], dist[i] = best_j, best_d
    return idx, dist


def _facets_on_interface(facets: np.ndarray, near_ids: np.ndarray) -> np.ndarray:
    """Rows of ``facets`` whose every corner node is in ``near_ids`` (1-based)."""
    if len(facets) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    n = int(facets.max())
    flag = np.zeros(n + 1, dtype=bool)
    flag[np.asarray(near_ids, dtype=np.int64)] = True
    return facets[flag[facets].all(axis=1)]


def detect_interfaces(result, tol: float | None = None,
                      use_kdtree: bool = True) -> list[dict]:
    """Find the interfaces between bodies that touch or nearly touch.

    For every unordered body pair (i < j) the boundary nodes of body i that lie
    within ``tol`` of body j's boundary are matched to their nearest node in j.
    Single-body meshes (and pairs with no near nodes) yield nothing.

    ``tol`` defaults to :func:`default_tol`. ``use_kdtree`` forces the numpy
    grid fallback when ``False`` (mainly for testing).

    Returns one dict per interface with:
      ``bodies``      (i, j) 0-based body indices;
      ``node_pairs``  (P, 2) 1-based ids [node in i, nearest node in j];
      ``segments_i``  (Si, 4) 1-based interface facets of body i (tris padded);
      ``segments_j``  (Sj, 4) the same for body j;
      ``tol``         the tolerance used.
    All node ids are 1-based into ``result.coords`` (see module docstring).
    """
    coords = np.asarray(result.coords, dtype=float)
    parts = np.asarray(result.elem_parts, dtype=np.int64)
    bodies = sorted(int(b) for b in np.unique(parts))
    if len(bodies) < 2:
        return []
    tol = default_tol(result) if tol is None else float(tol)

    bnd = {b: _body_boundary(result, b) for b in bodies}
    interfaces: list[dict] = []
    for i, j in combinations(bodies, 2):
        bi, bj = bnd[i], bnd[j]
        if len(bi.node_ids) == 0 or len(bj.node_ids) == 0:
            continue
        ci, cj = coords[bi.node_ids - 1], coords[bj.node_ids - 1]

        idx_ij, _ = _match_nodes(ci, cj, tol, use_kdtree)   # i -> nearest in j
        near_i = idx_ij >= 0
        if not near_i.any():
            continue
        node_pairs = np.column_stack(
            (bi.node_ids[near_i], bj.node_ids[idx_ij[near_i]]))

        idx_ji, _ = _match_nodes(cj, ci, tol, use_kdtree)   # j nodes near i
        near_j = idx_ji >= 0

        interfaces.append({
            "bodies": (i, j),
            "node_pairs": node_pairs,
            "segments_i": _facets_on_interface(bi.facets, bi.node_ids[near_i]),
            "segments_j": _facets_on_interface(bj.facets, bj.node_ids[near_j]),
            "tol": tol,
        })
    return interfaces


def interface_pairs(result, tol: float | None = None) -> list[tuple[int, int]]:
    """The touching body pairs of ``result`` as raw 0-based index tuples.

    A lower-level helper over :func:`detect_interfaces`: returns just the
    ``(i, j)`` body indices (0-based into ``result.elem_parts``, ``i < j``) of
    every pair whose boundaries touch or nearly touch within ``tol`` - dropping
    the matched-node/segment geometry. ``tol`` defaults to :func:`default_tol`.

    Returns ``[]`` for single-body meshes or when no bodies touch.
    """
    return [itf["bodies"] for itf in detect_interfaces(result, tol=tol)]


def contact_pairs(result, base_pid: int = 1, tol: float | None = None,
                  ctype: str = "automatic_surface_to_surface",
                  fs: float = 0.0) -> list[dict]:
    """Per-touching-pair scoped contact suggestions for ``write_k``.

    One dict per touching body pair - ready for the ``contact_pairs`` argument
    of :func:`dyna_writer.write_k` - so an assembly gets a single contact scoped
    to each pair of parts that actually touch, rather than one contact spanning
    everything.

    ``base_pid`` maps a 0-based body index to its LS-DYNA part id exactly as the
    CLI numbers parts: **body k -> PID ``base_pid + k``** (so body 0 -> base_pid,
    body 1 -> base_pid + 1, ...). For each touching pair ``(i, j)`` from
    :func:`interface_pairs` the dict is::

        {"slave_parts":  [base_pid + i],
         "master_parts": [base_pid + j],
         "type":         ctype,
         "fs":           fs,
         "title":        f"CONTACT_p{base_pid + i}_p{base_pid + j}"}

    ``ctype`` is the contact type suffix (default
    ``"automatic_surface_to_surface"``) and ``fs`` the static friction
    coefficient. Returns ``[]`` for single-body meshes or when no bodies touch.
    """
    return [
        {"slave_parts": [base_pid + i],
         "master_parts": [base_pid + j],
         "type": ctype,
         "fs": fs,
         "title": f"CONTACT_p{base_pid + i}_p{base_pid + j}"}
        for i, j in interface_pairs(result, tol=tol)
    ]


def _thin_by_spacing(points: np.ndarray, spacing: float) -> np.ndarray:
    """Greedy Poisson-style thinning: keep points at least ``spacing`` apart.
    Returns the kept row indices (into ``points``), in a stable spatial order."""
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    kept: list[int] = []
    kept_pts = np.empty((0, 3), dtype=float)
    for i in order:
        p = points[i]
        if len(kept_pts) and np.linalg.norm(kept_pts - p, axis=1).min() < spacing:
            continue
        kept.append(int(i))
        kept_pts = np.vstack((kept_pts, p))
    return np.asarray(kept, dtype=np.int64)


def spotweld_pairs(result, tol: float | None = None,
                   max_welds: int | None = None,
                   spacing: float | None = None,
                   use_kdtree: bool = True) -> list[tuple[int, int]]:
    """Node-id pairs for ``write_k(spotwelds=...)`` (-> ``*CONSTRAINED_SPOTWELD``).

    Collects the matched interface node pairs from :func:`detect_interfaces`.
    Every pair joins one node on body i to its near-coincident node on body j.

    ``spacing`` thins the raw (typically every-coincident-node) pattern to a
    minimum spot spacing so the result is a sensible weld pattern rather than a
    weld at every node. ``max_welds`` caps the count (evenly subsampled after
    thinning). Returns ``[(n1, n2), ...]`` with 1-based ids as in ``result``.
    """
    interfaces = detect_interfaces(result, tol=tol, use_kdtree=use_kdtree)
    if not interfaces:
        return []
    pairs = np.vstack([itf["node_pairs"] for itf in interfaces])
    coords = np.asarray(result.coords, dtype=float)
    # representative location of each weld: midpoint of its two (near) nodes
    pts = 0.5 * (coords[pairs[:, 0] - 1] + coords[pairs[:, 1] - 1])

    if spacing is not None and spacing > 0 and len(pairs):
        keep = _thin_by_spacing(pts, float(spacing))
        pairs = pairs[keep]
    else:
        # deterministic spatial order even without thinning
        pairs = pairs[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]

    if max_welds is not None and len(pairs) > max_welds:
        sel = np.unique(np.linspace(0, len(pairs) - 1, max_welds).round()
                        .astype(np.int64))
        pairs = pairs[sel]
    return [(int(a), int(b)) for a, b in pairs]


def tied_segment_sets(result, tol: float | None = None,
                      use_kdtree: bool = True) -> list[dict]:
    """``face_sets``-style segment sets for the interface facets.

    Returns a list of ``{"kind": "segment", "title": str, "segments": (S, 4)}``
    dicts (1-based ids, tris padded to a degenerate quad) - two per interface,
    one for each body's side of the contact - ready for ``write_k(face_sets=...)``
    (each becomes a ``*SET_SEGMENT``). Pair with a tied ``contacts`` entry (see
    :func:`tied_contact`) to weld the interfaces with a tied contact.
    """
    sets: list[dict] = []
    for itf in detect_interfaces(result, tol=tol, use_kdtree=use_kdtree):
        i, j = itf["bodies"]
        for body, segs in ((i, itf["segments_i"]), (j, itf["segments_j"])):
            if len(segs):
                sets.append({"kind": "segment",
                             "title": f"TIE interface {i}-{j} side {body}",
                             "segments": segs})
    return sets


def tied_contact(result, tol: float | None = None,
                 fs: float = 0.0, use_kdtree: bool = True):
    """Convenience: ``(face_sets_list, contacts_tuple)`` for a tied assembly.

    ``face_sets_list`` is :func:`tied_segment_sets`; ``contacts_tuple`` holds a
    single ``{"type": "tied_surface_to_surface", "fs": fs}`` entry (empty when
    there is no interface). ``write_k`` scopes that contact over all parts, so
    one entry ties every detected interface. Feed both straight into
    ``write_k(face_sets=face_sets_list, contacts=contacts_tuple)``.
    """
    face_sets = tied_segment_sets(result, tol=tol, use_kdtree=use_kdtree)
    contacts = (({"type": "tied_surface_to_surface", "fs": float(fs)},)
                if face_sets else ())
    return face_sets, contacts
