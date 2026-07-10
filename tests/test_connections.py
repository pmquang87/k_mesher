"""Tests for connections.py: automatic interface / spotweld / tied-contact
detection on a meshed multi-body assembly.

The assembly is the two-box test part (``make_two_bodies``) meshed with
``glue=False`` so the shared face at x = 20 keeps TWO distinct, near-coincident
node layers (one per body) - exactly the situation connection detection is for.

Run: python -m pytest tests/test_connections.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from k_mesher import connections
from k_mesher import dyna_writer
from k_mesher import mesher
from make_test_step import make, make_two_bodies

EX = os.path.join(ROOT, "examples")
OUT_DIR = os.path.join(ROOT, "tests", "out")
STEP = os.path.join(EX, "test_part.step")
STEP2 = os.path.join(EX, "test_two_bodies.step")

IFACE_X = 20.0          # the two boxes share the plane x = 20
_cache: dict = {}


def QUIET(_msg):
    pass


def _ensure():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(STEP):
        make(STEP)
    if not os.path.isfile(STEP2):
        make_two_bodies(STEP2)


def res_two_unglued():
    """Two touching boxes, NOT glued -> separate node layers at x = 20."""
    if "two" not in _cache:
        _ensure()
        _cache["two"] = mesher.mesh_step(
            mesher.MeshSettings(step_file=STEP2, size_max=6.0, glue=False),
            log=QUIET)
    return _cache["two"]


def res_single():
    """A single-body block (for the guard test)."""
    if "single" not in _cache:
        _ensure()
        _cache["single"] = mesher.mesh_step(
            mesher.MeshSettings(step_file=STEP, size_max=10.0), log=QUIET)
    return _cache["single"]


def _body_node_ids(res, body):
    """Set of 1-based node ids used by a given 0-based body."""
    mask = np.asarray(res.elem_parts) == body
    return set(int(n) for n in np.unique(res.elems[mask][:, :4]))


# --------------------------------------------------------------------------

def test_two_bodies_are_separate_layers():
    """Sanity: glue=False really leaves two coincident node layers."""
    res = res_two_unglued()
    assert res.stats["measure_label"] == "volume"
    assert set(np.unique(res.elem_parts)) == {0, 1}
    at_iface = np.abs(res.coords[:, 0] - IFACE_X) < 1e-6
    n_iface = int(at_iface.sum())
    n_unique = len(np.unique(np.round(res.coords[at_iface], 6), axis=0))
    # every interface location carries two nodes (one per body)
    assert n_iface > 20 and n_unique * 2 == n_iface


def test_detect_interfaces():
    res = res_two_unglued()
    ifaces = connections.detect_interfaces(res)
    assert len(ifaces) == 1, "exactly one interface between the two bodies"
    itf = ifaces[0]
    assert itf["bodies"] == (0, 1)
    tol = itf["tol"]

    pairs = itf["node_pairs"]
    # the shared 20x20 face has a plausible number of coincident nodes
    assert 20 <= len(pairs) <= 200
    ids0, ids1 = _body_node_ids(res, 0), _body_node_ids(res, 1)
    for a, b in pairs:
        assert a in ids0 and b in ids1, "pairs must join opposite bodies"
        d = np.linalg.norm(res.coords[a - 1] - res.coords[b - 1])
        assert d <= tol, "paired nodes must be within tol"
        # both endpoints lie on the shared plane
        assert abs(res.coords[a - 1, 0] - IFACE_X) < 1e-6
        assert abs(res.coords[b - 1, 0] - IFACE_X) < 1e-6

    # interface facets on both sides, all corners on the plane, padded to quads
    for key in ("segments_i", "segments_j"):
        segs = itf[key]
        assert len(segs) > 3 and segs.shape[1] == 4
        assert np.allclose(res.coords[segs[:, :3].ravel() - 1][:, 0], IFACE_X,
                           atol=1e-6)
        assert (segs[:, 2] == segs[:, 3]).all(), "tet faces -> padded tris"


def test_spotweld_pairs():
    res = res_two_unglued()
    welds = connections.spotweld_pairs(res)
    assert len(welds) > 10
    tol = connections.default_tol(res)
    ids0, ids1 = _body_node_ids(res, 0), _body_node_ids(res, 1)
    for a, b in welds:
        assert a in ids0 and b in ids1
        assert np.linalg.norm(res.coords[a - 1] - res.coords[b - 1]) <= tol

    # spacing thinning yields a sparser, valid subset
    thinned = connections.spotweld_pairs(res, spacing=8.0)
    assert 0 < len(thinned) < len(welds)
    tp = np.array([0.5 * (res.coords[a - 1] + res.coords[b - 1])
                   for a, b in thinned])
    # kept welds honour the requested minimum spacing
    for i in range(len(tp)):
        for j in range(i + 1, len(tp)):
            assert np.linalg.norm(tp[i] - tp[j]) >= 8.0 - 1e-9

    # max_welds caps the count
    capped = connections.spotweld_pairs(res, max_welds=5)
    assert len(capped) <= 5


def test_tied_segment_sets_and_writer_roundtrip():
    res = res_two_unglued()
    seg_sets = connections.tied_segment_sets(res)
    assert len(seg_sets) == 2, "one segment set per side of the one interface"
    for s in seg_sets:
        assert s["kind"] == "segment" and len(s["segments"]) > 3
        nodes = s["segments"][:, :3].ravel()
        assert np.allclose(res.coords[nodes - 1][:, 0], IFACE_X, atol=1e-6)

    face_sets, contacts = connections.tied_contact(res)
    assert face_sets and contacts == (
        {"type": "tied_surface_to_surface", "fs": 0.0},)

    welds = connections.spotweld_pairs(res, spacing=6.0)
    k_path = os.path.join(OUT_DIR, "connections.k")
    dyna_writer.write_k(
        k_path, res.coords, res.elems, pid=10,
        part_ids=10 + np.asarray(res.elem_parts),
        part_titles={10: "left box", 11: "right box"},
        spotwelds=tuple(welds), face_sets=tuple(face_sets), contacts=contacts)

    with open(k_path) as f:
        text = f.read()
    assert "*CONSTRAINED_SPOTWELD" in text
    assert "*CONTACT_TIED_SURFACE_TO_SURFACE" in text
    assert "*SET_SEGMENT_TITLE" in text
    # each detected weld must appear as a spotweld card node pair
    assert text.count("*CONSTRAINED_SPOTWELD") == len(welds)


def test_numpy_fallback_matches_kdtree():
    """The scipy KD-tree and the numpy grid fallback must agree."""
    res = res_two_unglued()
    kd = connections.detect_interfaces(res, use_kdtree=True)

    # force the numpy grid path even though scipy is installed
    saved = connections._HAVE_SCIPY
    connections._HAVE_SCIPY = False
    try:
        npf = connections.detect_interfaces(res, use_kdtree=True)
        # also exercise the explicit use_kdtree=False switch
        npf2 = connections.detect_interfaces(res, use_kdtree=False)
    finally:
        connections._HAVE_SCIPY = saved

    assert len(kd) == len(npf) == len(npf2) == 1
    a = {tuple(p) for p in kd[0]["node_pairs"]}
    b = {tuple(p) for p in npf[0]["node_pairs"]}
    c = {tuple(p) for p in npf2[0]["node_pairs"]}
    assert a == b == c, "fallback must find the same coincident node pairs"


def test_single_body_is_empty():
    res = res_single()
    assert set(np.unique(res.elem_parts)) == {0}
    assert connections.detect_interfaces(res) == []
    assert connections.spotweld_pairs(res) == []
    assert connections.tied_segment_sets(res) == []
    assert connections.tied_contact(res) == ([], ())


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            print(f"--- {name} ---")
            globals()[name]()
    print("OK")
