"""LS-DYNA keyword (.k) file reader.

A pragmatic parser for the subset of LS-DYNA keyword decks that ``k_mesher``
itself writes (see :mod:`dyna_writer`) plus common hand-edited variants, so
decks can be round-tripped, renumbered, merged, or inspected in Python.

Supported keywords
------------------
``*KEYWORD`` / ``*KEYWORD LONG=Y`` (field width is auto-detected but the
parser splits on whitespace, so both the standard 8/16/10 format and the
20-character LONG format read identically), ``*TITLE``, ``*NODE``,
``*ELEMENT_SOLID`` (both the one-line 10-field TET4/degenerate-hex form and
the two-line TET10 form), ``*ELEMENT_SHELL`` (triangles written with
``n4 == n3``), ``*PART``, ``*SECTION_SOLID`` / ``*SECTION_SHELL``,
``*MAT_ELASTIC`` / ``*MAT_RIGID`` (and a generic ``*MAT_*`` fallback),
``*SET_NODE_LIST[_TITLE]``, ``*SET_SEGMENT[_TITLE]``, ``*INCLUDE`` and
``*END``. Unknown keywords are skipped gracefully and recorded.

Design notes
------------
Every data line ``k_mesher`` emits is space-separated fixed-width, so the
reader simply splits on whitespace. This is robust to both the standard and
the LONG=Y widths and tolerates trailing spaces, blank lines and mixed-case
keywords. ``$`` lines are comments (``$#`` are column headers); both are
skipped. Degenerate-hex TET4 (``n5..n8`` repeating ``n4``) and triangle
shells (``n4 == n3``) collapse to their unique nodes (4 and 3 respectively)
via consecutive-duplicate removal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class KModel:
    """Parsed contents of an LS-DYNA keyword deck.

    Attributes
    ----------
    title:
        The ``*TITLE`` text (empty string if none).
    nodes:
        ``{nid: (x, y, z)}`` mapping (insertion-ordered by appearance).
    solids:
        list of ``(eid, pid, node_ids)`` where ``node_ids`` is a tuple of the
        element's unique nodes (4 for a TET4/degenerate hex, 10 for a TET10).
    shells:
        list of ``(eid, pid, node_ids)`` (3 for a triangle, 4 for a quad).
    parts:
        ``{pid: {"secid", "mid", "title"}}``.
    sections:
        ``{secid: {"type", "elform", ...}}``.
    materials:
        ``{mid: {"type", "params", "raw"}}``.
    node_sets:
        ``{sid: [nid, ...]}``.
    segment_sets:
        ``{sid: [(n1, n2, n3, n4), ...]}``.
    long_format:
        ``True`` when the deck declared ``*KEYWORD LONG=Y``.
    includes:
        list of filenames referenced by ``*INCLUDE`` cards.
    unknown_keywords:
        sorted-on-request list of keyword names that were skipped.
    """

    title: str = ""
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    solids: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    shells: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    parts: dict[int, dict] = field(default_factory=dict)
    sections: dict[int, dict] = field(default_factory=dict)
    materials: dict[int, dict] = field(default_factory=dict)
    node_sets: dict[int, list[int]] = field(default_factory=dict)
    segment_sets: dict[int, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    long_format: bool = False
    includes: list[str] = field(default_factory=list)
    unknown_keywords: list[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def element_count(self) -> int:
        return len(self.solids) + len(self.shells)

    def to_arrays(self, kind: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coords, elems)`` suitable for feeding back to ``write_k``.

        ``coords`` is an ``(N, 3)`` float array ordered by ascending node id;
        ``elems`` is an ``(M, K)`` **1-based** int array whose values index
        rows of ``coords`` (i.e. node ids are compacted to ``1..N`` in the
        same order). Elements with fewer nodes than the widest one in the set
        are padded by repeating their last node (matching the writer's
        convention of storing triangles as ``n4 == n3``). ``kind`` selects
        ``"solid"`` or ``"shell"``; it defaults to solids when present.
        """
        if kind is None:
            kind = "solid" if self.solids else "shell"
        elements = self.solids if kind == "solid" else self.shells

        order = sorted(self.nodes)
        remap = {nid: i + 1 for i, nid in enumerate(order)}
        coords = np.array([self.nodes[nid] for nid in order], dtype=float)
        if coords.size == 0:
            coords = coords.reshape(0, 3)

        if not elements:
            return coords, np.zeros((0, 0), dtype=np.int64)
        width = max(len(nodes) for _, _, nodes in elements)
        rows = []
        for _eid, _pid, nodes in elements:
            padded = list(nodes) + [nodes[-1]] * (width - len(nodes))
            rows.append([remap[n] for n in padded])
        return coords, np.array(rows, dtype=np.int64)


def _to_float(tok: str) -> float:
    """Parse a float, tolerating Fortran exponents like ``1.0-3`` / ``1.0+3``."""
    try:
        return float(tok)
    except ValueError:
        for i in range(1, len(tok)):
            if tok[i] in "+-" and tok[i - 1] not in "eE":
                return float(tok[:i] + "e" + tok[i:])
        raise


def _collapse(nodes: list[int]) -> tuple[int, ...]:
    """Drop consecutive duplicate node ids (degenerate-hex / triangle collapse)."""
    out: list[int] = []
    for n in nodes:
        if not out or n != out[-1]:
            out.append(n)
    return tuple(out)


def _block(lines: list[str], i: int) -> tuple[list[str], int]:
    """Collect non-comment, non-blank data lines until the next ``*`` keyword.

    Returns the stripped data lines and the index of the next keyword (or the
    end of file). ``$`` comment lines (including ``$#`` headers) and blank
    lines are skipped.
    """
    body: list[str] = []
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith("*"):
            break
        if s and not s.startswith("$"):
            body.append(s)
        i += 1
    return body, i


def read_k(path: str, resolve_includes: bool = False) -> KModel:
    """Read an LS-DYNA keyword deck at ``path`` into a :class:`KModel`.

    When ``resolve_includes`` is true, ``*INCLUDE`` fragments are located
    relative to the deck's directory, parsed, and merged into the returned
    model. ``k_mesher`` writes *INCLUDE fragments with global node/element
    numbering, so merging is a plain union of nodes/elements/sets with no
    renumbering. Missing include files are ignored (their names remain in
    ``.includes``).
    """
    with open(path) as f:
        lines = f.read().splitlines()

    model = KModel()
    unknown: set[str] = set()
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s or s.startswith("$"):
            i += 1
            continue
        if not s.startswith("*"):
            i += 1  # stray data outside any keyword
            continue
        kw = s.upper()
        name = kw.split()[0]
        i += 1

        if name == "*END":
            break
        if name == "*KEYWORD":
            if "LONG=Y" in kw.replace(" ", ""):
                model.long_format = True
            continue

        body, i = _block(lines, i)

        if name == "*TITLE":
            if body:
                model.title = body[0]
        elif name == "*NODE":
            _read_nodes(model, body)
        elif name == "*ELEMENT_SOLID":
            _read_solids(model, body)
        elif name == "*ELEMENT_SHELL":
            _read_shells(model, body)
        elif name == "*PART":
            _read_part(model, body)
        elif name.startswith("*SECTION_SOLID"):
            _read_section(model, body, "*SECTION_SOLID")
        elif name.startswith("*SECTION_SHELL"):
            _read_section(model, body, "*SECTION_SHELL")
        elif name.startswith("*MAT_"):
            _read_mat(model, body, name)
        elif name.startswith("*SET_NODE_LIST"):
            _read_node_set(model, body, "_TITLE" in name)
        elif name.startswith("*SET_SEGMENT"):
            _read_segment_set(model, body, "_TITLE" in name)
        elif name == "*INCLUDE":
            if body:
                model.includes.append(body[0])
        else:
            unknown.add(name)

    model.unknown_keywords = sorted(unknown)

    if resolve_includes and model.includes:
        base = os.path.dirname(os.path.abspath(path))
        for inc in model.includes:
            frag_path = inc if os.path.isabs(inc) else os.path.join(base, inc)
            if os.path.isfile(frag_path):
                _merge(model, read_k(frag_path, resolve_includes=True))
    return model


# -- per-keyword handlers -------------------------------------------------
def _read_nodes(model: KModel, body: list[str]) -> None:
    for row in body:
        t = row.split()
        if len(t) >= 4:
            model.nodes[int(t[0])] = (
                _to_float(t[1]), _to_float(t[2]), _to_float(t[3]))


def _read_solids(model: KModel, body: list[str]) -> None:
    j = 0
    while j < len(body):
        t = body[j].split()
        if len(t) >= 10:  # one-line TET4 / degenerate hex: eid pid n1..n8
            eid, pid = int(t[0]), int(t[1])
            nodes = _collapse([int(x) for x in t[2:]])
            j += 1
        elif len(t) == 2 and j + 1 < len(body):  # two-line TET10
            eid, pid = int(t[0]), int(t[1])
            nodes = tuple(int(x) for x in body[j + 1].split())
            j += 2
        else:
            j += 1
            continue
        model.solids.append((eid, pid, nodes))


def _read_shells(model: KModel, body: list[str]) -> None:
    for row in body:
        t = row.split()
        if len(t) >= 4:
            eid, pid = int(t[0]), int(t[1])
            nodes = _collapse([int(x) for x in t[2:6]])
            model.shells.append((eid, pid, nodes))


def _read_part(model: KModel, body: list[str]) -> None:
    if len(body) < 2:
        return
    title = body[0]
    t = body[1].split()
    pid = int(t[0])
    model.parts[pid] = {
        "secid": int(t[1]) if len(t) > 1 else None,
        "mid": int(t[2]) if len(t) > 2 else None,
        "title": title,
    }


def _read_section(model: KModel, body: list[str], sec_type: str) -> None:
    if not body:
        return
    t = body[0].split()
    secid = int(t[0])
    entry = {"type": sec_type,
             "elform": int(_to_float(t[1])) if len(t) > 1 else None}
    if sec_type == "*SECTION_SHELL" and len(body) > 1:
        th = body[1].split()
        if th:
            entry["thickness"] = _to_float(th[0])
    model.sections[secid] = entry


def _read_mat(model: KModel, body: list[str], name: str) -> None:
    if not body:
        return
    t = body[0].split()
    try:
        mid = int(t[0])
    except (ValueError, IndexError):
        mid = -(len(model.materials) + 1)  # synthetic key if unparseable
    params: dict[str, float] = {}
    if name in ("*MAT_ELASTIC", "*MAT_RIGID") and len(t) >= 4:
        params = {"ro": _to_float(t[1]), "e": _to_float(t[2]),
                  "pr": _to_float(t[3])}
        if name == "*MAT_ELASTIC" and len(t) >= 6:
            params["da"] = _to_float(t[4])
            params["db"] = _to_float(t[5])
    model.materials[mid] = {"type": name, "params": params, "raw": list(body)}


def _read_node_set(model: KModel, body: list[str], has_title: bool) -> None:
    rows = body[1:] if has_title else body
    if not rows:
        return
    sid = int(rows[0].split()[0])
    nids: list[int] = []
    for row in rows[1:]:
        nids.extend(int(x) for x in row.split())
    model.node_sets[sid] = nids


def _read_segment_set(model: KModel, body: list[str], has_title: bool) -> None:
    rows = body[1:] if has_title else body
    if not rows:
        return
    sid = int(rows[0].split()[0])
    segs: list[tuple[int, int, int, int]] = []
    for row in rows[1:]:
        t = [int(x) for x in row.split()]
        if len(t) >= 4:
            segs.append((t[0], t[1], t[2], t[3]))
    model.segment_sets[sid] = segs


def _merge(dst: KModel, src: KModel) -> None:
    """Merge a fragment ``src`` into ``dst`` (global numbering, union of ids)."""
    dst.nodes.update(src.nodes)
    dst.solids.extend(src.solids)
    dst.shells.extend(src.shells)
    dst.parts.update(src.parts)
    dst.sections.update(src.sections)
    dst.materials.update(src.materials)
    dst.node_sets.update(src.node_sets)
    dst.segment_sets.update(src.segment_sets)
    dst.long_format = dst.long_format or src.long_format
    for kwname in src.unknown_keywords:
        if kwname not in dst.unknown_keywords:
            dst.unknown_keywords.append(kwname)
    dst.unknown_keywords.sort()
