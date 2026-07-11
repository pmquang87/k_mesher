"""Results-visualization layer for k_mesher post-processing.

This module turns the result objects produced by :mod:`k_mesher.post`
(``D3plotResult`` / ``BinoutResult``) into pictures (PNG via matplotlib) and
ParaView-ready files (``.vtu`` via meshio), so a k_mesher user can close the
CAD -> mesh -> run -> results loop *visually* without leaving Python.

It does **not** re-implement any file reading: every array it consumes comes
from a :class:`k_mesher.post.D3plotResult` or
:class:`k_mesher.post.BinoutResult`.

matplotlib and meshio are imported lazily inside the helpers so that
``import post_viz`` succeeds even when they are absent; a friendly ImportError
("pip install matplotlib" / "pip install meshio") is raised only when a
plotting/writing function is actually called.

Public API
----------
    plot_energy(binout_result, path, quantities=None) -> str
    plot_displacement_history(d3plot_result, node_ids=None, path=..., id_base=1) -> str
    plot_part_energy(binout_result, path) -> str
    deformed_shape_png(d3plot_result, path, state=-1, scale=1.0, component="magnitude") -> str
    results_to_vtu(d3plot_result, path, elems=None, element_kind="solid", state=-1) -> str
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "plot_energy",
    "plot_displacement_history",
    "plot_part_energy",
    "deformed_shape_png",
    "results_to_vtu",
    "von_mises",
]

_MPL_HINT = "matplotlib is required for plotting; install it with 'pip install matplotlib'"
_MESHIO_HINT = "meshio is required for .vtu export; install it with 'pip install meshio'"

# The three global-energy channels plotted by default, in draw order.
_DEFAULT_ENERGY = ("total_energy", "kinetic_energy", "internal_energy")


# ---------------------------------------------------------------------------
# Lazy imports. Kept as the single indirection point so tests can monkeypatch.
# ---------------------------------------------------------------------------
def _require_pyplot():
    """Import matplotlib's Agg pyplot lazily with a friendly error.

    The non-interactive ``Agg`` backend is forced *before* importing pyplot so
    the functions work headless (no ``$DISPLAY``) and never pop a window.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(_MPL_HINT) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _require_meshio():
    """Import meshio lazily with a friendly error if it is not installed."""
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(_MESHIO_HINT) from exc
    return meshio


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def von_mises(stress: np.ndarray) -> np.ndarray:
    """Return the von Mises equivalent stress from Cauchy stress components.

    ``stress`` has its **last axis** of length 6 holding the LS-DYNA component
    order ``(xx, yy, zz, xy, yz, zx)``. Any leading axes are preserved, so an
    array of shape ``(..., 6)`` returns shape ``(...,)``.
    """
    stress = np.asarray(stress, dtype=float)
    if stress.shape[-1] != 6:
        raise ValueError(
            f"stress last axis must be 6 (xx,yy,zz,xy,yz,zx); got {stress.shape}"
        )
    sxx = stress[..., 0]
    syy = stress[..., 1]
    szz = stress[..., 2]
    sxy = stress[..., 3]
    syz = stress[..., 4]
    szx = stress[..., 5]
    dev = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
    shear = sxy ** 2 + syz ** 2 + szx ** 2
    return np.sqrt(0.5 * dev + 3.0 * shear)


def _displacement_magnitude(vectors: np.ndarray) -> np.ndarray:
    """L2 norm of displacement vectors along the last axis."""
    return np.linalg.norm(np.asarray(vectors, dtype=float), axis=-1)


def _element_von_mises(stress: np.ndarray, state: int) -> np.ndarray:
    """Reduce a per-element stress array to one von Mises value per element.

    LS-DYNA solid/shell stress arrays are shaped
    ``(n_states, n_elems, n_mid, 6)`` where ``n_mid`` is the number of history
    variables (solids) or through-thickness layers (shells). The values at the
    requested ``state`` are averaged over ``n_mid`` before the von Mises norm,
    giving one scalar per element. Arrays without the mid axis
    (``(n_states, n_elems, 6)``) are handled too.
    """
    arr = np.asarray(stress, dtype=float)
    frame = arr[state]  # (n_elems, n_mid, 6) or (n_elems, 6)
    if frame.ndim == 3:
        frame = frame.mean(axis=1)  # average over history vars / layers
    return von_mises(frame)  # (n_elems,)


# ---------------------------------------------------------------------------
# Energy plots (binout)
# ---------------------------------------------------------------------------
def plot_energy(
    binout_result,
    path: str,
    quantities: Optional[Sequence[str]] = None,
) -> str:
    """Plot global energy time-histories from a ``glstat`` binout to a PNG.

    Draws total / kinetic / internal energy versus time on one axis using the
    dict returned by ``binout_result.glstat_energy()``.

    Parameters
    ----------
    binout_result:
        A :class:`k_mesher.post.BinoutResult`.
    path:
        Destination PNG path (returned unchanged).
    quantities:
        Optional subset/order of channels to draw; defaults to
        ``("total_energy", "kinetic_energy", "internal_energy")``.

    Raises
    ------
    ValueError
        If no time vector or none of the requested channels are present.
    """
    energy = binout_result.glstat_energy()
    time = energy.get("time")
    if time is None:
        raise ValueError("glstat branch has no 'time' vector to plot against")
    time = np.asarray(time)

    channels = tuple(quantities) if quantities is not None else _DEFAULT_ENERGY

    plt = _require_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        drawn = 0
        for name in channels:
            series = energy.get(name)
            if series is None:
                continue
            ax.plot(time, np.asarray(series), label=name.replace("_", " "))
            drawn += 1
        if drawn == 0:
            raise ValueError(
                f"none of the requested energy channels {list(channels)} "
                "are present in the glstat branch"
            )
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        ax.set_title("Global energy time-history")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return path


def plot_part_energy(binout_result, path: str, variable: str = "internal_energy") -> str:
    """Plot per-part energy from a ``matsum`` binout to a PNG.

    Draws each part's energy history (a line per part) using
    ``binout_result.matsum_part_energy(variable)``, which is shaped
    ``(n_states, n_parts)``.

    Raises
    ------
    ValueError
        If the ``matsum`` branch has no usable data.
    """
    data = np.asarray(binout_result.matsum_part_energy(variable), dtype=float)
    if data.size == 0:
        raise ValueError(f"matsum branch has no '{variable}' data to plot")
    if data.ndim == 1:
        data = data[:, None]  # single part -> (n_states, 1)

    try:
        time = np.asarray(binout_result.time("matsum"), dtype=float)
        if time.shape[0] != data.shape[0]:
            raise ValueError
    except Exception:
        time = np.arange(data.shape[0], dtype=float)

    n_states, n_parts = data.shape

    plt = _require_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for part in range(n_parts):
            ax.plot(time, data[:, part], marker="o", label=f"part {part + 1}")
        ax.set_xlabel("time")
        ax.set_ylabel(variable.replace("_", " "))
        ax.set_title("Per-part energy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Displacement plots (d3plot)
# ---------------------------------------------------------------------------
def plot_displacement_history(
    d3plot_result,
    node_ids: Optional[Sequence[int]] = None,
    path: str = "displacement_history.png",
    id_base: int = 1,
) -> str:
    """Plot displacement-magnitude time-histories to a PNG.

    With ``node_ids=None`` the maximum nodal displacement magnitude over all
    nodes is drawn per state. Otherwise one curve is drawn per requested node,
    selected positionally via the ``id_base`` convention that
    :func:`k_mesher.post.map_node_results_to_set` uses (id ``i`` -> row
    ``i - id_base``).

    Raises
    ------
    ValueError
        If displacement or time data is missing, or a node id is out of range.
    """
    disp = d3plot_result.node_displacement
    if disp is None or np.asarray(disp).size == 0:
        raise ValueError("result has no node_displacement data")
    disp = np.asarray(disp, dtype=float)  # (n_states, n_nodes, 3)

    times = d3plot_result.times
    if times is None or np.asarray(times).size == 0:
        times = np.arange(disp.shape[0], dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    mags = _displacement_magnitude(disp)  # (n_states, n_nodes)

    plt = _require_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if node_ids is None:
            ax.plot(times, mags.max(axis=1), marker="o", label="max |disp|")
        else:
            ids = np.asarray(list(node_ids), dtype=np.int64)
            rows = ids - id_base
            if np.any(rows < 0):
                raise ValueError(
                    f"node id(s) below id_base={id_base} produced negative rows"
                )
            if np.any(rows >= mags.shape[1]):
                raise ValueError(
                    f"node id(s) map to rows beyond the {mags.shape[1]} nodes "
                    "in the result"
                )
            for nid, row in zip(ids, rows):
                ax.plot(times, mags[:, row], marker="o", label=f"node {int(nid)}")
        ax.set_xlabel("time")
        ax.set_ylabel("displacement magnitude")
        ax.set_title("Displacement time-history")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return path


def deformed_shape_png(
    d3plot_result,
    path: str,
    state: int = -1,
    scale: float = 1.0,
    component: str = "magnitude",
) -> str:
    """Render a 3D scatter of the deformed shape at ``state`` to a PNG.

    Node coordinates are displaced by ``displacement * scale`` at the requested
    state and drawn as a 3D scatter colored by displacement magnitude (or a
    single component ``"x"``/``"y"``/``"z"``). Robust for point clouds.

    Raises
    ------
    ValueError
        If coordinates or displacements are missing, or ``component`` is bad.
    """
    coords = d3plot_result.node_coordinates
    if coords is None or np.asarray(coords).size == 0:
        raise ValueError("result has no node_coordinates data")
    coords = np.asarray(coords, dtype=float)

    disp = d3plot_result.node_displacement
    if disp is None or np.asarray(disp).size == 0:
        raise ValueError("result has no node_displacement data")
    frame = np.asarray(disp, dtype=float)[state]  # (n_nodes, 3)

    comp_index = {"x": 0, "y": 1, "z": 2}
    if component == "magnitude":
        color = _displacement_magnitude(frame)
    elif component in comp_index:
        color = frame[:, comp_index[component]]
    else:
        raise ValueError(
            f"unknown component {component!r}; expected 'magnitude', 'x', 'y' or 'z'"
        )

    deformed = coords + frame * scale

    plt = _require_pyplot()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d proj

    fig = plt.figure(figsize=(7, 6))
    try:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            deformed[:, 0],
            deformed[:, 1],
            deformed[:, 2],
            c=color,
            cmap="viridis",
            depthshade=True,
        )
        fig.colorbar(sc, ax=ax, shrink=0.6, label=f"displacement {component}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Deformed shape (state {state}, scale {scale:g})")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# ParaView export (d3plot -> .vtu)
# ---------------------------------------------------------------------------
def results_to_vtu(
    d3plot_result,
    path: str,
    elems: Optional[np.ndarray] = None,
    element_kind: str = "solid",
    state: int = -1,
) -> str:
    """Write a ParaView ``.vtu`` of the results at ``state`` via meshio.

    The mesh geometry is the node coordinates displaced by the displacement at
    ``state`` (so the file shows the deformed shape). Point data always carries
    the displacement vector and its magnitude. When ``elems`` (k_mesher 1-based
    connectivity) is given and the matching element stress is present, per-cell
    von Mises stress is attached as cell data.

    Parameters
    ----------
    d3plot_result:
        A :class:`k_mesher.post.D3plotResult`.
    path:
        Destination ``.vtu`` path (returned unchanged).
    elems:
        Optional k_mesher 1-based connectivity ``(M, K)``. If given, the
        k_mesher -> meshio cell mapping is reused (via
        :func:`k_mesher.mesh_io.to_meshio`).
    element_kind:
        ``"solid"`` or ``"shell"`` — selects tet/shell mapping and which stress
        array (``solid_stress`` / ``shell_stress``) is read for cell data.
    state:
        State index to export (default: last state).

    Raises
    ------
    ValueError
        If coordinates or displacements are missing, or ``element_kind`` bad.
    """
    meshio = _require_meshio()

    coords = d3plot_result.node_coordinates
    if coords is None or np.asarray(coords).size == 0:
        raise ValueError("result has no node_coordinates data")
    coords = np.asarray(coords, dtype=float)

    disp = d3plot_result.node_displacement
    if disp is None or np.asarray(disp).size == 0:
        raise ValueError("result has no node_displacement data")
    frame = np.asarray(disp, dtype=float)[state]  # (n_nodes, 3)
    if frame.shape[0] != coords.shape[0]:
        raise ValueError(
            f"displacement rows ({frame.shape[0]}) do not match node count "
            f"({coords.shape[0]})"
        )

    magnitude = _displacement_magnitude(frame)
    point_data = {
        "displacement": frame,
        "displacement_magnitude": magnitude,
    }

    if element_kind not in ("solid", "shell"):
        raise ValueError(
            f"element_kind must be 'solid' or 'shell'; got {element_kind!r}"
        )

    if elems is None:
        # Point cloud: a "vertex" cell block per node keeps ParaView happy.
        n = coords.shape[0]
        cells = [("vertex", np.arange(n, dtype=np.int64).reshape(n, 1))]
        mesh = meshio.Mesh(coords + frame, cells, point_data=point_data)
        meshio.write(path, mesh)
        return path

    # Reuse the k_mesher -> meshio connectivity mapping.
    from k_mesher import mesh_io

    mesh = mesh_io.to_meshio(coords, elems, element_kind=element_kind)
    # Deform the geometry and attach nodal point data.
    mesh.points = coords + frame
    mesh.point_data = dict(point_data)

    stress = (
        d3plot_result.solid_stress if element_kind == "solid"
        else d3plot_result.shell_stress
    )
    if stress is not None and np.asarray(stress).size:
        vm = _element_von_mises(stress, state)  # (n_elems,)
        # Distribute the per-element von Mises across the (possibly split) cell
        # blocks in the same element order to_meshio produced them.
        cell_data = _split_cell_data(mesh, vm, elems, element_kind)
        if cell_data is not None:
            mesh.cell_data = {"von_mises": cell_data}

    meshio.write(path, mesh)
    return path


def _split_cell_data(mesh, values: np.ndarray, elems, element_kind: str):
    """Split a per-element scalar into per-block lists matching ``mesh.cells``.

    ``to_meshio`` keeps solids as one block, but splits a shell family into a
    "triangle" block then a "quad" block. This reproduces that split so a flat
    per-element array lines up with the meshio cell blocks.
    """
    values = np.asarray(values, dtype=float)
    total = sum(len(cb.data) for cb in mesh.cells)
    if values.shape[0] != total:
        # Element/stress count mismatch — skip cell data rather than mislabel.
        return None

    if element_kind == "solid":
        return [values]

    # Shell: to_meshio emits triangles (degenerate quads, n4 == n3) first,
    # then quads, preserving relative order within each group.
    conn = np.asarray(elems)
    is_tri = conn[:, 3] == conn[:, 2]
    out = []
    for cb in mesh.cells:
        if cb.type == "triangle":
            out.append(values[is_tri])
        elif cb.type == "quad":
            out.append(values[~is_tri])
        else:  # pragma: no cover - defensive
            out.append(values[: len(cb.data)])
    return out


def main(argv=None) -> int:
    """Console entry (``k-mesher-post``): make quick plots / a VTU from results.

    Reads an LS-DYNA d3plot (for displacement plots / deformed shape / VTU) or a
    binout (for energy plots) and writes PNG/VTU files.
    """
    import argparse

    from k_mesher import post

    p = argparse.ArgumentParser(
        prog="k-mesher-post",
        description="Visualize LS-DYNA results: energy / displacement plots, "
                    "deformed shape, or a ParaView .vtu (via lasso + matplotlib "
                    "+ meshio).")
    p.add_argument("result", help="a d3plot file (displacement/shape/vtu) or a "
                                  "binout file (energy)")
    p.add_argument("--kind", choices=["d3plot", "binout"], default="d3plot",
                   help="how to read the result file")
    p.add_argument("--energy", metavar="PNG",
                   help="binout: write the global energy time-history plot")
    p.add_argument("--part-energy", metavar="PNG",
                   help="binout: write the per-part energy plot")
    p.add_argument("--displacement", metavar="PNG",
                   help="d3plot: write the max-displacement time-history plot")
    p.add_argument("--deformed", metavar="PNG",
                   help="d3plot: write a deformed-shape image (last state)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="displacement scale factor for --deformed")
    p.add_argument("--vtu", metavar="VTU",
                   help="d3plot: write a ParaView .vtu of the last state")
    args = p.parse_args(argv)

    written = []
    if args.kind == "binout":
        res = post.read_binout(args.result)
        if args.energy:
            written.append(plot_energy(res, args.energy))
        if args.part_energy:
            written.append(plot_part_energy(res, args.part_energy))
    else:
        res = post.read_d3plot(args.result)
        if args.displacement:
            written.append(plot_displacement_history(res, path=args.displacement))
        if args.deformed:
            written.append(deformed_shape_png(res, args.deformed, scale=args.scale))
        if args.vtu:
            written.append(results_to_vtu(res, args.vtu))
    if not written:
        p.error("nothing to do: pass at least one output option "
                "(--energy/--part-energy/--displacement/--deformed/--vtu)")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
