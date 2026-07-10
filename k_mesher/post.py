"""Post-processing bridge: read LS-DYNA results back into k_mesher.

This module is a *thin* wrapper around `lasso-python
<https://pypi.org/project/lasso-python/>`_. It does **not** re-implement any
binary/ASCII parsing; it delegates every read to lasso and exposes a small,
well-named set of accessors so a k_mesher user can pull results out of a
``d3plot`` (binary state database) or a ``binout`` (ASCII/HDF-style time
history database) and map them back onto the parts and node-sets that
k_mesher authored -- closing the CAD -> mesh -> run -> results loop without
leaving Python.

lasso is imported lazily inside the loader helpers so that ``import post``
succeeds even when lasso is not installed; the ImportError is only raised when
you actually try to read a file.

Relevant lasso ArrayType keys used here (``from lasso.dyna import ArrayType``)::

    ArrayType.node_ids              -> "node_ids"
    ArrayType.node_coordinates      -> "node_coordinates"
    ArrayType.node_displacement     -> "node_displacement"
    ArrayType.global_timesteps      -> "timesteps"
    ArrayType.element_solid_stress  -> "element_solid_stress"
    ArrayType.element_shell_stress  -> "element_shell_stress"
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

__all__ = [
    "D3plotResult",
    "BinoutResult",
    "read_d3plot",
    "read_binout",
    "max_displacement",
    "peak_part_energy",
    "map_node_results_to_set",
]

_INSTALL_HINT = "lasso-python is required for post-processing: pip install lasso-python"


# ---------------------------------------------------------------------------
# Lazy lasso loaders. Tests monkeypatch these to inject fake D3plot/Binout
# classes, so keep them as the single indirection point to lasso.
# ---------------------------------------------------------------------------
def _load_d3plot_class() -> Any:
    """Return lasso's ``D3plot`` class, raising a clear ImportError if absent."""
    try:
        from lasso.dyna import D3plot
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(_INSTALL_HINT) from exc
    return D3plot


def _load_binout_class() -> Any:
    """Return lasso's ``Binout`` class, raising a clear ImportError if absent."""
    try:
        from lasso.dyna import Binout
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(_INSTALL_HINT) from exc
    return Binout


# ArrayType keys as plain strings, so this module does not need lasso imported
# to describe what it reads. These mirror ``lasso.dyna.ArrayType`` values.
_KEY_NODE_IDS = "node_ids"
_KEY_NODE_COORDS = "node_coordinates"
_KEY_NODE_DISP = "node_displacement"
_KEY_TIMESTEPS = "timesteps"
_KEY_SOLID_STRESS = "element_solid_stress"
_KEY_SHELL_STRESS = "element_shell_stress"


# ---------------------------------------------------------------------------
# d3plot
# ---------------------------------------------------------------------------
@dataclass
class D3plotResult:
    """Convenience view over a lasso ``D3plot.arrays`` dictionary.

    Only the common quantities are surfaced with clear names; the raw lasso
    arrays remain available through :attr:`arrays` for anything not wrapped.
    """

    arrays: dict
    path: Optional[str] = None

    def _get(self, key: str) -> Optional[np.ndarray]:
        val = self.arrays.get(key)
        return None if val is None else np.asarray(val)

    @property
    def node_ids(self) -> Optional[np.ndarray]:
        """LS-DYNA node ids, shape ``(n_nodes,)`` (ArrayType.node_ids)."""
        return self._get(_KEY_NODE_IDS)

    @property
    def node_coordinates(self) -> Optional[np.ndarray]:
        """Initial node coordinates, shape ``(n_nodes, 3)`` (ArrayType.node_coordinates)."""
        return self._get(_KEY_NODE_COORDS)

    @property
    def times(self) -> Optional[np.ndarray]:
        """Output time of each state, shape ``(n_states,)`` (ArrayType.global_timesteps)."""
        return self._get(_KEY_TIMESTEPS)

    @property
    def node_displacement(self) -> Optional[np.ndarray]:
        """Nodal displacement per state, shape ``(n_states, n_nodes, 3)``.

        Corresponds to ``ArrayType.node_displacement``.
        """
        return self._get(_KEY_NODE_DISP)

    @property
    def solid_stress(self) -> Optional[np.ndarray]:
        """Solid element stress if present, else ``None`` (ArrayType.element_solid_stress).

        Typical shape ``(n_states, n_solids, n_history_vars, 6)``.
        """
        return self._get(_KEY_SOLID_STRESS)

    @property
    def shell_stress(self) -> Optional[np.ndarray]:
        """Shell element stress if present, else ``None`` (ArrayType.element_shell_stress).

        Typical shape ``(n_states, n_shells, n_layers, 6)``.
        """
        return self._get(_KEY_SHELL_STRESS)

    @property
    def n_states(self) -> int:
        """Number of output states, inferred from the timesteps array."""
        t = self.times
        return 0 if t is None else int(t.shape[0])


def read_d3plot(path: str) -> D3plotResult:
    """Read an LS-DYNA ``d3plot`` database into a :class:`D3plotResult`.

    Parameters
    ----------
    path:
        Path to the ``d3plot`` root file (lasso auto-discovers ``d3plot01``,
        ``d3plot02`` ... siblings).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *path* is not a readable/valid d3plot database.
    ImportError
        If lasso-python is not installed.
    """
    if not path:
        raise ValueError("read_d3plot: an empty path was given")
    if not os.path.exists(path):
        raise FileNotFoundError(f"d3plot not found: {path}")

    d3plot_cls = _load_d3plot_class()
    try:
        plot = d3plot_cls(path)
    except (FileNotFoundError, ImportError):
        raise
    except Exception as exc:  # lasso raises a variety of parse/struct errors
        raise ValueError(
            f"failed to read '{path}' as an LS-DYNA d3plot: {exc}"
        ) from exc

    arrays = getattr(plot, "arrays", None)
    if arrays is None:
        raise ValueError(
            f"'{path}' was opened but exposes no '.arrays'; it is not a valid d3plot"
        )
    return D3plotResult(arrays=dict(arrays), path=path)


def max_displacement(result: D3plotResult, state: int = -1) -> float:
    """Return the largest nodal displacement magnitude at *state*.

    The L2 norm of each node's displacement vector is taken at the requested
    state (default: the last state) and the maximum over all nodes returned.

    Raises
    ------
    ValueError
        If the result has no displacement data.
    """
    disp = result.node_displacement
    if disp is None or disp.size == 0:
        raise ValueError("result has no node_displacement data")
    frame = disp[state]  # (n_nodes, 3)
    mags = np.linalg.norm(frame, axis=-1)
    return float(np.max(mags))


def map_node_results_to_set(
    result: D3plotResult,
    node_ids: Sequence[int],
    quantity: str = "displacement",
    state: int = -1,
    id_base: int = 1,
) -> np.ndarray:
    """Return *quantity* for a k_mesher node-set at a given state.

    This is the "map results back onto the sets I authored" capability.

    **Node-id base assumption.** k_mesher writes nodes to the ``.k`` deck with
    contiguous 1-based ids that match the mesh row order, and LS-DYNA preserves
    that order in the d3plot state arrays. Therefore a node-set id ``i`` maps to
    array row ``i - id_base`` (positional lookup). ``id_base`` defaults to ``1``
    to match the 1-based ids written to the deck; pass ``id_base=0`` if you are
    already handing 0-based row indices.

    Parameters
    ----------
    result:
        A :class:`D3plotResult`.
    node_ids:
        The k_mesher node-set ids (1-based as written to the ``.k`` deck).
    quantity:
        ``"displacement"`` -> displacement vectors at *state*, shape
        ``(len(node_ids), 3)``; ``"coordinates"`` -> initial coordinates, shape
        ``(len(node_ids), 3)``.
    state:
        State index for time-dependent quantities (default last state).
    id_base:
        The base of the incoming ids (default ``1``).

    Raises
    ------
    ValueError
        For an unknown *quantity*, missing data, or out-of-range ids.
    """
    ids = np.asarray(node_ids, dtype=np.int64)
    if ids.size == 0:
        return np.empty((0, 3))
    rows = ids - id_base
    if np.any(rows < 0):
        raise ValueError(
            f"node id(s) below id_base={id_base} produced negative row indices"
        )

    if quantity == "displacement":
        data = result.node_displacement
        if data is None:
            raise ValueError("result has no node_displacement data")
        frame = data[state]  # (n_nodes, 3)
    elif quantity == "coordinates":
        frame = result.node_coordinates
        if frame is None:
            raise ValueError("result has no node_coordinates data")
    else:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected 'displacement' or 'coordinates'"
        )

    n_nodes = frame.shape[0]
    if np.any(rows >= n_nodes):
        raise ValueError(
            f"node id(s) map to rows beyond the {n_nodes} nodes in the result"
        )
    return np.asarray(frame)[rows]


# ---------------------------------------------------------------------------
# binout
# ---------------------------------------------------------------------------
@dataclass
class BinoutResult:
    """Convenience view over a lasso ``Binout`` object.

    Wraps :meth:`lasso.dyna.Binout.read`. Use :meth:`get` for arbitrary
    branch/variable access, or the named convenience methods for the common
    databases (``glstat``, ``matsum``, ``rcforc``).
    """

    binout: Any
    path: Optional[str] = None

    def branches(self) -> list:
        """Top-level branches present in the binout (e.g. ``glstat``, ``matsum``)."""
        result = self.binout.read()
        return list(result) if result is not None else []

    def get(self, branch: str, name: Optional[str] = None) -> Any:
        """Read a branch, or a named variable within a branch.

        ``get("glstat")`` lists the variables in the ``glstat`` branch;
        ``get("glstat", "kinetic_energy")`` returns that time history array.
        Delegates directly to ``Binout.read(*args)``.
        """
        if name is None:
            return self.binout.read(branch)
        return self.binout.read(branch, name)

    def time(self, branch: str = "glstat") -> np.ndarray:
        """Return the time vector for *branch* (``read(branch, 'time')``)."""
        return np.asarray(self.get(branch, "time"))

    def glstat_energy(self) -> dict:
        """Return global energy time histories from the ``glstat`` branch.

        Keys: ``time``, ``total_energy``, ``kinetic_energy``, ``internal_energy``.
        Any variable absent in this run is returned as ``None``.
        """
        out = {"time": None}
        wanted = {
            "time": "time",
            "total_energy": "total_energy",
            "kinetic_energy": "kinetic_energy",
            "internal_energy": "internal_energy",
        }
        for key, var in wanted.items():
            try:
                out[key] = np.asarray(self.get("glstat", var))
            except Exception:
                out[key] = None
        return out

    def matsum_part_energy(self, variable: str = "internal_energy") -> np.ndarray:
        """Return per-part energy history from ``matsum``.

        Shape is typically ``(n_states, n_parts)``. ``variable`` is one of
        ``internal_energy``, ``kinetic_energy``, etc.
        """
        return np.asarray(self.get("matsum", variable))

    def rcforc_force(self, variable: str = "x_force") -> np.ndarray:
        """Return a contact-force time history from ``rcforc``.

        ``variable`` is e.g. ``x_force``, ``y_force``, ``z_force``.
        """
        return np.asarray(self.get("rcforc", variable))


def read_binout(path: str) -> BinoutResult:
    """Read an LS-DYNA ``binout`` database into a :class:`BinoutResult`.

    Parameters
    ----------
    path:
        Path to a ``binout`` file (or a glob such as ``binout*``).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist (and is not a glob matching existing files).
    ValueError
        If *path* is not a readable/valid binout database.
    ImportError
        If lasso-python is not installed.
    """
    if not path:
        raise ValueError("read_binout: an empty path was given")
    # Allow glob patterns (binout is often split into binout0000, binout0001...);
    # only enforce existence for a plain, non-glob path.
    is_glob = any(ch in path for ch in "*?[")
    if not is_glob and not os.path.exists(path):
        raise FileNotFoundError(f"binout not found: {path}")

    binout_cls = _load_binout_class()
    try:
        binout = binout_cls(path)
    except (FileNotFoundError, ImportError):
        raise
    except Exception as exc:
        raise ValueError(
            f"failed to read '{path}' as an LS-DYNA binout: {exc}"
        ) from exc
    return BinoutResult(binout=binout, path=path)


def peak_part_energy(
    result: BinoutResult,
    variable: str = "internal_energy",
    part: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Return the peak per-part energy from a ``matsum`` binout.

    With ``part=None`` the maximum over all parts and all states is returned as
    a float. Pass a 0-based *part* column index to get that part's peak instead.

    Raises
    ------
    ValueError
        If the ``matsum`` branch has no usable data.
    """
    data = np.asarray(result.matsum_part_energy(variable))
    if data.size == 0:
        raise ValueError(f"matsum branch has no '{variable}' data")
    if part is not None:
        col = data[..., part] if data.ndim >= 2 else data
        return float(np.max(np.abs(col)))
    return float(np.max(np.abs(data)))
