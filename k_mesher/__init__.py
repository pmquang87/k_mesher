"""k_mesher — mesh CAD geometry to LS-DYNA keyword (.k) decks.

Public library API. Import the meshing core, the LS-DYNA writer, the .k
reader, and the interop/automation helpers directly::

    import k_mesher
    settings = k_mesher.MeshSettings(step_file="part.step", element_type="TET4",
                                     size_max=8.0, size_min=0.0)
    result = k_mesher.mesh_step_auto(settings)
    k_mesher.write_k("part.k", result.coords, result.elems, mat={"e": 210000,
                     "pr": 0.3, "ro": 7.85e-9}, endtim=5.0)

The tkinter GUI (``k_mesher.gui``) is intentionally NOT imported here so that
``import k_mesher`` works in headless environments; import it explicitly if
you need it.
"""
from k_mesher._version import __version__
from k_mesher.dyna_writer import write_k, write_k_include, write_k_split
from k_mesher.k_reader import KModel, read_k
from k_mesher.mesher import (
    MeshResult,
    MeshSettings,
    critical_timestep,
    list_faces,
    mass_and_timestep,
    mesh_step,
    mesh_step_auto,
    midsurface_shell,
    select_nodes,
)

# Submodules with light or lazily-imported dependencies: safe to expose.
from k_mesher import connections, doe, mesh_io, post  # noqa: E402,F401

__all__ = [
    "__version__",
    "MeshSettings",
    "MeshResult",
    "mesh_step",
    "mesh_step_auto",
    "midsurface_shell",
    "list_faces",
    "select_nodes",
    "critical_timestep",
    "mass_and_timestep",
    "write_k",
    "write_k_include",
    "write_k_split",
    "read_k",
    "KModel",
    "connections",
    "mesh_io",
    "post",
    "doe",
]
