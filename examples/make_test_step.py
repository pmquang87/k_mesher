"""Create a simple test part (100 x 50 x 25 block with a through-hole,
centered on the origin) and save it as examples/test_part.step."""
import os

import gmsh


def make(path: str) -> None:
    gmsh.initialize()
    try:
        occ = gmsh.model.occ
        box = occ.addBox(-50, -25, -12.5, 100, 50, 25)
        hole = occ.addCylinder(0, 0, -20, 0, 0, 40, 10)
        occ.cut([(3, box)], [(3, hole)])
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def make_shell(path: str) -> None:
    """Surface-only model: the 6 faces of a 50 x 30 x 20 box, no solid
    (for shell meshing tests). Total area = 6200."""
    gmsh.initialize()
    try:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 50, 30, 20)
        occ.synchronize()
        occ.remove([(3, box)], recursive=False)  # keep only the faces
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def make_two_bodies(path: str) -> None:
    """Two boxes sharing the face at x=20 (for multi-part / glue tests)."""
    gmsh.initialize()
    try:
        occ = gmsh.model.occ
        occ.addBox(0, 0, 0, 20, 20, 20)
        occ.addBox(20, 0, 0, 20, 20, 20)
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def make_formats(base: str) -> None:
    """Write the block-with-hole test part (see make()) in the alternative CAD
    formats IGES and BREP, plus an STL tessellation, next to `base`.stl etc.
    (for the multi-format import tests). `base` is a path without extension."""
    gmsh.initialize()
    try:
        occ = gmsh.model.occ
        box = occ.addBox(-50, -25, -12.5, 100, 50, 25)
        hole = occ.addCylinder(0, 0, -20, 0, 0, 40, 10)
        occ.cut([(3, box)], [(3, hole)])
        occ.synchronize()
        for ext in ("iges", "brep"):
            gmsh.write(f"{base}.{ext}")
        # STL needs a surface mesh first
        gmsh.option.setNumber("Mesh.MeshSizeMax", 6.0)
        gmsh.model.mesh.generate(2)
        gmsh.write(f"{base}.stl")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    make(os.path.join(here, "test_part.step"))
    make_two_bodies(os.path.join(here, "test_two_bodies.step"))
    make_shell(os.path.join(here, "test_shell.step"))
    make_formats(os.path.join(here, "test_part"))
    print(f"Wrote test files to {here}")
