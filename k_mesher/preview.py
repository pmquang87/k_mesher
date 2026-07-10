"""Open a mesh file in the interactive Gmsh viewer.

Run as a separate process so the main GUI is not blocked:
    python preview.py <mesh.msh>
"""
import sys

import gmsh


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python preview.py <mesh.msh>")
        sys.exit(1)
    gmsh.initialize()
    try:
        gmsh.open(sys.argv[1])
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.option.setNumber("Mesh.VolumeEdges", 0)
        gmsh.fltk.run()
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
