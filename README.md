# k_mesher

[![tests](https://github.com/pmquang87/k_mesher/actions/workflows/ci.yml/badge.svg)](https://github.com/pmquang87/k_mesher/actions/workflows/ci.yml)

Mesh CAD geometry (STEP / IGES / BREP) or a surface tessellation
(STL / OBJ / PLY) into solid tetrahedra (TET4/TET10) or shells (TRI3/QUAD4)
and export an LS-DYNA keyword file (`.k`). Features: symmetry planes
(half/quarter/eighth models) with separate node-set and boundary-condition
options (symmetric / anti-symmetric / fixed / custom DOFs), per-body parts
with per-body materials (elastic or rigid), single-surface contact and
gravity load cards, defeaturing (remove holes/fillets), face sets with BCs
and loads (SPC / pressure / force), coordinate node sets (plane/box/sphere —
BCs and loads on STL/OBJ/PLY too), local mesh refinement (regions and
per-face sizes), LS-DYNA quality criteria with failed-element sets,
quality-driven auto-remeshing, mass properties with an explicit
critical-timestep estimate (plus an optional `*CONTROL_TIMESTEP` card and a
target-dt sizing report), shell integrity checks, mesh-only output for
`*INCLUDE` decks, per-part file export (standalone `.k` per body, or an
`*INCLUDE` assembly of mesh fragments plus a master deck), mesh export to
ParaView/Gmsh formats, automatic LONG=Y format when ids overflow the
standard fields, machine-readable statistics (JSON), parameter presets and
a batch queue with optional parallel workers.
Meshing is done with [gmsh](https://gmsh.info) (OpenCASCADE kernel), the GUI is
plain tkinter.

**Input formats** — STEP (`.step/.stp`), IGES (`.iges/.igs`) and BREP
(`.brep/.brp`) are B-rep CAD and support the full pipeline (solids, symmetry,
defeature, face sets). STL/OBJ/PLY are surface tessellations: meshed as-is with
shells, or — when the surface is watertight — tetrahedralized (TET4/TET10) by
reconstructing the enclosed volume. Tessellations still have no CAD geometry, so
symmetry/defeature/refinement need a B-rep. See
[docs/cad_formats.md](docs/cad_formats.md) for the full survey of CAD formats and
what is / isn't importable.

| Mesh parameters & refinement | Symmetry, face BCs/loads, defeature |
|---|---|
| ![Mesh tab](docs/screenshot_mesh.png) | ![Face sets and roles](docs/screenshot_faces.png) |

## Install & run

```
pip install -r requirements.txt
python main.py
```

Or install as a package — this puts the console scripts `k-mesher` (CLI),
`k-mesher-gui` (GUI), `k-mesher-doe` (mesh-convergence sweep) and
`k-mesher-convert` (FE-format conversion) on your PATH:

```
pip install .
k-mesher part.stp --size-max 8
k-mesher-gui
k-mesher-doe part.stp --sizes 12,8,4 --csv sweep.csv
k-mesher-convert mesh.msh mesh.k
```

Optional features live behind dependency extras: `pip install k-mesher[io]`
(meshio bridge), `[post]` (lasso-python results reader), `[extras]`
(scipy + matplotlib), or `[all]` for everything. The bridge modules and the
scipy/matplotlib code paths degrade gracefully when the optional dependency is
absent.

Or double-click / run `start_gui.py` — it automatically uses the project's
`.venv` interpreter (no console window) regardless of which Python starts it.

### Use as a library

`import k_mesher` exposes a small public API for driving meshing from Python:

```python
import k_mesher

settings = k_mesher.MeshSettings(size_max=8.0, size_min=1.0)
result = k_mesher.mesh_step_auto("part.stp", settings)   # -> MeshResult
k_mesher.write_k(result, "part.k")
```

`k_mesher.read_k` / `KModel` parse an existing `.k` deck, and the submodules
`k_mesher.connections` (spotweld / tied-contact detection), `k_mesher.doe`
(convergence sweeps), `k_mesher.mesh_io` (meshio bridge) and `k_mesher.post`
(lasso results reader) cover the rest. The package imports headless — the GUI
module is intentionally not pulled in by `import k_mesher`.

Command line (batch) use:

```
python -m k_mesher.mesh_cli part.stp --size-max 8 --size-min 1
python -m k_mesher.mesh_cli part.iges -o half.k --sym x --mat --elform 13
python -m k_mesher.mesh_cli part.brep --etype tet10 --algo hxt --sym x:0:+ --sym y:5:-
python -m k_mesher.mesh_cli part.stl --etype tri3 --thickness 1.2      # STL -> shells
python -m k_mesher.mesh_cli part.stl --etype tet4                      # watertight STL -> tets
python -m k_mesher.mesh_cli sheet.stp --etype quad4 --thickness 2.5
python -m k_mesher.mesh_cli part.stp --sym x --sym-constraint antisymmetric
python -m k_mesher.mesh_cli part.stp --sym x --no-sym-spc              # node set, no BC
python -m k_mesher.mesh_cli part.stp --sym x --sym-dofs 13             # custom SPC DOFs
python -m k_mesher.mesh_cli part.stp --sym x --sym-segset              # + plane segment set
python -m k_mesher.mesh_cli part.stp --list-faces
python -m k_mesher.mesh_cli part.stp --face-nodeset 7 --face-segset 12
python -m k_mesher.mesh_cli part.stp --refine-sphere 0:0:0:15:1.5 --face-size 6:2.0
python -m k_mesher.mesh_cli part.stp --defeature 7 --spc 1 --pressure 6:0.5 \
                            --force 4:z:-500 --implicit-cards --auto-refine
python -m k_mesher.mesh_cli part.stp --mesh-only -o part_mesh.k   # for *INCLUDE decks
python -m k_mesher.mesh_cli part.stp --mat --stats-json part_stats.json --title "bracket"
python -m k_mesher.mesh_cli asm.stp --glue --mat --part-mat 2:70000:0.33:2.7e-9
python -m k_mesher.mesh_cli asm.stp --contact 0.15 --tssfac 0.9 --gravity z:9810
python -m k_mesher.mesh_cli asm.stp --split-parts        # + one standalone .k per body
python -m k_mesher.mesh_cli asm.stp --split-include      # *INCLUDE fragments + master
python -m k_mesher.mesh_cli plate.stp --midsurface --mat   # thin plate -> midsurface shell
python -m k_mesher.mesh_cli asm.stp --auto-spotweld        # weld detected interfaces
python -m k_mesher.mesh_cli asm.stp --auto-spotweld 15     # + thin welds to 15 spacing
python -m k_mesher.mesh_cli asm.stp --tied-contact --connect-tol 0.05   # tie interfaces
python -m k_mesher.mesh_cli asm.stp --mat --part-rigid 2         # body 2 = *MAT_RIGID
python -m k_mesher.mesh_cli part.stl --etype tet4 --mat \
    --nset plane,z,0,spc=123 --nset sphere,0,0,40,15,force=z:-500
python -m k_mesher.mesh_cli part.stp --export part.vtk --mat --target-dt 5e-7
python -m k_mesher.mesh_cli asm.stp --mat --endtim 0.01 --mass-scale=-1e-6 \
    --control-energy --hourglass 5:0.05         # explicit control cards
python -m k_mesher.mesh_cli asm.stp --mat --d3plot-dt 1e-4 --database GLSTAT:1e-5 \
    --database MATSUM:1e-5                       # output requests
python -m k_mesher.mesh_cli asm.stp --mat --init-velocity 0:0:-5000 \
    --contact 0.1 --contact-type tied_surface_to_surface
python -m k_mesher.mesh_cli asm.stp --mat --rigidwall 0:0:-50:0:0:1:0.2 \
    --spotweld 101:202 --define-curve 99:0,0;1,1 \
    --prescribed-motion 5:3:0:99:1.0            # loads / BCs
python -m k_mesher.mesh_cli asm.stp --auto-contact --contact 0.1   # per-pair contact
python -m k_mesher.mesh_cli asm.stp --auto-contact tied_surface_to_surface
python -m k_mesher.mesh_cli part.stp --mat --cross-section 0:0:0:1:0:0:MIDCUT \
    --history-node 1,2,3 --history-solid 10,11    # section force + time history
python -m k_mesher.mesh_cli part.stp \
    --mat-model plastic_kinematic:210000:0.3:7.85e-9:1000:200   # plasticity material
python -m k_mesher.mesh_cli --version
```

Note: pass negative scientific-notation values with `=`, e.g.
`--mass-scale=-1e-6`, so argparse does not read them as a flag.

**Midsurface & automatic connections** (grouped under "mesh-time connections /
midsurface" in `-h`):

- `--midsurface` — for thin, roughly constant-thickness plate solids (the
  sheet-metal case), extract a midsurface **shell** mesh instead of solids
  (CAD B-rep only; rejected for STL/OBJ/PLY). This now also handles **curved**
  constant-thickness shells (single-curvature panels), not just flat plates.
  The detected wall thickness is printed and written on `*SECTION_SHELL`; an
  explicit `--thickness` overrides it. A shell `--etype` (tri3/quad4) is
  honoured, otherwise TRI3 is used.
- `--auto-spotweld [SPACING]` — detect the interfaces of a multi-body model and
  weld the coincident node pairs with `*CONSTRAINED_SPOTWELD`; the optional
  `SPACING` thins the pattern to that minimum spot spacing.
- `--tied-contact` — detect the interfaces and tie them with
  `*CONTACT_TIED_SURFACE_TO_SURFACE` over the interface `*SET_SEGMENT`s.
- `--connect-tol TOL` — node-matching tolerance shared by the two flags
  (default: 1e-3 of the bounding-box diagonal). On a single-body model the two
  connection flags find no interface and warn but do not fail. The connection
  cards are added to the single-file and `*INCLUDE`-master output only, not the
  standalone per-part split files.

**Assembly contacts / output / materials** (grouped under "assembly contacts /
output / materials" in `-h`):

- `--auto-contact [TYPE]` — detect the touching part pairs of a multi-body model
  and write one **scoped `*CONTACT_`** per pair (each part pair gets its own
  `*SET_PART_LIST` slave/master sets), instead of a single contact spanning the
  whole assembly. `TYPE` defaults to `automatic_surface_to_surface` (the other
  contact types are also accepted); the friction comes from `--contact FS` and
  the matching tolerance from `--connect-tol`. A single-body model warns and
  writes no card. Like the other connection cards it lands in the single-file
  and `*INCLUDE`-master output only, not the standalone per-part split files.
- `--cross-section X:Y:Z:NX:NY:NZ[:TITLE]` — a `*DATABASE_CROSS_SECTION_PLANE`
  section cut (in-plane point + normal) through the whole model; the required
  `*DATABASE_SECFORC` output request is added automatically. Repeatable.
- `--history-node`, `--history-solid`, `--history-shell` — comma-separated id
  lists (repeatable) written as `*DATABASE_HISTORY_NODE/_SOLID/_SHELL` time-
  history requests.
- `--mat-model MODEL:E:NU:RHO:SIGY[:ETAN]` — write a plasticity material in
  place of `--mat`: `plastic_kinematic` (`*MAT_PLASTIC_KINEMATIC`) or
  `piecewise` (`*MAT_PIECEWISE_LINEAR_PLASTICITY`).

`python -m k_mesher.mesh_cli -h` lists all options (the new LS-DYNA control / load
flags are grouped under "LS-DYNA control / loads"). GUI settings are remembered
between sessions in `k_mesher_settings.json` (written on every run and on
window close).

## Workflow (GUI)

1. **Files** — pick the CAD/mesh input (STEP/IGES/BREP or STL/OBJ/PLY); the
   `.k` output path is auto-filled.
2. **Mesh tab**
   - *Max/Min element size* in the model's length units.
   - *Refine by curvature* — targets N elements per full circle, so holes and
     fillets get smaller elements automatically.
   - *3D algorithm* — Delaunay (robust default), Frontal, or HXT (fast,
     multithreaded, recommended for large models).
   - *Geometry units* — leave on "File units" normally; force mm/m/etc. if a
     STEP file imports at the wrong scale.
   - *Heal geometry* — for dirty CAD (fixes degenerated/small edges and faces).
   - *Glue touching bodies* — merges coincident faces of touching solids so
     the mesh shares nodes at the interfaces (bodies behave as bonded).
   - *Local refinement regions* — sphere (cx, cy, cz, radius) or box
     (xmin..zmax) regions with their own element size, e.g. around a stress
     concentration. Keeps the node count down compared to global refinement.
3. **LS-DYNA output tab**
   - Element type:
     - **TET4** solid (ELFORM 10 or 13)
     - **TET10** solid (ELFORM 16 or 17) — for implicit stress analysis
       TET10 + ELFORM 16 is strongly recommended; linear tets are
       artificially stiff in bending.
     - **Shell TRI3** (ELFORM 4 or 17) — triangles on the model surfaces.
     - **Shell QUAD4** (ELFORM 16 or 2) — quad-dominant recombined mesh
       (a few triangles remain, written as degenerate quads).
     Shells take a *thickness* (written on `*SECTION_SHELL`, NIP = 5,
     SHRF = 0.8333). Surface-only STEP files are meshed directly with
     shells; for solids, the body's boundary surfaces are meshed.
   - Base part ID: each solid body in the STEP file becomes its own `*PART`
     (PID, PID+1, ...; shell meshes are one part). Start node/element/set IDs
     (the last renumbers node/segment/SPC/element sets) let you merge into
     assembly decks without ID clashes.
   - Optional `*MAT_ELASTIC` card (E, ν, ρ; defaults are steel in mm-t-s).
     Without it, define your own `*MAT_` with MID = base PID. Multi-body
     models can override the material **per body** — those parts get their
     own MID and `*MAT_ELASTIC` card, and the log reports per-part masses.
   - *Mesh-only output* — write an `*INCLUDE`-friendly file with only nodes,
     elements and sets (no `*PART`/`*SECTION`/`*MAT`/control cards); define
     those in the master deck. Combine with the start node/element/set IDs to
     merge several meshes without ID clashes.
   - *Per-body files* — two modes. **Standalone .k per body**: each body as
     its own self-contained file (`<output>_p<PID>[_<name>].k`) with
     compactly renumbered nodes, that body's material, and the sets filtered
     to the part (empty sets dropped; contact cards skipped since contact
     acts between parts; a face force's total is re-spread over each file's
     own face nodes). ***INCLUDE fragments + master deck**: the main output
     becomes a master deck (PART/MAT/contact/sets/`*INCLUDE` cards) and each
     body's mesh goes into a fragment that keeps the **global** numbering —
     every node defined in exactly one fragment — so the assembly rebuilds
     exactly and one part can be re-exported without touching the rest.
   - Per-body materials can be marked **rigid** (`*MAT_RIGID`; E/ν set the
     contact stiffness) — rigid parts are excluded from the timestep
     estimate.
   - *Contact between parts* — `*CONTACT_AUTOMATIC_SINGLE_SURFACE` over all
     parts with a friction coefficient, for assemblies whose bodies are not
     glued (glued bodies share nodes and need no contact).
   - *`*CONTROL_TIMESTEP`* with a chosen TSSFAC, and *gravity* via
     `*LOAD_BODY_X/Y/Z` (scaled unit ramp curve; a positive acceleration
     loads the negative axis, per the LS-DYNA base-acceleration convention).
   - Node/element ids that overflow the standard 8-character fields switch
     the file automatically to the LONG=Y keyword format (20-character
     fields); `--long-format` forces it.
4. **Symmetry & face sets tab**
   - Symmetry: tick X/Y/Z planes, set the plane position and which side to
     keep. The geometry is cut by boolean intersection *before* meshing, so
     the mesh conforms exactly to the plane. The node set and the boundary
     condition are now **separate options**:
     - *Write node set* — emit `*SET_NODE_LIST` for the symmetry-plane nodes.
     - *Apply boundary condition* — emit `*BOUNDARY_SPC_SET`, with a choice of
       constraint: **Symmetric** (default; fixes the normal translation and the
       two in-plane rotations), **Anti-symmetric** (the complement, for
       anti-symmetric loading), **Fixed** (all 6 DOFs), or **Custom DOFs** (type
       the DOF digits, e.g. `13`).

     Turn the BC off to get just the node set (e.g. to apply your own
     constraint, contact or coupling); turning the node set off while the BC is
     on still writes the node set the SPC references. *Write segment set*
     additionally emits a `*SET_SEGMENT` for the boundary faces lying on each
     symmetry plane (solids only — useful for contact or pressure on the cut
     face). (STL/OBJ/PLY inputs have no CAD geometry to cut, so symmetry is
     unavailable for them.)
   - Coordinate node sets: select nodes **by position** (plane, box or
     sphere) instead of by CAD face, with an optional SPC or force role.
     Because the selection runs on the mesh, this is the way to put BCs and
     loads on STL/OBJ/PLY inputs, which have no faces to scan.
   - Face sets: **Scan faces from geometry** lists all model faces with
     type, area and centroid (uses the current symmetry/glue settings —
     rescan after changing them). "Select small faces" finds candidates for
     defeaturing. Select faces (Ctrl-click) to write `*SET_NODE_LIST` and/or
     `*SET_SEGMENT`, or assign a **role**:
     - *Fix (SPC)* — `*BOUNDARY_SPC_SET` with chosen DOFs (1-6),
     - *Pressure* — `*SET_SEGMENT` + `*LOAD_SEGMENT_SET` (unit ramp curve,
       magnitude as scale factor),
     - *Force X/Y/Z* — `*SET_NODE_LIST` + `*LOAD_NODE_SET` (the total force
       is divided over the face nodes),
     - *Mesh size* — local element size on that face,
     - *Suppress (defeature)* — remove the face (hole, fillet, tangent
       sliver) from the solid with OpenCASCADE defeaturing before meshing.
       NOTE: defeaturing changes the face tags of the remaining faces —
       rescan to see the resulting geometry before assigning other roles.
5. **Batch & presets tab** — save/apply named parameter presets; queue
   several STEP files (each job snapshots the current settings) and run them
   unattended, optionally on several worker processes in parallel (each job
   runs in its own process — gmsh is single-instance per process).
6. **Generate mesh** — runs in the background; the log shows per-stage
   timings, element counts, mesh volume/area, mass + COG + inertia (per part
   when per-body materials are set) and an
   estimated explicit critical timestep dt = Lc/c (when a
   material is set; Lc is the worst element's characteristic length — tet
   minimum altitude, or (1+β)·area/longest-edge for shells — and c the
   material wave speed; apply your own TSSFAC). A dt distribution line
   (1% / 10% / median) shows whether the critical element is an outlier
   worth mass-scaling. Then a quality histogram
   (SICN, 1.0 is a perfect tet), an
   LS-DYNA-style **quality criteria table** (aspect ratio, SICN, warpage,
   min angle; failing elements are written as `*SET_SOLID`/`*SET_SHELL` for
   review in LS-PrePost), shell integrity checks (free edges, non-manifold
   edges, normal orientation — normals are auto-aligned), and the locations
   of badly shaped elements (usually dirty CAD spots — defeature them).
   "Auto-refine bad spots" remeshes up to 2 extra rounds with refinement
   spheres at the worst locations and keeps the best mesh (the preview always
   shows the kept mesh, and a failed refinement round falls back to the best
   earlier one); note that slivers caused by tangent faces cannot be refined
   away — defeature instead. "Preview last mesh" opens the interactive Gmsh
   viewer. On the command line, `--stats-json` writes all of these statistics
   (counts, quality criteria, mass properties, timestep estimate) to a JSON
   file for scripted pipelines.

## Output file contents

```
*KEYWORD / *TITLE     (*KEYWORD LONG=Y when ids need 20-char fields)
*CONTROL_TIMESTEP     (optional; chosen TSSFAC and/or DT2MS mass scaling)
*CONTROL_TERMINATION  (optional; --endtim)
*CONTROL_ENERGY       (optional; --control-energy)
*PART                 (one per solid body; SECID = base PID, MID = base PID
                       or the body's own PID with per-body materials;
                       HGID set when --hourglass is used)
*SECTION_SOLID /      (chosen ELFORM; SECTION_SHELL carries the thickness)
*SECTION_SHELL
*MAT_ELASTIC          (optional; one per referenced MID)
*HOURGLASS            (optional; --hourglass, referenced by every *PART)
*CONTACT_...          (optional; single-surface, or the --contact-type card;
                       friction on card 2)
*SET_PART_LIST /      (optional; --auto-contact writes one scoped *CONTACT_ per
*CONTACT_...           touching part pair via slave/master part-list sets)
*LOAD_BODY_X/Y/Z      (optional gravity; scaled unit ramp curve)
*NODE                 (I8 id, 3 x E16.9 coordinates)
*ELEMENT_SOLID        (TET4: one line, tet as degenerate hex;
                       TET10: two-line format, 10 nodes)
*ELEMENT_SHELL        (quads n1..n4, triangles as n1 n2 n3 n3)
*SET_NODE_LIST_TITLE  (symmetry planes and selected faces)
*BOUNDARY_SPC_SET     (one per symmetry plane)
*SET_SEGMENT_TITLE    (selected faces; triangles as degenerate quads)
*DEFINE_CURVE         (optional; --define-curve, user load curves)
*INITIAL_VELOCITY_GENERATION      (optional; --init-velocity)
*BOUNDARY_PRESCRIBED_MOTION_SET   (optional; --prescribed-motion)
*RIGIDWALL_PLANAR     (optional; --rigidwall)
*CONSTRAINED_SPOTWELD (optional; --spotweld, one per node pair)
*DATABASE_BINARY_D3PLOT / *DATABASE_<NAME>   (optional; --d3plot-dt / --database)
*DATABASE_CROSS_SECTION_PLANE     (optional; --cross-section, + auto SECFORC)
*DATABASE_HISTORY_NODE/_SOLID/_SHELL   (optional; --history-node/-solid/-shell)
*END
```

### LS-DYNA control / load cards

Beyond the mesh, `.k` output can carry a set of optional analysis cards
(CLI flags in parentheses; `run_job`/the GUI wire the same options):

- `*CONTROL_TERMINATION` — analysis end time (`--endtim`).
- `*CONTROL_TIMESTEP` — TSSFAC (`--tssfac`) and DT2MS mass scaling
  (`--mass-scale`, typically negative; forces the card even without a TSSFAC).
- `*CONTROL_ENERGY` — hourglass / sliding / rigidwall / Rayleigh energy
  tracking (`--control-energy`).
- `*HOURGLASS` — one card (`--hourglass IHQ[:QM]`) whose HGID is stamped on
  every `*PART`.
- `*DATABASE_BINARY_D3PLOT` and ASCII `*DATABASE_<NAME>` output requests
  (`--d3plot-dt`, repeatable `--database NAME:DT`).
- `*INITIAL_VELOCITY_GENERATION` over all nodes, translational plus optional
  rotational components (`--init-velocity VX:VY:VZ[:VXR:VYR:VZR]`).
- `*CONTACT_...` with a chosen type (`--contact-type`, used together with
  `--contact FS`): automatic single-surface / surface-to-surface, or tied
  surface-to-surface / nodes-to-surface.
- `*RIGIDWALL_PLANAR` (`--rigidwall`, repeatable), `*CONSTRAINED_SPOTWELD`
  (`--spotweld`, repeatable), user `*DEFINE_CURVE` (`--define-curve`,
  repeatable) and `*BOUNDARY_PRESCRIBED_MOTION_SET` (`--prescribed-motion`,
  repeatable).

These are writer-only cards, added to the single-file and `*INCLUDE`-master
output; the standalone per-part split files intentionally omit contact.

All tets are checked and reoriented for positive volume before writing.
TET10 node ordering follows the LS-DYNA convention (mid-side nodes 5-10).
Shell normals and segment set normals follow the gmsh surface orientation —
check the sign when applying pressure loads. The symmetry SPC constrains the
normal translation plus the two in-plane rotations, which is correct for
both solids and shells.

## Performance notes

The default Delaunay volume mesher and the Netgen optimizer are
**single-threaded** — on large models most of the wall time is one core at
100%. The log shows per-stage timings so you can see where time goes. For
big meshes:

- Switch the 3D algorithm to **HXT** — it is multithreaded (all cores) and
  usually several times faster; quality is comparable.
- Element count grows roughly with `1/size³`: halving the element size means
  ~8x more elements and far more than 8x meshing time. Prefer local
  refinement regions over a small global size.
- Disabling "Optimize element quality" skips the (serial) optimization passes;
  useful for quick previews of the sizing.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | root launcher, launches the GUI |
| `start_gui.py` | root launcher that relaunches via the project `.venv` (double-click friendly) |
| `k_mesher/__init__.py` | public library API (`import k_mesher`) |
| `k_mesher/_version.py` | single-source package version |
| `k_mesher/mesh_cli.py` | command-line interface for batch meshing (`python -m k_mesher.mesh_cli`) |
| `k_mesher/gui.py` | tkinter GUI |
| `k_mesher/job_runner.py` | GUI-independent mesh+write job execution (also used by the parallel batch workers) |
| `k_mesher/mesher.py` | gmsh meshing core (STEP/IGES/BREP/STL/OBJ/PLY import, symmetry, refinement, tet/shell extraction) |
| `k_mesher/dyna_writer.py` | LS-DYNA `.k` writer (single file, per-part files, *INCLUDE assemblies) |
| `k_mesher/k_reader.py` | LS-DYNA `.k` keyword reader (`read_k` / `KModel`) |
| `k_mesher/connections.py` | automatic connection detection between touching bodies (spotweld pairs / tied-contact segment sets) |
| `k_mesher/mesh_io.py` | meshio bridge: convert k_mesher meshes to/from other FE formats |
| `k_mesher/post.py` | post-processing bridge: read LS-DYNA results (d3plot/binout) back via lasso |
| `k_mesher/post_viz.py` | results visualization: plots (matplotlib PNG) and ParaView `.vtu` exports from `post` results |
| `k_mesher/doe.py` | design-of-experiments / mesh-convergence driver (meshing as code) |
| `k_mesher/preview.py` | standalone Gmsh viewer process |
| `pyproject.toml` | packaging (`pip install .` → `k-mesher` / `k-mesher-gui`) |
| `examples/make_test_step.py` | generates test parts (STEP + IGES/BREP/STL) |
| `tests/test_headless.py` | end-to-end tests, pytest-compatible (`pytest tests/ -v` or `python tests/test_headless.py`) |
| `tests/test_gui.py` | headless GUI tests (skipped when tkinter/display is unavailable; CI runs them under Xvfb) |
| `docs/cad_formats.md` | survey of CAD input formats and what is importable |

## Notes on element types

Hex meshing of arbitrary STEP solids is not something gmsh does
automatically — that would need transfinite/swept regions or an external hex
mesher, so solid meshes are tetrahedra only. (gmsh's tet-subdivision "all-hex"
mode exists but produces badly distorted hexes and is deliberately not
offered.) Shells are meshed on the model's surfaces: use them for
surface-only STEP exports or thin-walled parts. For thin, roughly
constant-thickness plate solids (sheet metal), `--midsurface` extracts a
midsurface shell mesh and sets the shell thickness from the measured wall
gap; general or strongly curved solids are not midsurfaced, so export the
midsurface from CAD in those cases.

### Python modules

Beyond the CLI/GUI, k_mesher can be driven as a library via `import k_mesher`
(see **Use as a library** above): `k_mesher.mesher` + `k_mesher.dyna_writer`
mesh and write, `k_mesher.connections` derives spotweld/tied-contact data from
a `MeshResult`, `k_mesher.mesh_io` converts meshes to/from other FE formats via
meshio, `k_mesher.post` reads LS-DYNA results back (lasso), `k_mesher.post_viz`
turns those results into plots / ParaView `.vtu` exports, and `k_mesher.doe`
runs element-size / parameter studies as code.
