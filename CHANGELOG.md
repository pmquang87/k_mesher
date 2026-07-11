# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-07-11

### Added

- **HEX8 solid meshing** (`--etype hex8`) — structured/transfinite hexahedra
  for box-like or sweepable solids, written as one-line `*ELEMENT_SOLID` rows
  with `*SECTION_SOLID` ELFORM 1 (default) or 2. Non-boxlike geometry,
  tessellation input, mixed (partially recombined) meshes and combinations
  with symmetry/defeature/refinements/face-sizes/boundary-layer are rejected
  with clear errors.
- **Boundary layers** (`--boundary-layer THICKNESS[:RATIO[:NLAYERS[:SIZEWALL]]]`
  plus `--bl-faces`, and `MeshSettings.boundary_layer`) — distance-graded
  near-wall sizing (Distance + Threshold fields) from selected or all wall
  faces, for TET and shell meshing.
- **Crash cards** in the writer, CLI and `run_job` kopts — lumped masses
  (`--point-mass` → `*ELEMENT_MASS`), spring/damper elements (`--spring` /
  `--damper` → `*ELEMENT_DISCRETE` + `*SECTION_DISCRETE` +
  `*MAT_SPRING_ELASTIC` / `*MAT_DAMPER_VISCOUS`), nodal rigid bodies
  (`--nodal-rigid-body` → `*CONSTRAINED_NODAL_RIGID_BODY`), global damping
  (`--damping` → `*DAMPING_GLOBAL`) and geometric rigid walls
  (`--rigidwall-sphere` / `--rigidwall-cylinder` →
  `*RIGIDWALL_GEOMETRIC_SPHERE` / `_CYLINDER`).
- **Multi-region midsurface extraction** — `--midsurface` now handles
  multi-region plate solids, not just a single detected wall.
- **Per-part shell thickness** — shell thickness can be set per part instead
  of one global `*SECTION_SHELL` value.

- **Importable `k_mesher` package** — the flat modules were repackaged into a
  `k_mesher/` package with a public API exposed via `import k_mesher`
  (`MeshSettings`, `MeshResult`, `mesh_step` / `mesh_step_auto`,
  `midsurface_shell`, `list_faces`, `select_nodes`, timestep/mass helpers, the
  `.k` reader/writer functions and the `KModel` type). The package imports
  headless; the GUI module is not pulled in.
- **New library modules** — `connections` (automatic spotweld / tied-contact
  detection between touching bodies), `mesh_io` (meshio bridge to/from other FE
  formats), `post` (lasso bridge to read LS-DYNA results back) and `doe`
  (mesh-convergence / parameter-study driver).
- **Midsurface shell extraction** (`--midsurface`) for thin, roughly
  constant-thickness plate solids, with the wall thickness measured and written
  on `*SECTION_SHELL`.
- **Automatic connection detection** in the CLI and GUI — `--auto-spotweld`
  (`*CONSTRAINED_SPOTWELD`) and `--tied-contact`
  (`*CONTACT_TIED_SURFACE_TO_SURFACE`) over detected interfaces, with a shared
  `--connect-tol` node-matching tolerance.
- **New console scripts** — `k-mesher-doe` (mesh-convergence sweep) and
  `k-mesher-convert` (FE-format conversion via meshio, writing `.k` through the
  LS-DYNA writer), alongside the existing `k-mesher` / `k-mesher-gui`.
- **Optional-dependency extras** — `k-mesher[io]` (meshio), `[post]`
  (lasso-python), `[extras]` (scipy + matplotlib) and `[all]`; the bridge
  modules and scipy/matplotlib code paths degrade gracefully when the optional
  dependency is absent.

### Changed

- The command-line tool is now invoked as `python -m k_mesher.mesh_cli` from a
  source checkout (or the `k-mesher` console script); the old
  `python mesh_cli.py` form no longer works because the module moved into the
  package. `python main.py` and `start_gui.py` remain root GUI launchers.

## [0.5.0]

### Added

- **LS-DYNA deck completeness** — broader keyword coverage in the deck writer,
  including control, database, hourglass and mass-scaling cards, initial
  velocity, prescribed motion, rigid walls, spotwelds and tied contacts, a
  generic material passthrough for arbitrary `*MAT` cards, and a more flexible
  `*DEFINE_CURVE` writer.
- **`.k` reader module** — a new module for parsing existing LS-DYNA `.k`
  keyword files.
- **`--version` flag** on the command-line interface (backed by the new
  single-source `_version` module).

### Changed

- **Mesher concurrency and robustness** — fixes around parallel/concurrent
  meshing and general robustness of the meshing pipeline.

### Infrastructure

- Single source of version truth in `_version.py`, consumed dynamically by the
  build backend.
- Enriched package metadata (authors, keywords, classifiers, project URLs).
- CI now tests across Python 3.10–3.13, collects coverage, cancels superseded
  runs, and builds/smoke-tests the wheel and console entry point.

## [0.4.0]

### Added

- `*INCLUDE` assemblies and multi-part assembly cards.
- Coordinate-driven node sets and rigid parts.
- Per-part `.k` export (`--split-parts` / GUI checkbox).
- Parallel batch processing and additional export options.
- `LONG=Y` deck format and timestep (dt) reporting.

## [0.3.0]

### Added

- IGES / BREP / STL geometry input in addition to STEP.
- Symmetry boundary-condition options.
- Set-ID control.

## [0.1.0]

### Added

- Initial release: mesh STEP geometry to LS-DYNA `.k` files
  (TET4 / TET10 solids, TRI3 / QUAD4 shells).

[Unreleased]: https://github.com/pmquang87/k_mesher/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.6.0
[0.5.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.5.0
[0.4.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.4.0
[0.3.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.3.0
[0.1.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.1.0
