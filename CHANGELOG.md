# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/pmquang87/k_mesher/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.5.0
[0.4.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.4.0
[0.3.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.3.0
[0.1.0]: https://github.com/pmquang87/k_mesher/releases/tag/v0.1.0
