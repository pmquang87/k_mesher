"""Design-of-experiments / mesh-convergence driver for k_mesher.

Meshing-as-code: vary the element size (or an arbitrary ``MeshSettings``
field) across a list of values, mesh each one, collect the per-run statistics
that ``mesher`` already computes, and report a convergence curve. This turns
k_mesher's single-run stats into a scriptable study.

The existing GUI batch queue runs independent *jobs* across worker processes;
this module is the library-level equivalent for *sweeps* of one geometry.

Cost / concurrency notes
------------------------
* Meshing many sizes is expensive: for a solid the element count grows roughly
  as ``1 / size**3``, so halving the element size multiplies the tet count by
  ~8 and the run time by more. Start coarse and refine only as far as the
  quantity of interest needs (see :func:`convergence`).
* gmsh is a process-global singleton (one live ``initialize()/finalize()``
  per process). This driver therefore runs every mesh **serially in-process**.
  To use more cores, run several sweeps across separate processes the way the
  batch queue does (e.g. ``concurrent.futures.ProcessPoolExecutor`` with a
  ``spawn`` context) - do not thread this driver.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import time
from dataclasses import dataclass, fields, replace

from k_mesher import mesher

# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Metrics for one mesh in a sweep.

    ``value`` is the value the swept ``field`` was set to for this run. On a
    failed run every metric stays ``None`` and ``error`` holds the message, so
    a single bad size never aborts the whole sweep.
    """
    field: str                       # the MeshSettings field that was varied
    value: object                    # the value it was set to for this run
    n_nodes: int | None = None
    n_elems: int | None = None
    quality_min: float | None = None
    char_length: float | None = None
    critical_timestep: float | None = None   # only when a material is given
    seconds: float | None = None             # wall-clock for this mesh
    error: str | None = None                 # message if the run failed


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def _field_names() -> set[str]:
    return {f.name for f in fields(mesher.MeshSettings)}


def _run_one(settings: mesher.MeshSettings, field: str, value,
             mat: dict | None, log) -> SweepResult:
    """Mesh a single settings variant and collect its metrics. Never raises
    for a meshing failure - the error is captured on the record."""
    rec = SweepResult(field=field, value=value)
    t0 = time.perf_counter()
    try:
        result = mesher.mesh_step_auto(settings, log=log)
    except Exception as e:                      # noqa: BLE001 - record & continue
        rec.seconds = time.perf_counter() - t0
        rec.error = f"{type(e).__name__}: {e}"
        log(f"Sweep {field}={value!r} failed: {rec.error}")
        return rec
    rec.seconds = time.perf_counter() - t0

    stats = result.stats
    rec.n_nodes = stats.get("n_nodes")
    rec.n_elems = stats.get("n_elems")
    rec.quality_min = stats.get("quality_min")
    rec.char_length = stats.get("char_length")
    if mat is not None:
        rec.critical_timestep = mesher.critical_timestep(
            stats, settings.element_type, mat)
    log(f"Sweep {field}={value!r}: {rec.n_elems} elements, "
        f"{rec.seconds:.2f} s")
    return rec


def param_sweep(settings: mesher.MeshSettings, field: str, values,
                mat: dict | None = None, log=print) -> list[SweepResult]:
    """Mesh ``settings`` once per value of an arbitrary ``MeshSettings`` field.

    ``field`` is any field name of :class:`mesher.MeshSettings` (e.g.
    ``"size_max"``, ``"element_type"``, ``"algorithm3d"``, ``"curvature_elems"``).
    For each ``value`` the settings are cloned with ``dataclasses.replace`` and
    meshed with :func:`mesher.mesh_step_auto`. When ``mat`` (an elastic
    material dict ``{"e", "pr", "ro"}``) is given, the explicit critical
    timestep is recorded too. Runs are serial and robust to a failing value.
    """
    valid = _field_names()
    if field not in valid:
        raise ValueError(
            f"{field!r} is not a MeshSettings field. Valid fields: "
            f"{', '.join(sorted(valid))}")
    records = []
    for value in values:
        variant = replace(settings, **{field: value})
        records.append(_run_one(variant, field, value, mat, log))
    return records


def size_sweep(settings: mesher.MeshSettings, sizes, mat: dict | None = None,
               log=print) -> list[SweepResult]:
    """Convergence sweep over the maximum element size (``size_max``).

    For each size in ``sizes`` the settings are cloned with
    ``size_max=size``; ``size_min`` is clamped so it never exceeds the new
    ``size_max`` (an inverted range makes gmsh error). Otherwise identical to
    :func:`param_sweep` with ``field="size_max"``.
    """
    records = []
    for size in sizes:
        size_min = min(settings.size_min, size)
        variant = replace(settings, size_max=size, size_min=size_min)
        records.append(_run_one(variant, "size_max", size, mat, log))
    return records


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------


@dataclass
class Convergence:
    """A metric's convergence curve over the successful runs of a sweep.

    ``values`` is the metric per successful run (sweep order). ``rel_change``
    is the relative change of each value from its predecessor
    (``abs(v[i] - v[i-1]) / abs(v[i-1])``); its length is ``len(values) - 1``.
    ``last_change`` is the final relative change - a scalar indicator that
    approaches 0 once the quantity of interest has stabilised.
    """
    metric: str
    values: list[float]
    rel_change: list[float]
    last_change: float | None = None


def convergence(records: list[SweepResult], metric: str = "n_elems") -> Convergence:
    """Extract one metric as a series plus a convergence indicator.

    ``metric`` is any numeric :class:`SweepResult` field (``"n_elems"``,
    ``"n_nodes"``, ``"quality_min"``, ``"char_length"``,
    ``"critical_timestep"``, ``"seconds"``). Records that failed or whose
    metric is missing are skipped. Inspect ``rel_change`` / ``last_change`` to
    see when successive refinements stop moving the quantity of interest.
    """
    if not any(f.name == metric for f in fields(SweepResult)):
        raise ValueError(f"{metric!r} is not a SweepResult field")
    values = [float(getattr(r, metric)) for r in records
              if r.error is None and getattr(r, metric) is not None]
    rel_change = [
        abs(values[i] - values[i - 1]) / abs(values[i - 1])
        if values[i - 1] != 0 else float("inf")
        for i in range(1, len(values))
    ]
    last = rel_change[-1] if rel_change else None
    return Convergence(metric=metric, values=values, rel_change=rel_change,
                       last_change=last)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def _rows(records: list[SweepResult]) -> list[dict]:
    return [dataclasses.asdict(r) for r in records]


def to_csv(records: list[SweepResult], path: str) -> str:
    """Write the sweep records as CSV (one row per run). Returns ``path``."""
    field_names = [f.name for f in fields(SweepResult)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for row in _rows(records):
            writer.writerow(row)
    return path


def to_json(records: list[SweepResult], path: str) -> str:
    """Write the sweep records as a JSON list of objects. Returns ``path``."""
    with open(path, "w") as fh:
        json.dump(_rows(records), fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# optional plotting (matplotlib is NOT a hard dependency)
# ---------------------------------------------------------------------------

def plot_convergence(records: list[SweepResult], metric: str, path: str) -> bool:
    """Write a PNG of ``metric`` vs. the swept value - only if matplotlib is
    importable. Returns ``True`` if a file was written, ``False`` if
    matplotlib is unavailable (import is lazy, so it is never required).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless backend, no display needed
        import matplotlib.pyplot as plt
    except Exception:                   # noqa: BLE001 - optional dependency
        return False

    good = [r for r in records
            if r.error is None and getattr(r, metric) is not None]
    xs = [r.value for r in good]
    ys = [getattr(r, metric) for r in good]
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(good[0].field if good else "value")
    ax.set_ylabel(metric)
    ax.set_title(f"Convergence of {metric}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    """Console entry (``k-mesher-doe``): run an element-size convergence sweep."""
    import argparse

    p = argparse.ArgumentParser(
        prog="k-mesher-doe",
        description="Mesh-convergence sweep: mesh a CAD file at several element "
                    "sizes and report how the mesh metrics evolve.")
    p.add_argument("input", help="CAD/mesh input (STEP/IGES/BREP/STL/...)")
    p.add_argument("--sizes", required=True,
                   help="comma-separated max element sizes, e.g. 12,8,4")
    p.add_argument("--etype", default="tet4",
                   choices=["tet4", "tet10", "tri3", "quad4"])
    p.add_argument("--csv", help="write the sweep records to this CSV file")
    p.add_argument("--json", help="write the sweep records to this JSON file")
    args = p.parse_args(argv)

    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    settings = mesher.MeshSettings(step_file=args.input,
                                   element_type=args.etype.upper(),
                                   size_max=max(sizes), size_min=0.0)
    records = size_sweep(settings, sizes)
    print(f"{'size':>10} {'nodes':>10} {'elems':>10} {'q_min':>8}  status")
    for r in records:
        if r.error:
            print(f"{r.value:>10g} {'':>10} {'':>10} {'':>8}  ERROR: {r.error}")
        else:
            print(f"{r.value:>10g} {r.n_nodes:>10} {r.n_elems:>10} "
                  f"{(r.quality_min or 0):>8.3f}  ok ({r.seconds:.1f}s)")
    if args.csv:
        to_csv(records, args.csv)
        print(f"Wrote {args.csv}")
    if args.json:
        to_json(records, args.json)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
