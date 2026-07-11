"""Tests for :mod:`post_viz`, the results-visualization layer.

We have no real d3plot/binout, so (mirroring ``tests/test_post.py``) synthetic
result objects are built: ``post.D3plotResult`` wraps a hand-made ``arrays``
dict, and ``post.BinoutResult`` wraps a tiny fake ``Binout`` with a ``read()``
dispatch. The PNG functions are checked by asserting a non-trivial file is
written (matplotlib forced to the headless Agg backend); ``results_to_vtu`` is
round-tripped through meshio. All tests are hermetic and fast.
"""
import os
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
pytest.importorskip("meshio", reason="meshio not installed")

matplotlib.use("Agg", force=True)  # headless: set backend before pyplot import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from k_mesher import post, post_viz  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / builders
# ---------------------------------------------------------------------------
class FakeBinout:
    """Tiny stand-in for ``lasso.dyna.Binout`` with a read() dispatch."""

    def __init__(self):
        self._db = {
            (): ["glstat", "matsum"],
            ("glstat", "time"): np.array([0.0, 1.0, 2.0]),
            ("glstat", "total_energy"): np.array([100.0, 100.0, 100.0]),
            ("glstat", "kinetic_energy"): np.array([0.0, 40.0, 10.0]),
            ("glstat", "internal_energy"): np.array([0.0, 55.0, 85.0]),
            ("matsum", "time"): np.array([0.0, 1.0, 2.0]),
            # (states=3, parts=2)
            ("matsum", "internal_energy"): np.array(
                [[0.0, 0.0], [30.0, 20.0], [45.0, 25.0]]
            ),
        }

    def read(self, *args):
        return self._db[tuple(args)]


def make_binout_result():
    return post.BinoutResult(binout=FakeBinout(), path="binout")


def make_d3plot_result(with_solid_stress=True):
    """A 5-node / 2-tet synthetic result with 2 states."""
    n_nodes = 5
    disp = np.zeros((2, n_nodes, 3))
    disp[1, 4] = [3.0, 4.0, 0.0]  # magnitude 5.0 at last state, node 5
    disp[1, 1] = [0.0, 0.0, 2.0]  # magnitude 2.0
    arrays = {
        "node_ids": np.array([1, 2, 3, 4, 5]),
        "node_coordinates": np.array(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
        ),
        "node_displacement": disp,
        "timesteps": np.array([0.0, 1.0]),
    }
    if with_solid_stress:
        # (n_states, n_elems=2, n_hist=1, 6). All-ones -> von Mises 3.0.
        arrays["element_solid_stress"] = np.ones((2, 2, 1, 6))
    return post.D3plotResult(arrays=arrays, path="d3plot")


# k_mesher 1-based tet connectivity for the 5-node mesh (2 tets).
SOLID_ELEMS = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])


def _assert_nonempty_png(path):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000  # a real PNG, not an empty stub
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic


# ---------------------------------------------------------------------------
# von Mises helper
# ---------------------------------------------------------------------------
def test_von_mises_uniaxial():
    # pure sigma_xx = 10 -> von Mises 10
    s = np.array([10.0, 0, 0, 0, 0, 0])
    assert post_viz.von_mises(s) == pytest.approx(10.0)


def test_von_mises_all_ones():
    s = np.ones(6)
    assert post_viz.von_mises(s) == pytest.approx(3.0)


def test_von_mises_bad_shape():
    with pytest.raises(ValueError, match="last axis must be 6"):
        post_viz.von_mises(np.ones(5))


# ---------------------------------------------------------------------------
# Energy plots
# ---------------------------------------------------------------------------
def test_plot_energy(tmp_path):
    out = tmp_path / "energy.png"
    ret = post_viz.plot_energy(make_binout_result(), str(out))
    assert ret == str(out)
    _assert_nonempty_png(str(out))


def test_plot_energy_subset(tmp_path):
    out = tmp_path / "energy_ke.png"
    post_viz.plot_energy(
        make_binout_result(), str(out), quantities=["kinetic_energy"]
    )
    _assert_nonempty_png(str(out))


def test_plot_energy_no_channels(tmp_path):
    out = tmp_path / "energy_none.png"
    with pytest.raises(ValueError, match="none of the requested"):
        post_viz.plot_energy(make_binout_result(), str(out), quantities=["bogus"])


def test_plot_part_energy(tmp_path):
    out = tmp_path / "part_energy.png"
    ret = post_viz.plot_part_energy(make_binout_result(), str(out))
    assert ret == str(out)
    _assert_nonempty_png(str(out))


# ---------------------------------------------------------------------------
# Displacement plots
# ---------------------------------------------------------------------------
def test_plot_displacement_history_max(tmp_path):
    out = tmp_path / "disp.png"
    ret = post_viz.plot_displacement_history(make_d3plot_result(), path=str(out))
    assert ret == str(out)
    _assert_nonempty_png(str(out))


def test_plot_displacement_history_nodes(tmp_path):
    out = tmp_path / "disp_nodes.png"
    post_viz.plot_displacement_history(
        make_d3plot_result(), node_ids=[2, 5], path=str(out)
    )
    _assert_nonempty_png(str(out))


def test_plot_displacement_history_out_of_range(tmp_path):
    out = tmp_path / "disp_bad.png"
    with pytest.raises(ValueError, match="beyond"):
        post_viz.plot_displacement_history(
            make_d3plot_result(), node_ids=[99], path=str(out)
        )


def test_plot_displacement_history_no_data(tmp_path):
    out = tmp_path / "disp_empty.png"
    empty = post.D3plotResult(arrays={}, path="d3plot")
    with pytest.raises(ValueError, match="no node_displacement"):
        post_viz.plot_displacement_history(empty, path=str(out))


def test_deformed_shape_png(tmp_path):
    out = tmp_path / "deformed.png"
    ret = post_viz.deformed_shape_png(make_d3plot_result(), str(out), scale=2.0)
    assert ret == str(out)
    _assert_nonempty_png(str(out))


def test_deformed_shape_png_component(tmp_path):
    out = tmp_path / "deformed_z.png"
    post_viz.deformed_shape_png(make_d3plot_result(), str(out), component="z")
    _assert_nonempty_png(str(out))


def test_deformed_shape_bad_component(tmp_path):
    out = tmp_path / "deformed_bad.png"
    with pytest.raises(ValueError, match="unknown component"):
        post_viz.deformed_shape_png(make_d3plot_result(), str(out), component="w")


# ---------------------------------------------------------------------------
# VTU export
# ---------------------------------------------------------------------------
def test_results_to_vtu_point_cloud(tmp_path):
    import meshio

    out = tmp_path / "cloud.vtu"
    ret = post_viz.results_to_vtu(make_d3plot_result(), str(out))
    assert ret == str(out)

    mesh = meshio.read(str(out))
    assert mesh.points.shape == (5, 3)
    assert "displacement" in mesh.point_data
    assert mesh.point_data["displacement"].shape == (5, 3)
    assert "displacement_magnitude" in mesh.point_data
    # last-state deformed coords = coords + disp; node 5 magnitude 5.0
    assert mesh.point_data["displacement_magnitude"][4] == pytest.approx(5.0)


def test_results_to_vtu_solid_with_stress(tmp_path):
    import meshio

    out = tmp_path / "solid.vtu"
    post_viz.results_to_vtu(
        make_d3plot_result(),
        str(out),
        elems=SOLID_ELEMS,
        element_kind="solid",
    )

    mesh = meshio.read(str(out))
    assert mesh.points.shape == (5, 3)
    assert "displacement" in mesh.point_data
    assert "displacement_magnitude" in mesh.point_data
    # von Mises cell data present, one value per tet (all-ones stress -> 3.0)
    assert "von_mises" in mesh.cell_data
    vm = np.concatenate([np.asarray(a) for a in mesh.cell_data["von_mises"]])
    assert vm.shape == (2,)
    np.testing.assert_allclose(vm, [3.0, 3.0])
    # deformed geometry: node 5 moved by (3,4,0) from (1,1,1)
    np.testing.assert_allclose(mesh.points[4], [4.0, 5.0, 1.0])


def test_results_to_vtu_solid_no_stress(tmp_path):
    import meshio

    out = tmp_path / "solid_nostress.vtu"
    result = make_d3plot_result(with_solid_stress=False)
    post_viz.results_to_vtu(result, str(out), elems=SOLID_ELEMS, element_kind="solid")

    mesh = meshio.read(str(out))
    assert "displacement" in mesh.point_data
    assert not mesh.cell_data.get("von_mises")  # no stress -> no cell data


def test_results_to_vtu_no_coords(tmp_path):
    out = tmp_path / "nocoords.vtu"
    empty = post.D3plotResult(arrays={"node_displacement": np.zeros((1, 1, 3))})
    with pytest.raises(ValueError, match="no node_coordinates"):
        post_viz.results_to_vtu(empty, str(out))


def test_results_to_vtu_bad_kind(tmp_path):
    out = tmp_path / "badkind.vtu"
    with pytest.raises(ValueError, match="element_kind"):
        post_viz.results_to_vtu(make_d3plot_result(), str(out), element_kind="beam")


# ---------------------------------------------------------------------------
# Friendly ImportError paths
# ---------------------------------------------------------------------------
def test_plot_energy_missing_matplotlib(tmp_path, monkeypatch):
    def _boom():
        raise ImportError(post_viz._MPL_HINT)

    monkeypatch.setattr(post_viz, "_require_pyplot", _boom)
    with pytest.raises(ImportError, match="pip install matplotlib"):
        post_viz.plot_energy(make_binout_result(), str(tmp_path / "e.png"))


def test_results_to_vtu_missing_meshio(tmp_path, monkeypatch):
    def _boom():
        raise ImportError(post_viz._MESHIO_HINT)

    monkeypatch.setattr(post_viz, "_require_meshio", _boom)
    with pytest.raises(ImportError, match="pip install meshio"):
        post_viz.results_to_vtu(make_d3plot_result(), str(tmp_path / "x.vtu"))
