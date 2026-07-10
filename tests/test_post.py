"""Tests for :mod:`post`, the lasso-python d3plot/binout bridge.

We do not have a real solver output file (generating a valid d3plot/binout is
impractical in CI), so the wrapper *structure* is checked by monkeypatching
``post``'s lasso loaders with fake D3plot/Binout objects that return synthetic
arrays. The error-handling paths are checked against a missing path and a
random-bytes file. All tests are hermetic: no network, no real solver.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("lasso")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from k_mesher import post  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeD3plot:
    """Stand-in for ``lasso.dyna.D3plot`` exposing a synthetic ``.arrays``."""

    def __init__(self, filepath):
        self.filepath = filepath
        n_nodes = 4
        # displacement grows with node index and state so maxima are predictable
        disp = np.zeros((2, n_nodes, 3))
        disp[1, 3] = [3.0, 4.0, 0.0]  # magnitude 5.0 at last state, node 4
        disp[1, 1] = [0.0, 0.0, 1.0]
        self.arrays = {
            "node_ids": np.array([10, 20, 30, 40]),
            "node_coordinates": np.array(
                [[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
            ),
            "node_displacement": disp,
            "timesteps": np.array([0.0, 1.0]),
            "element_shell_stress": np.ones((2, 5, 1, 6)),
        }


class FakeBinout:
    """Stand-in for ``lasso.dyna.Binout`` with a tiny read() dispatch."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._db = {
            (): ["glstat", "matsum", "rcforc"],
            ("glstat",): ["time", "total_energy", "kinetic_energy", "internal_energy"],
            ("glstat", "time"): np.array([0.0, 1.0, 2.0]),
            ("glstat", "total_energy"): np.array([100.0, 90.0, 80.0]),
            ("glstat", "kinetic_energy"): np.array([0.0, 40.0, 10.0]),
            ("glstat", "internal_energy"): np.array([0.0, 50.0, 70.0]),
            ("matsum",): ["internal_energy", "kinetic_energy"],
            # (states=3, parts=2)
            ("matsum", "internal_energy"): np.array(
                [[0.0, 0.0], [30.0, 20.0], [45.0, 25.0]]
            ),
            ("rcforc",): ["x_force"],
            ("rcforc", "x_force"): np.array([0.0, -12.0, 7.0]),
        }

    def read(self, *args):
        return self._db[tuple(args)]


@pytest.fixture
def d3plot_result(monkeypatch, tmp_path):
    monkeypatch.setattr(post, "_load_d3plot_class", lambda: FakeD3plot)
    fake_file = tmp_path / "d3plot"
    fake_file.write_bytes(b"placeholder")  # so os.path.exists passes
    return post.read_d3plot(str(fake_file))


@pytest.fixture
def binout_result(monkeypatch, tmp_path):
    monkeypatch.setattr(post, "_load_binout_class", lambda: FakeBinout)
    fake_file = tmp_path / "binout"
    fake_file.write_bytes(b"placeholder")
    return post.read_binout(str(fake_file))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_read_d3plot_missing_path():
    with pytest.raises(FileNotFoundError, match="d3plot not found"):
        post.read_d3plot("/no/such/d3plot")


def test_read_binout_missing_path():
    with pytest.raises(FileNotFoundError, match="binout not found"):
        post.read_binout("/no/such/binout")


def test_read_d3plot_empty_path():
    with pytest.raises(ValueError, match="empty path"):
        post.read_d3plot("")


def test_read_d3plot_bogus_file(tmp_path):
    bad = tmp_path / "junk.d3plot"
    bad.write_bytes(os.urandom(512))
    with pytest.raises(ValueError, match="failed to read"):
        post.read_d3plot(str(bad))


def test_read_binout_bogus_file(tmp_path):
    bad = tmp_path / "junk.binout"
    bad.write_bytes(os.urandom(512))
    with pytest.raises(ValueError, match="failed to read"):
        post.read_binout(str(bad))


# ---------------------------------------------------------------------------
# d3plot wrapper structure (via fake)
# ---------------------------------------------------------------------------
def test_d3plot_accessors(d3plot_result):
    r = d3plot_result
    np.testing.assert_array_equal(r.node_ids, [10, 20, 30, 40])
    assert r.node_coordinates.shape == (4, 3)
    np.testing.assert_array_equal(r.times, [0.0, 1.0])
    assert r.node_displacement.shape == (2, 4, 3)
    assert r.shell_stress is not None
    assert r.solid_stress is None  # not present in the fake
    assert r.n_states == 2


def test_max_displacement(d3plot_result):
    # node 4 at last state is (3,4,0) -> magnitude 5.0
    assert post.max_displacement(d3plot_result) == pytest.approx(5.0)
    # at the first state everything is zero
    assert post.max_displacement(d3plot_result, state=0) == pytest.approx(0.0)


def test_map_node_results_to_set_displacement(d3plot_result):
    # 1-based ids 2 and 4 -> rows 1 and 3
    out = post.map_node_results_to_set(d3plot_result, [2, 4])
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 1.0])  # row 1 last state
    np.testing.assert_array_equal(out[1], [3.0, 4.0, 0.0])  # row 3 last state


def test_map_node_results_to_set_coordinates(d3plot_result):
    out = post.map_node_results_to_set(d3plot_result, [1, 3], quantity="coordinates")
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 0.0])  # row 0
    np.testing.assert_array_equal(out[1], [2.0, 0.0, 0.0])  # row 2


def test_map_node_results_id_base_zero(d3plot_result):
    # id_base=0 means the ids are already 0-based row indices
    out = post.map_node_results_to_set(d3plot_result, [1, 3], id_base=0)
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 1.0])  # row 1
    np.testing.assert_array_equal(out[1], [3.0, 4.0, 0.0])  # row 3


def test_map_node_results_out_of_range(d3plot_result):
    with pytest.raises(ValueError, match="beyond"):
        post.map_node_results_to_set(d3plot_result, [99])


def test_map_node_results_unknown_quantity(d3plot_result):
    with pytest.raises(ValueError, match="unknown quantity"):
        post.map_node_results_to_set(d3plot_result, [1], quantity="temperature")


# ---------------------------------------------------------------------------
# binout wrapper structure (via fake)
# ---------------------------------------------------------------------------
def test_binout_branches_and_get(binout_result):
    assert binout_result.branches() == ["glstat", "matsum", "rcforc"]
    np.testing.assert_array_equal(
        binout_result.get("glstat", "kinetic_energy"), [0.0, 40.0, 10.0]
    )
    assert binout_result.get("matsum") == ["internal_energy", "kinetic_energy"]


def test_binout_glstat_energy(binout_result):
    energy = binout_result.glstat_energy()
    np.testing.assert_array_equal(energy["internal_energy"], [0.0, 50.0, 70.0])
    np.testing.assert_array_equal(energy["time"], [0.0, 1.0, 2.0])


def test_peak_part_energy_all(binout_result):
    # matsum internal_energy max over all parts/states is 45.0
    assert post.peak_part_energy(binout_result) == pytest.approx(45.0)


def test_peak_part_energy_single_part(binout_result):
    # part column 1 peak is 25.0
    assert post.peak_part_energy(binout_result, part=1) == pytest.approx(25.0)


def test_rcforc_force(binout_result):
    # abs peak is 12.0
    force = binout_result.rcforc_force()
    assert float(np.max(np.abs(force))) == pytest.approx(12.0)
