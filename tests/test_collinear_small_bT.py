import numpy as np

from lametlat.perturbative_matching.collinear_small_bT import (
    helicity_matching_kernel_rows_by_order,
    transversity_matching_kernel_rows,
    unpolarized_matching_kernel_rows_by_order,
)


def _grid():
    return np.array([0.25, 0.5, 0.75])


def test_unpolarized_nlo_smoke():
    x = _grid()
    kernel = unpolarized_matching_kernel_rows_by_order(x, x, range(len(x)), 0.1, orders=("NLO",))["NLO"]
    assert kernel.shape == (3, 3)
    assert np.all(np.isfinite(kernel))


def test_helicity_n3lo_smoke():
    x = _grid()
    kernel = helicity_matching_kernel_rows_by_order(x, x, range(len(x)), 0.1, orders=("N3LO",))["N3LO"]
    assert kernel.shape == (3, 3)
    assert np.all(np.isfinite(kernel))


def test_transversity_n3lo_smoke():
    x = _grid()
    kernel = transversity_matching_kernel_rows(x, x, range(len(x)), 0.1, order="N3LO")
    assert kernel.shape == (3, 3)
    assert np.all(np.isfinite(kernel))
