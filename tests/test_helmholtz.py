import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import to_device
from scipy.special import sph_harm_y_all, spherical_jn
from ultrasphere import create_spherical

from ultrasphere_harmonics._core._eigenfunction import Phase
from ultrasphere_harmonics._core._flatten import flatten_harmonics
from ultrasphere_harmonics._helmholtz import harmonics_regular_singular


@pytest.mark.parametrize("n_end", [1, 2, 12])  # scipy does not support n_end == 0
@pytest.mark.parametrize("k", [1, 2])
def test_match_scipy(n_end: int, xp: ArrayNamespaceFull, k: Array) -> None:
    c = create_spherical()
    shape = ()
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.c_ndim, *shape))
    x_spherical = c.from_cartesian(x)
    expected = sph_harm_y_all(
        n_end - 1,
        n_end - 1,
        to_device(x_spherical["theta"], "cpu"),
        to_device(x_spherical["phi"], "cpu"),
    ) * spherical_jn(
        np.arange(n_end)[:, None, ...],
        k * to_device(x_spherical["r"][None, None, ...], "cpu"),
    )
    expected = xp.moveaxis(xp.asarray(expected), (0, 1), (-2, -1))
    expected = flatten_harmonics(
        c,
        expected,
    )
    actual = harmonics_regular_singular(
        c,
        x_spherical,
        n_end=n_end,
        phase=Phase.NEGATIVE_LEGENDRE | Phase.CONDON_SHORTLEY,
        concat=True,
        expand_dims=True,
        flatten=True,
        type="regular",
        k=k,
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))
