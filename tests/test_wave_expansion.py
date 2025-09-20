import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from jacobi_poly import gegenbauer_all
from ultrasphere import SphericalCoordinates, create_spherical, create_standard
from ultrasphere.special import sjv

from ultrasphere_harmonics._wave_expansion import plane_wave_expansion_coef


@pytest.mark.parametrize(
    "c",
    [
        (create_spherical()),
        (create_standard(3)),
    ],
)
@pytest.mark.parametrize("n_end", [30])
def test_plane_wave_decomposition[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    xp: ArrayNamespaceFull,
) -> None:
    shape = (5,)
    r = xp.random.random_uniform(low=0, high=2, shape=shape)
    gamma = xp.random.random_uniform(low=0, high=xp.pi, shape=shape)
    k = xp.ones_like(r)
    expected = xp.exp(1j * k * r * xp.cos(gamma))
    n = xp.arange(n_end)[(None,) * len(shape) + (slice(None),)]
    coef = plane_wave_expansion_coef(n, e_ndim=c.e_ndim)
    actual = xp.sum(
        coef
        * sjv(
            n,
            xp.asarray(c.e_ndim),
            k[..., None] * r[..., None],
        )
        # * legendre(xp.cos(gamma), ndim=c.e_ndim, n_end=n_end)
        * gegenbauer_all(
            xp.cos(gamma), alpha=xp.asarray((c.e_ndim - 2) / 2), n_end=n_end
        ),
        axis=-1,
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))
