import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from jacobi_poly import lgamma


def plane_wave_expansion_coef(n: int | Array, *, e_ndim: int | Array) -> Array:
    """
    The coefficients of the plane wave expansion.

    Parameters
    ----------
    n : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    Array
        The coefficients for regular elementary wave solutions
        of degree n.

    """
    xp = array_namespace(n, e_ndim)
    return (
        1j**n
        * (2 * n + e_ndim - 2)
        / (e_ndim - 2)
        * xp.exp(lgamma(e_ndim / 2.0) + np.log(2) * ((e_ndim - 1) / 2))
        / xp.sqrt(xp.pi)
    )
