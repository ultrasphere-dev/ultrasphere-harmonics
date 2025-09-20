import array_api_extra as xpx
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from jacobi_poly import binom


def homogeneous_ndim_eq(n: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the homogeneous polynomials of degree equals to n.

    Parameters
    ----------
    n : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250 (8.7)

    """
    s_ndim = e_ndim - 1
    result = binom(n + s_ndim, s_ndim)
    xp = array_namespace(result)
    return xp.astype(xp.round(result), int)


def homogeneous_ndim_le(n_end: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the homogeneous polynomials of degree below n_end.

    Parameters
    ----------
    n_end : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250 (8.9)

    """
    try:
        xp = array_namespace(n_end, e_ndim)
    except TypeError:
        xp = np
    return xp.apply_where(
        n_end < 1,
        (n_end, e_ndim),
        lambda n_end, e_ndim: 0,
        lambda n_end, e_ndim: xpx.apply_where(
            n_end == 1,
            (n_end, e_ndim),
            lambda n_end, e_ndim: homogeneous_ndim_eq(0, e_ndim=e_ndim),
            lambda n_end, e_ndim: homogeneous_ndim_eq(n_end - 1, e_ndim=e_ndim + 1),
        ),
    )


def harm_n_ndim_eq(n: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the spherical harmonics of degree below n_end.

    Parameters
    ----------
    n : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251 (8.13)

    """
    try:
        xp = array_namespace(n, e_ndim)
    except TypeError:
        xp = np
    n = xp.asarray(n)
    e_ndim = xp.asarray(e_ndim)
    return xpx.apply_where(
        e_ndim > 2,
        (n, e_ndim),
        lambda n, e_ndim: xp.astype(
            xp.round(
                (2 * n + e_ndim - 2) / (e_ndim - 2) * binom(n + e_ndim - 3, e_ndim - 3)
            ),
            int,
        ),
        lambda n, e_ndim: xpx.apply_where(
            e_ndim == 1,
            (n,),
            lambda n: xp.where(n <= 1, 1, 0),
            lambda n: xp.where(n == 0, 1, 2),
        ),
    )


def harm_n_ndim_le(n_end: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the spherical harmonics of degree below n_end.

    Parameters
    ----------
    n_end : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251 (8.12)

    """
    try:
        xp = array_namespace(n_end, e_ndim)
    except TypeError:
        xp = np
    n_end = xp.asarray(n_end)
    e_ndim = xp.asarray(e_ndim)
    return xpx.apply_where(
        n_end < 1,
        (n_end, e_ndim),
        lambda n_end, e_ndim: 0,
        lambda n_end, e_ndim: xpx.apply_where(
            n_end == 1,
            (n_end, e_ndim),
            lambda n_end, e_ndim: harm_n_ndim_eq(0, e_ndim=e_ndim),
            lambda n_end, e_ndim: harm_n_ndim_eq(n_end - 1, e_ndim=e_ndim + 1),
        ),
    )
