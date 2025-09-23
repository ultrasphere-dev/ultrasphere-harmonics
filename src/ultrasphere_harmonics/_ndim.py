import array_api_extra as xpx
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from jacobi_poly import binom


def homogeneous_ndim_eq(n: int | Array, *, c_ndim: int | Array) -> int | Array:
    r"""
    The dimension of the homogeneous polynomials of degree equals to n.

    $$
    M(n, d) = \dim(\mathcal{P}_n (\mathbb{R}^{d})) = \binom{n + d - 1}{d - 1}
    $$

    Parameters
    ----------
    n : int | Array
        The degree.
    c_ndim : int | Array
        The dimension of the Cartesian space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250 (8.7)

    Example
    -------
    >>> homogeneous_ndim_eq(3, c_ndim=3)
    array(10)

    """
    s_ndim = c_ndim - 1
    result = binom(n + s_ndim, s_ndim)
    xp = array_namespace(result)
    return xp.asarray(xp.astype(xp.round(result), int))


def homogeneous_ndim_le(n_end: int | Array, *, c_ndim: int | Array) -> int | Array:
    r"""
    The dimension of the homogeneous polynomials of degree below n_end.

    $$
    \dim(\mathcal{P}_{< n} (\mathbb{R}^{d}))
    = \sum_{k=0}^{n-1} \dim(\mathcal{P}_k (\mathbb{R}^{d}))
    = \begin{cases}
    0 & (n < 1) \\
    M(n, 0) & (n = 1) \\
    M(n - 1, d + 1) &(\text{otherwise}) \\
    \end{cases}
    $$

    where

    $$
    M(n, d) = \dim(\mathcal{P}_n (\mathbb{R}^{d}))
    = \binom{n + d - 1}{d - 1}
    $$

    Parameters
    ----------
    n_end : int | Array
        The degree.
    c_ndim : int | Array
        The dimension of the Cartesian space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250 (8.9)

    Example
    -------
    >>> homogeneous_ndim_le(3, c_ndim=3)
    array(10)

    """
    try:
        xp = array_namespace(n_end, c_ndim)
    except TypeError:
        xp = np
    n_end = xp.asarray(n_end)
    c_ndim = xp.asarray(c_ndim)
    return xpx.apply_where(
        n_end < 1,
        (n_end, c_ndim),
        lambda n_end, c_ndim: 0,
        lambda n_end, c_ndim: xpx.apply_where(
            n_end == 1,
            (n_end, c_ndim),
            lambda n_end, c_ndim: homogeneous_ndim_eq(0, c_ndim=c_ndim),
            lambda n_end, c_ndim: homogeneous_ndim_eq(n_end - 1, c_ndim=c_ndim + 1),
        ),
    )


def harm_n_ndim_eq(n: int | Array, *, c_ndim: int | Array) -> int | Array:
    r"""
    The dimension of the spherical harmonics of degree equals to n.

    $$
    N(n, d) = \dim(\mathcal{H}_n (\mathbb{R}^{d}))
    =
    \begin{cases}
    \begin{cases}
    1 & (n = 0) \\
    0 & (n \geq 1) \\
    \end{cases} & (d = 1) \\
    \begin{cases}
    1 & (n = 0) \\
    2 & (n \geq 1) \\
    \end{cases} & (d = 2) \\
    \frac{2 n + d - 2}{d - 2} \binom{n + d - 3}{d - 3} & (d \geq 3) \\
    \end{cases}
    $$

    Parameters
    ----------
    n : int | Array
        The degree.
    c_ndim : int | Array
        The dimension of the Cartesian space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251 (8.13)

    Example
    -------
    >>> harm_n_ndim_eq(3, c_ndim=3)
    array(7)

    """
    try:
        xp = array_namespace(n, c_ndim)
    except TypeError:
        xp = np
    n = xp.asarray(n)
    c_ndim = xp.asarray(c_ndim)
    return xpx.apply_where(
        c_ndim > 2,
        (n, c_ndim),
        lambda n, c_ndim: xp.astype(
            xp.round(
                (2 * n + c_ndim - 2) / (c_ndim - 2) * binom(n + c_ndim - 3, c_ndim - 3)
            ),
            int,
        ),
        lambda n, c_ndim: xpx.apply_where(
            c_ndim == 1,
            (n,),
            lambda n: xp.where(n <= 1, 1, 0),
            lambda n: xp.where(n == 0, 1, 2),
        ),
    )


def harm_n_ndim_le(n_end: int | Array, *, c_ndim: int | Array) -> int | Array:
    r"""
    The dimension of the spherical harmonics of degree below n_end.

    $$
    \dim(\mathcal{H}_{< n} (\mathbb{R}^{d}))
    = \sum_{k=0}^{n-1} \dim(\mathcal{H}_k (\mathbb{R}^{d}))
    = \begin{cases}
    0 & (n < 1) \\
    N(0, d) & (n = 1) \\
    N(n - 1, d + 1) &(\text{otherwise}) \\
    \end{cases}
    $$

    Parameters
    ----------
    n_end : int | Array
        The degree.
    c_ndim : int | Array
        The dimension of the Cartesian space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251 (8.12)

    Example
    -------
    >>> harm_n_ndim_le(3, c_ndim=3)
    array(9)

    """
    try:
        xp = array_namespace(n_end, c_ndim)
    except TypeError:
        xp = np
    n_end = xp.asarray(n_end)
    c_ndim = xp.asarray(c_ndim)
    return xpx.apply_where(
        n_end < 1,
        (n_end, c_ndim),
        lambda n_end, c_ndim: 0,
        lambda n_end, c_ndim: xpx.apply_where(
            n_end == 1,
            (n_end, c_ndim),
            lambda n_end, c_ndim: harm_n_ndim_eq(0, c_ndim=c_ndim),
            lambda n_end, c_ndim: harm_n_ndim_eq(n_end - 1, c_ndim=c_ndim + 1),
        ),
    )
