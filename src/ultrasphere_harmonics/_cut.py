from collections.abc import Mapping
from typing import overload

from array_api._2024_12 import Array
from ultrasphere import SphericalCoordinates

from ._ndim import harm_n_ndim_le


@overload
def expand_cut[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    expansion: Mapping[TSpherical, Array],
    n_end: int,
) -> Mapping[TSpherical, Array]: ...


@overload
def expand_cut[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    expansion: Array,
    n_end: int,
) -> Array: ...


def expand_cut[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    expansion: Mapping[TSpherical, Array] | Array,
    n_end: int,
) -> Mapping[TSpherical, Array] | Array:
    """
    Cut the expansion coefficients to the maximum degree.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    expansion : Mapping[TSpherical, Array] | Array
        The expansion coefficients.
        If mapping, assume that the expansion is not expanded.
    n_end : int
        The maximum degree to cut.

    Returns
    -------
    Mapping[TSpherical, Array] | Array
        The cut expansion coefficients.

    Example
    -------
    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> from ultrasphere_harmonics import harmonics
    >>> c = create_spherical()
    >>> harm = harmonics(
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=3,
    ...     phase=0
    ... )
    >>> harm.shape
    (9,)

    >>> harm_cut = expand_cut(c, harm, n_end=2)
    >>> harm_cut.shape
    (4,)

    """
    if isinstance(expansion, Mapping):
        return {
            k: v[..., : int(harm_n_ndim_le(n_end, c_ndim=c.c_ndim))]
            for k, v in expansion.items()
        }
    return expansion[..., : int(harm_n_ndim_le(n_end, c_ndim=c.c_ndim))]
