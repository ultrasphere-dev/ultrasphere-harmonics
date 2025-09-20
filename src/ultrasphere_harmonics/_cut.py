from collections.abc import Mapping
from typing import overload

from array_api._2024_12 import Array
from ultrasphere import SphericalCoordinates

from ._ndim import harm_n_ndim_le


@overload
def expand_cut[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array],
    n_end: int,
) -> Mapping[TSpherical, Array]: ...


@overload
def expand_cut[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Array,
    n_end: int,
) -> Array: ...


def expand_cut[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array] | Array,
    n_end: int,
) -> Mapping[TSpherical, Array] | Array:
    """
    Cut the expansion coefficients to the maximum degree.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
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

    """
    if isinstance(expansion, Mapping):
        return {
            k: v[..., : int(harm_n_ndim_le(n_end, e_ndim=c.e_ndim))]
            for k, v in expansion.items()
        }
    return expansion[..., : int(harm_n_ndim_le(n_end, e_ndim=c.e_ndim))]
