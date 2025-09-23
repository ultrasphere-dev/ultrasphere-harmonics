from collections.abc import Mapping
from typing import Literal, overload

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ultrasphere import SphericalCoordinates
from ultrasphere.special import szv

from ultrasphere_harmonics._core._eigenfunction import Phase

from ._core import harmonics
from ._core._flatten import flatten_harmonics, index_array_harmonics


@overload
def harmonics_regular_singular_component[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...


@overload
def harmonics_regular_singular_component[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[True] = ...,
) -> Array: ...


def harmonics_regular_singular_component[TCartesian, TSpherical](  # type: ignore[misc]
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    /,
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: bool = True,
) -> Array | Mapping[TSpherical, Array]:
    r"""
    Regular or singular COMPONENT of harmonics (does not include $Y_n$).

    $$
    R_n (x) &:= j_n \left(\|x\|\right) Y_n^m \left(\frac{x}{\|x\|}\right) \\
    S_n (x) &:= h_n^{(1)} \left(\|x\|\right) Y_n^m \left(\frac{x}{\|x\|}\right)
    $$

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    spherical : Mapping[TSpherical | Literal['r'],
        Array] | Mapping[Literal['r'],
        Array]
        The spherical coordinates.
    k : Array
        The wavenumber. Must be positive.
    n_end : int
        The maximum degree of the harmonic.
    type : Literal['regular', 'singular', 'j', 'y', 'h1', 'h2']
        The type of the spherical Bessel/Hankel function.
    derivative : bool, optional
        Whether to return the directional derivative to r,
        in other words whether to return the derivative with respect to r,
        by default False
    expand_dims : bool, optional
        Whether to expand dimensions so that
        all values of the result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same
    flatten : bool, optional
        Whether to flatten the result, by default None
        If None, True iff concat is True.
    concat : bool, optional
        Whether to concatenate the results, by default True

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The regular or singular harmonics.

    Raises
    ------
    ValueError
        If the wavenumber is not positive.

    Example
    -------
    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> c = create_spherical()
    >>> Y = harmonics(
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=0,
    ... )
    >>> j = harmonics_regular_singular_component(
    ...     c,
    ...     {"r": np.asarray(1.0)},
    ...     n_end=2,
    ...     k=np.asarray(1.0),
    ...     type="regular",
    ... )
    >>> R = j * Y
    >>> np.round(R, 2)
    array([0.24+0.j  , 0.13+0.j  , 0.03+0.04j, 0.03-0.04j])

    """
    if flatten is None:
        flatten = concat
    if concat and not expand_dims:
        raise ValueError("expand_dims must be True if concat is True.")
    if flatten and not expand_dims:
        raise ValueError("expand_dims must be True if flatten is True.")
    xp = array_namespace(k, spherical["r"])
    extra_dims = spherical["r"].ndim
    n = index_array_harmonics(
        c,
        c.root,
        n_end=n_end,
        include_negative_m=True,
        xp=xp,
        expand_dims=expand_dims,
        flatten=False,
    )[(None,) * extra_dims + (...,)]

    kr = k * spherical["r"]
    kr = kr[(...,) + (None,) * c.s_ndim]

    if type == "regular":
        type = "j"
    elif type == "singular":
        type = "h1"
    val = szv(n, c.c_ndim, kr, type=type, derivative=derivative)
    # val = xp.nan_to_num(val, nan=0)
    if flatten:
        val = flatten_harmonics(c, val, n_end=n_end, include_negative_m=True)
    if not concat:
        return {"r": val}
    return val


@overload
def harmonics_regular_singular[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    /,
    *,
    n_end: int,
    k: Array,
    phase: Phase,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...


@overload
def harmonics_regular_singular[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    /,
    *,
    n_end: int,
    k: Array,
    phase: Phase,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[True] = ...,
) -> Array: ...


def harmonics_regular_singular[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    /,
    *,
    n_end: int,
    k: Array,
    phase: Phase,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: bool = True,
) -> Array | Mapping[TSpherical, Array]:
    r"""
    Regular or singular harmonics.

    $$
    R_n (x) &:= j_n \left(\|x\|\right) Y_n^m \left(\frac{x}{\|x\|}\right) \\
    S_n (x) &:= h_n^{(1)} \left(\|x\|\right) Y_n^m \left(\frac{x}{\|x\|}\right)
    $$

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    spherical : Mapping[TSpherical | Literal['r'],
        Array] | Mapping[Literal['r'],
        Array]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    k : Array
        The wavenumber. Must be positive.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    type : Literal['regular', 'singular', 'j', 'y', 'h1', 'h2']
        The type of the spherical Bessel/Hankel function.
    derivative : bool, optional
        Whether to return the directional derivative to r,
        in other words whether to return the derivative with respect to r,
        by default False
    expand_dims : bool, optional
        Whether to expand dimensions so that
        all values of the result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same
    flatten : bool, optional
        Whether to flatten the result, by default None
        If None, True iff concat is True.
    concat : bool, optional
        Whether to concatenate the results, by default True

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The regular or singular harmonics.

    Raises
    ------
    ValueError
        If the wavenumber is not positive.

    Example
    -------
    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> c = create_spherical()
    >>> R = harmonics_regular_singular(
    ...     c,
    ...     {"r": np.asarray(1.0), "theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=0,
    ...     k=np.asarray(1.0),
    ...     type="regular",
    ... )
    >>> np.round(R, 2)
    array([0.24+0.j  , 0.13+0.j  , 0.03+0.04j, 0.03-0.04j])

    """
    return harmonics(  # type: ignore[call-overload]
        c,
        spherical,
        n_end=n_end,
        phase=phase,
        include_negative_m=True,
        index_with_surrogate_quantum_number=False,
        expand_dims=expand_dims,
        flatten=flatten,
        concat=concat,
    ) * harmonics_regular_singular_component(  # type: ignore[call-overload]
        c,
        spherical,
        n_end=n_end,
        k=k,
        type=type,
        derivative=derivative,
        expand_dims=expand_dims,
        flatten=flatten,
        concat=concat,
    )
