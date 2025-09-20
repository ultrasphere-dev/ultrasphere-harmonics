from collections.abc import Mapping
from typing import Literal, overload

import array_api_extra as xpx
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from array_api_negative_index import to_symmetric
from shift_nth_row_n_steps._torch_like import create_slice
from ultrasphere import (
    BranchingType,
    SphericalCoordinates,
    get_child,
)

from ._assume import assume_n_end_and_include_negative_m_from_harmonics


def _index_array_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    node: TSpherical,
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    expand_dims: bool = True,
    include_negative_m: bool = True,
) -> Array:
    """
    The index of the eigenfunction corresponding to the node.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    node : TSpherical
        The node of the spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    expand_dims : bool, optional
        Whether to expand dimensions, by default True
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
    xp : ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array
        The index.

    """
    branching_type = c.branching_types[node]
    if branching_type == BranchingType.A and include_negative_m:
        result = to_symmetric(xp.arange(0, n_end), asymmetric=True)
    elif (
        branching_type == BranchingType.B
        or branching_type == BranchingType.BP
        or (branching_type == BranchingType.A and not include_negative_m)
    ):
        result = xp.arange(0, n_end)
    elif branching_type == BranchingType.C:
        # result = xp.arange(0, (n_end + 1) // 2)
        result = xp.arange(0, n_end)
    if expand_dims:
        idx = c.s_nodes.index(node)
        result = result[create_slice(c.s_ndim, [(idx, slice(None))], default=None)]
    return result


@overload
def _index_array_harmonics_all[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    /,
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = ...,
    expand_dims: bool = ...,
    as_array: Literal[False],
    mask: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...
@overload
def _index_array_harmonics_all[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    /,
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = ...,
    expand_dims: Literal[True] = ...,
    as_array: Literal[True],
    mask: bool = ...,
) -> Array: ...


def _index_array_harmonics_all[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    /,
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = True,
    expand_dims: bool = True,
    as_array: bool,
    mask: bool = False,
) -> Array | Mapping[TSpherical, Array]:
    """
    The all indices of the eigenfunction corresponding to the spherical coordinates.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
    expand_dims : bool, optional
        Whether to expand dimensions, by default True
        Must be True if as_array is True.
    as_array : bool, optional
        Whether to return as an array, by default False
    mask : bool, optional
        Whether to fill invalid quantum numbers with NaN, by default False
        Must be False if as_array is False.
    xp : ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        If as_array is True, the indices of shape
        [c.s_ndim,
        len(index_array_harmonics(c, node1)),
        ...,
        len(index_array_harmonics(c, node(c.s_ndim)))].
        If as_array is False, the dictionary of indices.

    Notes
    -----
        To check the indices where all quantum numbers match,
        `(numbers1 == numbers2).all(axis=0)`
        can be used.

    Raises
    ------
    ValueError
        If expand_dims is False and as_array is True.
        If mask is True and as_array is False.

    """
    if not expand_dims and as_array:
        raise ValueError("expand_dims must be True if as_array is True.")
    if mask and not as_array:
        raise ValueError("mask must be False if as_array is False.")
    index_arrays = {
        node: _index_array_harmonics(
            c,
            node,
            xp=xp,
            n_end=n_end,
            expand_dims=expand_dims,
            include_negative_m=include_negative_m,
        )
        for node in c.s_nodes
    }
    if as_array:
        result = xp.stack(
            xp.broadcast_arrays(*[index_arrays[node] for node in c.s_nodes]),
            axis=0,
        )
        if mask:
            result[
                :,
                ~flatten_mask_harmonics(
                    c, n_end=n_end, xp=xp, include_negative_m=include_negative_m
                ),
            ] = xp.nan
        return result
    return index_arrays


def flatten_mask_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    /,
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = True,
) -> Array:
    """
    Create a mask representing the valid combinations of the quantum numbers.

    Can be used to flatten the harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
    nodes : Iterable[TSpherical] | None, optional
        The nodes to consider, by default None
        If None, all nodes are considered.
    xp : ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array
        The mask.

    """
    index_arrays: Mapping[TSpherical, Array] = _index_array_harmonics_all(
        c,
        n_end=n_end,
        include_negative_m=include_negative_m,
        as_array=False,
        expand_dims=True,
        xp=xp,
    )
    mask = xp.ones((1,) * c.s_ndim, dtype=bool)
    for node, branching_type in c.branching_types.items():
        if branching_type == BranchingType.B:
            mask = mask & (
                xp.abs(index_arrays[get_child(c.G, node, "sin")]) <= index_arrays[node]
            )
        if branching_type == BranchingType.BP:
            mask = mask & (
                xp.abs(index_arrays[get_child(c.G, node, "cos")]) <= index_arrays[node]
            )
        if branching_type == BranchingType.C:
            value = (
                index_arrays[node]
                - xp.abs(index_arrays[get_child(c.G, node, "sin")])
                - xp.abs(index_arrays[get_child(c.G, node, "cos")])
            )
            mask = mask & (value % 2 == 0) & (value >= 0)

    shape = xpx.broadcast_shapes(
        *[index_array.shape for index_array in index_arrays.values()]
    )
    mask = xp.broadcast_to(mask, shape)
    return mask


def flatten_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Array,
    n_end: int | None = None,
    include_negative_m: bool | None = None,
    axis_end: int = -1,
) -> Array:
    """
    Flatten the harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    harmonics : Array
        The (unflattend) harmonics.
    n_end : int | None, optional
        The maximum degree of the harmonic, by default None
        If None, assume from the shape of harmonics.
    include_negative_m : bool | None, optional
        Whether to include negative m values, by default None
        If None, assume from the shape of harmonics.
    axis_end : int, optional
        The axis to flatten, by default -1
        Must be negative.

    Returns
    -------
    Array
        The flattened harmonics of shape (..., n_harmonics).

    """
    if axis_end >= 0:
        raise ValueError("axis_end must be negative.")
    xp = array_namespace(harmonics)
    if n_end is None or include_negative_m is None:
        n_end, include_negative_m = assume_n_end_and_include_negative_m_from_harmonics(
            c,
            harmonics.shape if axis_end == -1 else harmonics.shape[: axis_end + 1],
            flatten=False,
        )
    mask = flatten_mask_harmonics(
        c, n_end=n_end, xp=xp, include_negative_m=include_negative_m
    )
    shape = xpx.broadcast_shapes(harmonics.shape, mask.shape + (1,) * (-axis_end - 1))
    harmonics = xp.broadcast_to(harmonics, shape)
    return harmonics[(..., mask) + (slice(None),) * (-axis_end - 1)]


def unflatten_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Array,
    *,
    include_negative_m: bool = True,
) -> Array:
    """
    Unflatten the harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    harmonics : Array
        The flattened harmonics.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True

    Returns
    -------
    Array
        The unflattened harmonics of shape (..., n_1, n_2, ..., n_(c.s_ndim)).

    """
    xp = array_namespace(harmonics)
    n_end, _ = assume_n_end_and_include_negative_m_from_harmonics(
        c, harmonics, flatten=True
    )
    mask = flatten_mask_harmonics(
        c, n_end=n_end, xp=xp, include_negative_m=include_negative_m
    )
    shape = (*harmonics.shape[:-1], *mask.shape)
    result = xp.zeros(shape, dtype=harmonics.dtype, device=harmonics.device)
    result[..., mask] = harmonics
    return result


def index_array_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    node: TSpherical,
    /,
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    expand_dims: bool = True,
    include_negative_m: bool = True,
    flatten: bool = False,
) -> Array:
    """
    The index of the eigenfunction corresponding to the node.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    node : TSpherical
        The node of the spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    expand_dims : bool, optional
        Whether to expand dimensions, by default True
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
        If None, True iff concat is True.
    flatten : bool, optional
        Whether to flatten the result, by default False
    xp : ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array
        The index.

    """
    if flatten and not expand_dims:
        raise ValueError("expand_dims must be True if flatten is True.")
    index_array = _index_array_harmonics(
        c,
        node,
        n_end=n_end,
        xp=xp,
        expand_dims=expand_dims,
        include_negative_m=include_negative_m,
    )
    if flatten:
        return flatten_harmonics(
            c, index_array, n_end=n_end, include_negative_m=include_negative_m
        )
    return index_array


@overload
def index_array_harmonics_all[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = ...,
    expand_dims: bool,
    as_array: Literal[False],
    mask: Literal[False] = ...,
    flatten: bool | None = ...,
) -> Mapping[TSpherical, Array]: ...
@overload
def index_array_harmonics_all[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = ...,
    expand_dims: Literal[True],
    as_array: Literal[True],
    mask: bool = ...,
    flatten: bool | None = ...,
) -> Array: ...


def index_array_harmonics_all[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end: int,
    xp: ArrayNamespaceFull,
    include_negative_m: bool = True,
    expand_dims: bool,
    as_array: bool,
    mask: bool = False,
    flatten: bool | None = None,
) -> Array | Mapping[TSpherical, Array]:
    """
    The all indices of the eigenfunction corresponding to the spherical coordinates.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
    expand_dims : bool, optional
        Whether to expand dimensions, by default True
        Must be True if as_array is True.
    as_array : bool, optional
        Whether to return as an array, by default False
    mask : bool, optional
        Whether to fill invalid quantum numbers with NaN, by default False
        Must be False if as_array is False.
    flatten : bool, optional
        Whether to flatten the result, by default None
        If None, True iff as_array is True.
    xp : ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        If as_array is True, the indices of shape
        [c.s_ndim,
        len(index_array_harmonics(c, node1)),
        ...,
        len(index_array_harmonics(c, node(c.s_ndim)))].
        If as_array is False, the dictionary of indices.

    Notes
    -----
        To check the indices where all quantum numbers match,
        `(numbers1 == numbers2).all(axis=0)`
        can be used.

    Raises
    ------
    ValueError
        If expand_dims is False and as_array is True.
        If mask is True and as_array is False.

    """
    if flatten is None:
        flatten = as_array
    if flatten and not expand_dims:
        raise ValueError("expand_dims must be True if flatten is True.")
    index_arrays = _index_array_harmonics_all(  # type: ignore[call-overload]
        c,
        n_end=n_end,
        xp=xp,
        include_negative_m=include_negative_m,
        as_array=as_array,
        expand_dims=expand_dims,
        mask=mask,
    )
    if flatten:
        if as_array:
            return flatten_harmonics(c, index_arrays)
        return {
            node: flatten_harmonics(c, index_array)
            for node, index_array in index_arrays.items()
        }
    return index_arrays
