from collections.abc import Mapping
from typing import Literal, overload

from array_api._2024_12 import Array
from ultrasphere import BranchingType, SphericalCoordinates, get_child
from ultrasphere._coordinates import TEuclidean, TSpherical

from ._concat import concat_harmonics
from ._eigenfunction import Phase, type_a, type_b, type_bdash, type_c
from ._expand_dim import expand_dims_harmonics
from ._flatten import flatten_harmonics


def _harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    n_end: int,
    *,
    phase: Phase,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
) -> Mapping[TSpherical, Array] | Array:
    """
    Calculate the spherical harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    spherical : Mapping[TSpherical, Array]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
        If True, the m values are [0, 1, ..., n_end-1, -n_end+1, ..., -1],
        and starts from 0, not -n_end+1.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False

    Returns
    -------
    Array
        The spherical harmonics.

    """
    result = {}
    for node in c.s_nodes:
        value = spherical[node]
        if node == "r":
            continue
        if node not in c.s_nodes:
            raise ValueError(f"Key {node} is not in c.s_nodes {c.s_nodes}.")
        if c.branching_types[node] == BranchingType.A:
            result[node] = type_a(
                value,
                n_end=n_end,
                phase=phase,
                include_negative_m=include_negative_m,
            )
        elif c.branching_types[node] == BranchingType.B:
            result[node] = type_b(
                value,
                n_end=n_end,
                s_beta=c.S[get_child(c.G, node, "sin")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_beta_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "sin")] == BranchingType.A,
            )
        elif c.branching_types[node] == BranchingType.BP:
            result[node] = type_bdash(
                value,
                n_end=n_end,
                s_alpha=c.S[get_child(c.G, node, "cos")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_alpha_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "cos")] == BranchingType.A,
            )
        elif c.branching_types[node] == BranchingType.C:
            result[node] = type_c(
                value,
                n_end=n_end,
                s_alpha=c.S[get_child(c.G, node, "cos")],
                s_beta=c.S[get_child(c.G, node, "sin")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_alpha_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "cos")] == BranchingType.A,
                is_beta_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "sin")] == BranchingType.A,
            )
        else:
            raise ValueError(f"Invalid branching type {c.branching_types[node]}.")
    return result


@overload
def harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    /,
    *,
    n_end: int,
    phase: Phase,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[True] = ...,
) -> Array: ...


@overload
def harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    /,
    *,
    n_end: int,
    phase: Phase,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...


def harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    /,
    *,
    n_end: int,
    phase: Phase,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: bool = True,
) -> Mapping[TSpherical, Array] | Array:
    """
    Calculate the spherical harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    spherical : Mapping[TSpherical, Array]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
        If True, the m values are [0, 1, ..., n_end-1, -n_end+1, ..., -1],
        and starts from 0, not -n_end+1.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
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
    Array
        The spherical harmonics.

    """
    if flatten is None:
        flatten = concat
    if index_with_surrogate_quantum_number and expand_dims:
        raise ValueError(
            "expand_dims must be False if index_with_surrogate_quantum_number is True."
        )
    if concat and not expand_dims:
        raise ValueError("expand_dims must be True if concat is True.")
    if flatten and not expand_dims:
        raise ValueError("expand_dims must be True if flatten is True.")

    result = _harmonics(
        c,
        spherical,
        n_end,
        phase=phase,
        include_negative_m=include_negative_m,
        index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
    )
    if expand_dims:
        result = expand_dims_harmonics(c, result)
    if concat:
        result = concat_harmonics(c, result)
    if flatten:
        if concat:
            result = flatten_harmonics(c, result)
        else:
            result = {k: flatten_harmonics(c, v) for k, v in result.items()}
    return result
