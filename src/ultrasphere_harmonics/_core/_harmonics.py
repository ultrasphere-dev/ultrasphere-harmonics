from collections.abc import Mapping
from typing import Literal, overload

from array_api._2024_12 import Array
from ultrasphere import BranchingType, SphericalCoordinates, get_child
from ultrasphere._coordinates import TCartesian, TSpherical

from ._concat import concat_harmonics
from ._eigenfunction import Phase, type_a, type_b, type_bdash, type_c
from ._expand_dim import expand_dims_harmonics
from ._flatten import flatten_harmonics


def _harmonics(
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c : SphericalCoordinates[TSpherical, TCartesian]
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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c : SphericalCoordinates[TSpherical, TCartesian]
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
        - if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        - if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

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

    Example
    -------

    Basic usage
    ^^^^^^^^^^^

    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> c = create_spherical()
    >>> harm = harmonics( # flattened output
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=0
    ... )
    >>> np.round(harm, 2)
    array([0.28+0.j  , 0.43+0.j  , 0.09+0.14j, 0.09-0.14j])

    Unflattened output
    ^^^^^^^^^^^^^^^^^^

    >>> harm = harmonics(
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=0,
    ...     flatten=False,
    ... )
    >>> np.round(harm, 2)
    array([[0.28+0.j  , 0.  +0.j  , 0.  +0.j  ],
           [0.43+0.j  , 0.09+0.14j, 0.09-0.14j]])

    Unflattened mapping output
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> harm = harmonics(
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=0,
    ...     concat=False
    ... )
    >>> {k: np.round(harm[k], 2) for k in c.s_nodes}
    {'theta': array([[0.71, 0.  , 0.  ],
           [1.07, 0.42, 0.42]]), 'phi': array([[0.4 +0.j  , 0.22+0.34j, 0.22-0.34j]])}

    Using different phase convention to match scipy.special.sph_harm
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Call `harmonics` with `phase=3` (negative legendre + Condon-Shortley phase)

    >>> harm = harmonics(
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=3 # negative legendre + Condon-Shortley phase
    ... )
    >>> np.round(harm, 2)
    array([ 0.28+0.j  ,  0.43+0.j  , -0.09-0.14j,  0.09-0.14j])

    Call `scipy.special.sph_harm_y_all`

    >>> from scipy.special import sph_harm_y_all
    >>> harm_scipy = sph_harm_y_all(1, 1, np.asarray(0.5), np.asarray(1.0))
    >>> np.round(harm_scipy, 2)
    array([[ 0.28+0.j  ,  0.  +0.j  ,  0.  +0.j  ],
           [ 0.43+0.j  , -0.09-0.14j,  0.09-0.14j]])

    Flatten the scipy output to compare with `harmonics`

    >>> from ultrasphere_harmonics import flatten_harmonics
    >>> harm_scipy_flatten = flatten_harmonics(c, harm_scipy)
    >>> np.round(harm_scipy_flatten, 2)
    array([ 0.28+0.j  ,  0.43+0.j  , -0.09-0.14j,  0.09-0.14j])

    Check if they are close

    >>> import array_api_extra as xpx
    >>> np.all(xpx.isclose(harm, harm_scipy_flatten))
    np.True_

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
