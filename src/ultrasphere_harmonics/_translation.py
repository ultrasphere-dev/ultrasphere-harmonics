from collections.abc import Mapping
from typing import Literal

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from gumerov_expansion_coefficients import translational_coefficients
from ultrasphere import SphericalCoordinates, get_child

from ultrasphere_harmonics._core._eigenfunction import Phase, minus_1_power

from ._core import concat_harmonics, expand_dims_harmonics
from ._core._flatten import (
    flatten_harmonics,
    index_array_harmonics,
)
from ._core._harmonics import harmonics
from ._expansion import (
    expand,
)
from ._helmholtz import harmonics_regular_singular


def _harmonics_translation_coef_plane_wave[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    cartesian: Mapping[TCartesian, Array],
    *,
    n_end: int,
    n_end_add: int,
    phase: Phase,
    k: Array,
) -> Array:
    r"""
    Translation coefficients between same type of elementary solutions.

    Returns $(R|R) = (S|S)$, where

    .. math::
        R(x + t) = \sum_n (R|R)_n(t) R(x) \\
        S(x + t) = \sum_n (S|S)_n(t) S(x)

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    cartesian : Mapping[TCartesian, Array]
        The translation vector in cartesian coordinates.
        Each array must have the same shape (...,).
    n_end : int
        The maximum degree of the harmonic.
    n_end_add : int
        The maximum degree of the harmonic to be summed over.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    k : Array
        The wavenumber.

    Returns
    -------
    Array
        The translation coefficients of shape (..., N, N).
        Axis -1 is to be summed over with the elementary solutions
        to get translated elementary solution
        which quantum number is axis -2 indices.

    """
    xp = array_namespace(*[cartesian[k] for k in c.c_nodes])
    _, k = xp.broadcast_arrays(cartesian[c.c_nodes[0]], k)
    n = index_array_harmonics(
        c, c.root, n_end=n_end, xp=xp, expand_dims=True, flatten=True
    )[:, None]
    ns = index_array_harmonics(
        c, c.root, n_end=n_end_add, xp=xp, expand_dims=True, flatten=True
    )[None, :]

    def to_expand(spherical: Mapping[TSpherical, Array]) -> Array:
        # returns [spherical1,...,sphericalN,user1,...,userM,harmn]
        # [spherical1,...,sphericalN,harmn]
        Y = harmonics(
            c,
            spherical,
            n_end=n_end,
            phase=phase,
            expand_dims=True,
            concat=True,
            flatten=True,
        )
        x = c.to_cartesian(spherical)
        ndim_user = cartesian[c.c_nodes[0]].ndim
        ndim_spherical = c.s_ndim
        ip = xp.sum(
            xp.stack(
                xp.broadcast_arrays(
                    *[
                        cartesian[i][
                            (None,) * ndim_spherical + (slice(None),) * ndim_user
                        ]
                        * x[i][(slice(None),) * ndim_spherical + (None,) * ndim_user]
                        for i in c.c_nodes
                    ]
                ),
                axis=0,
            ),
            axis=0,
        )
        # [spherical1,...,sphericalN,user1,...,userM]
        e = xp.exp(1j * k[(None,) * ndim_spherical + (slice(None),) * ndim_user] * ip)
        result = (
            Y[(slice(None),) * ndim_spherical + (None,) * ndim_user + (slice(None),)]
            * e[(slice(None),) * (ndim_spherical + ndim_user) + (None,)]
        )
        return result

    # returns [user1,...,userM,harmn,harmn']
    return (-1j) ** (n - ns) * expand(
        c,
        to_expand,
        does_f_support_separation_of_variables=False,
        n=n_end + n_end_add - 1,
        n_end=n_end_add,
        phase=phase,
        xp=xp,
    )


def harmonics_twins_expansion[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    *,
    n_end_1: int,
    n_end_2: int,
    phase: Phase,
    xp: ArrayNamespaceFull,
    conj_1: bool = False,
    conj_2: bool = False,
) -> Array:
    """
    Expansion coefficients of the twins of the harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    n_end_1 : int
        The maximum degree of the harmonic
        for the first harmonics.
    n_end_2 : int
        The maximum degree of the harmonic
        for the second harmonics.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    xp : ArrayNamespaceFull
        The array namespace.
    conj_1 : bool
        Whether to conjugate the first harmonics.
        by default False
    conj_2 : bool
        Whether to conjugate the second harmonics.
        by default False

    Returns
    -------
    Array
        The expansion coefficients of the twins of shape
        [*1st quantum number, *2nd quantum number, *3rd quantum number]
        and dim `3 * c.s_ndim` and of dtype float, not complex.
        The n_end for 1st quantum number is `n_end_1`,
        The n_end for 2nd quantum number is `n_end_2`,
        The n_end for 3rd quantum number is `n_end_1 + n_end_2 - 1`.
        (not `n_end_1` or `n_end_2`)

    Notes
    -----
    To get ∫Y_{n1}(x)Y_{n2}(x)Y_{n3}(x)dx
    (integral involving three harmonics),
    one may use
    `harmonics_twins_expansion(conj_1=True, conj_2=True)`

    """

    def to_expand(spherical: Mapping[TSpherical, Array]) -> Array:
        # returns [theta,n1,...,nN,nsummed1,...,nsummedN]
        # Y(n)Y*(nsummed)
        Y1 = harmonics(
            c,
            spherical,
            n_end=n_end_1,
            phase=phase,
            expand_dims=True,
            concat=False,
        )
        Y1 = {k: v[(...,) + (None,) * c.s_ndim] for k, v in Y1.items()}
        if conj_1:
            Y1 = {k: xp.conj(v) for k, v in Y1.items()}
        Y2 = harmonics(
            c,
            spherical,
            n_end=n_end_2,
            phase=phase,
            expand_dims=True,
            concat=False,
        )
        Y2 = {
            k: v[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim]
            for k, v in Y2.items()
        }
        if conj_2:
            Y2 = {k: xp.conj(v) for k, v in Y2.items()}
        return {k: Y1[k] * Y2[k] for k in c.s_nodes}

    # returns [user1,...,userM,n1,...,nN,np1,...,npN]
    result: Mapping[TSpherical, Array] = expand(
        c,
        to_expand,
        does_f_support_separation_of_variables=True,
        n=n_end_1 + n_end_2 - 1,  # at least n_end + 2
        n_end=n_end_1 + n_end_2 - 1,
        phase=phase,
        xp=xp,
    )
    result = expand_dims_harmonics(c, result)
    result = concat_harmonics(c, result)
    result = xp.real(result)
    result = flatten_harmonics(c, result)
    result = flatten_harmonics(
        c,
        result,
        axis_end=-2,  # , n_end=n_end_2, include_negative_m=True
    )
    result = flatten_harmonics(
        c,
        result,
        axis_end=-3,  # , n_end=n_end_1, include_negative_m=True
    )
    return result


def _harmonics_translation_coef_triplet[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: Mapping[TSpherical | Literal["r"], Array],
    *,
    n_end: int,
    n_end_add: int,
    phase: Phase,
    k: Array,
    is_type_same: bool,
) -> Array:
    r"""
    Translation coefficients between same or different type of elementary solutions.

    If is_type_same is True, returns $(R|R) = (S|S)$.
    If is_type_same is False, returns $(S|R)$.

    .. math::
        R(x + t) = \sum_n (R|R)_n(t) R(x) \\
        S(x + t) = \sum_n (S|S)_n(t) S(x) \\
        S(x + t) = \sum_n (S|R)_n(t) R(x)

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    spherical : Mapping[TSpherical, Array]
        The translation vector in spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    n_end_add : int
        The maximum degree of the harmonic to be summed over.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    k : Array
        The wavenumber.
    is_type_same : bool
        Whether the type of the elementary solutions is same.

    Returns
    -------
    Array
        The translation coefficients of shape (..., ndim, ndim).
        The last axis are to be summed over with the elementary solutions
        to get translated elementary solution which quantum number corresponds to
        the second last axis indices.

    """
    xp = array_namespace(*[spherical[k] for k in c.s_nodes])
    # [user1,...,userM,n1,...,nN,nsummed1,...,nsummedN,ntemp1,...,ntempN]
    n = index_array_harmonics(
        c, c.root, n_end=n_end, expand_dims=True, xp=xp, flatten=True
    )[:, None, None]
    ns = index_array_harmonics(
        c, c.root, n_end=n_end_add, expand_dims=True, xp=xp, flatten=True
    )[None, :, None]
    ntemp = index_array_harmonics(
        c, c.root, n_end=n_end + n_end_add - 1, expand_dims=True, xp=xp, flatten=True
    )[None, None, :]

    # returns [user1,...,userM,n1,...,nN,np1,...,npN]
    coef = (2 * xp.pi) ** (c.c_ndim / 2) * xp.sqrt(2 / xp.pi)
    t_RS = harmonics_regular_singular(
        c,
        spherical,
        n_end=n_end + n_end_add - 1,
        phase=phase,
        expand_dims=True,
        concat=True,
        k=k,
        type="regular" if is_type_same else "singular",
        flatten=True,
    )
    expansion = harmonics_twins_expansion(
        c,
        n_end_1=n_end,
        n_end_2=n_end_add,
        phase=phase,
        conj_1=False,
        conj_2=True,
        xp=xp,
    )
    return coef * xp.sum(
        (-1j) ** (n - ns - ntemp) * t_RS[..., None, None, :] * expansion,
        axis=-1,
    )


def harmonics_translation_coef[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    spherical: Mapping[TSpherical | Literal["r"], Array],
    *,
    n_end: int,
    n_end_add: int,
    phase: Phase,
    k: Array,
    is_type_same: bool,
    method: Literal["gumerov", "plane_wave", "triplet"] | None = None,
) -> Array:
    r"""
    Translation coefficients between same or different type of elementary solutions.

    If is_type_same is True, returns $(R|R) = (S|S)$.
    If is_type_same is False, returns $(S|R)$.

    .. math::
        R(x + t) = \sum_n (R|R)_n(t) R(x) \\
        S(x + t) = \sum_n (S|S)_n(t) S(x) \\
        S(x + t) = \sum_n (S|R)_n(t) R(x)

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    spherical : Mapping[TSpherical, Array]
        The translation vector in spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    n_end_add : int
        The maximum degree of the harmonic to be summed over.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    k : Array
        The wavenumber.
    is_type_same : bool
        Whether the type of the elementary solutions is same.
    method : Literal["gumerov", "plane_wave", "triplet"] | None
        The method to compute the translation coefficients.
        If None, the fastest method is chosen automatically.
        "gumerov" is only available for branching type "ba".
        "plane_wave" is only available when `is_type_same` is True.

    Returns
    -------
    Array
        The translation coefficients of shape (..., ndim, ndim).
        The last axis are to be summed over with the elementary solutions
        to get translated elementary solution which quantum number corresponds to
        the second last axis indices.

    Example
    -------
    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> c = create_spherical()
    >>> t = np.asarray([2, -7, 1])
    >>> coef = harmonics_translation_coef(
    ...     c,
    ...     c.from_cartesian(t),
    ...     n_end=2,
    ...     n_end_add=2,
    ...     phase=0,
    ...     k=np.asarray(1.0),
    ...     is_type_same=True,
    ... )
    >>> np.round(coef, 2)
    array([[ 0.12+0.j  ,  0.01+0.j  ,  0.02+0.06j,  0.02-0.06j],
           [-0.01+0.j  , -0.01+0.j  ,  0.01+0.04j,  0.01-0.04j],
           [-0.02+0.06j,  0.01-0.04j,  0.18+0.j  , -0.17-0.11j],
           [-0.02-0.06j,  0.01+0.04j, -0.17+0.11j,  0.18+0.j  ]])

    >>> x = np.asarray([-1.0, 1.0, 0.0])
    >>> y = x + t
    >>> coef = harmonics_translation_coef(
    ...     c,
    ...     c.from_cartesian(t),
    ...     n_end=2,
    ...     n_end_add=6,
    ...     phase=0,
    ...     k=np.asarray(1.0),
    ...     is_type_same=True,
    ... )
    >>> R_x = harmonics_regular_singular(
    ...     c,
    ...     c.from_cartesian(x),
    ...     n_end=6,
    ...     phase=0,
    ...     k=np.asarray(1.0),
    ...     type="regular",
    ...     concat=True,
    ... )
    >>> R_y_approx = np.sum(coef * R_x[..., None, :], axis=-1)
    >>> R_y = harmonics_regular_singular(
    ...     c,
    ...     c.from_cartesian(y),
    ...     n_end=2,
    ...     phase=0,
    ...     k=np.asarray(1.0),
    ...     type="regular",
    ...     concat=True,
    ... )
    >>> np.round(R_y_approx, 8)
    array([-0.00541026-0.j        , -0.01301699-0.j        ,
           -0.00919779+0.05521981j, -0.00919779-0.05521981j])
    >>> np.round(R_y, 8)
    array([-0.00542242+0.j        , -0.01301453+0.j        ,
           -0.00920266+0.05521598j, -0.00920266-0.05521598j])

    """
    phase = Phase(phase)
    if method is None:
        if c.branching_types_expression_str == "ba":
            method = "gumerov"
        elif is_type_same:
            method = "plane_wave"
        else:
            method = "triplet"
    if method == "gumerov":
        if c.branching_types_expression_str == "ba":
            result = translational_coefficients(
                k * spherical["r"],
                spherical[c.root],
                spherical[get_child(c.G, c.root, "sin")],
                n_end=max(n_end, n_end_add),
                same=is_type_same,
            ).T[: n_end**2, : n_end_add**2]
            if phase == Phase.CONDON_SHORTLEY:
                return result
            xp = array_namespace(result)
            m = index_array_harmonics(
                c, get_child(c.G, c.root, "sin"), n_end=n_end, xp=xp, flatten=True
            )[:, None]
            m_add = index_array_harmonics(
                c, get_child(c.G, c.root, "sin"), n_end=n_end_add, xp=xp, flatten=True
            )[None, :]
            if phase == Phase(0):
                result *= minus_1_power(m + m_add)
            elif phase == Phase.NEGATIVE_LEGENDRE:
                result *= minus_1_power(
                    (xp.abs(m) + m) // 2 + (xp.abs(m_add) + m_add) // 2
                )
            elif phase == (Phase.NEGATIVE_LEGENDRE | Phase.CONDON_SHORTLEY):  # type: ignore[unreachable]
                result *= minus_1_power(
                    (xp.abs(m) - m) // 2 + (xp.abs(m_add) - m_add) // 2
                )
            return result
        else:
            raise NotImplementedError()
    elif method == "plane_wave":
        if not is_type_same:
            raise NotImplementedError(
                "plane_wave method only supports is_type_same=True"
            )
        return _harmonics_translation_coef_plane_wave(
            c,
            cartesian=c.to_cartesian(spherical, as_array=True),
            n_end=n_end,
            n_end_add=n_end_add,
            phase=phase,
            k=k,
        )
    elif method == "triplet":
        return _harmonics_translation_coef_triplet(
            c,
            spherical,
            n_end=n_end,
            n_end_add=n_end_add,
            phase=phase,
            k=k,
            is_type_same=is_type_same,
        )
    else:
        raise ValueError(f"Invalid method {method}.")
