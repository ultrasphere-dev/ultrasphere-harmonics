from collections.abc import Callable, Mapping
from typing import Any, Literal, overload

import array_api_extra as xpx
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from ultrasphere import (
    SphericalCoordinates,
    integrate,
)

from ._core import assume_n_end_and_include_negative_m_from_harmonics, harmonics
from ._core._eigenfunction import Phase
from ._core._eigenfunction import ndim_harmonics as ndim_harmonics_


@overload
def expand[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array] | Array,
        ]
        | Mapping[TSpherical, Array]
        | Array
    ),
    does_f_support_separation_of_variables: Literal[True],
    n_end: int,
    n: int,
    *,
    phase: Phase,
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Mapping[TSpherical, Array]: ...


@overload
def expand[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array] | Array,
        ]
        | Mapping[TSpherical, Array]
        | Array
    ),
    does_f_support_separation_of_variables: Literal[False],
    n_end: int,
    n: int,
    *,
    phase: Phase,
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Array: ...


def expand[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array] | Array,
        ]
        | Mapping[TSpherical, Array]
        | Array
    ),
    does_f_support_separation_of_variables: bool,
    n_end: int,
    n: int,
    *,
    phase: Phase,
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Array | Mapping[TSpherical, Array]:
    r"""
    Calculate the expansion coefficients of the function over the hypersphere.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    f : Callable[ [Mapping[TSpherical, Array]],
        Mapping[TSpherical, Array] | Array, ]
       | Mapping[TSpherical, Array] | Array
        The function to integrate or the values of the function.
        In case of vectorized function, the function should add extra
        axis to the last dimension, not the first dimension.
    does_f_support_separation_of_variables : bool
        Whether the function supports separation of variables.
        This could significantly reduce the computational cost.
    n : int
        The number of integration points.

        Must be equal to or larger than n_end.

        Must be large enough AGAINST f, as this method
        does not use adaptive integration. For example,
        consider expanding

        $$
        f(θ) = e^{2Nθ}
        $$

        with $n=N$.

        >>> from ultrasphere import create_polar
        >>> from array_api_compat import numpy as np
        >>> n = 3
        >>> expansion = expand(
        ...     create_polar(),
        ...     lambda x: np.exp(1j * (2 * n) * x["phi"]) / np.sqrt(2 * np.pi),
        ...     does_f_support_separation_of_variables=False,
        ...     n=n,
        ...     n_end=n,
        ...     phase=False,
        ...     xp=np,
        ... )
        >>> np.round(expansion, 2)
        array([ 1.-0.j,  0.+0.j,  0.+0.j,  0.+0.j, -0.+0.j])

        This result claims that

        $$
        f(\phi) = 1 + \sum_{\|m\|\geq3} a_m e^{im\phi}, a_m \in \mathbb{C}
        $$

        , which is incorrect.
    n_end : int
        The maximum degree of the harmonic.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any, optional
        The device, by default None
    dtype : Any, optional
        The data type, by default None

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The expanded value of shape (..., ndim)

    Example
    -------
    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> c = create_spherical()
    >>> coef = expand(
    ...     c,
    ...     lambda spherical: np.sin(spherical["theta"]) * np.sin(spherical["phi"]),
    ...     n_end=2,
    ...     n=3,
    ...     does_f_support_separation_of_variables=False,
    ...     phase=0,
    ...     xp=np,
    ... )
    >>> np.round(coef, 2) + 0.0
    array([0.+0.j  , 0.+0.j  , 0.-1.45j, 0.+1.45j])

    >>> coef_fast = expand(
    ...     c,
    ...     lambda spherical: {
    ...         "theta": np.sin(spherical["theta"]),
    ...         "phi": np.sin(spherical["phi"])
    ...     },
    ...     n_end=2,
    ...     n=3,
    ...     does_f_support_separation_of_variables=True,
    ...     phase=0,
    ...     xp=np,
    ... )
    >>> {k: np.round(coef_fast[k], 2) for k in c.s_nodes}
    {'theta': array([[1.13, 0.  ],
           [0.  , 1.15],
           [0.  , 1.15]]), 'phi': array([0.+0.j  , 0.-1.25j, 0.+1.25j])}

    >>> from ultrasphere import create_polar
    >>> expansion = expand(
    ...     create_polar(),
    ...     lambda x: np.exp(1j * (n - 1) * x["phi"]) / np.sqrt(2 * np.pi),
    ...     does_f_support_separation_of_variables=False,
    ...     n=n,
    ...     n_end=n,
    ...     phase=False,
    ...     xp=np,
    ... )
    >>> np.round(expansion, 2) + 0.0
    array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])

    >>> expansion = expand(
    ...     create_polar(),
    ...     lambda x: np.exp(1j * n * x["phi"]) / np.sqrt(2 * np.pi),
    ...     does_f_support_separation_of_variables=False,
    ...     n=n,
    ...     n_end=n,
    ...     phase=False,
    ...     xp=np,
    ... )
    >>> np.round(expansion, 2) + 0.0
    array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    """
    if n < n_end:
        raise ValueError(
            f"n={n} < n_end={n_end}, which would lead to incorrect results."
        )

    def inner(
        xs: Mapping[TSpherical, Array],
    ) -> Mapping[TSpherical, Array]:
        # calculate f
        if isinstance(f, Callable):  # type: ignore
            try:
                val = f(xs)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Error occurred while evaluating {f=}") from e
        else:
            val = f

        # calculate harmonics
        Y = harmonics(  # type: ignore[call-overload]
            c,
            xs,
            n_end=n_end,
            phase=phase,
            expand_dims=not does_f_support_separation_of_variables,
            concat=not does_f_support_separation_of_variables,
        )

        # multiply f and harmonics
        # (C,complex conjugate) is star-algebra
        if isinstance(val, Mapping):
            if not does_f_support_separation_of_variables:
                raise ValueError(
                    "val is Mapping but "
                    "does_f_support_separation_of_variables "
                    "is False."
                )
            result = {}
            for node in c.s_nodes:
                value = val[node]
                # val: theta(node),u1,...,uM
                # harmonics: theta(node),harm1,...,harmNnode
                # result: theta(node),u1,...,uM,harm1,...,harmNnode
                xpx.broadcast_shapes(value.shape[:1], Y[node].shape[:1])
                ndim_val = value.ndim - 1
                ndim_harm = ndim_harmonics_(c, node)
                value = value[(...,) + (None,) * (ndim_harm)]
                harm = Y[node][
                    (slice(None),) + (None,) * ndim_val + (slice(None),) * ndim_harm
                ]
                result[node] = value * xp.conj(harm)
        else:
            if does_f_support_separation_of_variables:
                raise ValueError(
                    "val is not Mapping but "
                    "does_f_support_separation_of_variables "
                    "is True."
                )
            # val: theta1,...,thetaN,u1,...,uM
            # harmonics: theta1,...,thetaN,harm
            # res: theta1,...,thetaN,u1,...,uM,harm
            xpx.broadcast_shapes(val.shape[: c.s_ndim], Y.shape[: c.s_ndim])
            ndim_val = val.ndim - c.s_ndim
            val = val[..., None]
            Y = Y[(slice(None),) * c.s_ndim + (None,) * ndim_val + (slice(None),)]
            result = val * xp.conj(Y)

        return result

    return integrate(
        c,
        inner,
        does_f_support_separation_of_variables,
        n,
        device=device,
        dtype=dtype,
        xp=xp,
    )


@overload
def expand_evaluate[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    expansion: Mapping[TSpherical, Array],
    spherical: Mapping[TSpherical, Array],
    *,
    phase: Phase,
) -> Mapping[TSpherical, Array]: ...


@overload
def expand_evaluate[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    expansion: Array,
    spherical: Mapping[TSpherical, Array],
    *,
    phase: Phase,
) -> Array: ...


def expand_evaluate[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    expansion: Mapping[TSpherical, Array] | Array,
    spherical: Mapping[TSpherical, Array],
    *,
    phase: Phase,
) -> Array | Mapping[TSpherical, Array]:
    """
    Evaluate the expansion at the spherical coordinates.

    Make sure to compare with values on the SPHERE,
    not in the ball.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    expansion : Mapping[TSpherical, Array] | Array
        The expansion coefficients of shape (*shape_e, ndim).
        If not mapping, assume that the expansion is flatten.
        If mapping, assume that the expansion is not expanded.
    spherical : Mapping[TSpherical, Array]
        The spherical coordinates of shape (*shape_s, c.s_ndim).
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The evaluated value of shape (*shape_s, *shape_e).

    """
    is_mapping = isinstance(expansion, Mapping)
    xp = (
        array_namespace(*[expansion[k] for k in c.s_nodes])
        if is_mapping
        else array_namespace(expansion)
    )
    n_end, _ = assume_n_end_and_include_negative_m_from_harmonics(c, expansion)
    if "r" in spherical:
        raise ValueError("Passing points not on the sphere is not supported.")
    Y = harmonics(  # type: ignore[call-overload]
        c,
        spherical,
        n_end=n_end,
        phase=phase,
        expand_dims=not is_mapping,
        concat=not is_mapping,
        flatten=not is_mapping,
    )
    if is_mapping:
        result: dict[TSpherical, Array] = {}
        for node in c.s_nodes:
            expansion_ = expansion[node]
            Y_ = Y[node]
            # expansion: f1,...,fL,harm1,...,harmNnode
            # harmonics: u1,...,uM,harm1,...,harmNnode
            # result: u1,...,uM,f1,...,fL
            ndim_harmonics = ndim_harmonics_(c, node)
            ndim_expansion = expansion_.ndim - ndim_harmonics
            ndim_extra_harmonics = Y_.ndim - ndim_harmonics
            expansion_ = Y_[
                (None,) * (ndim_extra_harmonics)
                + (slice(None),) * (ndim_expansion + ndim_harmonics)
            ]
            Y_ = Y_[
                (slice(None),) * ndim_extra_harmonics
                + (None,) * ndim_expansion
                + (slice(None),) * ndim_harmonics
            ]
            result_ = Y_ * expansion_
            for _ in range(ndim_harmonics):
                result_ = xp.sum(result_, axis=-1)
            result[node] = result
        return result
    if isinstance(expansion, Mapping):
        raise AssertionError()
    # expansion: f1,...,fL,harm
    # harmonics: u1,...,uM,harm
    # result: u1,...,uM,f1,...,fL
    ndim_expansion = expansion.ndim - 1
    ndim_extra_harmonics = Y.ndim - 1
    expansion = expansion[
        (None,) * (ndim_extra_harmonics) + (slice(None),) * (ndim_expansion + 1)
    ]
    Y = Y[
        (slice(None),) * ndim_extra_harmonics
        + (None,) * ndim_expansion
        + (slice(None),)
    ]
    result = xp.sum(Y * expansion, axis=-1)
    return result
