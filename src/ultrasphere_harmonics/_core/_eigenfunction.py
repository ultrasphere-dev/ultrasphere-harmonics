from enum import STRICT, Flag, auto

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from array_api_negative_index import to_symmetric
from jacobi_poly import jacobi_all, jacobi_normalization_constant
from shift_nth_row_n_steps import shift_nth_row_n_steps
from ultrasphere import BranchingType, SphericalCoordinates


class Phase(Flag, boundary=STRICT):
    """Adjust phase (±) of the spherical harmonics, mainly to match conventions."""

    CONDON_SHORTLEY = auto()
    """Whether to apply the Condon-Shortley phase.

    It just multiplies the result by $(-1)^m$.

    By using this phase, in quantum mechanics,
    spherical harmonics with ladder operator applied
    become spherical harmonics times always positive real numbers.

    scipy.special.sph_harm_y uses the Condon-Shortley phase."""

    NEGATIVE_LEGENDRE = auto()
    r"""Whether to use $P_l^m$ or $P_{l}^{\left\|m\right\|}$ for negative m.

    If False, $Y^{-m}_{l} = \overline{Y^{m}_{l}}$.
    If True, $Y^{-m}_{l} = (-1)^m \overline{Y^{m}_{l}}$.

    scipy.special.sph_harm_y uses P_l^m."""

    @classmethod
    def all(cls) -> list["Phase"]:
        """Return all possible combinations of the Phase flags."""
        return [
            cls(0),
            cls.CONDON_SHORTLEY,
            cls.NEGATIVE_LEGENDRE,
            cls.CONDON_SHORTLEY | cls.NEGATIVE_LEGENDRE,
        ]


def minus_1_power(x: Array, /) -> Array:
    """
    $(-1)^x$.

    Parameters
    ----------
    x : Array
        The exponent.

    Returns
    -------
    Array
        $(-1)^x$

    """
    return 1 - 2 * (x % 2)


def type_a(
    theta: Array,
    n_end: int,
    *,
    phase: Phase,
    include_negative_m: bool = True,
) -> Array:
    r"""
    Eigenfunction for type a node.

    $$
    \psi_n (\theta) := \frac{1}{\sqrt{2\pi}} e^{i m \theta}
    $$

    Parameters
    ----------
    theta : Array
        [0, 2π)
    n_end : int
        The maximum degree of the harmonic.
    phase : Phase
        Adjust phase (±) of the spherical harmonics, mainly to match conventions.
        See `Phase` for details.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
        If True, the m values are [0, 1, ..., n_end-1, -n_end+1, ..., -1],
        and starts from 0, not -n_end+1.

    Returns
    -------
    Array
        The result of the eigenfunction.

    Reference
    ---------
    Cohl, H. S. (2013).
    Fourier, Gegenbauer and Jacobi Expansions for a Power-Law Fundamental Solution
    of the Polyharmonic Equation and Polyspherical Addition Theorems.
    Symmetry, Integrability and Geometry: Methods and Applications.
    https://doi.org/10.3842/SIGMA.2013.042

    """
    xp = array_namespace(theta)
    m = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (slice(None),)
    ]
    if include_negative_m:
        m = to_symmetric(m, axis=-1, asymmetric=True, conjugate=False)
    res = xp.exp(
        xp.asarray(
            1j,
            dtype=(
                xp.complex64
                if theta.dtype in [xp.complex64, xp.float32]
                else xp.complex128
            ),
            device=theta.device,
        )
        * m
        * theta[..., None]
    ) / xp.sqrt(xp.asarray(2 * xp.pi))
    phase = Phase(phase)
    if Phase.CONDON_SHORTLEY in phase:
        if Phase.NEGATIVE_LEGENDRE in phase:
            res *= minus_1_power((xp.abs(m) + m) // 2)
        else:
            res *= minus_1_power(m)
    else:
        if Phase.NEGATIVE_LEGENDRE in phase:
            res *= minus_1_power((xp.abs(m) - m) // 2)
    return res


def type_b(
    theta: Array,
    *,
    n_end: int,
    s_beta: Array | int,
    index_with_surrogate_quantum_number: bool = False,
    is_beta_type_a_and_include_negative_m: bool = False,
    fill_value: float = 0,
) -> Array:
    r"""
    Eigenfunction for type b node.

    $$
    \alpha := l_\beta + \frac{s_\beta}{2} \\
    \psi_{n,l_\beta}^{s_\beta}(\theta) :=
    N^{(\alpha,\alpha)}_n
    \sin^{l_\beta} theta P_n^{(\alpha,\alpha)}(\cos \theta)
    $$

    Parameters
    ----------
    theta : Array
        [0, π]
    n_end : int
        Positive integer, l - l_beta, where l is the quantum number of this node.
    s_beta : Array
        The number of non-leaf child nodes of the node beta.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    is_beta_type_a_and_include_negative_m : bool, optional
        Whether the node beta is type a and include negative m, by default False
    fill_value : float, optional
        The value to fill for the indices that are not possible, by default 0

    Returns
    -------
    Array
        If index_with_surrogate_quantum_number is True,
        [..., l_beta, n] of size (..., n_end, n_end)
        Otherwise,
        [..., l_beta, l] of size (..., n_end, n_end), if l < l_beta value is 0.

    Reference
    ---------
    Cohl, H. S. (2013).
    Fourier, Gegenbauer and Jacobi Expansions for a Power-Law Fundamental Solution
    of the Polyharmonic Equation and Polyspherical Addition Theorems.
    Symmetry, Integrability and Geometry: Methods and Applications.
    https://doi.org/10.3842/SIGMA.2013.042

    """
    xp = array_namespace(theta)
    if isinstance(s_beta, int):
        s_beta = xp.asarray(s_beta, dtype=theta.dtype, device=theta.device)
    # using broadcasting may cause problems, we have to be very careful here
    l_beta = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (slice(None),)
    ]
    n = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (None, slice(None))
    ]
    alpha = l_beta + s_beta[..., None] / 2
    res = (
        jacobi_normalization_constant(
            alpha=alpha[..., None], beta=alpha[..., None], n=n
        )
        * (xp.sin(theta[..., None, None]) ** l_beta[..., None])
        * jacobi_all(n_end=n_end, alpha=alpha, beta=alpha, x=xp.cos(theta[..., None]))
    )
    if not index_with_surrogate_quantum_number:
        # [l_beta, n] -> [l_beta, l = n + l_beta]
        res = shift_nth_row_n_steps(
            res,
            axis_row=-2,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
    if is_beta_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-2, asymmetric=False, conjugate=False)
    return res


def type_bdash(
    theta: Array,
    *,
    n_end: int,
    s_alpha: Array | int,
    index_with_surrogate_quantum_number: bool = False,
    is_alpha_type_a_and_include_negative_m: bool = False,
    fill_value: float = 0,
) -> Array:
    r"""
    Eigenfunction for type b node.

    $$
    \beta := l_\alpha + \frac{s_\alpha}{2} \\
    \psi_{n,l_\alpha}^{s_\alpha}(\theta) :=
    N^{(\beta,\beta)}_n
    \cos^{l_\alpha} \theta P_n^{(\beta,\beta)}(\sin \theta)
    $$

    Parameters
    ----------
    theta : Array
        [-π/2, π/2]
    n_end : int
        Positive integer, l - l_alpha, where l is the quantum number of this node.
    s_alpha : Array
        The number of non-leaf child nodes of the node alpha.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    is_alpha_type_a_and_include_negative_m : bool, optional
        Whether the node alpha is type a and include negative m, by default False
    fill_value : float, optional
        The value to fill for the indices that are not possible, by default 0

    Returns
    -------
    Array
        If index_with_surrogate_quantum_number is True,
        [..., l_alpha, n] of size (..., n_end, n_end)
        Otherwise,
        [..., l_alpha, l] of size (..., n_end, n_end), if l < l_alpha value is 0.

    Reference
    ---------
    Cohl, H. S. (2013).
    Fourier, Gegenbauer and Jacobi Expansions for a Power-Law Fundamental Solution
    of the Polyharmonic Equation and Polyspherical Addition Theorems.
    Symmetry, Integrability and Geometry: Methods and Applications.
    https://doi.org/10.3842/SIGMA.2013.042

    """
    xp = array_namespace(theta)
    if isinstance(s_alpha, int):
        s_alpha = xp.asarray(s_alpha, dtype=theta.dtype, device=theta.device)
    l_alpha = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (slice(None),)
    ]
    n = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (None, slice(None))
    ]
    beta = l_alpha + s_alpha[..., None] / 2
    res = (
        jacobi_normalization_constant(alpha=beta[..., None], beta=beta[..., None], n=n)
        * (xp.cos(theta[..., None, None]) ** l_alpha[..., None])
        * jacobi_all(n_end=n_end, alpha=beta, beta=beta, x=xp.sin(theta[..., None]))
    )
    if not index_with_surrogate_quantum_number:
        res = shift_nth_row_n_steps(
            res,
            axis_row=-2,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
    # [l_alpha, n] -> [l_alpha, l = n + l_alpha]
    if is_alpha_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-2, asymmetric=False, conjugate=False)
    return res


def type_c(
    theta: Array,
    *,
    n_end: int,
    s_alpha: Array | int,
    s_beta: Array | int,
    index_with_surrogate_quantum_number: bool = False,
    is_alpha_type_a_and_include_negative_m: bool = False,
    is_beta_type_a_and_include_negative_m: bool = False,
    fill_value: float = 0,
) -> Array:
    r"""
    Eigenfunction for type c node.

    $$
    \alpha := l_\beta + \frac{s_\beta}{2} \\
    \beta := l_\alpha + \frac{s_\alpha}{2} \\
    \psi_{n,l_\alpha,l_\beta}^{s_\alpha,s_\beta}(\theta) :=
    2^{\frac{\alpha+\beta}{2}+1}
    N_n^{(\alpha,\beta)}
    (\sin \theta)^{l_\beta} (\cos \theta)^{l_\alpha} P_n^{(\alpha,\beta)}(\cos 2\theta)
    $$

    Parameters
    ----------
    theta : Array
        [0, π/2]
    n_end : int
        Positive integer, (l - l_alpha - l_beta) / 2,
        where l is the quantum number of this node.
    s_alpha : Array
        The number of non-leaf child nodes of the node alpha.
    s_beta : Array
        The number of non-leaf child nodes of the node beta.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    is_alpha_type_a_and_include_negative_m : bool, optional
        Whether the node alpha is type a and include negative m, by default False
    is_beta_type_a_and_include_negative_m : bool, optional
        Whether the node beta is type a and include negative m, by default False
    fill_value : float, optional
        The value to fill for the indices that are not possible, by default 0

    Returns
    -------
    Array
        [..., l_alpha, l_beta, l] of size (..., n_end, n_end),
        if l < l_alpha + l_beta value is 0.

    Reference
    ---------
    Cohl, H. S. (2013).
    Fourier, Gegenbauer and Jacobi Expansions for a Power-Law Fundamental Solution
    of the Polyharmonic Equation and Polyspherical Addition Theorems.
    Symmetry, Integrability and Geometry: Methods and Applications.
    https://doi.org/10.3842/SIGMA.2013.042

    """
    xp = array_namespace(theta)
    if isinstance(s_alpha, int):
        s_alpha = xp.asarray(s_alpha, dtype=theta.dtype, device=theta.device)
    if isinstance(s_beta, int):
        s_beta = xp.asarray(s_beta, dtype=theta.dtype, device=theta.device)
    l_alpha = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (slice(None), None)
    ]  # 2d
    l_beta = xp.arange(0, n_end, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (None, slice(None))
    ]  # 2d
    n = xp.arange(0, (n_end + 1) // 2, dtype=theta.dtype, device=theta.device)[
        (None,) * (theta.ndim) + (None, None, slice(None))
    ]  # 3d
    alpha = l_alpha + s_alpha[..., None, None] / 2  # 2d
    beta = l_beta + s_beta[..., None, None] / 2  # 2d
    res = (
        2 ** ((alpha + beta) / 2 + 1)[..., None]
        * jacobi_normalization_constant(
            alpha=alpha[..., None], beta=beta[..., None], n=n
        )
        * (xp.sin(theta[..., None, None, None]) ** l_beta[..., None])
        * (xp.cos(theta[..., None, None, None]) ** l_alpha[..., None])
        * jacobi_all(
            n_end=(n_end + 1) // 2,
            alpha=beta,
            beta=alpha,  # this is weird but correct
            x=xp.cos(2 * theta[..., None, None]),
        )
    )
    # n_end = 3 -> max l = 2 -> max jacobi order = 1 -> jacobi n_end = 2
    # n_end = 4 -> max l = 3 -> max jacobi order = 1 -> jacobi n_end = 2
    # http://kuiperbelt.la.coocan.jp/sf/egan/Diaspora/atomic-orbital/
    # laplacian/4D-2.html
    if not index_with_surrogate_quantum_number:
        # complicated reshaping
        # [l_alpha, l_beta, n] -> [l_alpha, l_beta, l = 2n + l_alpha + l_beta]
        # 1. [l_alpha, l_beta, n] -> [l_alpha, l_beta, 2n]
        # add zeros to the left for each row, i.e. [1, 2, 3] -> [1, 0, 2, 0, 3, 0]
        res_expaneded = xp.zeros((*res.shape[:-1], n_end))
        res_expaneded[..., ::2] = res
        # 2. [l_alpha, l_beta, 2n] -> [l_alpha, l_beta, 2n + l_alpha]
        res_expaneded = shift_nth_row_n_steps(
            res_expaneded,
            axis_row=-3,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
        # 3. [l_alpha, l_beta, 2n + l_alpha] ->
        # [l_alpha, l_beta, 2n + l_alpha + l_beta]
        res = shift_nth_row_n_steps(
            res_expaneded,
            axis_row=-2,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
    if is_alpha_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-3, asymmetric=False, conjugate=False)
    if is_beta_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-2, asymmetric=False, conjugate=False)
    return res


def ndim_harmonics[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    node: TSpherical,
) -> int:
    r"""
    The number of dimensions of the eigenfunction corresponding to the node.

    $$
    \begin{cases}
    1 & \text{if branching type is A} \\
    2 & \text{if branching type is B or B'} \\
    3 & \text{if branching type is C} \\
    \end{cases}
    $$

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    node : TSpherical
        The node of the spherical coordinates.

    Returns
    -------
    int
        The number of dimensions.

    """
    return {
        BranchingType.A: 1,
        BranchingType.B: 2,
        BranchingType.BP: 2,
        BranchingType.C: 3,
    }[c.branching_types[node]]
