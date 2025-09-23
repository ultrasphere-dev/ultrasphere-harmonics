from pathlib import Path
from typing import Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import (
    SphericalCoordinates,
    create_from_branching_types,
    create_spherical,
)

from ultrasphere_harmonics._core import Phase
from ultrasphere_harmonics._core._flatten import unflatten_harmonics
from ultrasphere_harmonics._helmholtz import (
    harmonics_regular_singular,
)
from ultrasphere_harmonics._translation import harmonics_translation_coef


def test_harmonics_translation_coef_gumerov_table(xp: ArrayNamespaceFull) -> None:
    if "torch" in xp.__name__:
        pytest.skip("round_cpu not implemented in torch")
    # Gumerov, N.A., & Duraiswami, R. (2001). Fast, Exact,
    # and Stable Computation of Multipole Translation and
    # Rotation Coefficients for the 3-D Helmholtz Equation.
    # got completely same results as the table in 12.3 Example
    c = create_spherical()
    x = xp.asarray([-1.0, 1.0, 0.0])
    t = xp.asarray([2.0, -7.0, 1.0])
    y = xp.add(x, t)
    x_spherical = c.from_cartesian(x)
    y_spherical = c.from_cartesian(y)
    t_spherical = c.from_cartesian(t)
    k = xp.asarray(1)

    n_end = 6
    for n_end_add in [1, 3, 5, 7, 9]:
        y_RS = harmonics_regular_singular(
            c,
            y_spherical,
            k=k,
            n_end=n_end,
            phase=Phase(0),
            concat=True,
            expand_dims=True,
            type="singular",
        )
        x_RS = harmonics_regular_singular(
            c,
            x_spherical,
            k=k,
            n_end=n_end_add,
            phase=Phase(0),
            concat=True,
            expand_dims=True,
            type="regular",
        )
        # expected (y)
        expected = y_RS

        # actual
        coef = harmonics_translation_coef(
            c,
            t_spherical,
            n_end=n_end,
            n_end_add=n_end_add,
            k=k,
            phase=Phase(0),
            is_type_same=False,
            method="triplet",
        )
        actual = xp.sum(
            x_RS[..., None, :] * coef,
            axis=-1,
        )
        expected = unflatten_harmonics(c, expected)
        actual = unflatten_harmonics(c, actual)
        print(xp.round(expected[5, 2], decimals=6), xp.round(actual[5, 2], decimals=6))


@pytest.mark.parametrize(
    "c",
    [
        (create_from_branching_types("a")),
        (create_spherical()),
    ],
)
@pytest.mark.parametrize("n_end, n_end_add", [(4, 6)])
@pytest.mark.parametrize("phase", Phase.all())
@pytest.mark.parametrize(
    "from_,to_",
    [("regular", "regular"), ("singular", "singular"), ("regular", "singular")],
)
@pytest.mark.parametrize(
    "method",
    ["gumerov", "plane_wave", "triplet", None],
)
def test_harmonics_translation_coef[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    n_end: int,
    n_end_add: int,
    phase: Phase,
    from_: Literal["regular", "singular"],
    to_: Literal["regular", "singular"],
    xp: ArrayNamespaceFull,
    method: Literal["gumerov", "plane_wave", "triplet"],
) -> None:
    if method == "gumerov" and c.branching_types_expression_str != "ba":
        pytest.skip("gumerov method only supports ba branching type")
    if method == "plane_wave" and from_ != to_:
        pytest.skip("plane_wave method only supports from_=to_")

    # get x, t, y := x + t
    x = xp.arange(c.c_ndim)
    t = xp.flip(xp.arange(c.c_ndim))
    k = 1.0
    if (from_, to_) == ("singular", "singular"):
        # |t| < |x| (if too close, the result would be inaccurate)
        t = t * 0.1
        assert (
            xp.linalg.vector_norm(t, axis=0) < xp.linalg.vector_norm(x, axis=0)
        ).all()
    elif (from_, to_) == ("regular", "singular"):
        # |t| > |x| (if too close, the result would be inaccurate)
        t = t * 10
        assert (
            xp.linalg.vector_norm(t, axis=0) > xp.linalg.vector_norm(x, axis=0)
        ).all()
    elif (from_, to_) == ("regular", "regular"):
        # accurate everywhere
        pass

    # t = xp.zeros_like(t)
    y = x + t
    t_spherical = c.from_cartesian(t)
    x_spherical = c.from_cartesian(x)
    y_spherical = c.from_cartesian(y)

    y_RS = harmonics_regular_singular(
        c,
        y_spherical,
        k=k,
        n_end=n_end,
        phase=phase,
        concat=True,
        expand_dims=True,
        type=to_,
    )
    x_RS = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        n_end=n_end_add,
        phase=phase,
        concat=True,
        expand_dims=True,
        type=from_,
    )
    # expected (y)
    expected = y_RS

    # actual
    coef = harmonics_translation_coef(
        c,
        t_spherical,
        n_end=n_end,
        n_end_add=n_end_add,
        k=k,
        phase=phase,
        is_type_same=from_ == to_,
        method=method,
    )
    # cannot be replaced with vecdot because both is complex
    actual = xp.sum(
        x_RS[..., None, :] * coef,
        axis=-1,
    )
    if (from_, to_) == ("singular", "singular"):
        pytest.skip("singular case does not converge in real world computation")
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))


def test_dataset_coef() -> None:
    import numpy as np

    cartesian = np.array([2, -7, 1])
    c = create_spherical()
    spherical = c.from_cartesian(cartesian)
    Path("tests/.cache").mkdir(exist_ok=True)
    for phase in Phase.all():
        for is_same_type in [True, False]:
            coef = harmonics_translation_coef(
                c,
                spherical,
                n_end=3,
                n_end_add=3,
                phase=phase,
                is_type_same=is_same_type,
                method="triplet",
                k=1.0,
            )
            np.savetxt(
                f"tests/.cache/translation_coef_{is_same_type}_{phase}.csv".replace(
                    "|", "_"
                ),
                coef,
                delimiter=",",
            )
        for type_ in ["regular", "singular"]:
            RS = harmonics_regular_singular(  # type: ignore[call-overload]
                c,
                spherical,
                n_end=4,
                k=1.0,
                phase=phase,
                concat=True,
                expand_dims=True,
                type=type_,
            )
            np.savetxt(
                f"tests/.cache/{type_}_{phase}.csv".replace("|", "_"),
                RS,
                delimiter=",",
            )
