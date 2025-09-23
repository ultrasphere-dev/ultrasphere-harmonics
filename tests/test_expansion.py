from collections.abc import Mapping
from pathlib import Path

import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import is_torch_namespace
from matplotlib import pyplot as plt
from ultrasphere import (
    SphericalCoordinates,
    create_from_branching_types,
    create_hopf,
    create_spherical,
    create_standard,
    roots,
)

from ultrasphere_harmonics._core import Phase, harmonics
from ultrasphere_harmonics._core._eigenfunction import ndim_harmonics
from ultrasphere_harmonics._cut import expand_cut
from ultrasphere_harmonics._expansion import expand, expand_evaluate
from ultrasphere_harmonics._ndim import harm_n_ndim_le

PATH = Path("tests/.cache/")
Path.mkdir(PATH, exist_ok=True)


@pytest.mark.parametrize(
    "c",
    [
        (create_spherical()),
        (create_standard(3)),
        (create_standard(4)),
        (create_hopf(2)),
    ],
)
@pytest.mark.parametrize("n_end", [3, 4])
@pytest.mark.parametrize("phase", Phase.all())
@pytest.mark.parametrize("concat", [True, False])
def test_orthogonal_expand[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    n_end: int,
    phase: Phase,
    concat: bool,
    xp: ArrayNamespaceFull,
) -> None:
    def f(spherical: Mapping[TSpherical, Array]) -> Array:
        return harmonics(  # type: ignore[call-overload]
            c,
            spherical,
            n_end=n_end,
            phase=phase,
            concat=concat,
            expand_dims=concat,
        )

    actual = expand(  # type: ignore[call-overload]
        c,
        f,
        n=2 * n_end - 1,
        n_end=n_end,
        does_f_support_separation_of_variables=not concat,
        phase=phase,
        xp=xp,
        dtype=xp.float32,
        device=None,
    )
    if not concat:
        if is_torch_namespace(xp):
            pytest.skip("torch.nonzero is not array API compatible")
        for key, value in actual.items():
            # assert quantum numbers are the same for non-zero values
            expansion_nonzero = xp.moveaxis(
                xp.asarray(xp.nonzero(xp.abs(value) > 1e-3)), 0, 1
            )
            assert expansion_nonzero.shape[1] == ndim_harmonics(c, key) * 2
            l, r = (
                expansion_nonzero[:, : ndim_harmonics(c, key)],
                expansion_nonzero[:, ndim_harmonics(c, key) :],
            )
            idx = xp.squeeze(
                xp.asarray(xp.nonzero((l[:-1, :] == r[:-1, :]).all(axis=-1))), 0
            )
            assert xp.all(l[idx, :] == r[idx, :])
    else:
        expected = xp.eye(
            int(harm_n_ndim_le(n_end, c_ndim=c.c_ndim)), dtype=xp.complex64
        )
        assert xp.all(xpx.isclose(actual, expected, rtol=1e-6, atol=1e-6))


@pytest.mark.parametrize(
    "name, c, n_end",
    [
        ("spherical", create_spherical(), 5),
        ("standard-2", create_standard(2), 7),
        ("standard-2'", create_from_branching_types("bpa"), 10),
        ("standard-3", create_standard(3), 6),
        ("hoph-2", create_hopf(2), 6),
    ],
)
@pytest.mark.parametrize("phase", Phase.all())
def test_approximate[TSpherical, TCartesian](
    name: str,
    c: SphericalCoordinates[TSpherical, TCartesian],
    n_end: int,
    phase: Phase,
    xp: ArrayNamespaceFull,
) -> None:
    k = xp.arange(c.c_ndim) / c.c_ndim

    def f(s: Mapping[TSpherical, Array]) -> Array:
        x = c.to_cartesian(s, as_array=True)
        # k is complex
        return xp.exp(
            1j
            * xp.vecdot(
                x,
                xp.astype(k, x.dtype)[(slice(None),) + (None,) * (x.ndim - 1)],
                axis=0,
            )
        )

    spherical, _ = roots(c, 1, expand_dims_x=True, xp=xp)
    expected = f(spherical)
    error = {}
    expansion = expand(
        c,
        f,
        n=n_end,
        n_end=n_end,
        does_f_support_separation_of_variables=False,
        phase=phase,
        xp=xp,
    )
    for n_end_c in np.linspace(1, n_end, 5):
        n_end_c = int(n_end_c)
        expansion_cut = expand_cut(c, expansion, n_end_c)
        approx = expand_evaluate(
            c,
            expansion_cut,
            spherical,
            phase=phase,
        )
        error[n_end_c] = xp.mean(xp.abs(approx - expected))
    fig, ax = plt.subplots()
    ax.plot(list(error.keys()), list(error.values()))
    ax.set_xlabel("Degree")
    ax.set_ylabel("MAE")
    ax.set_title(f"Spherical Harmonics Expansion Error for {c}")
    ax.set_yscale("log")
    fig.savefig(PATH / f"{name}-approximate.png")
    assert error[max(error.keys())] < 2e-3
