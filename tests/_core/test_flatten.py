from collections.abc import Mapping
from typing import Any

import array_api_extra as xpx
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from ultrasphere import (
    SphericalCoordinates,
    create_hopf,
    create_random,
    create_spherical,
)
from ultrasphere._integral import roots

from ultrasphere_harmonics._core import harmonics as harmonics_
from ultrasphere_harmonics._core._eigenfunction import Phase
from ultrasphere_harmonics._core._flatten import (
    _index_array_harmonics_all,
    flatten_harmonics,
    unflatten_harmonics,
)


@pytest.mark.parametrize(
    "c",
    [
        create_random(1),
        create_random(2),
        create_spherical(),
        create_hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_index_array_harmonics_all[TCartesian, TSpherical](
    c: SphericalCoordinates[TSpherical, TCartesian],
    n_end: int,
    xp: ArrayNamespaceFull,
    device: Any,
) -> None:
    iall_concat = _index_array_harmonics_all(
        c,
        n_end=n_end,
        include_negative_m=False,
        expand_dims=True,
        as_array=True,
        xp=xp,
        device=device,
    )
    iall: Mapping[TSpherical, Array] = _index_array_harmonics_all(
        c,
        n_end=n_end,
        include_negative_m=False,
        expand_dims=True,
        as_array=False,
        xp=xp,
    )
    assert iall_concat.shape == (
        c.s_ndim,
        *xpx.broadcast_shapes(*[v.shape for v in iall.values()]),
    )
    for i, s_node in enumerate(c.s_nodes):
        # the shapes not necessarily match, so all_equal cannot be used
        assert xp.all(iall_concat[i] == iall[s_node])


@pytest.mark.parametrize(
    "c",
    [
        create_random(1),
        create_random(2),
        create_spherical(),
        create_hopf(2),
    ],
)
@pytest.mark.parametrize("phase", Phase.all())
def test_flatten_unflatten_harmonics[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    xp: ArrayNamespaceFull,
    phase: Phase,
    device: Any,
) -> None:
    n_end = 4
    harmonics = harmonics_(
        c,
        roots(c, n=n_end, expand_dims_x=True, xp=xp, device=device)[0],
        n_end=n_end,
        phase=phase,
        concat=True,
        expand_dims=True,
        flatten=False,
    )
    flattened = flatten_harmonics(c, harmonics)
    unflattened = unflatten_harmonics(c, flattened)
    assert xp.all(harmonics == unflattened)
