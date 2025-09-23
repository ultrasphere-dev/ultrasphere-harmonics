from collections.abc import Mapping

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ultrasphere import SphericalCoordinates


def concat_harmonics[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    harmonics: Mapping[TSpherical, Array],
) -> Array:
    """
    Concatenate the mapping of expanded harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    harmonics : Mapping[TSpherical, Array]
        The expanded harmonics.

    Returns
    -------
    Array
        The concatenated harmonics.

    Example
    -------
    >>> from array_api_compat import numpy as np
    >>> from ultrasphere import create_spherical
    >>> from ultrasphere_harmonics import harmonics
    >>> c = create_spherical()
    >>> harm = harmonics(
    ...     c,
    ...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)},
    ...     n_end=2,
    ...     phase=0,
    ...     concat=False,
    ... )
    >>> {k: np.round(harm[k], 2) for k in c.s_nodes}
    {'theta': array([[0.71, 0.  , 0.  ],
           [1.07, 0.42, 0.42]]), 'phi': array([[0.4 +0.j  , 0.22+0.34j, 0.22-0.34j]])}

    >>> np.round(concat_harmonics(c, harm), 2)
    array([[0.28+0.j  , 0.  +0.j  , 0.  +0.j  ],
           [0.43+0.j  , 0.09+0.14j, 0.09-0.14j]])

    """
    xp = array_namespace(*[harmonics[k] for k in c.s_nodes])
    try:
        if c.s_ndim == 0:
            return xp.asarray(1)
        return xp.prod(
            xp.stack(xp.broadcast_arrays(*[harmonics[k] for k in c.s_nodes]), axis=0),
            axis=0,
        )
    except Exception as e:
        shapes = {k: v.shape for k, v in harmonics.items()}
        raise RuntimeError(f"Error occurred while concatenating {shapes=}") from e
