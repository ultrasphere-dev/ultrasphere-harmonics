from collections.abc import Mapping

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ultrasphere import SphericalCoordinates


def concat_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Mapping[TSpherical, Array],
) -> Array:
    """
    Concatenate the mapping of expanded harmonics.

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TEuclidean]
        The spherical coordinates.
    harmonics : Mapping[TSpherical, Array]
        The expanded harmonics.

    Returns
    -------
    Array
        The concatenated harmonics.

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
