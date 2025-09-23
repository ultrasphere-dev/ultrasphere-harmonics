from collections.abc import Mapping

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ultrasphere import BranchingType, SphericalCoordinates, get_child


def _expand_dim_harmoncis[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    node: TSpherical,
    harmonics: Array,
) -> Array:
    """
    Expand the dimension of the harmonics.

    Expand the dimension so that
    all values of the harmonics() result dictionary
    are commomly indexed by the same s_nodes, by default False

    For example, if spherical coordinates,
    - if True, the result will be indexed {"phi": [m], "theta": [m, n]}
    - if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

    Note that the values will not be repeated
    therefore the computational cost will be the same

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    node : TSpherical
        The node of the spherical coordinates.
    harmonics : Array
        The harmonics (eigenfunctions).

    Returns
    -------
    Array
        The expanded harmonics.
        The shapes does not need to be either
        same or broadcastable.

    """
    xp = array_namespace(harmonics)
    idx_node = c.s_nodes.index(node)
    branching_type = c.branching_types[node]
    if branching_type == BranchingType.A:
        moveaxis = {0: idx_node}
    elif branching_type == BranchingType.B:
        idx_sin_child = c.s_nodes.index(get_child(c.G, node, "sin"))
        moveaxis = {
            0: idx_sin_child,
            1: idx_node,
        }
    elif branching_type == BranchingType.BP:
        idx_cos_child = c.s_nodes.index(get_child(c.G, node, "cos"))
        moveaxis = {
            0: idx_cos_child,
            1: idx_node,
        }
    elif branching_type == BranchingType.C:
        idx_cos_child = c.s_nodes.index(get_child(c.G, node, "cos"))
        idx_sin_child = c.s_nodes.index(get_child(c.G, node, "sin"))
        moveaxis = {0: idx_cos_child, 1: idx_sin_child, 2: idx_node}
    value_additional_ndim = harmonics.ndim - len(moveaxis)
    moveaxis = {
        k + value_additional_ndim: v + value_additional_ndim
        for k, v in moveaxis.items()
    }
    adding_ndim = c.s_ndim - len(moveaxis)
    harmonics = harmonics[(...,) + (None,) * adding_ndim]
    return xp.moveaxis(harmonics, list(moveaxis.keys()), list(moveaxis.values()))


def expand_dims_harmonics[TSpherical, TCartesian](
    c: SphericalCoordinates[TSpherical, TCartesian],
    harmonics: Mapping[TSpherical, Array],
) -> Mapping[TSpherical, Array]:
    """
    Expand dimensions of the harmonics.

    Expand dimensions so that
    all values of the harmonics() result dictionary
    are commomly indexed by the same s_nodes, by default False

    For example, if spherical coordinates,
    - if True, the result will be indexed {"phi": [m], "theta": [m, n]}
    - if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

    Note that the values will not be repeated
    therefore the computational cost will be the same

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates.
    harmonics : Mapping[TSpherical, Array]
        The dictionary of harmonics (eigenfunctions).

    Returns
    -------
    Mapping[TSpherical, Array]
        The expanded harmonics.
        The shapes does not need to be either
        same or broadcastable.

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
    ...     expand_dims=False,
    ... )
    >>> {k: harm[k].shape for k in c.s_nodes} # not broadcastable
    {'theta': (3, 2), 'phi': (3,)}

    >>> harm = expand_dims_harmonics(c, harm)
    >>> {k: harm[k].shape for k in c.s_nodes} # broadcastable
    {'theta': (2, 3), 'phi': (1, 3)}

    """
    result: dict[TSpherical, Array] = {}
    for node in c.s_nodes:
        result[node] = _expand_dim_harmoncis(c, node, harmonics[node])
    return result
