from collections.abc import Mapping
from typing import Any

import array_api_extra as xpx
import cyclopts
import matplotlib.pyplot as plt
import open3d as o3d
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from array_api_compat import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from ultrasphere import create_standard, roots

from ultrasphere_harmonics._core._eigenfunction import Phase
from ultrasphere_harmonics._cut import expand_cut
from ultrasphere_harmonics._expansion import expand, expand_evaluate

app = cyclopts.App(__name__)


def bunny_mesh_dist(direction: Array, /, *, max_radius: float = 100) -> Array:
    """
    Compute the distance to the bunny mesh along the given direction.

    Parameters
    ----------
    direction : Array
        The direction to cast the ray of shape (..., 3).
    max_radius : float, optional
        The maximum radius of the mesh, by default 100

    Returns
    -------
    Array
        The distance to the mesh along the given direction of shape (...,).

    """
    data = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(data.path).translate([0.03, -0.08, 0])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    outgoing = False
    xp = array_namespace(direction)
    if outgoing:
        rays = xp.concat([direction, xp.zeros_like(direction)], axis=-1)
    else:
        rays = xp.concat([direction * max_radius, -direction], axis=-1)
    rays_ = o3d.core.Tensor(np.asarray(rays), dtype=o3d.core.Dtype.Float32)
    answer = scene.cast_rays(rays_)
    t_hit = xp.asarray(answer["t_hit"].numpy())
    if not outgoing:
        t_hit = max_radius - t_hit
    t_hit = xp.clip(t_hit, 0, max_radius)
    t_hit = xpx.nan_to_num(t_hit)
    return t_hit


def bunny_mesh_isin(x: Array, /) -> Array:
    """
    Check if the points are inside the bunny mesh.

    Parameters
    ----------
    x : Array
        The points to check of shape (..., 3).

    Returns
    -------
    Array
        A boolean array of shape (...) indicating if the points are inside the mesh.

    """
    xp = array_namespace(x)
    x_norm = xp.linalg.vector_norm(x, axis=-1, keepdims=True)
    direction = x / x_norm
    dist = bunny_mesh_dist(direction)
    return x_norm[..., 0] < dist


@app.command()
def _plot_3d(
    *,
    n_plot: int = 100,
    n_end: int = 40,
    r: float = 100,
    phase: Phase = Phase(0),  # noqa
    xp: ArrayNamespaceFull = np,
) -> None:
    """Visualize the spherical harmonics expansion."""
    c = create_standard(2)

    def f(spherical: Mapping[Any, Array]) -> Array:
        """Get the distance to the surface."""
        euclidean = c.to_euclidean(spherical)
        return bunny_mesh_dist(euclidean)

    expansion = expand(c, f, False, n_end, 2 * n_end, phase=phase, xp=xp)
    spherical, _ = roots(c, n_plot, expand_dims_x=True, xp=xp)
    keys = ("ground_truth", *tuple(range(n_end)))
    xs = []
    rmax = 0
    for key in tqdm(keys, desc="Evaluating the cut expansion"):
        if key == "ground_truth":
            r = f(spherical)
            rmax = xp.max(r)
            label = "Ground Truth"
        else:
            key = int(key)
            r = xp.real(
                expand_evaluate(
                    c, expand_cut(c, expansion, key), spherical, phase=phase
                )
            )
            label = f"Degree: {key}"
        xs.append(
            {
                "x": c.to_euclidean(spherical | {"r": r}),
                "label": label,
                "key": key,
            }
        )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    def animate(data: dict[str, Any]) -> None:
        ax.clear()
        ax.view_init(elev=45, azim=45, roll=120)
        ax.scatter3D(
            data["x"][0], data["x"][1], data["x"][2], c=data["x"][2], cmap="viridis"
        )
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_zlim(-rmax, rmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(data["label"])

    anim = FuncAnimation(fig, animate, frames=xs, repeat=False, interval=200)
    anim.save("spherical_harmonics_expanation.gif", writer="pillow")
    anim.save("spherical_harmonics_expanation.mp4", writer="ffmpeg")


# @app.command()
# def plot_4d(
#     *,
#     n_plot: int = 100,
#     n_end: int = 40,
#     r: float = 100,
#     phase: Phase = Phase(0),
#     xp: ArrayNamespaceFull = np,
#     frontend: Literal["matplotlib", "plotly"] = "plotly",
# ) -> None:
#     """Visualize the spherical harmonics expansion."""
#     if n_integrate < n_end:
#         raise ValueError(
#             f"n_integrate({n_integrate}) must be greater than n_end({n_end}), "
#             "otherwise the result will not converge."
#         )
#     if n_plot < n_end:
#         raise ValueError(
#             f"n_plot({n_plot}) must be greater than n_end({n_end}), "
#             "otherwise you may not see the result."
#         )

#     if type == "3d":
#         c = create_standard(2)
#     else:
#         c = create_standard(3)


#     def f(spherical: Mapping[Any, Array]) -> Array:
#         """Get the distance to the surface."""
#         xp = array_namespace(*spherical.values())
#         euclidean = c.to_euclidean(spherical)
#         if type == "3d":
#             return bunny_mesh_dist(euclidean)
#         else:
#             # stereographic projection
#             denom = 1 - euclidean[0, ...]
#             direction = xp.stack((
#                 euclidean[1] / denom,
#                 euclidean[2] / denom,
#                 euclidean[3] / denom,
#             ), axis=0)
#             return bunny_mesh_isin(direction)

#     expansion = expand(
#         c,
#         f,
#         n_end=n_end,
#         n=n_integrate,
#         does_f_support_separation_of_variables=False,
#         condon_shortley_phase=False,
#     )

#     # plot coordinates
#     if type == "3d":
#         spherical, _ = roots(c, n_plot, expand_dims_x=True, xp=xp)
#     else:
#         euclidean = xp.meshgrid(
#             (xp.linspace(-1, 1, n_plot),) * 3, indexing="ij"
#         )
#         spherical = c.from_euclidean(euclidean)

#     # compute the expansion
#     keys = ("ground_truth",) + tuple(range(n_end))
#     xs = {}
#     ws = {}
#     for key in tqdm(keys, desc="Evaluating the cut expansion"):
#         if type == "3d":
#             if key == "ground_truth":
#                 r = f(spherical)
#             else:
#                 key = int(key)
#                 r = expand_evaluate(
#                     c,
#                     expand_cut(expansion, key),
#                     spherical,
#                 ).real
#             xs[key] = c.to_euclidean(spherical | {"r": r})
#         else:

#             if key == "ground_truth":
#                 w = f(spherical)
#             else:
#                 w = expand_evaluate(
#                     c,
#                     expand_cut(expansion, key),
#                     spherical,
#                 ).real


#     if frontend == "matplotlib":
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#         def animate(coordinates: dict[str, Array]) -> None:
#             n, x, y, z = (
#                 coordinates["n"],
#                 coordinates["x"],
#                 coordinates["y"],
#                 coordinates["z"],
#             )
#             n, x, y, z = (
#                 xp.to_numpy(n),
#                 xp.to_numpy(x),
#                 xp.to_numpy(y),
#                 xp.to_numpy(z),
#             )
#             n_end = n.flatten()[0]
#             ax.clear()
#             ax.view_init(elev=45, azim=45, roll=120)
#             ax.scatter3D(x, y, z, c=z, cmap="viridis")
#             ax.set_xlim(lims["x"])
#             ax.set_ylim(lims["y"])
#             ax.set_zlim(lims["z"])
#             ax.set_xlabel("x")
#             ax.set_ylabel("y")
#             ax.set_zlabel("z")
#             if xp.isnan(n_end):
#                 ax.set_title("Original")
#             else:
#                 ax.set_title(f"Degree: {n_end}")
#             return

#         anim = FuncAnimation(
#             fig, animate, frames=coordinates, repeat=False, interval=200
#         )
#         anim.save("spherical_harmonics_expanation.gif", writer="pillow")
#         anim.save("spherical_harmonics_expanation.mp4", writer="ffmpeg")
