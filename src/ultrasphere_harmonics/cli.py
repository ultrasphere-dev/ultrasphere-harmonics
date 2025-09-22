from collections.abc import Mapping
from typing import Any, Literal

import array_api_extra as xpx
import cyclopts
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import plotly.express as px
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from array_api_compat import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from ultrasphere import create_from_branching_types, create_standard, random_ball, shn1

from ultrasphere_harmonics._core._eigenfunction import Phase
from ultrasphere_harmonics._cut import expand_cut
from ultrasphere_harmonics._expansion import expand, expand_evaluate
from ultrasphere_harmonics._ndim import harm_n_ndim_le

from ._core import index_array_harmonics
from ._helmholtz import harmonics_regular_singular

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
    mesh = o3d.io.read_triangle_mesh(data.path).translate([0.06, -0.12, -0.02])
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
    return t_hit * 10


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
def expand_bunny(
    *,
    n_plot: int = 10000,
    n_end: int = 20,
    phase: Phase = 0,  # type: ignore
    xp: ArrayNamespaceFull = np,
    frontend: Literal["matplotlib", "plotly"] = "matplotlib",
) -> None:
    """Visualize the spherical harmonics expansion of Stanford Bunny."""
    c = create_standard(2)

    def f(spherical: Mapping[Any, Array]) -> Array:
        """Get the distance to the surface."""
        xp = array_namespace(*spherical.values())
        euclidean = c.to_euclidean(spherical)
        return bunny_mesh_dist(
            xp.stack(xp.broadcast_arrays(*[euclidean[i] for i in c.e_nodes]), axis=-1)
        )

    expansion = expand(c, f, False, n_end, 2 * n_end, phase=phase, xp=xp)
    euclidean = random_ball(c, shape=(n_plot,), xp=xp, surface=True)
    spherical = c.from_euclidean(euclidean)
    del spherical["r"]
    keys = ("ground_truth", *tuple(range(1, n_end + 1)))
    data = []
    for key in tqdm(keys, desc="Evaluating the cut expansion"):
        if key == "ground_truth":
            r = f(spherical)
            label = "Ground Truth\nBasis Count: ∞"
        else:
            key = int(key)
            r = xp.real(
                expand_evaluate(
                    c, expand_cut(c, expansion, key), spherical, phase=phase
                )
            )
            label = (
                f"Max Degree: {key - 1:02d}\n"
                f"Basis Count: {harm_n_ndim_le(key - 1, e_ndim=c.e_ndim):03d}"
            )
        data.append(
            {
                "x": c.to_euclidean(spherical | {"r": r}),
                "label": label,
                "key": key,
            }
        )

    if frontend == "plotly":
        # [t, x, y, z]
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "x": d["x"][0],
                        "y": d["x"][1],
                        "z": d["x"][2],
                        "label": d["label"],
                    }
                )
                for d in data
            ]
        )
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="z",
            animation_frame="label",
            range_x=[data[0]["x"][0].min(), data[0]["x"][0].max()],
            range_y=[data[0]["x"][1].min(), data[0]["x"][1].max()],
            range_z=[data[0]["x"][2].min(), data[0]["x"][2].max()],
        )
        fig.update_layout(
            scene={
                "aspectmode": "cube",
            }
        )
        fig.update_traces(marker={"size": 3})
        fig.write_html("expand_bunny.html")
        return

    plt.style.use("dark_background")
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"}, figsize=(4, 4), layout="constrained"
    )

    def animate(data_: dict[str, Any]) -> None:
        ax.clear()
        ax.view_init(elev=45, azim=45, roll=120)
        ax.scatter3D(
            data_["x"][0], data_["x"][1], data_["x"][2], c=data_["x"][2], cmap="viridis"
        )
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xlim(data[0]["x"][0].min(), data[0]["x"][0].max())
        ax.set_ylim(data[0]["x"][1].min(), data[0]["x"][1].max())
        ax.set_zlim(data[0]["x"][2].min(), data[0]["x"][2].max())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(data_["label"])

    anim = FuncAnimation(fig, animate, frames=data, repeat=False, interval=1000 // 3)
    anim.save("expand_bunny.gif", writer="pillow")
    anim.save("expand_bunny.mp4", writer="ffmpeg")


@app.command()
def expand_bunny_4d(
    *,
    n_plot: int = 10000,
    n_end: int = 14,
    phase: Phase = 0,  # type: ignore
    xp: ArrayNamespaceFull = np,
    threshold: float | None = 0.5,
) -> None:
    """Visualize the spherical harmonics expansion of Stanford Bunny."""
    c = create_standard(3)

    def f(spherical: Mapping[Any, Array]) -> Array:
        """Get the distance to the surface."""
        xp = array_namespace(*spherical.values())
        euclidean = c.to_euclidean(spherical)
        # stereographic projection
        denom = 1 - euclidean[0]
        direction = xp.stack(
            xp.broadcast_arrays(
                euclidean[1] / denom,
                euclidean[2] / denom,
                euclidean[3] / denom,
            ),
            axis=-1,
        )
        return bunny_mesh_isin(direction)

    expansion = expand(
        c,
        f,
        False,
        n_end,
        2 * n_end,
        phase=phase,
        xp=xp,
    )

    # plot coordinates
    euclidean = random_ball(create_standard(2), shape=(n_plot,), xp=xp, surface=False)
    r = xp.linalg.vector_norm(euclidean, axis=0)
    denom = r**2 + 1
    euclidean_proj = [
        (r**2 - 1) / denom,
        2 * euclidean[0] / denom,
        2 * euclidean[1] / denom,
        2 * euclidean[2] / denom,
    ]
    spherical_proj = c.from_euclidean(euclidean_proj)
    del spherical_proj["r"]

    # compute the expansion
    keys = ("ground_truth", *tuple(range(1, n_end + 1)))
    data = []
    for key in tqdm(keys, desc="Evaluating the cut expansion"):
        if key == "ground_truth":
            w = f(spherical_proj)
            label = "Ground Truth\nBasis Count: ∞"
        else:
            w = xp.real(
                expand_evaluate(
                    c,
                    expand_cut(c, expansion, key),
                    spherical_proj,
                    phase=phase,
                )
            )
            label = (
                f"Max Degree: {key - 1:02d}\n"
                f"Basis Count: {harm_n_ndim_le(key - 1, e_ndim=c.e_ndim):04d}"
            )
        if threshold is not None:
            w = (w > threshold).astype(w.dtype)
        data.append(
            {
                "w": w,
                "key": key,
                "label": label,
            }
        )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"}, figsize=(4, 4), layout="constrained"
    )

    def animate(data: dict[str, Any]) -> None:
        ax.clear()
        ax.view_init(elev=45, azim=45, roll=120)
        ax.scatter3D(
            euclidean[0],
            euclidean[1],
            euclidean[2],
            s=data["w"] * 10,
            c=euclidean[2],
            cmap="viridis",
        )
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(data["label"])
        return

    anim = FuncAnimation(fig, animate, frames=data, repeat=False, interval=1000 // 3)
    anim.save("expand_bunny_4d.gif", writer="pillow")
    anim.save("expand_bunny_4d.mp4", writer="ffmpeg")


@app.command()
def scattering(branching_types: str, n_end: int = 6, k: float = 1) -> None:
    """Visualize scattering from a sound-soft sphere."""
    c = create_from_branching_types(branching_types)

    def uin(spherical: Mapping[Any, Array]) -> Array:
        euclidean = c.to_euclidean(spherical, as_array=True)
        xp = array_namespace(euclidean[0])
        return shn1(
            xp.asarray(0),
            xp.asarray(c.e_ndim),
            k
            * xp.linalg.vector_norm(
                euclidean
                - xp.asarray([2.0] + [0.0] * (c.e_ndim - 1))[
                    (...,) + (None,) * (euclidean.ndim - 1)
                ],
                axis=0,
            ),
        )

    xp = np
    phase = Phase(0)
    expansion_coef = expand(c, uin, False, n_end, n_end, phase=phase, xp=xp)
    euclidean = xp.meshgrid(
        *((xp.linspace(-3, 3, 100),) * 2 + (xp.asarray([0.0]),)), indexing="ij"
    )
    euclidean = xp.stack([xp.reshape(xi, (-1,)) for xi in euclidean])
    euclidean = euclidean[:, xp.linalg.vector_norm(euclidean, axis=0) > 1.0]
    spherical = c.from_euclidean(euclidean)
    uin_v = uin(spherical)
    n = index_array_harmonics(c, c.root, n_end=n_end, xp=xp, flatten=True)
    uscat_v = -xp.sum(
        1
        / shn1(n, xp.asarray(c.e_ndim), xp.asarray(k))
        * expansion_coef
        * harmonics_regular_singular(
            c, spherical, n_end=n_end, type="singular", k=k, phase=phase
        ),
        axis=-1,
    )
    utot_v = uin_v + uscat_v
    vmin = min(
        xp.min(xp.real(uin_v)), xp.min(xp.real(uscat_v)), xp.min(xp.real(utot_v))
    )
    vmax = max(
        xp.max(xp.real(uin_v)), xp.max(xp.real(uscat_v)), xp.max(xp.real(utot_v))
    )
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    ax[0].scatter(
        euclidean[0],
        euclidean[1],
        c=xp.real(uin_v),
        s=1,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax[0].set_title("Incident Wave")
    ax[1].scatter(
        euclidean[0],
        euclidean[1],
        c=xp.real(uscat_v),
        s=1,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1].set_title("Scattered Wave")
    sc = ax[2].scatter(
        euclidean[0],
        euclidean[1],
        c=xp.real(utot_v),
        s=1,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax[2].set_title("Total Wave")
    fig.colorbar(sc, ax=ax.ravel().tolist())
    fig.savefig(f"scattering_{branching_types}.png")
