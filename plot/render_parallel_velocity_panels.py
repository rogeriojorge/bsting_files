from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from boutdata.collect import collect
from boututils.datafile import DataFile
from matplotlib.animation import FuncAnimation


PLOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLOT_DIR.parent
DATA_DIR = REPO_ROOT / "run_stellarator" / "data"
GRID_PATH = DATA_DIR / "Dommaschk.fci.nc"
OUTPUT_DIR = PLOT_DIR / "outputs"
MOVIE_PATH = OUTPUT_DIR / "parallel_velocity_panel_movie.mp4"
SNAPSHOT_PATH = OUTPUT_DIR / "parallel_velocity_panel_snapshots.png"


def _load_data():
    shard_paths = sorted(DATA_DIR.glob("BOUT.dmp.[0-9]*.nc"))
    combined_path = DATA_DIR / "BOUT.dmp.nc"
    combined_mtime = combined_path.stat().st_mtime if combined_path.exists() else float("-inf")

    if shard_paths and max(path.stat().st_mtime for path in shard_paths) >= combined_mtime:
        ve = np.asarray(collect("Ve", path=str(DATA_DIR), prefix="BOUT.dmp", xguards=True, yguards=True, info=False))
        with DataFile(str(shard_paths[0])) as data_file:
            t_array = np.asarray(data_file.read("t_array"))
        print(f"Using MPI shard dumps from {DATA_DIR}")
    else:
        with DataFile(str(combined_path)) as data_file:
            ve = np.asarray(data_file.read("Ve"))
            t_array = np.asarray(data_file.read("t_array"))
        print(f"Using combined dump from {combined_path}")

    with DataFile(str(GRID_PATH)) as grid_file:
        r = np.asarray(grid_file.read("R"))
        z = np.asarray(grid_file.read("Z"))

    nx = min(ve.shape[1], r.shape[0])
    ny = min(ve.shape[2], r.shape[1])
    nz = min(ve.shape[3], r.shape[2])
    return ve[:, :nx, :ny, :nz], t_array, r[:nx, :ny, :nz], z[:nx, :ny, :nz]


def _panel_indices(ny: int) -> np.ndarray:
    return np.unique(np.linspace(0, ny - 1, 4, dtype=int))


def _time_indices(nt: int) -> np.ndarray:
    return np.array([0, nt // 2, nt - 1], dtype=int)


def _ve_bounds(ve: np.ndarray, panel_indices: np.ndarray) -> tuple[float, float]:
    ve_subset = ve[:, :, panel_indices, :]
    vmin = float(np.nanpercentile(ve_subset, 5))
    vmax = float(np.nanpercentile(ve_subset, 95))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_panel(ax, r_slice, z_slice, values, title, vmin, vmax):
    mesh = ax.pcolormesh(r_slice, z_slice, values, shading="gouraud", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    return mesh


def create_parallel_velocity_movie(ve: np.ndarray, t_array: np.ndarray, r: np.ndarray, z: np.ndarray) -> None:
    nx, ny, nz = r.shape
    panel_indices = _panel_indices(ny)
    vmin, vmax = _ve_bounds(ve, panel_indices)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.flatten()
    meshes = []

    for ax, y_idx in zip(axes, panel_indices):
        mesh = _plot_panel(
            ax,
            r[:, y_idx, :],
            z[:, y_idx, :],
            ve[0, :, y_idx, :],
            rf"Parallel Velocity $\phi={2*np.pi*y_idx/ny:.2f}$ rad",
            vmin,
            vmax,
        )
        fig.colorbar(mesh, ax=ax)
        meshes.append(mesh)

    total_frames = min(len(t_array), 100)
    time_indices = np.linspace(0, len(t_array) - 1, total_frames, dtype=int)

    def update(frame_idx):
        t_idx = time_indices[frame_idx]
        for mesh, y_idx in zip(meshes, panel_indices):
            mesh.set_array(ve[t_idx, :, y_idx, :].ravel())
        fig.suptitle(f"Parallel Velocity at FCI Panels | t = {t_array[t_idx]:.2e} s", fontsize=18)
        return tuple(meshes)

    print(f"Rendering parallel velocity movie ({total_frames} frames)...")
    animation = FuncAnimation(fig, update, frames=total_frames, blit=False)
    animation.save(str(MOVIE_PATH), writer="ffmpeg", fps=10, dpi=120)
    plt.close(fig)
    print(f"Parallel velocity movie saved to {MOVIE_PATH}")


def create_parallel_velocity_snapshots(ve: np.ndarray, t_array: np.ndarray, r: np.ndarray, z: np.ndarray) -> None:
    nx, ny, nz = r.shape
    panel_indices = _panel_indices(ny)
    snapshot_indices = _time_indices(len(t_array))
    vmin, vmax = _ve_bounds(ve, panel_indices)

    fig, axes = plt.subplots(3, len(panel_indices), figsize=(18, 12), constrained_layout=True)

    for row_idx, t_idx in enumerate(snapshot_indices):
        for col_idx, y_idx in enumerate(panel_indices):
            ax = axes[row_idx, col_idx]
            mesh = _plot_panel(
                ax,
                r[:, y_idx, :],
                z[:, y_idx, :],
                ve[t_idx, :, y_idx, :],
                rf"t={t_array[t_idx]:.2e} s, $\phi={2*np.pi*y_idx/ny:.2f}$",
                vmin,
                vmax,
            )
            if col_idx == len(panel_indices) - 1:
                fig.colorbar(mesh, ax=ax)

    fig.suptitle("Parallel Velocity Snapshots at Beginning, Middle, and End", fontsize=18)
    fig.savefig(SNAPSHOT_PATH, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Parallel velocity snapshots saved to {SNAPSHOT_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading data for parallel velocity panels...")
    ve, t_array, r, z = _load_data()
    create_parallel_velocity_movie(ve, t_array, r, z)
    create_parallel_velocity_snapshots(ve, t_array, r, z)


if __name__ == "__main__":
    main()