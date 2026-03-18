from __future__ import annotations

import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from boututils.datafile import DataFile

try:
    import pyvista as pv
except ImportError:
    pv = None


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ZOIDBERG = REPO_ROOT / "external" / "zoidberg"
RUN_DIR = Path(__file__).resolve().parent
for extra_path in [RUN_DIR, EXTERNAL_ZOIDBERG]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from dommaschk_grid_utils import dommaschk  # noqa: E402
from zoidberg import field as zb_field  # noqa: E402
from zoidberg import zoidberg as zz  # noqa: E402


DATA_DIR = RUN_DIR / "data"
GRID_PATH = DATA_DIR / "Dommaschk.fci.nc"
PANEL_PATH = RUN_DIR / "dommaschk_grid_panels.png"
FCI_PANEL_PATH = RUN_DIR / "dommaschk_fci_maps_panels.png"
VTK_STEM = RUN_DIR / "dommaschk_grid_vtk"
PARAVIEW_DIR = RUN_DIR / "paraview_exports"
PANEL_SURFACES_VTK_PATH = PARAVIEW_DIR / "dommaschk_panel_surfaces.vtm"
FORWARD_STITCHED_SHELLS_VTK_PATH = PARAVIEW_DIR / "dommaschk_stitched_shells.vtm"
BACKWARD_STITCHED_SHELLS_VTK_PATH = PARAVIEW_DIR / "dommaschk_stitched_shells_backward.vtm"
WRITE_DIAGNOSTICS = False
OBSOLETE_OUTPUTS = [
    RUN_DIR / "dommaschk_mapped_surfaces.png",
    RUN_DIR / "dommaschk_zero_alignment.png",
    PARAVIEW_DIR / "dommaschk_3d_shells.vtm",
    PARAVIEW_DIR / "dommaschk_outer_surface.vts",
    PARAVIEW_DIR / "dommaschk_zero_trace.vtm",
]
DIAGNOSTIC_OUTPUTS = [
    PANEL_PATH,
    FCI_PANEL_PATH,
    VTK_STEM.with_suffix(".vts"),
    PANEL_SURFACES_VTK_PATH,
    PANEL_SURFACES_VTK_PATH.with_suffix(""),
    FORWARD_STITCHED_SHELLS_VTK_PATH,
    FORWARD_STITCHED_SHELLS_VTK_PATH.with_suffix(""),
    BACKWARD_STITCHED_SHELLS_VTK_PATH,
    BACKWARD_STITCHED_SHELLS_VTK_PATH.with_suffix(""),
]


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "grid.color": "#d0d6dc",
            "grid.alpha": 0.35,
        }
    )


def _load_grid(grid_path: Path) -> dict[str, np.ndarray]:
    with DataFile(str(grid_path)) as grid_file:
        return {name: grid_file.read(name) for name in grid_file.list()}


def _coelho_coefficients() -> np.ndarray:
    coefficients = np.zeros((6, 10, 4))
    coefficients[5, 2, 1] = 1.5
    coefficients[5, 2, 2] = 1.5
    coefficients[5, 4, 1] = 10.0
    coefficients[5, 9, 0] = -7.5e9
    coefficients[5, 9, 3] = 7.5e9
    return coefficients


def _coelho_noislands_coefficients() -> np.ndarray:
    coefficients = np.zeros((6, 5, 4))
    coefficients[5, 2, 1] = 1.5
    coefficients[5, 2, 2] = 1.5
    coefficients[5, 4, 1] = 10.0
    return coefficients


def _default_coefficients() -> np.ndarray:
    coefficients = np.zeros((6, 5, 4))
    coefficients[5, 2, 1] = 0.4
    coefficients[5, 2, 2] = 0.4
    return coefficients


def _resolve_dommaschk_coefficients(choice) -> np.ndarray:
    if isinstance(choice, np.ndarray):
        return np.asarray(choice, dtype=float)
    if choice == "Coelho":
        return _coelho_coefficients()
    if choice == "Coelho_noislands":
        return _coelho_noislands_coefficients()
    return _default_coefficients()


def _make_dommaschk_field(choice, xcentre: float = 1.0, btor: float = 1.0):
    coefficients = _resolve_dommaschk_coefficients(choice)
    return zb_field.DommaschkPotentials(coefficients, R_0=xcentre, B_0=btor)


def _plot_mesh_overlay(ax, r_slice, z_slice, color="#2a3f5f", lw=0.45, alpha=0.65):
    for x_idx in range(r_slice.shape[0]):
        ax.plot(r_slice[x_idx, :], z_slice[x_idx, :], color=color, lw=lw, alpha=alpha)
    for z_idx in range(r_slice.shape[1]):
        ax.plot(r_slice[:, z_idx], z_slice[:, z_idx], color=color, lw=lw, alpha=alpha)


def _make_3d_axes_equal(ax, x, y, z):
    x_limits = np.array([np.nanmin(x), np.nanmax(x)])
    y_limits = np.array([np.nanmin(y), np.nanmax(y)])
    z_limits = np.array([np.nanmin(z), np.nanmax(z)])

    spans = np.array(
        [x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]]
    )
    centers = np.array(
        [x_limits.mean(), y_limits.mean(), z_limits.mean()]
    )
    radius = 0.55 * spans.max()

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _structured_grid_from_xyz(x, y, z):
    if pv is None:
        raise RuntimeError("PyVista is not available")
    if x.ndim == 2:
        x = x[:, None, :]
        y = y[:, None, :]
        z = z[:, None, :]
    grid = pv.StructuredGrid()
    grid.points = np.c_[x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
    grid.dimensions = x.shape
    return grid


def _make_shell_strip(current_r, current_z, mapped_r, mapped_z, phi0, phi1):
    x = np.vstack([current_r * np.cos(phi0), mapped_r * np.cos(phi1)])
    y = np.vstack([current_r * np.sin(phi0), mapped_r * np.sin(phi1)])
    z = np.vstack([current_z, mapped_z])
    return x, y, z


def _make_scalar_strip(start_values: np.ndarray, end_values: np.ndarray) -> np.ndarray:
    return np.vstack([start_values, end_values])


def _remove_stale_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _remove_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _unwrap_periodic_values(values: np.ndarray, period: float) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    unwrapped = np.asarray(values, dtype=float).copy()
    for idx in range(1, unwrapped.size):
        delta = unwrapped[idx] - unwrapped[idx - 1]
        if delta > period / 2.0:
            unwrapped[idx:] -= period
        elif delta < -period / 2.0:
            unwrapped[idx:] += period
    return unwrapped


def _periodic_interpolate_missing(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    index_period: int,
    value_period: float | None = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size < 2 or valid_indices.size == result.size:
        return result

    valid_values = result[valid_mask]
    if value_period is not None:
        valid_values = _unwrap_periodic_values(valid_values, value_period)
        extended_values = np.concatenate(
            [valid_values - value_period, valid_values, valid_values + value_period]
        )
    else:
        extended_values = np.tile(valid_values, 3)

    extended_indices = np.concatenate(
        [valid_indices - index_period, valid_indices, valid_indices + index_period]
    )
    missing_indices = np.flatnonzero(~valid_mask)
    result[missing_indices] = np.interp(missing_indices, extended_indices, extended_values)
    if value_period is not None:
        result %= value_period
    return result


def _repair_invalid_boundary_corner_traces(grid_path: Path, repair_x: int = 1) -> dict[str, int]:
    with DataFile(str(grid_path), write=True) as grid_file:
        forward_xt = np.asarray(grid_file.read("forward_xt_prime"))
        forward_zt = np.asarray(grid_file.read("forward_zt_prime"))
        forward_r = np.asarray(grid_file.read("forward_R"))
        forward_z = np.asarray(grid_file.read("forward_Z"))
        backward_xt = np.asarray(grid_file.read("backward_xt_prime"))
        backward_zt = np.asarray(grid_file.read("backward_zt_prime"))
        backward_r = np.asarray(grid_file.read("backward_R"))
        backward_z = np.asarray(grid_file.read("backward_Z"))

        nx, ny, nz = forward_xt.shape
        repair_x = int(np.clip(repair_x, 0, nx - 1))
        summary: dict[str, int] = {}

        for prefix, xt, zt, mapped_r, mapped_z in [
            ("forward", forward_xt, forward_zt, forward_r, forward_z),
            ("backward", backward_xt, backward_zt, backward_r, backward_z),
        ]:
            invalid = (~np.isfinite(xt[repair_x])) | (xt[repair_x] < 0.0) | (xt[repair_x] > nx - 1)
            summary[f"{prefix}_before"] = int(np.count_nonzero(invalid))

            for y_idx in range(ny):
                bad_mask = invalid[y_idx]
                if not np.any(bad_mask):
                    continue
                valid_mask = ~bad_mask
                if np.count_nonzero(valid_mask) < 2:
                    continue

                xt[repair_x, y_idx, :] = _periodic_interpolate_missing(
                    xt[repair_x, y_idx, :], valid_mask, index_period=nz
                )
                zt[repair_x, y_idx, :] = _periodic_interpolate_missing(
                    zt[repair_x, y_idx, :],
                    valid_mask,
                    index_period=nz,
                    value_period=float(nz),
                )
                mapped_r[repair_x, y_idx, :] = _periodic_interpolate_missing(
                    mapped_r[repair_x, y_idx, :], valid_mask, index_period=nz
                )
                mapped_z[repair_x, y_idx, :] = _periodic_interpolate_missing(
                    mapped_z[repair_x, y_idx, :], valid_mask, index_period=nz
                )

            xt[repair_x] = np.clip(xt[repair_x], 0.0, nx - 1.0)
            zt[repair_x] %= float(nz)

            invalid_after = (~np.isfinite(xt[repair_x])) | (xt[repair_x] < 0.0) | (xt[repair_x] > nx - 1)
            summary[f"{prefix}_after"] = int(np.count_nonzero(invalid_after))

        grid_file.write("forward_xt_prime", forward_xt)
        grid_file.write("forward_zt_prime", forward_zt)
        grid_file.write("forward_R", forward_r)
        grid_file.write("forward_Z", forward_z)
        grid_file.write("backward_xt_prime", backward_xt)
        grid_file.write("backward_zt_prime", backward_zt)
        grid_file.write("backward_R", backward_r)
        grid_file.write("backward_Z", backward_z)

    return summary


def _summarize_invalid_traces(grid_path: Path, target_x: int = 1) -> dict[str, int]:
    with DataFile(str(grid_path)) as grid_file:
        forward_xt = np.asarray(grid_file.read("forward_xt_prime"))
        backward_xt = np.asarray(grid_file.read("backward_xt_prime"))

    nx = forward_xt.shape[0]
    target_x = int(np.clip(target_x, 0, nx - 1))

    summary = {}
    for prefix, xt in [("forward", forward_xt), ("backward", backward_xt)]:
        invalid = (~np.isfinite(xt)) | (xt < 0.0) | (xt > nx - 1)
        summary[f"{prefix}_invalid_total"] = int(np.count_nonzero(invalid))
        summary[f"{prefix}_invalid_x{target_x}"] = int(np.count_nonzero(invalid[target_x]))
        summary[f"{prefix}_invalid_x0"] = int(np.count_nonzero(invalid[0]))
    return summary


def _straighten_boundary_traces(grid_path: Path, boundary_count: int = 2) -> None:
    with DataFile(str(grid_path), write=True) as grid_file:
        forward_xt = np.asarray(grid_file.read("forward_xt_prime"))
        backward_xt = np.asarray(grid_file.read("backward_xt_prime"))

        for x_idx in range(min(boundary_count, forward_xt.shape[0], backward_xt.shape[0])):
            forward_xt[x_idx, :, :] = float(x_idx)
            backward_xt[x_idx, :, :] = float(x_idx)

        grid_file.write("forward_xt_prime", forward_xt)
        grid_file.write("backward_xt_prime", backward_xt)


def _nearest_periodic_shift(delta: int, period: int) -> int:
    if delta > period // 2:
        return delta - period
    if delta < -(period // 2):
        return delta + period
    return delta


def _compute_fci_zero_alignment(forward_zt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = forward_zt.shape
    radial_start = max(0, nx - 4)
    zero_indices = np.zeros(ny, dtype=int)

    for y_idx in range(ny):
        data = forward_zt[radial_start:, y_idx, :]
        wrapped_distance = np.minimum(np.abs(data), np.abs(data - nz))
        score = np.nanmean(wrapped_distance, axis=0)
        zero_indices[y_idx] = int(np.argmin(score))

    zero_unwrapped = np.zeros(ny, dtype=int)
    zero_unwrapped[0] = int(zero_indices[0])
    for y_idx in range(1, ny):
        delta = _nearest_periodic_shift(int(zero_indices[y_idx] - zero_indices[y_idx - 1]), nz)
        zero_unwrapped[y_idx] = zero_unwrapped[y_idx - 1] + delta

    reference = int(zero_unwrapped[0])
    rolls = np.array(
        [reference - int(idx) for idx in zero_unwrapped],
        dtype=int,
    )
    return zero_indices, zero_unwrapped, rolls


def _compute_fci_zero_rolls(forward_zt: np.ndarray) -> np.ndarray:
    return _compute_fci_zero_alignment(forward_zt)[2]


def _apply_z_rolls(data: np.ndarray, rolls: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(data)
    for y_idx, roll in enumerate(rolls):
        aligned[:, y_idx, :] = np.roll(data[:, y_idx, :], int(roll), axis=-1)
    return aligned


def create_publication_panels(grid_path: Path, panel_path: Path, field) -> None:
    grid = _load_grid(grid_path)

    r = np.asarray(grid["R"])
    z = np.asarray(grid["Z"])
    b = np.asarray(grid["B"])
    ly = np.asarray(grid["Ly"])
    forward_xt = np.asarray(grid["forward_xt_prime"])
    forward_zt = np.asarray(grid["forward_zt_prime"])
    forward_r = np.asarray(grid["forward_R"])
    forward_z = np.asarray(grid["forward_Z"])
    dagp_volume = np.asarray(grid["dagp_fv_volume"])
    rolls = _compute_fci_zero_rolls(forward_zt)
    r_aligned = _apply_z_rolls(r, rolls)
    z_aligned = _apply_z_rolls(z, rolls)
    b_aligned = _apply_z_rolls(b, rolls)
    forward_r_aligned = _apply_z_rolls(forward_r, rolls)
    forward_z_aligned = _apply_z_rolls(forward_z, rolls)

    nx, ny, nz = r.shape
    y0 = 0
    sample_slices = np.unique(np.linspace(0, ny - 1, 4, dtype=int))

    r0 = r[:, y0, :]
    z0 = z[:, y0, :]
    b0 = b[:, y0, :]
    ly0 = ly[:, y0, :]
    fxt0 = forward_xt[:, y0, :]
    fzt0 = forward_zt[:, y0, :]
    vol0 = dagp_volume[:, y0, :]

    phi = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    x3 = r_aligned * np.cos(phi[None, :, None])
    y3 = r_aligned * np.sin(phi[None, :, None])
    z3 = z_aligned

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15])

    ax_b = fig.add_subplot(gs[0, 0])
    pcm_b = ax_b.pcolormesh(r0, z0, b0, shading="gouraud", cmap="cividis")
    _plot_mesh_overlay(ax_b, r0, z0)
    fig.colorbar(pcm_b, ax=ax_b, pad=0.02, shrink=0.9, label="|B| [T]")
    ax_b.set_title("Poloidal Mesh and Magnetic Field")
    ax_b.set_xlabel("R [m]")
    ax_b.set_ylabel("Z [m]")
    ax_b.set_aspect("equal")

    ax_slices = fig.add_subplot(gs[0, 1])
    colors = ["#0f4c5c", "#e36414", "#6a994e", "#7b2cbf"]
    outer_idx = nx - 1
    inner_idx = max(1, nx // 4)
    for idx, y_idx in enumerate(sample_slices):
        label = rf"$\phi={phi[y_idx]:.2f}$ rad"
        ax_slices.plot(
            r[outer_idx, y_idx, :],
            z[outer_idx, y_idx, :],
            color=colors[idx],
            lw=2.0,
            label=label,
        )
        ax_slices.plot(
            r[inner_idx, y_idx, :],
            z[inner_idx, y_idx, :],
            color=colors[idx],
            lw=1.0,
            alpha=0.6,
        )
    ax_slices.set_title("Toroidal Progression of Flux Surfaces")
    ax_slices.set_xlabel("R [m]")
    ax_slices.set_ylabel("Z [m]")
    ax_slices.set_aspect("equal")
    ax_slices.grid(True)
    ax_slices.legend(frameon=False, fontsize=9, loc="best")

    ax_3d = fig.add_subplot(gs[0, 2], projection="3d")
    shell_indices = np.unique(np.linspace(0, nx - 1, 4, dtype=int))
    shell_alphas = [0.25, 0.35, 0.55, 0.9]
    b_norm = plt.Normalize(vmin=np.nanmin(b_aligned), vmax=np.nanmax(b_aligned))
    b_cmap = plt.cm.cividis
    for shell_idx, alpha in zip(shell_indices, shell_alphas):
        for y_idx in range(ny):
            next_idx = (y_idx + 1) % ny
            x_strip, y_strip, z_strip = _make_shell_strip(
                r_aligned[shell_idx, y_idx, :],
                z_aligned[shell_idx, y_idx, :],
                forward_r_aligned[shell_idx, y_idx, :],
                forward_z_aligned[shell_idx, y_idx, :],
                phi[y_idx],
                phi[next_idx],
            )
            b_strip = _make_scalar_strip(
                field.Bmag(
                    r_aligned[shell_idx, y_idx, :],
                    z_aligned[shell_idx, y_idx, :],
                    np.full(nz, phi[y_idx]),
                ),
                field.Bmag(
                    forward_r_aligned[shell_idx, y_idx, :],
                    forward_z_aligned[shell_idx, y_idx, :],
                    np.full(nz, phi[next_idx]),
                ),
            )
            ax_3d.plot_surface(
                x_strip,
                y_strip,
                z_strip,
                facecolors=b_cmap(b_norm(b_strip)),
                rstride=1,
                cstride=3,
                shade=False,
                alpha=alpha,
                linewidth=0.0,
                antialiased=True,
            )
    colorbar = plt.cm.ScalarMappable(norm=b_norm, cmap=b_cmap)
    colorbar.set_array([])
    fig.colorbar(colorbar, ax=ax_3d, pad=0.06, shrink=0.8, label="|B| [T]")
    ax_3d.set_title("3D Toroidal Geometry (forward-stitched, colored by |B|)")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.view_init(elev=28, azim=-55)
    _make_3d_axes_equal(ax_3d, x3, y3, z3)

    ax_fxt = fig.add_subplot(gs[1, 0])
    pcm_fxt = ax_fxt.pcolormesh(r0, z0, fxt0, shading="gouraud", cmap="viridis")
    fig.colorbar(pcm_fxt, ax=ax_fxt, pad=0.02, shrink=0.9, label="forward_xt_prime")
    ax_fxt.set_title("Forward FCI Radial Map")
    ax_fxt.set_xlabel("R [m]")
    ax_fxt.set_ylabel("Z [m]")
    ax_fxt.set_aspect("equal")

    ax_fzt = fig.add_subplot(gs[1, 1])
    pcm_fzt = ax_fzt.pcolormesh(r0, z0, fzt0, shading="gouraud", cmap="magma")
    fig.colorbar(pcm_fzt, ax=ax_fzt, pad=0.02, shrink=0.9, label="forward_zt_prime")
    ax_fzt.set_title("Forward FCI Poloidal Map")
    ax_fzt.set_xlabel("R [m]")
    ax_fzt.set_ylabel("Z [m]")
    ax_fzt.set_aspect("equal")

    ax_diag = fig.add_subplot(gs[1, 2])
    pcm_diag = ax_diag.pcolormesh(r0, z0, ly0, shading="gouraud", cmap="plasma")
    fig.colorbar(pcm_diag, ax=ax_diag, pad=0.02, shrink=0.9, label="Ly")
    contours = ax_diag.contour(
        r0,
        z0,
        vol0,
        levels=6,
        colors="#1d3557",
        linewidths=0.9,
        alpha=0.9,
    )
    ax_diag.clabel(contours, inline=True, fontsize=8, fmt="%.2e")
    ax_diag.set_title("FCI Metric Diagnostics")
    ax_diag.set_xlabel("R [m]")
    ax_diag.set_ylabel("Z [m]")
    ax_diag.set_aspect("equal")
    ax_diag.text(
        0.02,
        0.98,
        (
            f"grid: {nx} x {ny} x {nz}\n"
            f"B range: [{np.nanmin(b):.2f}, {np.nanmax(b):.2f}] T\n"
            f"Ly range: [{np.nanmin(ly):.3e}, {np.nanmax(ly):.3e}]"
        ),
        transform=ax_diag.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 5},
    )

    fig.suptitle("Dommaschk FCI Grid Diagnostics", fontsize=18, y=1.01)
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)


def create_fci_panels(grid_path: Path, panel_path: Path) -> None:
    grid = _load_grid(grid_path)

    r = np.asarray(grid["R"])
    z = np.asarray(grid["Z"])
    forward_xt = np.asarray(grid["forward_xt_prime"])
    forward_zt = np.asarray(grid["forward_zt_prime"])
    backward_xt = np.asarray(grid["backward_xt_prime"])
    backward_zt = np.asarray(grid["backward_zt_prime"])
    ly = np.asarray(grid["Ly"])
    jacobian = np.asarray(grid["J"])
    b = np.asarray(grid["B"])
    bxcv = np.asarray(grid["bxcv"])

    y0 = 0
    r0 = r[:, y0, :]
    z0 = z[:, y0, :]

    datasets = [
        (forward_xt[:, y0, :], "Forward xt'", "viridis"),
        (forward_zt[:, y0, :], "Forward zt'", "magma"),
        (backward_xt[:, y0, :], "Backward xt'", "viridis"),
        (backward_zt[:, y0, :], "Backward zt'", "magma"),
        (ly[:, y0, :], "Ly", "plasma"),
        (jacobian[:, y0, :], "J", "cividis"),
        (b[:, y0, :], "|B|", "inferno"),
        (bxcv[:, y0, :], "|b x kappa|", "cubehelix"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for ax, (data, title, cmap) in zip(axes.flat, datasets):
        pcm = ax.pcolormesh(r0, z0, data, shading="gouraud", cmap=cmap)
        _plot_mesh_overlay(ax, r0, z0, lw=0.25, alpha=0.35)
        fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")

    fig.suptitle("Forward/Backward FCI Maps and Metric Terms", fontsize=18, y=1.01)
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)


def _build_stitched_shell_blocks(
    shell_indices: np.ndarray,
    phi: np.ndarray,
    base_r: np.ndarray,
    base_z: np.ndarray,
    mapped_r: np.ndarray,
    mapped_z: np.ndarray,
    field,
    output_path: Path,
    direction: str,
) -> Path:
    if pv is None:
        raise RuntimeError("PyVista is not available")

    ny = base_r.shape[1]
    shell_blocks = pv.MultiBlock()
    step = 1 if direction == "forward" else -1

    for shell_idx in shell_indices:
        for y_idx in range(ny):
            target_idx = (y_idx + step) % ny
            x_strip, y_strip, z_strip = _make_shell_strip(
                base_r[shell_idx, y_idx, :],
                base_z[shell_idx, y_idx, :],
                mapped_r[shell_idx, y_idx, :],
                mapped_z[shell_idx, y_idx, :],
                phi[y_idx],
                phi[target_idx],
            )
            b_strip = _make_scalar_strip(
                field.Bmag(
                    base_r[shell_idx, y_idx, :],
                    base_z[shell_idx, y_idx, :],
                    np.full(base_r.shape[-1], phi[y_idx]),
                ),
                field.Bmag(
                    mapped_r[shell_idx, y_idx, :],
                    mapped_z[shell_idx, y_idx, :],
                    np.full(base_r.shape[-1], phi[target_idx]),
                ),
            )
            strip_grid = _structured_grid_from_xyz(x_strip, y_strip, z_strip)
            strip_grid["B"] = b_strip.ravel(order="F")
            strip_grid["shell_index"] = np.full(strip_grid.n_points, shell_idx)
            strip_grid["source_y_index"] = np.full(strip_grid.n_points, y_idx)
            strip_grid["target_y_index"] = np.full(strip_grid.n_points, target_idx)
            strip_grid["phi_start"] = np.full(strip_grid.n_points, phi[y_idx])
            strip_grid["phi_end"] = np.full(strip_grid.n_points, phi[target_idx])
            shell_blocks[f"{direction}_shell_{shell_idx}_strip_{y_idx:02d}"] = strip_grid

    shell_blocks.save(output_path)
    return output_path


def export_paraview_surfaces(grid_path: Path, output_dir: Path, field) -> list[Path]:
    if pv is None:
        print("ParaView surface export skipped: PyVista is not available")
        return []

    output_dir.mkdir(exist_ok=True)
    grid = _load_grid(grid_path)

    r = np.asarray(grid["R"])
    z = np.asarray(grid["Z"])
    b = np.asarray(grid["B"])
    ly = np.asarray(grid["Ly"])
    jacobian = np.asarray(grid["J"])
    forward_xt = np.asarray(grid["forward_xt_prime"])
    forward_zt = np.asarray(grid["forward_zt_prime"])
    backward_xt = np.asarray(grid["backward_xt_prime"])
    backward_zt = np.asarray(grid["backward_zt_prime"])
    bxcv = np.asarray(grid["bxcv"])
    forward_r = np.asarray(grid["forward_R"])
    forward_z = np.asarray(grid["forward_Z"])
    backward_r = np.asarray(grid["backward_R"])
    backward_z = np.asarray(grid["backward_Z"])
    _, _, rolls = _compute_fci_zero_alignment(forward_zt)
    r_aligned = _apply_z_rolls(r, rolls)
    z_aligned = _apply_z_rolls(z, rolls)
    b_aligned = _apply_z_rolls(b, rolls)
    ly_aligned = _apply_z_rolls(ly, rolls)
    jacobian_aligned = _apply_z_rolls(jacobian, rolls)
    forward_r_aligned = _apply_z_rolls(forward_r, rolls)
    forward_z_aligned = _apply_z_rolls(forward_z, rolls)
    backward_r_aligned = _apply_z_rolls(backward_r, rolls)
    backward_z_aligned = _apply_z_rolls(backward_z, rolls)

    nx, ny, _ = r.shape
    phi = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)

    shell_indices = np.unique(np.linspace(0, nx - 1, 4, dtype=int))
    shell_paths = []

    panel_blocks = pv.MultiBlock()
    bxcv_aligned = _apply_z_rolls(bxcv, rolls)
    for y_idx in range(ny):
        angle = phi[y_idx]
        r_panel = r_aligned[:, y_idx : y_idx + 1, :]
        z_panel = z_aligned[:, y_idx : y_idx + 1, :]
        x_panel = r_panel * np.cos(angle)
        y_panel = r_panel * np.sin(angle)
        panel_grid = _structured_grid_from_xyz(x_panel, y_panel, z_panel)
        panel_grid["B"] = b_aligned[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["forward_xt_prime"] = forward_xt[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["forward_zt_prime"] = forward_zt[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["backward_xt_prime"] = backward_xt[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["backward_zt_prime"] = backward_zt[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["Ly"] = ly_aligned[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["J"] = jacobian_aligned[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["bxcv"] = bxcv_aligned[:, y_idx : y_idx + 1, :].ravel(order="F")
        panel_grid["panel_phi"] = np.full(panel_grid.n_points, angle)
        panel_grid["panel_y_index"] = np.full(panel_grid.n_points, y_idx)
        panel_blocks[f"panel_{y_idx:02d}"] = panel_grid
    panel_blocks.save(PANEL_SURFACES_VTK_PATH)
    shell_paths.append(PANEL_SURFACES_VTK_PATH)

    shell_paths.append(
        _build_stitched_shell_blocks(
            shell_indices,
            phi,
            r_aligned,
            z_aligned,
            forward_r_aligned,
            forward_z_aligned,
            field,
            FORWARD_STITCHED_SHELLS_VTK_PATH,
            "forward",
        )
    )
    shell_paths.append(
        _build_stitched_shell_blocks(
            shell_indices,
            phi,
            r_aligned,
            z_aligned,
            backward_r_aligned,
            backward_z_aligned,
            field,
            BACKWARD_STITCHED_SHELLS_VTK_PATH,
            "backward",
        )
    )

    return shell_paths


def export_vtk(grid_path: Path, vtk_stem: Path) -> Path | None:
    if not getattr(zz, "have_evtk", False):
        print("VTK export skipped: PyEVTK is not available in this environment")
        return None

    grid = _load_grid(grid_path)
    r = np.asarray(grid["R"])
    z = np.asarray(grid["Z"])
    b = np.asarray(grid["B"])
    ly = np.asarray(grid["Ly"])
    forward_zt = np.asarray(grid["forward_zt_prime"])
    _, _, rolls = _compute_fci_zero_alignment(forward_zt)
    r_aligned = _apply_z_rolls(r, rolls)
    z_aligned = _apply_z_rolls(z, rolls)
    b_aligned = _apply_z_rolls(b, rolls)
    ly_aligned = _apply_z_rolls(ly, rolls)

    _, ny, _ = r.shape
    phi = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    x = r_aligned * np.cos(phi[None, :, None])
    y = r_aligned * np.sin(phi[None, :, None])

    try:
        vtk_path = zz.gridToVTK(
            str(vtk_stem),
            x,
            y,
            z_aligned,
            pointData={"B": b_aligned, "Ly": ly_aligned},
        )
        return Path(vtk_path)
    except Exception as exc:
        print(f"VTK export skipped: {exc}")
        return None


def main() -> None:
    _set_publication_style()
    DATA_DIR.mkdir(exist_ok=True)
    _remove_stale_outputs(OBSOLETE_OUTPUTS)
    if WRITE_DIAGNOSTICS:
        PARAVIEW_DIR.mkdir(exist_ok=True)
    else:
        _remove_paths(DIAGNOSTIC_OUTPUTS)

    # Preset options for Dommaschk coefficient sets.
    # dommaschk_choice = None
    dommaschk_choice = "Coelho"
    # dommaschk_choice = "Coelho_noislands"

    # Uncomment and pass a custom NumPy coefficient matrix instead of a preset.
    # custom_C = np.zeros((6, 5, 4))
    # custom_C[5, 2, 1] = 0.8
    # custom_C[5, 2, 2] = 0.8
    # custom_C[5, 4, 1] = 4.0
    # coefficient_set = custom_C

    coefficient_set = dommaschk_choice
    magnetic_field = _make_dommaschk_field(coefficient_set)

    print(f"Generating Dommaschk grid at {GRID_PATH} using preset: {coefficient_set}")
    dommaschk(
        nx=68,
        ny=16,
        nz=128,
        C=coefficient_set,
        fname=str(GRID_PATH),
        curvilinear=True,
        rectangular=False,
        calc_curvature=True,
        smooth_curvature=False,
        return_iota=True,
        write_iota=True,
    )
    repair_summary = _repair_invalid_boundary_corner_traces(GRID_PATH, repair_x=1)
    invalid_summary = _summarize_invalid_traces(GRID_PATH, target_x=1)
    print(
        "Applied targeted x=1 trace repair: "
        f"forward {repair_summary['forward_before']} -> {repair_summary['forward_after']}, "
        f"backward {repair_summary['backward_before']} -> {repair_summary['backward_after']}"
    )
    print(
        "Invalid trace summary: "
        f"forward total={invalid_summary['forward_invalid_total']} x0={invalid_summary['forward_invalid_x0']} x1={invalid_summary['forward_invalid_x1']}; "
        f"backward total={invalid_summary['backward_invalid_total']} x0={invalid_summary['backward_invalid_x0']} x1={invalid_summary['backward_invalid_x1']}"
    )

    print(f"Grid ready for Hermes at {GRID_PATH}")
    if WRITE_DIAGNOSTICS:
        print(f"Creating publication-style grid panels: {PANEL_PATH}")
        create_publication_panels(GRID_PATH, PANEL_PATH, magnetic_field)
        print(f"Creating focused FCI panels: {FCI_PANEL_PATH}")
        create_fci_panels(GRID_PATH, FCI_PANEL_PATH)

        vtk_path = export_vtk(GRID_PATH, VTK_STEM)
        if vtk_path:
            print(f"VTK export written to {vtk_path}")

        paraview_paths = export_paraview_surfaces(GRID_PATH, PARAVIEW_DIR, magnetic_field)
        for path in paraview_paths:
            print(f"ParaView export written to {path}")

        print(f"Diagnostics figure saved to {PANEL_PATH}")
        print(f"FCI diagnostics figure saved to {FCI_PANEL_PATH}")


if __name__ == "__main__":
    main()