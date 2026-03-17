from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from boututils.datafile import DataFile
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from tqdm import tqdm

from zoidberg import field as zb_field
from zoidberg import fieldtracer as zb_fieldtracer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "run_stellarator" / "data"
PARAVIEW_DIR = BASE_DIR / "run_stellarator" / "paraview_exports"
GRID_PATH = DATA_DIR / "Dommaschk.fci.nc"
SQUASHED_PATH = DATA_DIR / "BOUT.dmp.nc"
MOVIE_PATH = BASE_DIR / "te_3d_pyvista.mp4"
TRACED_SURFACES_PATH = PARAVIEW_DIR / "traced_movie_surfaces.vtm"

X_CENTRE = 1.0
B_TOR = 1.0
TOROIDAL_TURNS = 16
TOROIDAL_SAMPLES_PER_TURN = 96
WINDOW_SIZE = [1280, 1008]
SURFACE_FIELD_NAMES = ["Te", "Ne", "Ve", "Pe", "Vh+", "NVh+"]


def _coelho_coefficients() -> np.ndarray:
    coefficients = np.zeros((6, 10, 4))
    coefficients[5, 2, 1] = 1.5
    coefficients[5, 2, 2] = 1.5
    coefficients[5, 4, 1] = 10.0
    coefficients[5, 9, 0] = -7.5e9
    coefficients[5, 9, 3] = 7.5e9
    return coefficients


def _structured_grid_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> pv.StructuredGrid:
    if x.ndim == 2:
        x = x[:, None, :]
        y = y[:, None, :]
        z = z[:, None, :]
    grid = pv.StructuredGrid()
    grid.points = np.c_[x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
    grid.dimensions = x.shape
    return grid


def _polyline_from_points(points: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData(points)
    poly.lines = np.hstack([[len(points)], np.arange(len(points), dtype=np.int32)])
    return poly


def _build_inverse_plane_maps(
    r: np.ndarray,
    z: np.ndarray,
) -> list[tuple[LinearNDInterpolator, LinearNDInterpolator, NearestNDInterpolator, NearestNDInterpolator]]:
    nx, ny, nz = r.shape
    x_coords = np.broadcast_to(np.arange(nx)[:, None], (nx, nz)).ravel()
    z_coords = np.broadcast_to(np.arange(nz)[None, :], (nx, nz)).ravel()
    inverse_maps = []
    for y_idx in range(ny):
        points = np.column_stack([r[:, y_idx, :].ravel(), z[:, y_idx, :].ravel()])
        inverse_maps.append(
            (
                LinearNDInterpolator(points, x_coords, fill_value=np.nan),
                LinearNDInterpolator(points, z_coords, fill_value=np.nan),
                NearestNDInterpolator(points, x_coords),
                NearestNDInterpolator(points, z_coords),
            )
        )
    return inverse_maps


def _evaluate_inverse_map(
    inverse_map: tuple[LinearNDInterpolator, LinearNDInterpolator, NearestNDInterpolator, NearestNDInterpolator],
    r_values: np.ndarray,
    z_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_linear, z_linear, x_nearest, z_nearest = inverse_map
    x_query = np.asarray(x_linear(r_values, z_values), dtype=float)
    z_query = np.asarray(z_linear(r_values, z_values), dtype=float)

    invalid = np.isnan(x_query) | np.isnan(z_query)
    if np.any(invalid):
        x_query[invalid] = np.asarray(x_nearest(r_values[invalid], z_values[invalid]), dtype=float)
        z_query[invalid] = np.asarray(z_nearest(r_values[invalid], z_values[invalid]), dtype=float)
    return x_query, z_query


def _sample_plane(field_2d: np.ndarray, x_query: np.ndarray, z_query: np.ndarray) -> np.ndarray:
    x0 = np.floor(x_query).astype(int)
    z0 = np.floor(z_query).astype(int)
    x1 = np.clip(x0 + 1, 0, field_2d.shape[0] - 1)
    z1 = np.clip(z0 + 1, 0, field_2d.shape[1] - 1)
    x0 = np.clip(x0, 0, field_2d.shape[0] - 1)
    z0 = np.clip(z0, 0, field_2d.shape[1] - 1)

    fx = np.clip(x_query - x0, 0.0, 1.0)
    fz = np.clip(z_query - z0, 0.0, 1.0)

    v00 = field_2d[x0, z0]
    v10 = field_2d[x1, z0]
    v01 = field_2d[x0, z1]
    v11 = field_2d[x1, z1]

    return (
        (1.0 - fx) * (1.0 - fz) * v00
        + fx * (1.0 - fz) * v10
        + (1.0 - fx) * fz * v01
        + fx * fz * v11
    )


def _sample_volume_on_surface(
    volume: np.ndarray,
    r_surface: np.ndarray,
    z_surface: np.ndarray,
    phi_surface: np.ndarray,
    inverse_maps: list[tuple[LinearNDInterpolator, LinearNDInterpolator, NearestNDInterpolator, NearestNDInterpolator]],
) -> np.ndarray:
    _, ny, _ = volume.shape
    yperiod = 2.0 * np.pi
    dphi = yperiod / ny
    sampled = np.empty_like(r_surface)

    for phi_idx in range(phi_surface.shape[1]):
        phi_mod = np.remainder(phi_surface[0, phi_idx], yperiod)
        lower_idx = int(np.floor(phi_mod / dphi)) % ny
        upper_idx = (lower_idx + 1) % ny
        frac = (phi_mod - lower_idx * dphi) / dphi

        r_column = r_surface[:, phi_idx]
        z_column = z_surface[:, phi_idx]
        x_lower, z_lower = _evaluate_inverse_map(inverse_maps[lower_idx], r_column, z_column)
        x_upper, z_upper = _evaluate_inverse_map(inverse_maps[upper_idx], r_column, z_column)

        lower_values = _sample_plane(volume[:, lower_idx, :], x_lower, z_lower)
        upper_values = _sample_plane(volume[:, upper_idx, :], x_upper, z_upper)
        sampled[:, phi_idx] = (1.0 - frac) * lower_values + frac * upper_values

    return sampled


def _trace_surface(field, start_r: np.ndarray, start_z: np.ndarray) -> dict[str, np.ndarray]:
    tracer = zb_fieldtracer.FieldTracer(field)
    phi_trace = np.linspace(
        0.0,
        TOROIDAL_TURNS * 2.0 * np.pi,
        TOROIDAL_TURNS * TOROIDAL_SAMPLES_PER_TURN,
        endpoint=False,
    )
    traced = tracer.follow_field_lines(start_r, start_z, phi_trace, rtol=1e-10)
    r_surface = traced[:, :, 0].T
    z_surface = traced[:, :, 1].T
    phi_surface = np.broadcast_to(phi_trace[None, :], r_surface.shape)
    x_surface = r_surface * np.cos(phi_surface)
    y_surface = r_surface * np.sin(phi_surface)
    b_surface = field.Bmag(r_surface, z_surface, np.remainder(phi_surface, 2.0 * np.pi))
    return {
        "r": r_surface,
        "z": z_surface,
        "phi": phi_surface,
        "x": x_surface,
        "y": y_surface,
        "b": b_surface,
    }


def _traced_lines_path(surface_name: str) -> Path:
    return PARAVIEW_DIR / f"traced_field_lines_{surface_name}.vtm"


def _surface_specs(nx: int) -> list[tuple[str, slice, tuple[int, int]]]:
    middle_width = max(8, nx // 4)
    middle_start = min(max(4, nx // 3), max(4, nx - middle_width - 2))
    middle_stop = min(nx - 2, middle_start + middle_width)
    outer_width = max(10, nx // 4)
    outer_start = max(middle_stop + 2, nx - outer_width)
    return [
        ("middle", slice(middle_start, middle_stop), (-1, 0)),
        ("outer", slice(outer_start, nx), (2, -3)),
    ]


def _extract_surface_segments(
    surface: dict[str, np.ndarray],
    ny: int,
    plane_bounds: tuple[int, int],
) -> list[dict[str, np.ndarray]]:
    phi_full = surface["phi"][0]
    phi_mod = np.remainder(phi_full, 2.0 * np.pi)
    turn_index = np.floor(phi_full / (2.0 * np.pi)).astype(int)
    dphi = 2.0 * np.pi / ny

    start_plane, end_plane = plane_bounds
    start_plane %= ny
    end_plane %= ny

    if end_plane == (start_plane + 1) % ny:
        gap_start = start_plane * dphi
        gap_end = end_plane * dphi

        def keep_mask(values: np.ndarray) -> np.ndarray:
            if gap_start < gap_end:
                return ~((values >= gap_start) & (values < gap_end))
            return ~((values >= gap_start) | (values < gap_end))
    else:
        start_angle = start_plane * dphi
        end_angle = end_plane * dphi

        def keep_mask(values: np.ndarray) -> np.ndarray:
            return (values >= start_angle) & (values <= end_angle)

    segments: list[dict[str, np.ndarray]] = []
    for turn in np.unique(turn_index):
        turn_mask = turn_index == turn
        turn_indices = np.flatnonzero(turn_mask)
        keep_indices = turn_indices[keep_mask(phi_mod[turn_mask])]
        if keep_indices.size == 0:
            continue

        split_points = np.where(np.diff(keep_indices) > 1)[0] + 1
        for index_group in np.split(keep_indices, split_points):
            if index_group.size == 0:
                continue
            segment = {}
            for key, value in surface.items():
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == phi_full.size:
                    segment[key] = value[:, index_group]
                else:
                    segment[key] = value
            segments.append(segment)

    return segments


def _export_paraview_outputs(surface_records: list[dict[str, object]]) -> list[Path]:
    PARAVIEW_DIR.mkdir(exist_ok=True)

    surface_blocks = pv.MultiBlock()
    line_paths: list[Path] = []

    for record in surface_records:
        name = record["name"]
        x_trace = record["trace_x"]
        y_trace = record["trace_y"]
        z_trace = record["trace_z"]
        line_blocks = pv.MultiBlock()

        for line_idx in range(x_trace.shape[0]):
            points = np.column_stack([x_trace[line_idx], y_trace[line_idx], z_trace[line_idx]])
            line_blocks[f"{name}_line_{line_idx:03d}"] = _polyline_from_points(points)
        line_path = _traced_lines_path(name)
        line_blocks.save(line_path)
        line_paths.append(line_path)

        for segment_index, segment in enumerate(record["segments"]):
            surface_grid = _structured_grid_from_xyz(segment["x"], segment["y"], segment["z"])
            surface_grid["B"] = segment["b"].ravel(order="F")
            for field_name, field_values in segment["sampled_last_fields"].items():
                surface_grid[f"{field_name}_last"] = field_values.ravel(order="F")
            surface_grid["surface_index"] = np.full(surface_grid.n_points, int(record["surface_index"]))
            surface_grid["turn_index"] = np.full(surface_grid.n_points, segment_index)
            surface_blocks[f"{name}_segment_{segment_index:02d}"] = surface_grid

    surface_blocks.save(TRACED_SURFACES_PATH)
    return line_paths


def main() -> None:
    print("--- Starting traced-surface PyVista visualization ---")

    with DataFile(str(SQUASHED_PATH)) as data_file:
        field_data = {
            field_name: np.asarray(data_file.read(field_name))
            for field_name in SURFACE_FIELD_NAMES
        }
        t_array = data_file.read("t_array")

    with DataFile(str(GRID_PATH)) as grid_file:
        r = np.asarray(grid_file.read("R"))
        z = np.asarray(grid_file.read("Z"))

    reference_field = field_data["Te"]
    nx = min(reference_field.shape[1], r.shape[0])
    ny = min(reference_field.shape[2], r.shape[1])
    nz = min(reference_field.shape[3], r.shape[2])
    field_data = {
        field_name: values[:, :nx, :ny, :nz]
        for field_name, values in field_data.items()
    }
    r = r[:nx, :ny, :nz]
    z = z[:nx, :ny, :nz]

    start_r_line = r[:, 0, 0]
    start_z_line = z[:, 0, 0]

    field = zb_field.DommaschkPotentials(_coelho_coefficients(), R_0=X_CENTRE, B_0=B_TOR)
    inverse_maps = _build_inverse_plane_maps(r, z)
    surface_records: list[dict[str, object]] = []

    for surface_index, (name, radial_slice, plane_bounds) in enumerate(_surface_specs(nx)):
        traced_full = _trace_surface(field, start_r_line[radial_slice], start_z_line[radial_slice])
        segments = _extract_surface_segments(traced_full, ny, plane_bounds)
        for segment in segments:
            segment["sampled_last_fields"] = {
                field_name: _sample_volume_on_surface(values[-1], segment["r"], segment["z"], segment["phi"], inverse_maps)
                for field_name, values in field_data.items()
            }
        surface_records.append(
            {
                "name": name,
                "surface_index": surface_index,
                "segments": segments,
                "plane_bounds": plane_bounds,
                "trace_x": traced_full["x"],
                "trace_y": traced_full["y"],
                "trace_z": traced_full["z"],
            }
        )

    line_paths = _export_paraview_outputs(surface_records)

    te_all = field_data["Te"]
    global_min = float(np.nanmin(te_all))
    global_max = float(np.nanmax(te_all))
    vmin = global_min
    vmax = float(np.nanpercentile(te_all[-1], 90))
    if vmax <= vmin:
        vmax = vmin + 1.0
    print(
        f"Data range: {global_min:.2f} to {global_max:.2f}. "
        f"Using Color Scale: {vmin:.2f} to {vmax:.2f} eV"
    )

    outer_surface = next(record for record in surface_records if record["name"] == "outer")
    middle_surface = next(record for record in surface_records if record["name"] == "middle")
    outer_x = np.concatenate([segment["x"].ravel() for segment in outer_surface["segments"]])
    outer_y = np.concatenate([segment["y"].ravel() for segment in outer_surface["segments"]])
    outer_z = np.concatenate([segment["z"].ravel() for segment in outer_surface["segments"]])
    x_center = float(np.nanmean(outer_x))
    y_center = float(np.nanmean(outer_y))
    z_center = float(np.nanmean(outer_z))
    grid_size = max(
        float(np.nanmax(outer_x) - np.nanmin(outer_x)),
        float(np.nanmax(outer_y) - np.nanmin(outer_y)),
        float(np.nanmax(outer_z) - np.nanmin(outer_z)),
    )

    pv.set_jupyter_backend("none")
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.open_movie(str(MOVIE_PATH), framerate=15)

    scalar_bar_args = {
        "title": "Te [eV]",
        "title_font_size": 20,
        "label_font_size": 16,
        "shadow": True,
        "fmt": "%.1f",
    }

    total_frames = 100
    time_indices = np.linspace(0, len(t_array) - 1, total_frames, dtype=int)
    orbit_radius = grid_size * 1.72
    orbit_height = grid_size * 0.5
    orbit_start = np.deg2rad(-118.0)
    orbit_end = np.deg2rad(22.0)

    for frame_idx, t_idx in enumerate(tqdm(time_indices, desc="Rendering Frames")):
        plotter.clear()

        for record in [outer_surface, middle_surface]:
            for segment_index, segment in enumerate(record["segments"]):
                te_surface = _sample_volume_on_surface(
                    te_all[t_idx],
                    segment["r"],
                    segment["z"],
                    segment["phi"],
                    inverse_maps,
                )
                frame_mesh = _structured_grid_from_xyz(segment["x"], segment["y"], segment["z"])
                frame_mesh["Te"] = te_surface.ravel(order="F")
                frame_mesh["B"] = segment["b"].ravel(order="F")
                show_bar = record["name"] == "middle" and segment_index == 0
                plotter.add_mesh(
                    frame_mesh,
                    scalars="Te",
                    cmap="jet",
                    clim=[vmin, vmax],
                    scalar_bar_args=scalar_bar_args if show_bar else None,
                    show_scalar_bar=show_bar,
                    opacity=1.0,
                )

        plotter.add_text(
            f"Te on traced field-line surfaces | t = {t_array[t_idx]:.2e} s",
            font_size=18,
            position="upper_left",
        )

        progress = frame_idx / (total_frames - 1)
        orbit_angle = orbit_start * (1.0 - progress) + orbit_end * progress
        cam_pos = (
            x_center + orbit_radius * np.cos(orbit_angle),
            y_center + orbit_radius * np.sin(orbit_angle),
            z_center + orbit_height,
        )
        plotter.camera_position = [cam_pos, (x_center, y_center, z_center), (0, 0, 1)]
        plotter.write_frame()

    plotter.close()
    print(f"--- PyVista visualization complete: {MOVIE_PATH} ---")
    for line_path in line_paths:
        print(f"--- Traced field lines exported to: {line_path} ---")
    print(f"--- Traced surfaces exported to: {TRACED_SURFACES_PATH} ---")


if __name__ == "__main__":
    main()