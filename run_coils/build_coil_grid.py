from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ZOIDBERG = REPO_ROOT / "external" / "zoidberg"
RUN_DOMMASCHK = REPO_ROOT / "run_dommaschk"
DEFAULT_SIMSOPT_SRC = REPO_ROOT.parents[1] / "simsopt" / "src"
for extra_path in [EXTERNAL_ZOIDBERG, RUN_DOMMASCHK]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from dommaschk_grid_utils import calc_curvilinear_curvature  # noqa: E402
import zoidberg as zb  # noqa: E402


RUN_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN_DIR / "data"
GRID_PATH = DATA_DIR / "coils.fci.nc"
CURVES_JSON = RUN_DIR / "simsopt_inputs" / "circurves_opt.json"
SURFACE_JSON = RUN_DIR / "simsopt_inputs" / "qfmsurf_opt.json"

NX = 68
NY = 16
NZ = 128
SECTION_POINTS = 256
INNER_FRACTION_OF_MINOR_RADIUS = 1.0 / 3.0
COIL_CURRENT = 1.0e5


def _import_simsopt_load():
    try:
        from simsopt import load
        return load
    except ImportError:
        candidate_paths = []
        env_path = os.environ.get("SIMSOPT_SRC")
        if env_path:
            candidate_paths.append(Path(env_path))
        candidate_paths.append(DEFAULT_SIMSOPT_SRC)
        for candidate in candidate_paths:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        from simsopt import load
        return load


def _load_surfaces():
    load = _import_simsopt_load()
    outer_surface = load(str(SURFACE_JSON))
    inner_surface = load(str(SURFACE_JSON))
    inward_offset = float(outer_surface.minor_radius()) * INNER_FRACTION_OF_MINOR_RADIUS
    inner_surface.extend_via_projected_normal(-inward_offset)
    return outer_surface, inner_surface, inward_offset


def _surface_line(surface, phi: float, npoints: int) -> zb.rzline.RZline:
    section = np.asarray(surface.cross_section(phi / (2.0 * np.pi), thetas=npoints))
    radius = np.sqrt(section[:, 0] ** 2 + section[:, 1] ** 2)
    height = section[:, 2]
    return zb.rzline.line_from_points(
        radius,
        height,
        is_sorted=True,
    ).equallySpaced(n=max(32, NZ // 4))


def build_grid(nx: int = NX, ny: int = NY, nz: int = NZ, current: float = COIL_CURRENT) -> Path:
    outer_surface, inner_surface, inward_offset = _load_surfaces()
    field = zb.field.SimsoptBiotSavart(CURVES_JSON, currents=current)
    ycoords = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)

    inner_lines = [_surface_line(inner_surface, phi, SECTION_POINTS) for phi in ycoords]
    outer_lines = [_surface_line(outer_surface, phi, SECTION_POINTS) for phi in ycoords]

    poloidal_grids = [
        zb.poloidal_grid.grid_elliptic(
            inner_line,
            outer_line,
            nx,
            nz,
            restrict_size=2560,
            align=0,
            inner_ort=1,
            inner_maxmode=4,
            nx_inner=0,
            nx_outer=0,
        )
        for inner_line, outer_line in zip(inner_lines, outer_lines)
    ]

    grid = zb.grid.Grid(poloidal_grids, ycoords, 2.0 * np.pi, yperiodic=True)
    maps = zb.make_maps(grid, field)

    DATA_DIR.mkdir(exist_ok=True)
    with zb.zoidberg.MapWriter(str(GRID_PATH), metric2d=False) as writer:
        writer.add_grid_field(grid, field)
        writer.add_maps(maps)
        writer.add_dagp()

    calc_curvilinear_curvature(str(GRID_PATH), field, grid)

    print(f"Loaded coil curves from {CURVES_JSON}")
    print(f"Loaded QFM boundary from {SURFACE_JSON}")
    print(f"Applied inward projected-normal offset of {inward_offset:.6f} m")
    print(f"Wrote coil-driven grid to {GRID_PATH}")
    return GRID_PATH


def main() -> None:
    build_grid()


if __name__ == "__main__":
    main()