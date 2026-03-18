from __future__ import annotations

from pathlib import Path

from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
COIL_GRID = REPO_ROOT / "run_coils" / "data" / "coils.fci.nc"


def test_coil_grid_contains_parallel_jacobians() -> None:
    assert COIL_GRID.exists(), f"Expected generated coil grid at {COIL_GRID}"

    with Dataset(COIL_GRID) as dataset:
        variables = dataset.variables
        assert "J" in variables
        assert "forward_J" in variables
        assert "backward_J" in variables