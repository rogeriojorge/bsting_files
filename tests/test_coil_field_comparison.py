from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_coils.compare_fieldlines import compare_fieldlines, compare_magnetic_field_samples


def test_simsopt_biot_savart_field_matches_direct_samples() -> None:
    metrics = compare_magnetic_field_samples()
    assert metrics["max_abs_component_error"] < 1.0e-12
    assert metrics["max_rel_component_error"] < 1.0e-12


def test_zoidberg_fieldline_trace_matches_simsopt(tmp_path: Path) -> None:
    metrics = compare_fieldlines(
        output_plot=tmp_path / "fieldline_comparison.png",
        output_metrics=tmp_path / "fieldline_comparison_metrics.json",
    )
    assert metrics["max_line_error"] < 3.0e-3
    assert metrics["mean_line_error"] < 1.0e-3