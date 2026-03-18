from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ZOIDBERG_ROOT = REPO_ROOT / "external" / "zoidberg"
DEFAULT_SIMSOPT_SRC = REPO_ROOT.parents[1] / "simsopt" / "src"
if str(ZOIDBERG_ROOT) not in sys.path:
    sys.path.insert(0, str(ZOIDBERG_ROOT))

from zoidberg import field as zb_field  # noqa: E402
from zoidberg import fieldtracer as zb_fieldtracer  # noqa: E402


RUN_DIR = Path(__file__).resolve().parent
CURVES_JSON = RUN_DIR / "simsopt_inputs" / "circurves_opt.json"
SURFACE_JSON = RUN_DIR / "simsopt_inputs" / "qfmsurf_opt.json"
OUTPUT_DIR = RUN_DIR / "outputs"
DEFAULT_PLOT_PATH = OUTPUT_DIR / "fieldline_comparison.png"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "fieldline_comparison_metrics.json"
DEFAULT_CURRENT = 1.0e5
DEFAULT_PHIS = np.linspace(0.2, 1.2, 6)


def _import_simsopt_dependencies():
    try:
        import simsopt
        from simsopt.field import BiotSavart, Coil, Current, compute_fieldlines

        return simsopt, BiotSavart, Coil, Current, compute_fieldlines
    except ImportError:
        candidate_paths = []
        env_path = os.environ.get("SIMSOPT_SRC")
        if env_path:
            candidate_paths.append(Path(env_path))
        candidate_paths.append(DEFAULT_SIMSOPT_SRC)
        for candidate in candidate_paths:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        import simsopt
        from simsopt.field import BiotSavart, Coil, Current, compute_fieldlines

        return simsopt, BiotSavart, Coil, Current, compute_fieldlines


def _load_direct_biot_savart(current: float = DEFAULT_CURRENT):
    simsopt, BiotSavart, Coil, Current, _ = _import_simsopt_dependencies()
    loaded = simsopt.load(CURVES_JSON)
    if hasattr(loaded, "set_points_cyl") and hasattr(loaded, "B_cyl"):
        return loaded

    if np.isscalar(current):
        currents = [float(current)] * len(loaded)
    else:
        currents = [float(value) for value in current]

    coils = [Coil(curve, Current(1) * current_value) for curve, current_value in zip(loaded, currents)]
    return BiotSavart(coils)


def _load_surface():
    simsopt, _, _, _, _ = _import_simsopt_dependencies()
    return simsopt.load(SURFACE_JSON)


def _midplane_start_points(surface, nfieldlines: int) -> tuple[np.ndarray, np.ndarray]:
    section = np.asarray(surface.cross_section(0.0, thetas=512))
    radius = np.sqrt(section[:, 0] ** 2 + section[:, 1] ** 2)
    height = section[:, 2]
    nearest = np.argsort(np.abs(height))[:16]
    midplane_radius = radius[nearest]
    inner_radius = float(np.min(midplane_radius))
    outer_radius = float(np.max(midplane_radius))
    margin = 0.18 * (outer_radius - inner_radius)
    start_r = np.linspace(inner_radius + margin, outer_radius - margin, nfieldlines)
    start_z = np.zeros(nfieldlines)
    return start_r, start_z


def compare_magnetic_field_samples(
    phis: np.ndarray | None = None,
    current: float = DEFAULT_CURRENT,
) -> dict[str, float]:
    if phis is None:
        phis = DEFAULT_PHIS

    direct_bs = _load_direct_biot_savart(current=current)
    zoidberg_field = zb_field.SimsoptBiotSavart(CURVES_JSON, currents=current)
    surface = _load_surface()
    start_r, _ = _midplane_start_points(surface, 4)

    sample_r = np.repeat(start_r, 3)
    sample_z = np.tile(np.array([-0.02, 0.0, 0.02]), len(start_r))
    sample_phi = np.repeat(np.asarray(phis[: min(3, len(phis))], dtype=float), len(sample_r))
    sample_r = np.tile(sample_r, min(3, len(phis)))
    sample_z = np.tile(sample_z, min(3, len(phis)))

    points_cyl = np.column_stack([sample_r, sample_phi, sample_z])
    direct_bs.set_points_cyl(points_cyl)
    direct_field = np.asarray(direct_bs.B_cyl(), dtype=float)
    zoidberg_components = np.column_stack(
        [
            zoidberg_field.Bxfunc(sample_r, sample_z, sample_phi),
            zoidberg_field.Byfunc(sample_r, sample_z, sample_phi),
            zoidberg_field.Bzfunc(sample_r, sample_z, sample_phi),
        ]
    )

    abs_error = np.abs(zoidberg_components - direct_field)
    rel_error = abs_error / np.maximum(np.abs(direct_field), 1.0e-12)
    return {
        "max_abs_component_error": float(np.max(abs_error)),
        "mean_abs_component_error": float(np.mean(abs_error)),
        "max_rel_component_error": float(np.max(rel_error)),
        "mean_rel_component_error": float(np.mean(rel_error)),
    }


def _trace_simsopt_hits(
    direct_bs,
    start_r: np.ndarray,
    start_z: np.ndarray,
    phis: np.ndarray,
) -> np.ndarray:
    _, _, _, _, compute_fieldlines = _import_simsopt_dependencies()
    phis = np.asarray(phis, dtype=float)
    base_tmax = float(max(2.0, phis[-1] * 2.0))

    hits = None
    for scale in [1.0, 2.0, 4.0, 8.0]:
        _, hits = compute_fieldlines(
            direct_bs,
            start_r,
            start_z,
            tmax=base_tmax * scale,
            tol=1.0e-12,
            phis=phis.tolist(),
        )
        if all(line_hits.shape[0] >= len(phis) for line_hits in hits):
            break
    else:
        raise RuntimeError("SIMSOPT field-line tracing did not reach all requested toroidal planes")

    points = np.empty((len(start_r), len(phis), 2), dtype=float)
    for line_idx, line_hits in enumerate(hits):
        order = np.argsort(line_hits[:, 1].astype(int))
        ordered_hits = line_hits[order][: len(phis)]
        expected = np.arange(len(phis))
        if not np.array_equal(ordered_hits[:, 1].astype(int), expected):
            raise RuntimeError("SIMSOPT field-line hits are missing requested toroidal slices")
        x_coord = ordered_hits[:, 2]
        y_coord = ordered_hits[:, 3]
        z_coord = ordered_hits[:, 4]
        points[line_idx, :, 0] = np.sqrt(x_coord**2 + y_coord**2)
        points[line_idx, :, 1] = z_coord
    return points


def _trace_zoidberg_hits(
    start_r: np.ndarray,
    start_z: np.ndarray,
    phis: np.ndarray,
    current: float,
) -> np.ndarray:
    tracer = zb_fieldtracer.FieldTracer(zb_field.SimsoptBiotSavart(CURVES_JSON, currents=current))
    y_values = np.concatenate([[0.0], np.asarray(phis, dtype=float)])
    traced = tracer.follow_field_lines(start_r, start_z, y_values, rtol=1.0e-12)
    return np.transpose(traced[1:, :, :], (1, 0, 2))


def compare_fieldlines(
    output_plot: Path | None = DEFAULT_PLOT_PATH,
    output_metrics: Path | None = DEFAULT_METRICS_PATH,
    nfieldlines: int = 4,
    phis: np.ndarray | None = None,
    current: float = DEFAULT_CURRENT,
) -> dict[str, float]:
    if phis is None:
        phis = DEFAULT_PHIS
    phis = np.asarray(phis, dtype=float)

    surface = _load_surface()
    start_r, start_z = _midplane_start_points(surface, nfieldlines)
    direct_bs = _load_direct_biot_savart(current=current)
    simsopt_hits = _trace_simsopt_hits(direct_bs, start_r, start_z, phis)
    zoidberg_hits = _trace_zoidberg_hits(start_r, start_z, phis, current=current)

    delta = zoidberg_hits - simsopt_hits
    point_error = np.linalg.norm(delta, axis=2)
    radial_error = np.abs(delta[:, :, 0])
    vertical_error = np.abs(delta[:, :, 1])

    metrics = {
        **compare_magnetic_field_samples(phis=phis, current=current),
        "max_radial_error": float(np.max(radial_error)),
        "mean_radial_error": float(np.mean(radial_error)),
        "max_vertical_error": float(np.max(vertical_error)),
        "mean_vertical_error": float(np.mean(vertical_error)),
        "max_line_error": float(np.max(point_error)),
        "mean_line_error": float(np.mean(point_error)),
    }

    if output_plot is not None:
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
        for line_idx in range(nfieldlines):
            axes[0].plot(phis, simsopt_hits[line_idx, :, 0], lw=2.0, label=f"SIMSOPT line {line_idx}")
            axes[0].plot(phis, zoidberg_hits[line_idx, :, 0], ls="--", lw=1.6, label=f"Zoidberg line {line_idx}")
            axes[1].plot(phis, simsopt_hits[line_idx, :, 1], lw=2.0)
            axes[1].plot(phis, zoidberg_hits[line_idx, :, 1], ls="--", lw=1.6)
            axes[2].plot(phis, point_error[line_idx, :], lw=2.0, label=f"line {line_idx}")

        axes[0].set_ylabel("R [m]")
        axes[0].set_title("Field-line hit comparison: R vs toroidal angle")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].set_ylabel("Z [m]")
        axes[1].set_title("Field-line hit comparison: Z vs toroidal angle")
        axes[1].grid(True, alpha=0.3)
        axes[2].set_xlabel("Toroidal angle [rad]")
        axes[2].set_ylabel("|delta(R,Z)| [m]")
        axes[2].set_title("Pointwise difference between SIMSOPT and Zoidberg traces")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(ncol=2, fontsize=8)
        figure.savefig(output_plot, dpi=180)
        plt.close(figure)

    if output_metrics is not None:
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    return metrics


def main() -> None:
    metrics = compare_fieldlines()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved comparison plot to {DEFAULT_PLOT_PATH}")
    print(f"Saved metrics to {DEFAULT_METRICS_PATH}")


if __name__ == "__main__":
    main()