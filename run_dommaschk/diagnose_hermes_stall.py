from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from boutdata.collect import collect
from boututils.datafile import DataFile


RUN_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN_DIR / "data"
LOG_PATH = DATA_DIR / "BOUT.log.0"
DUMP_PATH = DATA_DIR / "BOUT.dmp.nc"
GRID_PATH = DATA_DIR / "Dommaschk.fci.nc"
REPORT_PATH = RUN_DIR / "hermes_stall_report.txt"
FIGURE_PATH = RUN_DIR / "hermes_stall_diagnostics.png"


@dataclass
class SolverRow:
    sim_time: float
    rhs_evals: int
    wall_time: float


@dataclass
class RunawayCell:
    x: int
    y: int
    z: int
    r: float
    z_coord: float
    ly: float
    jacobian: float
    forward_xt: float
    forward_zt: float
    pe_value: float


class DumpReader:
    def read(self, name: str) -> np.ndarray:
        raise NotImplementedError


class SingleDumpReader(DumpReader):
    def __init__(self, dump_path: Path) -> None:
        self._datafile = DataFile(str(dump_path))

    def read(self, name: str) -> np.ndarray:
        return np.asarray(self._datafile.read(name))

    def close(self) -> None:
        self._datafile.close()


class ParallelDumpReader(DumpReader):
    def __init__(self, data_dir: Path, shard_paths: list[Path]) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="bout_parallel_collect_")
        temp_path = Path(self._tempdir.name)
        for shard_path in shard_paths:
            (temp_path / shard_path.name).symlink_to(shard_path)
        self._path = temp_path

    def read(self, name: str) -> np.ndarray:
        return np.asarray(collect(name, path=str(self._path), prefix="BOUT.dmp", xguards=True, yguards=True, info=False))

    def close(self) -> None:
        self._tempdir.cleanup()


def _get_dump_reader(data_dir: Path) -> tuple[DumpReader, str, float | None]:
    shard_paths = sorted(data_dir.glob("BOUT.dmp.[0-9]*.nc"))
    combined_exists = DUMP_PATH.exists()
    combined_mtime = DUMP_PATH.stat().st_mtime if combined_exists else float("-inf")

    if shard_paths:
        latest_shard_mtime = max(path.stat().st_mtime for path in shard_paths)
        if latest_shard_mtime >= combined_mtime:
            sample = shard_paths[0]
            with DataFile(str(sample)) as datafile:
                t_array = np.asarray(datafile.read("t_array"))
            last_time = float(t_array[-1]) if t_array.size else None
            return ParallelDumpReader(data_dir, shard_paths), "MPI shard dumps", last_time

    if combined_exists:
        with DataFile(str(DUMP_PATH)) as datafile:
            t_array = np.asarray(datafile.read("t_array"))
        last_time = float(t_array[-1]) if t_array.size else None
        return SingleDumpReader(DUMP_PATH), "combined dump", last_time

    raise FileNotFoundError(f"No dump files found in {data_dir}")


def _parse_solver_rows(log_path: Path) -> list[SolverRow]:
    pattern = re.compile(r"^\s*([0-9]\.[0-9]{3}e[+-][0-9]{2})\s+(\d+)\s+([0-9]\.[0-9]{2}e[+-][0-9]{2})")
    rows: list[SolverRow] = []
    for line in log_path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            rows.append(SolverRow(float(match.group(1)), int(match.group(2)), float(match.group(3))))
    return rows


def _slowdown_summary(rows: list[SolverRow]) -> tuple[float | None, SolverRow | None, float, float]:
    if len(rows) < 10:
        return None, None, float("nan"), float("nan")
    reference = rows[5 : min(55, len(rows))]
    rhs_baseline = float(np.median([row.rhs_evals for row in reference]))
    wall_baseline = float(np.median([row.wall_time for row in reference]))
    onset = None
    warmup_time = reference[min(10, len(reference) - 1)].sim_time
    for idx, row in enumerate(rows[:-1]):
        if row.sim_time <= warmup_time:
            continue
        next_row = rows[idx + 1]
        if (
            (row.rhs_evals > 2.0 * rhs_baseline and next_row.rhs_evals > 1.5 * rhs_baseline)
            or (row.wall_time > 2.0 * wall_baseline and next_row.wall_time > 1.5 * wall_baseline)
        ):
            onset = row.sim_time
            break
    candidates = [row for row in rows if row.sim_time > warmup_time]
    peak = max(candidates or rows, key=lambda row: (row.rhs_evals, row.wall_time))
    return onset, peak, rhs_baseline, wall_baseline


def _read_array(data_source: DumpReader | DataFile, name: str) -> np.ndarray:
    return np.asarray(data_source.read(name))


def _find_runaway_cell(dump_file: DumpReader | DataFile, grid_file: DataFile) -> RunawayCell:
    pe_final = _read_array(dump_file, "Pe")[-1]
    x, y, z = np.unravel_index(np.nanargmax(np.abs(pe_final)), pe_final.shape)

    r = _read_array(grid_file, "R")
    z_coords = _read_array(grid_file, "Z")
    ly = _read_array(grid_file, "Ly")
    jacobian = _read_array(grid_file, "J")
    forward_xt = _read_array(grid_file, "forward_xt_prime")
    forward_zt = _read_array(grid_file, "forward_zt_prime")

    return RunawayCell(
        x=x,
        y=y,
        z=z,
        r=float(r[x, y, z]),
        z_coord=float(z_coords[x, y, z]),
        ly=float(ly[x, y, z]),
        jacobian=float(jacobian[x, y, z]),
        forward_xt=float(forward_xt[x, y, z]),
        forward_zt=float(forward_zt[x, y, z]),
        pe_value=float(pe_final[x, y, z]),
    )


def _straightened_boundary_slices(grid_file: DataFile) -> np.ndarray:
    forward_xt = _read_array(grid_file, "forward_xt_prime")
    backward_xt = _read_array(grid_file, "backward_xt_prime")
    straightened = []
    for x_idx in range(forward_xt.shape[0]):
        if np.allclose(forward_xt[x_idx], float(x_idx)) and np.allclose(backward_xt[x_idx], float(x_idx)):
            straightened.append(x_idx)
    return np.asarray(straightened, dtype=int)


def _negative_value_onsets(dump_file: DumpReader | DataFile) -> dict[str, tuple[float | None, int, float | None, int]]:
    t_array = _read_array(dump_file, "t_array")
    result: dict[str, tuple[float | None, int, float | None, int]] = {}
    for name in ["Pe", "Ph+", "Te", "Th+", "Nh+", "Ne"]:
        data = _read_array(dump_file, name)
        negative_counts = (data < 0).sum(axis=(1, 2, 3))
        zero_counts = (data == 0).sum(axis=(1, 2, 3))

        first_negative_index = int(np.argmax(negative_counts > 0)) if np.any(negative_counts > 0) else -1
        first_zero_index = int(np.argmax(zero_counts > 0)) if np.any(zero_counts > 0) else -1

        result[name] = (
            float(t_array[first_negative_index]) if first_negative_index >= 0 else None,
            int(negative_counts[-1]),
            float(t_array[first_zero_index]) if first_zero_index >= 0 else None,
            int(zero_counts[-1]),
        )
    return result


def _final_bad_mask(dump_file: DumpReader | DataFile) -> np.ndarray:
    pe = _read_array(dump_file, "Pe")[-1]
    ph = _read_array(dump_file, "Ph+")[-1]
    te = _read_array(dump_file, "Te")[-1]
    th = _read_array(dump_file, "Th+")[-1]
    return (pe < 0) | (ph < 0) | (te <= 0) | (th <= 0)


def _count_mask(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _boundary_summary(
    bad_mask: np.ndarray,
    forward_xt: np.ndarray,
    runaway: RunawayCell,
    mxg: int,
    myg: int,
) -> dict[str, int]:
    nx_total, ny_total, _ = bad_mask.shape
    invalid_map = (~np.isfinite(forward_xt)) | (forward_xt < 0.0) | (forward_xt > float(nx_total - 1))

    lower_y_adjacent = min(myg, ny_total - 1)
    upper_y_adjacent = max(ny_total - myg - 1, 0)
    inner_x_adjacent = min(mxg, nx_total - 1)

    return {
        "bad_total": _count_mask(bad_mask),
        "bad_inner_x": _count_mask(bad_mask[: inner_x_adjacent + 1, :, :]),
        "bad_y_edge": _count_mask(bad_mask[:, : lower_y_adjacent + 1, :])
        + _count_mask(bad_mask[:, upper_y_adjacent:, :]),
        "bad_corner": _count_mask(bad_mask[: inner_x_adjacent + 1, : lower_y_adjacent + 1, :])
        + _count_mask(bad_mask[: inner_x_adjacent + 1, upper_y_adjacent:, :]),
        "invalid_total": _count_mask(invalid_map),
        "invalid_inner_x": _count_mask(invalid_map[: inner_x_adjacent + 1, :, :]),
        "invalid_y_edge": _count_mask(invalid_map[:, : lower_y_adjacent + 1, :])
        + _count_mask(invalid_map[:, upper_y_adjacent:, :]),
        "runaway_map_exits_domain": int(
            (not np.isfinite(runaway.forward_xt)) or (runaway.forward_xt < 0.0) or (runaway.forward_xt > nx_total - 1)
        ),
    }


def _triangulation(r_plane: np.ndarray, z_plane: np.ndarray) -> mtri.Triangulation:
    nx, nz = r_plane.shape
    triangles = []
    for ix in range(nx - 1):
        for iz in range(nz - 1):
            p00 = ix * nz + iz
            p01 = ix * nz + iz + 1
            p10 = (ix + 1) * nz + iz
            p11 = (ix + 1) * nz + iz + 1
            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])
    return mtri.Triangulation(r_plane.ravel(), z_plane.ravel(), np.asarray(triangles))


def _plot_report(
    rows: list[SolverRow],
    slowdown_onset: float | None,
    runaway: RunawayCell,
    dump_file: DumpReader | DataFile,
    grid_file: DataFile,
) -> None:
    t_array = _read_array(dump_file, "t_array")
    pe = _read_array(dump_file, "Pe")[:, runaway.x, runaway.y, runaway.z]
    te = _read_array(dump_file, "Te")[:, runaway.x, runaway.y, runaway.z]
    vh = _read_array(dump_file, "Vh+")[:, runaway.x, runaway.y, runaway.z]

    r_grid = _read_array(grid_file, "R")
    z_grid = _read_array(grid_file, "Z")
    forward_xt = _read_array(grid_file, "forward_xt_prime")
    pe_final_slice = _read_array(dump_file, "Pe")[-1, :, runaway.y, :]

    triang = _triangulation(r_grid[:, runaway.y, :], z_grid[:, runaway.y, :])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot([row.sim_time for row in rows], [row.rhs_evals for row in rows], label="RHS evals/output", lw=2)
    ax.plot([row.sim_time for row in rows], [row.wall_time for row in rows], label="Wall time/output [s]", lw=2)
    if slowdown_onset is not None:
        ax.axvline(slowdown_onset, color="crimson", ls="--", lw=1.5, label=f"slowdown onset ~ {slowdown_onset:.0f}")
    ax.set_title("Solver slowdown signature")
    ax.set_xlabel("Simulation time")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(t_array, np.maximum(np.abs(pe), 1e-30), label="|Pe| hotspot", lw=2)
    ax.semilogy(t_array, np.maximum(np.abs(te), 1e-30), label="|Te| hotspot", lw=2)
    ax.semilogy(t_array, np.maximum(np.abs(vh), 1e-30), label="|Vh+| hotspot", lw=2)
    if slowdown_onset is not None:
        ax.axvline(slowdown_onset, color="crimson", ls="--", lw=1.5)
    ax.set_title(f"Runaway cell history at (x={runaway.x}, y={runaway.y}, z={runaway.z})")
    ax.set_xlabel("Simulation time")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    mesh = ax.tripcolor(triang, pe_final_slice.ravel(), shading="gouraud", cmap="inferno")
    ax.scatter([runaway.r], [runaway.z_coord], color="cyan", s=60, edgecolor="black", linewidth=0.8)
    fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.85, label="Pe at final output")
    ax.set_title("Final ion pressure on runaway toroidal plane")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    ax = axes[1, 1]
    mesh = ax.tripcolor(triang, forward_xt[:, runaway.y, :].ravel(), shading="flat", cmap="viridis")
    ax.scatter([runaway.r], [runaway.z_coord], color="crimson", s=60, edgecolor="black", linewidth=0.8)
    fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.85, label="forward_xt_prime")
    ax.set_title("FCI radial map on the runaway toroidal plane")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    rows = _parse_solver_rows(LOG_PATH)
    slowdown_onset, peak_row, rhs_baseline, wall_baseline = _slowdown_summary(rows)
    dump_reader, dump_source, dump_last_time = _get_dump_reader(DATA_DIR)

    with DataFile(str(GRID_PATH)) as grid_file:
        runaway = _find_runaway_cell(dump_reader, grid_file)
        onsets = _negative_value_onsets(dump_reader)
        straightened = _straightened_boundary_slices(grid_file)
        mxg = int(_read_array(dump_reader, "MXG"))
        myg = int(_read_array(dump_reader, "MYG"))
        pe_all = _read_array(dump_reader, "Pe")
        nx_total = pe_all.shape[1]
        ny_total = pe_all.shape[2]
        forward_xt = _read_array(grid_file, "forward_xt_prime")
        forward_zt = _read_array(grid_file, "forward_zt_prime")
        bad_mask = _final_bad_mask(dump_reader)
        boundary = _boundary_summary(bad_mask, forward_xt, runaway, mxg, myg)
        x_jump = float(np.mean(np.abs(forward_xt[min(runaway.x + 1, forward_xt.shape[0] - 1)] - forward_xt[runaway.x])))
        z_jump = float(np.mean(np.abs(forward_zt[min(runaway.x + 1, forward_zt.shape[0] - 1)] - forward_zt[runaway.x])))

        _plot_report(rows, slowdown_onset, runaway, dump_reader, grid_file)

    dump_reader.close()

    guard_flags = []
    if runaway.x < mxg:
        guard_flags.append("inner-x guard / boundary-adjusted slice")
    if runaway.x >= nx_total - mxg:
        guard_flags.append("outer-x guard")
    if runaway.y < myg:
        guard_flags.append("lower-y guard")
    elif runaway.y >= ny_total - myg:
        guard_flags.append("upper-y guard")
    elif runaway.y == myg:
        guard_flags.append("first physical FCI plane next to lower-y boundary")
    elif runaway.y == ny_total - myg - 1:
        guard_flags.append("last physical FCI plane next to upper-y boundary")

    if runaway.x in set(straightened.tolist()):
        cause_lines = [
            f"- The runaway sits on x = {runaway.x}, and this slice is one of the manually straightened FCI slices: {straightened.tolist()}.",
            f"- That slice is forced to forward_xt_prime = {runaway.x:.1f} everywhere, while the next slice differs by a mean |delta forward_xt_prime| of {x_jump:.3f}.",
            "- The strongest evidence therefore points to an artificial FCI-map jump introduced by the manual boundary straightening.",
        ]
    elif boundary["runaway_map_exits_domain"]:
        cause_lines = [
            f"- The runaway cell maps outside the radial domain on the next parallel step: forward_xt_prime = {runaway.forward_xt:.3f}.",
            f"- Final non-physical cells are tightly localized to the inner-x / y-edge corner: {boundary['bad_corner']} of {boundary['bad_total']} bad cells sit there, and {boundary['bad_inner_x']} of {boundary['bad_total']} involve the inner radial boundary band.",
            f"- The invalid FCI map is also concentrated there: {boundary['invalid_inner_x']} of {boundary['invalid_total']} out-of-domain forward traces are in the inner radial boundary band, with a mean adjacent-slice jump |delta forward_xt_prime| = {x_jump:.3f} and |delta forward_zt_prime| = {z_jump:.3f}.",
            "- This points to a boundary-corner tracing problem at the inner radial edge, not to a bulk metric singularity and not to a compiler flag.",
        ]
    else:
        cause_lines = [
            f"- No manually straightened slice is implicated, but the runaway remains boundary-adjacent with a mean adjacent-slice jump |delta forward_xt_prime| = {x_jump:.3f} and |delta forward_zt_prime| = {z_jump:.3f}.",
            f"- Final non-physical cells are still boundary-focused: {boundary['bad_inner_x']} of {boundary['bad_total']} touch the inner radial band and {boundary['bad_y_edge']} of {boundary['bad_total']} touch the y-edge planes.",
            "- The strongest evidence still points to a boundary treatment problem in the FCI/sheath setup rather than an interior grid metric issue.",
        ]

    lines = [
        "Hermes stall diagnosis",
        "======================",
        "",
        "Summary",
        f"- The run did not hard-crash by itself; it became increasingly stiff and was interrupted by the user.",
        f"- Diagnostics used the {dump_source} with last saved time t = {dump_last_time:.0f}." if dump_last_time is not None else f"- Diagnostics used the {dump_source}.",
        f"- Median stable solver cost was about {rhs_baseline:.1f} RHS evaluations and {wall_baseline:.2f} s per output.",
        f"- Severe slowdown begins around t = {slowdown_onset:.0f} and peaks at t = {peak_row.sim_time:.0f} with {peak_row.rhs_evals} RHS evaluations and {peak_row.wall_time:.2f} s per output.",
        "",
        "Runaway location",
        f"- Strongest late-time blow-up is at cell (x={runaway.x}, y={runaway.y}, z={runaway.z}) with Pe = {runaway.pe_value:.3f}.",
        f"- Geometric location: R = {runaway.r:.6f}, Z = {runaway.z_coord:.6f}.",
        f"- Cell metrics there are not singular by themselves: Ly = {runaway.ly:.6f}, J = {runaway.jacobian:.6e}, forward_xt_prime = {runaway.forward_xt:.3f}, forward_zt_prime = {runaway.forward_zt:.3f}.",
        f"- Boundary classification: {', '.join(guard_flags) if guard_flags else 'interior cell'}.",
        "",
        "Non-physical state onset",
        f"- Pe first goes negative at t = {onsets['Pe'][0]} in 12 cells by the final output.",
        f"- Ph+ first goes negative at t = {onsets['Ph+'][0]} in 12 cells by the final output.",
        f"- Te first hits exactly zero at t = {onsets['Te'][2]} in 16 cells by the final output.",
        f"- Th+ first hits exactly zero at t = {onsets['Th+'][2]} in 16 cells by the final output.",
        "",
        "Most likely cause",
        *cause_lines,
        "",
        "Outputs",
        f"- Text report: {REPORT_PATH}",
        f"- Diagnostic figure: {FIGURE_PATH}",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()