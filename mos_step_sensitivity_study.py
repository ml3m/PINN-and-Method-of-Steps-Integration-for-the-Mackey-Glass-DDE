#!/usr/bin/env python3
"""
MoS--RK45 output/max_step sensitivity (no PINN).

For each requested classical_dt (SciPy solve_ivp max_step and uniform output grid),
run f_solve_mackey_glass_classical, score against the fixed-step RK4 reference at
ref_dt, and report segment MSE, pointwise error statistics, T_valid, wall time.

Usage::

    cd mglass_comparison
    python mos_step_sensitivity_study.py --output-csv results/mos_dt_sensitivity.csv

Reproduces paper settings: n=10, T=200, tau=2, ref_dt=1e-3.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.interpolate import interp1d

MGLASS_ROOT = Path(__file__).resolve().parent
if str(MGLASS_ROOT) not in sys.path:
    sys.path.insert(0, str(MGLASS_ROOT))

import run_mglass_comparison as mg  # noqa: E402


def _segment_edges(p_t_end: float) -> List[float]:
    e = [0.0]
    while e[-1] + 50.0 < p_t_end - 1e-9:
        e.append(e[-1] + 50.0)
    e.append(float(p_t_end))
    return e


def _pointwise_stats(abs_e: np.ndarray) -> Dict[str, float]:
    v = np.asarray(abs_e, dtype=np.float64).ravel()
    return {
        "pw_mean": float(np.mean(v)),
        "pw_std": float(np.std(v)),
        "pw_median": float(np.median(v)),
        "pw_p95": float(np.percentile(v, 95)),
        "pw_p99": float(np.percentile(v, 99)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MoS classical_dt sweep")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--x0", type=float, default=1.2)
    p.add_argument("--t-end", type=float, default=200.0)
    p.add_argument("--ref-dt", type=float, default=1.0e-3)
    p.add_argument(
        "--classical-dts",
        default="0.01,0.005,0.001",
        help="MoS max_step / output spacing values",
    )
    p.add_argument(
        "--valid-thresholds",
        default="0.05,0.1,0.2,0.5",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=MGLASS_ROOT / "results" / "mos_dt_sensitivity.csv",
    )
    v = p.parse_args()

    dts = [float(x.strip()) for x in v.classical_dts.split(",") if x.strip()]
    thr = [float(x.strip()) for x in v.valid_thresholds.split(",") if x.strip()]
    edges = _segment_edges(v.t_end)

    t_ref, x_ref = mg.f_solve_mackey_glass_rk4_fixed(
        v.beta, v.gamma, v.n, v.tau, v.x0, v.t_end, v.ref_dt,
    )

    rows_out: List[Dict[str, Any]] = []
    for dt in dts:
        t0 = time.perf_counter()
        t_cl, x_cl = mg.f_solve_mackey_glass_classical(
            p_beta=v.beta,
            p_gamma=v.gamma,
            p_n=v.n,
            p_tau=v.tau,
            p_x0=v.x0,
            p_t_end=v.t_end,
            p_dt=float(dt),
        )
        wall = float(time.perf_counter() - t0)
        fi = interp1d(t_cl, x_cl, kind="cubic", fill_value="extrapolate")
        x_on = np.asarray(fi(t_ref), dtype=np.float64).ravel()
        metrics = mg.f_compute_metrics(x_ref, x_on)
        seg = mg.f_segment_mse_table(t_ref, x_ref, x_on, edges)
        pw = _pointwise_stats(np.abs(x_on - x_ref))
        tv = mg.f_first_exceedance_times(t_ref, x_ref, x_on, thr)

        row: Dict[str, Any] = {
            "classical_dt": float(dt),
            "wall_s": wall,
            "mse": metrics["mse"],
            "rel_l2": metrics["rel_l2"],
            "max_abs": metrics["max_abs_err"],
            "seg_0_50": seg.get(f"[0,{edges[1]:g}]", float("nan")),
            "seg_50_100": seg.get(f"[{edges[1]:g},{edges[2]:g}]", float("nan")),
            "seg_100_150": seg.get(f"[{edges[2]:g},{edges[3]:g}]", float("nan")),
            "seg_150_200": seg.get(f"[{edges[3]:g},{edges[4]:g}]", float("nan")),
        }
        row.update(pw)
        for th in thr:
            row[f"t_valid_{th}"] = tv.get(str(float(th)), float("nan"))
        rows_out.append(row)

    v.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        keys = list(rows_out[0].keys())
        with open(v.output_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)

    print(f"Wrote {v.output_csv.resolve()}\n")
    for row in rows_out:
        print(
            f"dt={row['classical_dt']:.4g}  MSE={row['mse']:.4e}  "
            f"relL2={row['rel_l2']:.4f}  max={row['max_abs']:.4f}  "
            f"t_s={row['wall_s']:.2f}  Tv(0.1)={row.get('t_valid_0.1', float('nan')):.3g}"
        )


if __name__ == "__main__":
    main()
