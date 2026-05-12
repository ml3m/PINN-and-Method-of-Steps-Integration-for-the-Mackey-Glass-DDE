#!/usr/bin/env python3
"""
RK4 reference self-convergence (no PyTorch training).

For each coarse fixed-step RK4(dt), integrate the Mackey--Glass DDE on [0, T],
interpolate onto the trajectory from the finest dt, and report MSE / rel l^2_N /
max |e| vs that finest grid. This justifies using the finest trajectory as the
numerical "truth" in MoS vs PINN comparisons.

Usage (from this directory):

    python reference_rk4_convergence_study.py \\
        --output-csv results/reference_rk4_convergence.csv

Depends on run_mglass_comparison.py (imports NumPy/SciPy; loads PyTorch module).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

MGLASS_ROOT = Path(__file__).resolve().parent
if str(MGLASS_ROOT) not in sys.path:
    sys.path.insert(0, str(MGLASS_ROOT))

import run_mglass_comparison as mg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="RK4(dt) convergence vs finest grid")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--x0", type=float, default=1.2)
    p.add_argument("--t-end", type=float, default=200.0)
    p.add_argument(
        "--dt-finest",
        type=float,
        default=1.0e-3,
        help="Ground-truth RK4 step (paper default 1e-3).",
    )
    p.add_argument(
        "--dts",
        default="0.01,0.005,0.002,0.001",
        help="Comma-separated RK4 steps (must include --dt-finest).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=MGLASS_ROOT / "results" / "reference_rk4_convergence.csv",
    )
    v = p.parse_args()

    dts = sorted({float(x.strip()) for x in v.dts.split(",") if x.strip()}, reverse=True)
    if v.dt_finest not in dts:
        dts = sorted(set(dts) | {float(v.dt_finest)}, reverse=True)

    rows = mg.f_reference_rk4_convergence_rows(
        v.beta, v.gamma, v.n, v.tau, v.x0, v.t_end, v.dt_finest, dts
    )

    v.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(v.output_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "n",
                "dt",
                "mse_vs_finest",
                "rel_l2_vs_finest",
                "max_abs_vs_finest",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {v.output_csv.resolve()}")
    print(f"Finest dt={v.dt_finest}, n={v.n}, T={v.t_end}")
    print(f"{'dt':>12} {'MSE':>14} {'rel_L2':>10} {'max|e|':>10}")
    for r in rows:
        print(
            f"{r['dt']:12.6g} {r['mse_vs_finest']:14.4e} "
            f"{r['rel_l2_vs_finest']:10.4g} {r['max_abs_vs_finest']:10.4g}"
        )


if __name__ == "__main__":
    main()
