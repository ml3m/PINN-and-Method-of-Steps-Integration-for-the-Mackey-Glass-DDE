#!/usr/bin/env python3
"""
Aggregate per-seed ablation_aggregate.csv rows into mean±std summaries.

Reads one batch directory (same ``batch_stamp`` folder written by
``run_ablation_matrix.py``), groups by ``run_id`` and ``phase_id``,
reports primary metrics and seed-matched Phase II contrasts when two
runs share the same seed set.

Example::

    cd mglass_comparison
    python analyze_ablation.py --batch-dir results/ablation_runs/20260101_120000
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

MROOT = Path(__file__).resolve().parent


def _f_collect_rows(p_csv: Path) -> List[Dict[str, str]]:
    with open(p_csv, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f_safe_float(cell: str) -> float:
    try:
        v = float(cell)
        if math.isnan(v):
            return float("nan")
        return v
    except (TypeError, ValueError):
        return float("nan")


def _f_mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.array([v for v in vals if v == v], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _f_paired_delta(
    rows_a: Dict[int, Dict[str, Any]],
    rows_b: Dict[int, Dict[str, Any]],
    key: str,
) -> Tuple[float, float]:
    deltas = []
    for s, ra in rows_a.items():
        if s not in rows_b:
            continue
        va, vb = ra.get(key), rows_b[s].get(key)
        if va != va or vb != vb:
            continue
        deltas.append(va - vb)
    arr = np.array(deltas, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ablation_aggregate.csv")
    ap.add_argument("--batch-dir", type=str, required=True, help="Stamp directory with aggregate CSV.")
    args = ap.parse_args()

    p_batch = Path(args.batch_dir).expanduser().resolve()
    if not p_batch.is_dir():
        alt = MROOT / args.batch_dir
        if alt.is_dir():
            p_batch = alt.resolve()
        else:
            raise SystemExit(f"No such directory: {args.batch_dir}")

    p_csv = p_batch / "ablation_aggregate.csv"
    if not p_csv.is_file():
        raise SystemExit(f"Missing aggregate file: {p_csv}")

    rows = _f_collect_rows(p_csv)
    by_run: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)

    out_lines: List[str] = []
    for run_id in sorted(by_run.keys()):
        chunk = by_run[run_id]
        phase = chunk[0].get("phase_id", "")
        seeds = sorted(int(r["seed"]) for r in chunk)
        mse_v = [_f_safe_float(r.get("mse", "")) for r in chunk]
        tv1 = [_f_safe_float(r.get("tvalid_0.1", "")) for r in chunk]
        m_m, m_s = _f_mean_std(mse_v)
        t_m, t_s = _f_mean_std(tv1)
        wall_v = [_f_safe_float(r.get("wall_s", "")) for r in chunk]
        w_w_m, w_w_s = _f_mean_std(wall_v)
        line = (
            f"{run_id} | phase={phase} | seeds={seeds} | "
            f"MSE={m_m:.6g}±{m_s:.3g} | Tvalid0.1={t_m:.4g}±{t_s:.3g} "
            f"| wall_s={w_w_m:.4g}±{w_w_s:.3g}s"
        )
        print(line)
        out_lines.append(line)

    out_txt = p_batch / "ablation_summary_mean_std.txt"
    out_txt.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    phase2_ids = sorted(r for r in by_run.keys() if r.startswith("II_"))
    paired_path = p_batch / "phase_II_seed_matched_delta.txt"
    plist: List[str] = []
    if len(phase2_ids) >= 2:
        def _indexed(bucket: List[Dict[str, str]]) -> Dict[int, Dict[str, Any]]:
            outm: Dict[int, Dict[str, Any]] = {}
            for rr in bucket:
                s = int(rr["seed"])
                outm[s] = {
                    "mse": _f_safe_float(rr.get("mse", "")),
                    "tvalid_0.1": _f_safe_float(rr.get("tvalid_0.1", "")),
                }
            return outm

        refs = {rid: _indexed(by_run[rid]) for rid in phase2_ids}
        base = refs.get("II_J0_S0_oracle_coarse_dt")
        if base is not None:
            for other in phase2_ids:
                if other == "II_J0_S0_oracle_coarse_dt":
                    continue
                dm, ds = _f_paired_delta(refs[other], base, "mse")
                plist.append(
                    f"{other} minus II_J0_S0 paired MSE delta "
                    f"mean ± std across seeds: {dm:.6g} ± {ds:.3g}",
                )

    paired_path.write_text(
        "\n".join(plist) if plist else "(no Phase II pairwise contrasts computed)\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_txt}")
    print(f"Wrote {paired_path}")


if __name__ == "__main__":
    main()
