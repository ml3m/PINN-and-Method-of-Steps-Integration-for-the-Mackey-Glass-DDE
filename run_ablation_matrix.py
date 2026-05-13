#!/usr/bin/env python3
"""
Batch driver for reproducible PINN ablation studies on Mackey-Glass.

Reads a YAML manifest (see ``configs/ablations/``), merges ``merge_patch``
into ``base_config`` via ``f_deep_merge_config``, trains with fixed metric
references, writes ``run_manifest.json`` plus one CSV row appended to an
aggregate file.

Typical invocation (full Phase I, GPU recommended)::

    cd mglass_comparison
    python run_ablation_matrix.py --manifest configs/ablations/manifest_phase_I.yaml

Smoke::

    python run_ablation_matrix.py \\
        --manifest configs/ablations/manifest_smoke.yaml --overwrite
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

MROOT = Path(__file__).resolve().parent
if str(MROOT) not in sys.path:
    sys.path.insert(0, str(MROOT))

import run_mglass_comparison as mg


DEFAULT_AGG_COLUMNS = (
    [
        "batch_stamp",
        "run_id",
        "phase_id",
        "seed",
        "wall_s",
        "git_rev",
        "torch",
        "cuda_available",
        "gpu_name",
        "ref_ck16",
        "ref_dt",
        "mse",
        "rel_l2",
        "max_abs_err",
        "status",
        "failure_reason",
    ]
)


def _f_sha16(p_arr: np.ndarray) -> str:
    buf = np.asarray(p_arr, dtype=np.float64).tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def _f_git_revision() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _f_torch_env() -> Dict[str, str]:
    import torch

    gpu = ""
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda-available"
    return {
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "gpu_name": gpu,
    }


def _f_aggregate_row_path(p_root: Path) -> Path:
    p_root.mkdir(parents=True, exist_ok=True)
    return p_root / "ablation_aggregate.csv"


def _f_append_aggregate(
    p_csv: Path, fieldnames: List[str], row: Dict[str, Any],
) -> None:
    existed = p_csv.is_file()
    with open(p_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not existed:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mackey-Glass PINN ablation manifest runner")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(MROOT / "configs" / "ablations" / "manifest_phase_I.yaml"),
        help="YAML manifest describing base config and runs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete per-seed checkpoint dir before training if present.",
    )
    v = ap.parse_args()

    man_path = Path(v.manifest).expanduser().resolve()
    if not man_path.is_file():
        alt = MROOT / v.manifest
        if alt.is_file():
            man_path = alt.resolve()
        else:
            raise SystemExit(f"Manifest not found: {v.manifest}")

    with open(man_path, "r", encoding="utf-8") as fh:
        manifest_doc: Dict[str, Any] = yaml.safe_load(fh)

    base_path = manifest_doc["base_config_path"]
    base_resolved = mg.f_resolve_bundle_config_path(str(base_path))
    with open(base_resolved, "r", encoding="utf-8") as fh:
        base_yaml: Dict[str, Any] = yaml.safe_load(fh)

    out_root_rel = manifest_doc.get("output_root", "results/ablation_runs")
    out_root = Path(mg.f_resolve_bundle_output_dir(str(out_root_rel)))
    failure_mse_mul = float(manifest_doc.get("failure_mse_multiplier", 10.0))

    beta = float(base_yaml["problem"]["beta_true"])
    gamma = float(base_yaml["problem"]["gamma_true"])
    tau = float(base_yaml["problem"]["tau"])
    x0_hist = float(base_yaml["problem"].get("initial_x_history", 1.2))
    t_total = float(base_yaml["data"]["t_total"])
    n_val_global = float(manifest_doc["n_value"])

    ref_dt = float(manifest_doc.get("ref_dt", 0.001))
    trials: List[Dict[str, Any]] = manifest_doc.get("runs", [])
    stamp = manifest_doc.get("batch_stamp") or _dt.datetime.now(
        tz=_dt.timezone.utc,
    ).strftime("%Y%m%d_%H%M%S")

    v_t_ref, v_x_ref = mg.f_solve_mackey_glass_rk4_fixed(
        beta, gamma, n_val_global, tau, x0_hist, t_total, ref_dt,
    )
    ref_ck = _f_sha16(v_x_ref)

    l_all_mse_seed: Dict[str, List[float]] = {}

    git_rev = _f_git_revision()
    torch_env = _f_torch_env()

    default_th = [0.05, 0.1, 0.2]
    th_list = manifest_doc.get("valid_thresholds", default_th)

    agg_fields = list(manifest_doc.get("aggregate_columns", DEFAULT_AGG_COLUMNS))
    for th in th_list:
        key = f"tvalid_{float(th)}"
        if key not in agg_fields:
            agg_fields.append(key)

    for trial in trials:
        run_id = str(trial["run_id"]).strip()
        phase_lbl = str(trial.get("phase_id", "")).strip()
        merge_patch = trial.get("merge_patch") or {}
        seeds_raw = trial.get("seeds")
        if seeds_raw is None:
            seeds_raw = manifest_doc["default_seeds"]
        seeds_list = [int(s) for s in seeds_raw]

        for seed_val in seeds_list:
            merged = mg.f_deep_merge_config(base_yaml, merge_patch)
            merged["_seed"] = seed_val
            n_for_ckpt = float(merged["problem"].get("n", n_val_global))

            run_dir = out_root / stamp / run_id / f"seed_{seed_val}"
            merged["_output_dir"] = str(run_dir.resolve())

            ckpt_rel = run_dir / f"checkpoints_n{n_for_ckpt:g}"
            if v.overwrite and ckpt_rel.is_dir():
                shutil.rmtree(ckpt_rel)

            t0_wall = time.perf_counter()
            merged_record = yaml.safe_dump(merged)
            diag: Dict[str, Any] = {"status": "UNKNOWN", "failure_reason": ""}
            fd: Dict[str, float] = {}

            try:
                d_out = mg.f_train_pinn_mackey_glass(
                    merged,
                    n_for_ckpt,
                    p_metric_t_ref=v_t_ref,
                    p_junction_t_ref=v_t_ref,
                    p_junction_x_ref=v_x_ref,
                )
            except Exception as ex:
                diag["status"] = "EXCEPTION"
                diag["failure_reason"] = repr(ex)
                d_out = {}

            elapsed = time.perf_counter() - t0_wall

            u_metric = np.asarray(d_out.get("u_pred_metric_ref")).ravel()
            metrics = {
                "mse": float("nan"),
                "rel_l2": float("nan"),
                "max_abs_err": float("nan"),
            }

            if u_metric.size == v_x_ref.size:
                metrics = mg.f_compute_metrics(v_x_ref, u_metric)
                fd = mg.f_first_exceedance_times(v_t_ref, v_x_ref, u_metric, th_list)

            if diag["status"] == "EXCEPTION":
                pass
            elif metrics["mse"] != metrics["mse"] or np.isinf(metrics["mse"]):
                diag["status"] = "FAILED"
                diag["failure_reason"] = "bad_mse"
            elif metrics["mse"] > 1e6:
                diag["status"] = "FAILED"
                diag["failure_reason"] = "exploded_mse"
            else:
                diag["status"] = "SUCCESS"

            l_all_mse_seed.setdefault(run_id, []).append(metrics.get("mse", float("nan")))

            t_valid_vals = {f"tvalid_{k}": vv for k, vv in fd.items()}

            row_flat: Dict[str, Any] = {
                "batch_stamp": stamp,
                "run_id": run_id,
                "phase_id": phase_lbl,
                "seed": seed_val,
                "wall_s": elapsed,
                "git_rev": git_rev,
                **torch_env,
                "ref_ck16": ref_ck,
                "ref_dt": ref_dt,
                **metrics,
                **t_valid_vals,
                "status": diag["status"],
                "failure_reason": diag["failure_reason"],
            }

            agg_path = _f_aggregate_row_path(out_root / stamp)
            run_dir.mkdir(parents=True, exist_ok=True)
            mf = run_dir / "run_manifest.json"
            with open(mf, "w", encoding="utf-8") as jf:
                json.dump(
                    {
                        "run_id": run_id,
                        "phase_id": phase_lbl,
                        "seed": seed_val,
                        "merge_patch_yaml": yaml.safe_dump(merge_patch),
                        "merged_yaml": merged_record,
                        "git_revision": git_rev,
                        **torch_env,
                        "reference_x_ck16": ref_ck,
                        "metrics": metrics,
                        "diag": diag,
                        "trial_valid_times": fd,
                    },
                    jf,
                    indent=2,
                )

            _f_append_aggregate(agg_path, agg_fields, row_flat)

        mids_run = sorted([x for x in l_all_mse_seed.get(run_id, []) if x == x])
        med_run = mids_run[len(mids_run) // 2] if mids_run else float("nan")
        rb = manifest_doc.get("baseline_run_id_for_failure")
        if rb and isinstance(rb, str) and failure_mse_mul > 1:
            base_list = sorted(
                [x for x in l_all_mse_seed.get(rb, []) if x == x],
            )
            bm = base_list[len(base_list) // 2] if base_list else float("nan")
            if bm == bm and bm > 0 and med_run == med_run and med_run > failure_mse_mul * bm:
                print(
                    f"[warn] {run_id} median_mse={med_run:g} exceeds "
                    f"{failure_mse_mul}× baseline {bm:g} ({rb})",
                )

    agg_final = _f_aggregate_row_path(out_root / stamp)
    print(f"Finished batch {stamp}. Outputs under {out_root / stamp}")
    print(f"Aggregate CSV: {agg_final}")


if __name__ == "__main__":
    main()
