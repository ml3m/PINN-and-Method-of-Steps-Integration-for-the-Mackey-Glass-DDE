#!/usr/bin/env python3
"""
Run run_mglass_comparison.py for several random seeds, aggregate metrics,
and plot variability (PINN training stochasticity).

Each seed writes to:  <base-dir>/seed_<seed>/

After all runs succeed:
  - multi_seed_metrics.csv
  - multi_seed_summary.json   (mean / std / median per n)
  - multi_seed_mse_bar.png
  - multi_seed_rel_l2_bar.png
  - multi_seed_timeseries_overlay.png
  - multi_seed_abs_error_overlay.png
  - multi_seed_learning_curves.png  (mean ± std ribbon over training steps; total / data / physics loss)

Example:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 run_mglass_multi_seed.py \\
    --config configs/config_mackey_glass_t20.yaml \\
    --base-output results/multi_seed_t20 \\
    --n-values 10 \\
    --n-seeds 10 --seed-start 1000

ROCm / venv: default --python is the interpreter running this script. A TensorFlow-only
.venv has no torch — install dependencies from this folder's ``requirements.txt`` (and
``requirements-rocm.txt`` for AMD GPUs), or pass ``--python`` to a ``python3`` that has PyTorch.

Or explicit seeds:
  python3 run_mglass_multi_seed.py --seeds 42,123,456,789,1024,2024,3141,5555,9999,12345
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MGLASS_COMPARISON_ROOT = Path(__file__).resolve().parent


def f_resolve_seed_config(p: Path) -> Path:
    p = Path(p).expanduser()
    if p.is_file():
        return p.resolve()
    alt = MGLASS_COMPARISON_ROOT / p
    if alt.is_file():
        return alt.resolve()
    return p.resolve()


def f_resolve_seed_base_output(p: Path) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (MGLASS_COMPARISON_ROOT / p).resolve()


def f_resolve_plot_n(
    v_rows: List[Dict[str, Any]],
    v_plot_n_arg: Optional[float],
    v_ns_from_cli: List[float],
) -> Optional[float]:
    if v_plot_n_arg is not None:
        return float(v_plot_n_arg)
    if len(v_ns_from_cli) == 1:
        return float(v_ns_from_cli[0])
    if v_rows:
        return float(min(r["n"] for r in v_rows))
    return None


def f_parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed PINN runs + aggregation")
    p.add_argument(
        "--config",
        type=Path,
        default=MGLASS_COMPARISON_ROOT / "configs" / "config_mackey_glass.yaml",
        help="YAML passed to run_mglass_comparison.py",
    )
    p.add_argument(
        "--base-output",
        type=Path,
        default=MGLASS_COMPARISON_ROOT / "results" / "multi_seed",
        help="Parent directory; each seed uses base-output/seed_<id>/",
    )
    p.add_argument("--n-values", default="10", help="Comma-separated n (Hill exponents)")
    p.add_argument(
        "--extra-args",
        default="",
        help='Extra args for run_mglass_comparison.py, e.g. \'--skip-pinn\' (quoted)',
    )
    p.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds (overrides --n-seeds / --seed-start if set)",
    )
    p.add_argument("--n-seeds", type=int, default=10, help="Number of seeds if --seeds empty")
    p.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="First seed when using --n-seeds (1000..1000+n-1)",
    )
    p.add_argument("--python", default=sys.executable, help="Python interpreter")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only scan base-output/seed_*/ and rebuild CSV/JSON/plots",
    )
    p.add_argument(
        "--no-runs",
        action="store_true",
        help="With --aggregate-only: skip subprocesses (default for aggregate-only)",
    )
    p.add_argument(
        "--ribbon-band",
        choices=("std", "sem", "ci95_mean"),
        default="std",
        help="Shading: std = mean±std across seeds; sem = mean±SE(mean); "
        "ci95_mean = 95%% CI for the mean (t-distribution, K>=2).",
    )
    p.add_argument(
        "--no-faint-runs",
        action="store_true",
        help="Do not draw faint per-seed loss curves behind the ribbon",
    )
    p.add_argument(
        "--plot-n",
        type=float,
        default=None,
        help="Hill exponent n for learning-curve ribbons and trajectory overlays "
        "(default: smallest n found in seed folders)",
    )
    return p.parse_args()


def f_resolve_seeds(v_args: argparse.Namespace) -> List[int]:
    if v_args.seeds.strip():
        return [int(x.strip()) for x in v_args.seeds.split(",") if x.strip()]
    return list(range(v_args.seed_start, v_args.seed_start + v_args.n_seeds))


def f_run_one(
    py: str,
    config: Path,
    out_dir: Path,
    n_values: str,
    seed: int,
    extra_argv: List[str],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        "-u",
        str(MGLASS_COMPARISON_ROOT / "run_mglass_comparison.py"),
        "--config",
        str(config),
        "--n-values",
        n_values,
        "--output-dir",
        str(out_dir),
        "--seed",
        str(seed),
    ]
    cmd.extend(extra_argv)
    env = os.environ.copy()
    env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    print("\n" + "=" * 72)
    print("RUN:", " ".join(cmd))
    print("=" * 72 + "\n", flush=True)
    r = subprocess.run(cmd, cwd=str(MGLASS_COMPARISON_ROOT), env=env)
    return int(r.returncode)


def f_pick_row_for_n(data: Dict[Any, Any], v_n_target: Optional[float]) -> Optional[Dict[str, Any]]:
    if not data:
        return None
    keys_sorted = sorted(data.keys(), key=float)
    if v_n_target is None:
        return data[keys_sorted[0]]
    if v_n_target in data:
        return data[v_n_target]
    for k in data:
        if abs(float(k) - float(v_n_target)) < 1e-9:
            return data[k]
    return None


def f_load_pickle_rows(seed_dir: Path) -> List[Dict[str, Any]]:
    pkl = seed_dir / "mglass_run.pkl"
    if not pkl.is_file():
        return []
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)
    # seed from folder name seed_1234
    try:
        seed = int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        seed = -1
    rows: List[Dict[str, Any]] = []
    for n_key, r in data.items():
        mp = r.get("metrics_pinn") or {}
        mc = r.get("metrics_classical") or {}
        v_pinn = r.get("valid_prediction_time_pinn") or {}
        v_cl = r.get("valid_prediction_time_classical") or {}
        w_tr = r.get("wall_time_pinn_train", r.get("wall_time_pinn", np.nan))
        w_if = r.get("wall_time_pinn_infer", np.nan)
        rows.append(
            {
                "seed": seed,
                "seed_dir": str(seed_dir),
                "n": float(n_key),
                "mse_pinn": float(mp.get("mse", np.nan)),
                "rel_l2_pinn": float(mp.get("rel_l2", np.nan)),
                "max_abs_pinn": float(mp.get("max_abs_err", np.nan)),
                "mse_classical": float(mc.get("mse", np.nan)),
                "rel_l2_classical": float(mc.get("rel_l2", np.nan)),
                "wall_pinn_s": float(r.get("wall_time_pinn", np.nan)),
                "wall_pinn_train_s": float(w_tr) if w_tr is not None else float("nan"),
                "wall_pinn_infer_s": float(w_if) if w_if is not None else float("nan"),
                "wall_classical_s": float(r.get("wall_time_classical", np.nan)),
                "t_valid_pinn_0.1": float(v_pinn.get("0.1", np.nan)),
                "t_valid_classical_0.1": float(v_cl.get("0.1", np.nan)),
                "valid_pinn_json": json.dumps(v_pinn) if v_pinn else "",
                "valid_classical_json": json.dumps(v_cl) if v_cl else "",
            }
        )
    return rows


def f_aggregate_base(v_base: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for d in sorted(v_base.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed_"):
            continue
        all_rows.extend(f_load_pickle_rows(d))

    summary: Dict[str, Any] = {"by_n": {}}
    by_n: Dict[float, List[Dict[str, Any]]] = {}
    for row in all_rows:
        n = row["n"]
        by_n.setdefault(n, []).append(row)

    for n, lst in sorted(by_n.items(), key=lambda x: x[0]):
        mse = np.array([x["mse_pinn"] for x in lst], dtype=np.float64)
        rl2 = np.array([x["rel_l2_pinn"] for x in lst], dtype=np.float64)
        mse = mse[np.isfinite(mse)]
        rl2 = rl2[np.isfinite(rl2)]
        key = str(int(n)) if n == int(n) else str(n)
        summary["by_n"][key] = {
            "runs": len(mse),
            "mse_pinn": {
                "mean": float(np.mean(mse)) if mse.size else None,
                "std": float(np.std(mse, ddof=1)) if mse.size > 1 else (float(np.std(mse)) if mse.size else 0.0),
                "median": float(np.median(mse)) if mse.size else None,
                "min": float(np.min(mse)) if mse.size else None,
                "max": float(np.max(mse)) if mse.size else None,
                "values": mse.tolist(),
            },
            "rel_l2_pinn": {
                "mean": float(np.mean(rl2)) if rl2.size else None,
                "std": float(np.std(rl2, ddof=1)) if rl2.size > 1 else (float(np.std(rl2)) if rl2.size else 0.0),
                "median": float(np.median(rl2)) if rl2.size else None,
                "min": float(np.min(rl2)) if rl2.size else None,
                "max": float(np.max(rl2)) if rl2.size else None,
                "values": rl2.tolist(),
            },
        }
    return all_rows, summary


def f_write_csv(v_rows: List[Dict[str, Any]], v_path: Path) -> None:
    if not v_rows:
        print("No rows to write to CSV.")
        return
    keys = [
        "seed",
        "n",
        "mse_pinn",
        "rel_l2_pinn",
        "max_abs_pinn",
        "mse_classical",
        "rel_l2_classical",
        "wall_pinn_s",
        "wall_pinn_train_s",
        "wall_pinn_infer_s",
        "wall_classical_s",
        "t_valid_pinn_0.1",
        "t_valid_classical_0.1",
        "valid_pinn_json",
        "valid_classical_json",
        "seed_dir",
    ]
    with open(v_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in sorted(v_rows, key=lambda r: (r["n"], r["seed"])):
            w.writerow(row)
    print(f"Wrote {v_path}")


def f_plot_bars(v_summary: Dict[str, Any], v_base: Path, metric_key: str, title: str, fname: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_n = v_summary.get("by_n", {})
    if not by_n:
        return
    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for k in sorted(by_n.keys(), key=lambda x: float(x)):
        block = by_n[k].get(metric_key, {})
        if block.get("mean") is None:
            continue
        labels.append(f"n={k}")
        means.append(float(block["mean"]))
        stds.append(float(block["std"]))

    if not means:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", ecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PINN " + ("MSE" if "mse" in metric_key else "relative L2"))
    ax.set_title(title + "\n(mean ± sample std over seeds)")
    fig.tight_layout()
    out = v_base / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def f_plot_overlay_timeseries(v_base: Path, v_n_target: Optional[float], fname: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: List[Tuple[int, np.ndarray, np.ndarray]] = []
    t_ref0: Optional[np.ndarray] = None
    x_ref0: Optional[np.ndarray] = None
    t_cl0: Optional[np.ndarray] = None
    x_cl0: Optional[np.ndarray] = None

    for d in sorted(v_base.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed_"):
            continue
        pkl = d / "mglass_run.pkl"
        if not pkl.is_file():
            continue
        with open(pkl, "rb") as fh:
            data = pickle.load(fh)
        keys = list(data.keys())
        if not keys:
            continue
        n_sel = v_n_target if v_n_target is not None else float(keys[0])
        r = data.get(n_sel) or data.get(float(n_sel))
        if r is None:
            for k in keys:
                if abs(float(k) - float(n_sel)) < 1e-6:
                    r = data[k]
                    break
        if r is None:
            continue
        try:
            seed = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        tt = np.asarray(r["t_pinn_test"]).flatten()
        xx = np.asarray(r["x_pinn"]).flatten()
        if tt.size == 0 or xx.size == 0:
            continue
        series.append((seed, tt, xx))
        if t_ref0 is None:
            t_ref0 = np.asarray(r["t_ref"]).flatten()
            x_ref0 = np.asarray(r["x_ref"]).flatten()
            t_cl0 = np.asarray(r["t_classical"]).flatten()
            x_cl0 = np.asarray(r["x_classical"]).flatten()

    if not series or t_ref0 is None:
        print("Skipping timeseries overlay (no PINN series found).")
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t_ref0, x_ref0, "k-", linewidth=0.9, alpha=0.35, label="Reference (RK4)")
    ax.plot(t_cl0, x_cl0, "b-", linewidth=1.0, alpha=0.7, label="Classical (MoS-RK45)")
    for seed, tt, xx in sorted(series, key=lambda s: s[0]):
        ax.plot(tt, xx, color="crimson", linewidth=0.55, alpha=0.35)
    ax.plot([], [], color="crimson", linewidth=1.5, label="PINN (each seed)")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    ax.set_title(
        f"PINN trajectories over seeds (n = {v_n_target if v_n_target is not None else '—'})"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = v_base / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def f_plot_overlay_abs_error(v_base: Path, v_n_target: Optional[float], fname: str) -> None:
    from scipy.interpolate import interp1d

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: List[Tuple[int, np.ndarray, np.ndarray]] = []
    t_ref0: Optional[np.ndarray] = None
    x_ref0: Optional[np.ndarray] = None

    for d in sorted(v_base.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed_"):
            continue
        pkl = d / "mglass_run.pkl"
        if not pkl.is_file():
            continue
        with open(pkl, "rb") as fh:
            data = pickle.load(fh)
        keys = list(data.keys())
        if not keys:
            continue
        n_sel = v_n_target if v_n_target is not None else float(keys[0])
        r = data.get(n_sel)
        if r is None:
            for k in keys:
                if abs(float(k) - float(n_sel)) < 1e-6:
                    r = data[k]
                    break
        if r is None:
            continue
        try:
            seed = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        tt = np.asarray(r["t_pinn_test"]).flatten()
        xx = np.asarray(r["x_pinn"]).flatten()
        if t_ref0 is None:
            t_ref0 = np.asarray(r["t_ref"]).flatten()
            x_ref0 = np.asarray(r["x_ref"]).flatten()
        x_ref_i = interp1d(t_ref0, x_ref0, kind="cubic", fill_value="extrapolate")
        err = np.abs(xx - x_ref_i(tt))
        series.append((seed, tt, err))

    if not series:
        print("Skipping abs-error overlay.")
        return

    fig, ax = plt.subplots(figsize=(11, 4.0))
    for seed, tt, err in sorted(series, key=lambda s: s[0]):
        ax.semilogy(tt, err + 1e-16, color="crimson", linewidth=0.55, alpha=0.4)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|x_{\mathrm{PINN}}(t) - x_{\mathrm{ref}}(t)|$")
    ax.set_title(
        f"Absolute error vs. time (n = {v_n_target if v_n_target is not None else '—'})"
    )
    fig.tight_layout()
    out = v_base / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def f_collect_loss_matrices(
    v_base: Path,
    v_n_target: Optional[float],
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Stack (n_seeds, T) loss arrays aligned to a common min length."""
    keys_series = ("loss_history", "data_loss_history", "physics_loss_history")
    per_seed: Dict[int, Dict[str, np.ndarray]] = {}
    for d in sorted(v_base.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed_"):
            continue
        try:
            seed = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        pkl = d / "mglass_run.pkl"
        if not pkl.is_file():
            continue
        with open(pkl, "rb") as fh:
            data = pickle.load(fh)
        row = f_pick_row_for_n(data, v_n_target)
        if row is None:
            continue
        series: Dict[str, np.ndarray] = {}
        valid = True
        for k in keys_series:
            raw = row.get(k)
            if not raw:
                valid = False
                break
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                valid = False
                break
            series[k] = arr
        if not valid:
            continue
        per_seed[seed] = series

    if not per_seed:
        return {}, []

    seeds_sorted = sorted(per_seed.keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys_series:
        t_min = min(per_seed[s][k].shape[0] for s in seeds_sorted)
        stacks = [per_seed[s][k][:t_min] for s in seeds_sorted]
        out[k] = np.stack(stacks, axis=0)

    return out, seeds_sorted


def f_plot_learning_curves_ribbon(
    v_base: Path,
    v_n_target: Optional[float],
    *,
    ribbon_band: str = "std",
    show_faint_runs: bool = True,
    fname: str = "multi_seed_learning_curves.png",
) -> None:
    """Mean ± variability across seeds over training steps (ribbon / learning curve)."""
    from scipy import stats

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats, seeds = f_collect_loss_matrices(v_base, v_n_target)
    if not mats:
        print("Skipping learning-curve ribbon (no loss histories found).")
        return

    spec = [
        ("loss_history", "Total loss", "#c0392b"),
        ("data_loss_history", "Data loss", "#2980b9"),
        ("physics_loss_history", "Physics residual loss", "#27ae60"),
    ]

    fig, axes = plt.subplots(len(spec), 1, figsize=(10, 8.5), sharex=True)
    if len(spec) == 1:
        axes = [axes]
    k_seeds = mats["loss_history"].shape[0]
    n_str = "all n" if v_n_target is None else f"n = {v_n_target:g}"
    fig.suptitle(
        f"Training curves (mean ± band over {k_seeds} seeds, {n_str})\n"
        f"band = {ribbon_band}",
        fontsize=11,
    )

    for ax, (key, label, color) in zip(axes, spec):
        y = mats[key]
        t = np.arange(y.shape[1])
        mean_c = np.mean(y, axis=0)
        if k_seeds > 1:
            std_c = np.std(y, axis=0, ddof=1)
            sem_c = std_c / np.sqrt(k_seeds)
        else:
            std_c = np.zeros_like(mean_c)
            sem_c = np.zeros_like(mean_c)

        if ribbon_band == "std":
            lo, hi = mean_c - std_c, mean_c + std_c
            band_label = r"$\pm$ std"
        elif ribbon_band == "sem":
            lo, hi = mean_c - sem_c, mean_c + sem_c
            band_label = r"$\pm$ SE"
        else:
            t_crit = float(stats.t.ppf(0.975, df=max(1, k_seeds - 1)))
            hw = t_crit * sem_c
            lo, hi = mean_c - hw, mean_c + hw
            band_label = r"95% CI (mean)"

        eps = 1e-30
        lo = np.maximum(lo, eps)
        hi = np.maximum(hi, eps)
        mean_pl = np.maximum(mean_c, eps)

        if show_faint_runs and k_seeds <= 20:
            for i in range(k_seeds):
                ax.semilogy(
                    t,
                    np.maximum(y[i], eps),
                    color=color,
                    alpha=0.12,
                    linewidth=0.7,
                )

        ax.fill_between(t, lo, hi, color=color, alpha=0.22, linewidth=0, label=band_label)
        ax.semilogy(t, mean_pl, color=color, linewidth=2.0, label="mean")

        ax.set_ylabel(label)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Training step (Adam iterations + L-BFGS steps, concatenated)")
    fig.tight_layout()
    out = v_base / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def f_run_aggregate_plots(
    v_base: Path,
    v_summary: Dict[str, Any],
    v_n_plot: Optional[float],
    *,
    ribbon_band: str,
    show_faint_runs: bool,
) -> None:
    f_plot_bars(v_summary, v_base, "mse_pinn", "PINN test MSE vs. RK4 reference", "multi_seed_mse_bar.png")
    f_plot_bars(
        v_summary,
        v_base,
        "rel_l2_pinn",
        "PINN relative L2 error vs. reference",
        "multi_seed_rel_l2_bar.png",
    )
    f_plot_overlay_timeseries(v_base, v_n_plot, "multi_seed_timeseries_overlay.png")
    f_plot_overlay_abs_error(v_base, v_n_plot, "multi_seed_abs_error_overlay.png")
    f_plot_learning_curves_ribbon(
        v_base,
        v_n_plot,
        ribbon_band=ribbon_band,
        show_faint_runs=show_faint_runs,
        fname="multi_seed_learning_curves.png",
    )


def f_child_has_torch(v_py: str) -> bool:
    r = subprocess.run(
        [v_py, "-c", "import torch"],
        cwd=str(MGLASS_COMPARISON_ROOT),
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def main() -> int:
    v_args = f_parse_args()
    v_base = f_resolve_seed_base_output(v_args.base_output)
    v_cfg = f_resolve_seed_config(v_args.config)
    v_base.mkdir(parents=True, exist_ok=True)

    extra = [x for x in v_args.extra_args.split() if x.strip()]

    if v_args.aggregate_only:
        v_rows, v_summary = f_aggregate_base(v_base)
        f_write_csv(v_rows, v_base / "multi_seed_metrics.csv")
        with open(v_base / "multi_seed_summary.json", "w") as fh:
            json.dump(v_summary, fh, indent=2)
        print(f"Wrote {v_base / 'multi_seed_summary.json'}")
        n_plot = f_resolve_plot_n(v_rows, v_args.plot_n, [])
        f_run_aggregate_plots(
            v_base,
            v_summary,
            n_plot,
            ribbon_band=v_args.ribbon_band,
            show_faint_runs=not v_args.no_faint_runs,
        )
        return 0

    if not f_child_has_torch(str(v_args.python)):
        print(
            "Error: PyTorch is not importable with "
            f"{v_args.python!r}. Child runs need torch.\n"
            f"  • In the bundle root: pip install -r {MGLASS_COMPARISON_ROOT / 'requirements.txt'}\n"
            "    (that file installs a CPU PyTorch build suitable for reproduction.)\n"
            "  • For AMD ROCm or NVIDIA CUDA, reinstall torch per README.rst and "
            "requirements-rocm.txt / https://pytorch.org/get-started/\n"
            "  • Or pass --python to an interpreter where torch already works.",
            file=sys.stderr,
        )
        return 1

    seeds = f_resolve_seeds(v_args)
    failed: List[int] = []
    for s in seeds:
        sub = v_base / f"seed_{s}"
        code = f_run_one(
            str(v_args.python),
            v_cfg,
            sub,
            v_args.n_values.strip(),
            s,
            extra,
        )
        if code != 0:
            failed.append(s)
            print(f"[WARN] Seed {s} exited with code {code}", file=sys.stderr)

    v_rows, v_summary = f_aggregate_base(v_base)
    f_write_csv(v_rows, v_base / "multi_seed_metrics.csv")
    with open(v_base / "multi_seed_summary.json", "w") as fh:
        json.dump(v_summary, fh, indent=2)
    print(f"Wrote {v_base / 'multi_seed_summary.json'}")

    ns = [float(x.strip()) for x in v_args.n_values.split(",") if x.strip()]
    n_plot = f_resolve_plot_n(v_rows, v_args.plot_n, ns)
    f_run_aggregate_plots(
        v_base,
        v_summary,
        n_plot,
        ribbon_band=v_args.ribbon_band,
        show_faint_runs=not v_args.no_faint_runs,
    )

    if failed:
        print(f"Failed seeds: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
