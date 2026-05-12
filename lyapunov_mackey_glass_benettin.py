#!/usr/bin/env python3
"""
Largest Lyapunov exponent (continuous-time scale) for Mackey--Glass trajectories
under the **same frozen-delay explicit RK4 step** as ``f_solve_mackey_glass_rk4_fixed``.

State on the delay line: z = (x_k, x_{k-1}, ..., x_{k-m})^T with m = tau/h.
The Benettin algorithm uses sparse J_k @ q (only O(m) work per step).

Usage::

    cd mglass_comparison
    python lyapunov_mackey_glass_benettin.py --output-csv results/lyapunov_n10.csv
    python lyapunov_mackey_glass_benettin.py \\
        --pinn-checkpoints-dir mglass_comparison/results/multi_seed_n10_t200/seed_42/checkpoints_n10 \\
        --config configs/config_mackey_glass_t200_windowed.yaml
    python lyapunov_mackey_glass_benettin.py \\
        --multi-seed-root mglass_comparison/results/multi_seed_n10_t200 \\
        --config configs/config_mackey_glass_t200_windowed.yaml \\
        --output-csv mglass_comparison/results/lyapunov_n10_per_seed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

MGLASS_ROOT = Path(__file__).resolve().parent
if str(MGLASS_ROOT) not in sys.path:
    sys.path.insert(0, str(MGLASS_ROOT))

import run_mglass_comparison as mg  # noqa: E402


def f_rk4_step_frozen(
    x: float,
    x_tau: float,
    h: float,
    beta: float,
    gamma: float,
    n_hill: float,
) -> float:
    def _f_rhs(xt: float, xdel: float) -> float:
        return beta * xdel / (1.0 + abs(xdel) ** n_hill) - gamma * xt

    k1 = _f_rhs(x, x_tau)
    k2 = _f_rhs(x + 0.5 * h * k1, x_tau)
    k3 = _f_rhs(x + 0.5 * h * k2, x_tau)
    k4 = _f_rhs(x + h * k3, x_tau)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def f_dphi_frozen_centered(
    x: float,
    x_tau: float,
    h: float,
    beta: float,
    gamma: float,
    n_hill: float,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """Centered finite differences ∂Φ/∂x, ∂Φ/∂x_tau for one frozen-delay RK4 step."""

    def _phi(a: float, b: float) -> float:
        return f_rk4_step_frozen(a, b, h, beta, gamma, n_hill)

    d_dx = (_phi(x + eps, x_tau) - _phi(x - eps, x_tau)) / (2.0 * eps)
    d_dtau = (_phi(x, x_tau + eps) - _phi(x, x_tau - eps)) / (2.0 * eps)
    return float(d_dx), float(d_dtau)


def f_J_matvec_frozen(
    q: np.ndarray,
    dphi_dx: float,
    dphi_dxtau: float,
) -> np.ndarray:
    """Jacobian of delay-line map (first row: Φ w.r.t. x, x_tau; shift down)."""
    out = np.zeros_like(q)
    out[0] = dphi_dx * q[0] + dphi_dxtau * q[-1]
    out[1:] = q[:-1]
    return out


def f_largest_lyapunov_benettin(
    x_traj: np.ndarray,
    h: float,
    delay_steps: int,
    beta: float,
    gamma: float,
    n_hill: float,
    burn_in_steps: int,
    eps_fd: float = 1e-7,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Largest Lyapunov exponent (1/s) for frozen-delay RK4 map along x_traj."""
    x_traj = np.asarray(x_traj, dtype=np.float64).ravel()
    m = int(delay_steps)
    if m < 1:
        raise ValueError("delay_steps must be positive (integer delay / grid).")
    if x_traj.size < m + 2:
        raise ValueError("trajectory too short for delay embedding")
    n_steps = x_traj.size - 1
    k0 = max(m, int(burn_in_steps))
    if k0 >= n_steps - 1:
        raise ValueError("burn-in consumes trajectory; lower burn_in_steps")

    if rng is None:
        rng = np.random.default_rng(0)
    q = rng.standard_normal(m + 1)
    q /= np.linalg.norm(q) + 1e-30

    acc_log: float = 0.0
    n_used = 0
    for k in range(k0, n_steps):
        xk = float(x_traj[k])
        xkm = float(x_traj[k - m])
        d0, dtm = f_dphi_frozen_centered(
            xk, xkm, h, beta, gamma, n_hill, eps=eps_fd,
        )
        v = f_J_matvec_frozen(q, d0, dtm)
        nv = float(np.linalg.norm(v))
        if nv < 1e-300 or not math.isfinite(nv):
            raise RuntimeError(f"Benettin norm collapsed at k={k}")
        acc_log += math.log(nv)
        q = v / nv
        n_used += 1

    t_spanned = float(n_used) * h
    lambda1 = acc_log / t_spanned if t_spanned > 0 else float("nan")
    return {
        "lambda1": float(lambda1),
        "n_steps_used": float(n_used),
        "t_spanned": t_spanned,
        "delay_steps": float(m),
        "h": float(h),
    }


def f_load_stitched_pinn_on_ref_grid(
    checkpoints_dir: Path,
    config_path: Path,
    n_hill: float,
    t_grid: np.ndarray,
    device_str: str = "cpu",
) -> np.ndarray:
    """Rebuild window networks from ``window_*.pt`` and stitch on ``t_grid``."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    v_layers_cfg = cfg["network"]["layers"]
    v_hidden_size = int(v_layers_cfg[1]) if len(v_layers_cfg) > 2 else 256
    v_hidden_layers = len(v_layers_cfg) - 2
    v_activation = str(cfg["network"].get("activation", "sine"))
    v_siren_init = cfg["network"].get("initialization", "siren") == "siren"
    v_fourier = bool(cfg["network"].get("fourier_features", True))
    v_fourier_dim = int(cfg["network"].get("fourier_dim", 64))
    v_fourier_scale = float(cfg["network"].get("fourier_scale", 5.0))

    import torch

    device = torch.device(device_str)
    paths = sorted(
        glob.glob(str(checkpoints_dir / "window_*.pt")),
        key=lambda p: int(Path(p).stem.split("_")[1]),
    )
    if not paths:
        raise FileNotFoundError(f"No window_*.pt in {checkpoints_dir}")

    models: List[Tuple[float, float, Any]] = []
    for pth in paths:
        ck = torch.load(pth, map_location=device, weights_only=False)
        model = mg.MackeyGlassPINN(
            hidden_layers=v_hidden_layers,
            hidden_size=v_hidden_size,
            activation=v_activation,
            siren_init=v_siren_init,
            fourier_features=v_fourier,
            fourier_dim=v_fourier_dim,
            fourier_scale=v_fourier_scale,
        ).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        w0 = float(ck["win_t0"])
        w1 = float(ck["win_t1"])
        models.append((w0, w1, model))

    x_pinn = mg.f_stitch_pinn_on_grid(models, t_grid, device)
    return np.asarray(x_pinn, dtype=np.float64).ravel()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--x0", type=float, default=1.2)
    p.add_argument("--t-end", type=float, default=200.0)
    p.add_argument("--ref-dt", type=float, default=1e-3)
    p.add_argument("--burn-in-fraction", type=float, default=0.15)
    p.add_argument(
        "--pinn-checkpoints-dir",
        type=Path,
        default=None,
        help="Directory with window_*.pt (e.g. seed_42/checkpoints_n10)",
    )
    p.add_argument(
        "--multi-seed-root",
        type=Path,
        default=None,
        help="Parent folder containing seed_*/checkpoints_n10 (runs all seeds).",
    )
    p.add_argument(
        "--checkpoints-subdir",
        type=str,
        default="checkpoints_n10",
        help="Under each seed_* directory (default checkpoints_n10).",
    )
    p.add_argument(
        "--lyapunov-seed",
        type=int,
        default=42,
        help="RNG seed for Benettin initial tangent vector (same for all orbits).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=MGLASS_ROOT / "configs" / "config_mackey_glass_t200_windowed.yaml",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=MGLASS_ROOT / "results" / "lyapunov_n10.csv",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for PINN ensemble summary (multi-seed mode).",
    )
    args = p.parse_args()

    if args.multi_seed_root is not None and args.pinn_checkpoints_dir is not None:
        raise SystemExit("Use either --multi-seed-root or --pinn-checkpoints-dir, not both.")

    h = float(args.ref_dt)
    _, x_ref = mg.f_solve_mackey_glass_rk4_fixed(
        args.beta, args.gamma, args.n, args.tau,
        args.x0, args.t_end, h,
    )
    m = int(round(args.tau / h))
    if abs(m * h - args.tau) > 1e-9:
        raise SystemExit("tau/ref_dt must be integer ratio for this delay-line map.")
    burn = int(args.burn_in_fraction * (x_ref.size - 1))

    ly_rng = np.random.default_rng(int(args.lyapunov_seed))
    d_ref = f_largest_lyapunov_benettin(
        x_ref, h, m, args.beta, args.gamma, args.n, burn, rng=ly_rng,
    )
    print("Reference trajectory (fine RK4):")
    print(f"  lambda_1 ≈ {d_ref['lambda1']:.6f} /s  (T={d_ref['t_spanned']:.3f}, steps={int(d_ref['n_steps_used'])})")

    rows: List[Dict[str, Any]] = [
        {
            "trajectory": "rk4_ref",
            "seed": "",
            **d_ref,
        },
    ]

    def _one_pinn_row(ckpt_dir: Path, seed_label: str) -> Dict[str, Any]:
        t_grid = np.linspace(0.0, args.t_end, x_ref.size)
        ly_rng = np.random.default_rng(int(args.lyapunov_seed))
        x_pn = f_load_stitched_pinn_on_ref_grid(
            ckpt_dir,
            args.config,
            args.n,
            t_grid,
            device_str=args.device,
        )
        if x_pn.shape != x_ref.shape:
            raise SystemExit("PINN stitch length mismatch")
        d_pn = f_largest_lyapunov_benettin(
            x_pn, h, m, args.beta, args.gamma, args.n, burn, rng=ly_rng,
        )
        return {"trajectory": "pinn_stitched", "seed": seed_label, **d_pn}

    if args.multi_seed_root is not None:
        root = args.multi_seed_root
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        seed_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name.startswith("seed_")
        )
        lam_pinn: List[float] = []
        for sd in seed_dirs:
            cdir = sd / args.checkpoints_subdir
            if not cdir.is_dir():
                print(f"  [skip] no {cdir}")
                continue
            label = sd.name.replace("seed_", "")
            d_pn = _one_pinn_row(cdir, label)
            rows.append(d_pn)
            lam_pinn.append(float(d_pn["lambda1"]))
            print(f"  seed {label}: lambda_1 ≈ {d_pn['lambda1']:.6f} /s")
        if lam_pinn:
            arr = np.asarray(lam_pinn, dtype=np.float64)
            summ = {
                "pinn_lambda1_mean": float(np.mean(arr)),
                "pinn_lambda1_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "pinn_lambda1_min": float(np.min(arr)),
                "pinn_lambda1_max": float(np.max(arr)),
                "pinn_lambda1_values": [float(x) for x in arr],
                "seeds_processed": [r["seed"] for r in rows if r["trajectory"] == "pinn_stitched"],
                "lyapunov_seed": int(args.lyapunov_seed),
                "rk4_lambda1": float(d_ref["lambda1"]),
            }
            jpath = args.summary_json
            if jpath is None:
                jpath = args.output_csv.with_name(
                    args.output_csv.stem + "_summary.json",
                )
            jpath.parent.mkdir(parents=True, exist_ok=True)
            with open(jpath, "w", encoding="utf-8") as jf:
                json.dump(summ, jf, indent=2)
            print(
                f"  PINN ensemble: mean {summ['pinn_lambda1_mean']:.6f} /s, "
                f"std {summ['pinn_lambda1_std']:.6f}, "
                f"[{summ['pinn_lambda1_min']:.6f}, {summ['pinn_lambda1_max']:.6f}]",
            )
            print(f"Wrote {jpath.resolve()}")

    elif args.pinn_checkpoints_dir is not None:
        d_pn = _one_pinn_row(args.pinn_checkpoints_dir, "")
        rows.append(d_pn)
        print("Stitched PINN (same map linearized along surrogate):")
        print(f"  lambda_1 ≈ {d_pn['lambda1']:.6f} /s  (T={d_pn['t_spanned']:.3f}, steps={int(d_pn['n_steps_used'])})")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["trajectory", "seed", "lambda1", "n_steps_used", "t_spanned", "delay_steps", "h"]
    with open(args.output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
