#!/usr/bin/env python3
"""
Mackey-Glass PINN comparison: Classical DDE Solver vs PINN for Mackey-Glass Equation.

PyTorch PINN with ROCm GPU acceleration (AMD RX 6700S / gfx1032).

Produces KdV-style comparison figures:
  - Solution heatmap u(t) with marked training snapshot locations
  - Training data overlays at selected time windows
  - Quantitative comparison table (MSE, relative discrete l^2_N on grid, wall time)
  - Delay-coordinate embedding visualizations

Usage (from the ``mglass_comparison/`` directory, or with absolute paths):

    python run_mglass_comparison.py \\
        [--config configs/config_mackey_glass_t200_windowed.yaml] \\
        [--n-values 10] [--output-dir results/my_run] \\
        [--classical-dt 0.01] [--ref-dt 0.001] \\
        [--reference-convergence] [--valid-thresholds 0.05,0.1,0.2]

YAML recipes live in ``mglass_comparison/configs/``. By default, figures and ``mglass_run.pkl``
are written under ``mglass_comparison/results/`` (see ``--output-dir``).

    The ``config_mackey_glass.yaml`` file keeps a shorter ``[0,100]``
    single-window setup; ``config_mackey_glass_t200_windowed.yaml`` is
    the windowed recipe on ``[0,200]`` used for the IEEE paper figures,
    with ``L=25``, ``O=5``.
    ``config_mackey_glass_t100_windowed.yaml`` is the shorter ``[0,100]`` windowed benchmark.
    Window construction is in ``f_build_time_windows``.

Batch ablations (junction mode, supervision orbit, coupled delay gradients, curriculum,
``n_lbfgs`` overrides) load YAML manifests via ``run_ablation_matrix.py`` and merge
dictionaries using ``f_deep_merge_config``.

Re-export overlay or heatmap from a finished run (see ``--only-3d-overlay`` /
``--only-heatmap``)::

    python run_mglass_comparison.py --only-heatmap \\
        --output-dir results/my_run --n-values 10

For a Navier-Stokes-style multi-term loss vs iteration figure (plus IC-band
breakdown and a three-stack IC/modifiers plot), use ``--only-loss-convergence``.

``--only-heatmap`` reads ``mglass_run.pkl`` (same as ``--only-3d-overlay``).
If ``OUTPUT_DIR/mglass_run.pkl`` is missing but a nested
``mglass_comparison/<relative-output>/mglass_run.pkl`` exists (duplicate bundle folder),
it is picked up automatically; otherwise use ``--from-pkl``.

Writes ``n10_3d_overlay.png``, ``n10_3d_overlay_pov_*.png`` for the other cameras,
and ``n10_3d_overlay_2x2.png`` (PDFs too unless ``--no-pdf``).

``mglass_run.pkl`` is resolved as above unless you pass an explicit ``--from-pkl`` path.
"""

import os
import random
import re

# ROCm override MUST be set before importing torch
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import sys
import math
import json
import time
import copy
import pickle
import platform
import argparse
import csv
import glob
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as _e:
    v_root = Path(__file__).resolve().parent
    raise SystemExit(
        "PyTorch is required but not installed.\n\n"
        "From the directory that contains this script, run:\n"
        f"  pip install -r {v_root / 'requirements.txt'}\n\n"
        "That file installs a CPU build of PyTorch suitable for reproduction on any\n"
        "machine. For GPU instructions see README.rst and requirements-rocm.txt.\n\n"
        f"Original error: {_e}"
    ) from _e

import matplotlib
# Batch jobs: Agg (default). Interactive tools: set MGLASS_MPL_INTERACTIVE=1 before import.
if os.environ.get("MGLASS_MPL_INTERACTIVE", "").lower() in ("1", "true", "yes"):
    matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"), force=True)
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Root of the Reproducibility bundle (this directory).
MGLASS_COMPARISON_ROOT = Path(__file__).resolve().parent

# If False (see ``--no-pdf``), skip PDF export — ``bbox_inches="tight"`` PDFs are slow.
_MGLASS_SAVE_PDF = True


def f_resolve_bundle_config_path(raw: str) -> str:
    """Resolve YAML path: CWD, then ``mglass_comparison/`` bundle (``MGLASS_COMPARISON_ROOT``)."""
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())
    alt = MGLASS_COMPARISON_ROOT / raw
    if alt.is_file():
        return str(alt.resolve())
    return str(p.resolve())


def f_resolve_bundle_output_dir(raw: str) -> str:
    """Place relative output paths under ``mglass_comparison/`` unless absolute."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((MGLASS_COMPARISON_ROOT / p).resolve())


def f_resolve_mglass_run_pkl_for_export(
    resolved_output_dir: str,
    from_pkl: Optional[str],
    mode_label: str,
) -> Tuple[str, str]:
    """Locate ``mglass_run.pkl`` for export-only CLI modes.

    Relative ``--output-dir`` resolves under ``MGLASS_COMPARISON_ROOT``. Some runs
    store artifacts under ``mglass_comparison/results/…`` inside that bundle
    (duplicate ``mglass_comparison`` segment). When the default pickle is missing,
    we try that nested folder before exiting with a hint.
    """
    v_od = Path(resolved_output_dir).resolve()
    if from_pkl is not None:
        v_manual = Path(from_pkl).expanduser()
        if not v_manual.is_absolute():
            v_manual = (MGLASS_COMPARISON_ROOT / v_manual).resolve()
        else:
            v_manual = v_manual.resolve()
        if not v_manual.is_file():
            raise SystemExit(f"{mode_label}: pickle not found (--from-pkl): {v_manual}")
        return str(v_manual), str(v_manual.parent)

    v_prim = v_od / "mglass_run.pkl"
    if v_prim.is_file():
        return str(v_prim), str(v_od)

    v_rel_od: Optional[Path]
    try:
        v_rel_od = v_od.relative_to(MGLASS_COMPARISON_ROOT.resolve())
    except ValueError:
        v_rel_od = None

    if v_rel_od is not None:
        v_nested_od = (MGLASS_COMPARISON_ROOT / "mglass_comparison" / v_rel_od).resolve()
        v_nested = v_nested_od / "mglass_run.pkl"
        if v_nested.is_file():
            print(
                f"{mode_label}: using nested pickle ({v_nested}) "
                f"(not present at {v_prim})",
            )
            return str(v_nested), str(v_nested_od)

    v_hints = ""
    if v_rel_od is not None:
        v_hints = (
            "\nRuns may live under a duplicated bundle folder.\nTry either:\n"
            f"  --output-dir mglass_comparison/{v_rel_od.as_posix()}\n"
            "  --from-pkl /absolute/path/to/mglass_run.pkl\n"
        )
    else:
        v_hints = "\nPass an absolute pickle path:\n  --from-pkl /path/to/mglass_run.pkl\n"
    raise SystemExit(f"{mode_label}: pickle not found: {v_prim}{v_hints}")


def f_deep_merge_config(base: Any, patch: Any) -> Any:
    """Recursively overlay dict ``patch`` onto ``copy.deepcopy(base)``."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out: Dict[str, Any] = (
        copy.deepcopy(base) if isinstance(base, dict) else {}
    )
    if not isinstance(out, dict):
        out = {}
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = f_deep_merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Classical DDE Solver (Method of Steps + RK45)
# ═══════════════════════════════════════════════════════════════════════════════

def f_solve_mackey_glass_classical(
    p_beta: float,
    p_gamma: float,
    p_n: float,
    p_tau: float,
    p_x0: float,
    p_t_end: float,
    p_dt: float = 0.01,
    p_rtol: float = 1e-9,
    p_atol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Mackey-Glass DDE using method-of-steps with adaptive RK45.

    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    On each delay-length interval [k*tau, (k+1)*tau], the delayed term
    x(t-tau) is known from the previous interval and supplied via cubic
    interpolation of the stored history buffer.

    Returns:
        (t_array, x_array) on a uniform grid with spacing p_dt.
    """
    v_n_intervals = int(np.ceil(p_t_end / p_tau))
    v_t_dense = np.arange(0.0, p_t_end + p_dt * 0.5, p_dt)

    v_history_t = np.array([-p_tau, 0.0])
    v_history_x = np.array([p_x0, p_x0])
    v_interp_history = interp1d(
        v_history_t, v_history_x, kind="linear", fill_value=p_x0, bounds_error=False
    )

    l_all_t = [0.0]
    l_all_x = [p_x0]

    v_x_current = p_x0

    for v_k in range(v_n_intervals):
        v_t_start = v_k * p_tau
        v_t_stop = min((v_k + 1) * p_tau, p_t_end)
        if v_t_start >= p_t_end:
            break

        v_interp_fn = v_interp_history

        def f_rhs(p_t, p_y, _interp=v_interp_fn):
            v_x = p_y[0]
            v_x_delayed = float(_interp(p_t - p_tau))
            v_dxdt = (
                p_beta * v_x_delayed / (1.0 + abs(v_x_delayed) ** p_n)
                - p_gamma * v_x
            )
            return [v_dxdt]

        v_t_eval_local = v_t_dense[
            (v_t_dense > v_t_start + 1e-14) & (v_t_dense <= v_t_stop + 1e-14)
        ]
        if len(v_t_eval_local) == 0:
            continue

        v_sol = solve_ivp(
            f_rhs,
            [v_t_start, v_t_stop],
            [v_x_current],
            method="RK45",
            t_eval=v_t_eval_local,
            rtol=p_rtol,
            atol=p_atol,
            max_step=p_dt,
        )

        if not v_sol.success:
            warnings.warn(f"RK45 failed on interval [{v_t_start:.2f}, {v_t_stop:.2f}]: {v_sol.message}")
            break

        l_all_t.extend(v_sol.t.tolist())
        l_all_x.extend(v_sol.y[0].tolist())
        v_x_current = v_sol.y[0, -1]

        v_combined_t = np.array(l_all_t)
        v_combined_x = np.array(l_all_x)
        v_interp_history = interp1d(
            v_combined_t, v_combined_x, kind="cubic", fill_value="extrapolate"
        )

    v_t_out = np.array(l_all_t)
    v_x_out = np.array(l_all_x)
    return v_t_out, v_x_out


def f_solve_mackey_glass_rk4_fixed(
    p_beta: float,
    p_gamma: float,
    p_n: float,
    p_tau: float,
    p_x0: float,
    p_t_end: float,
    p_dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step RK4 DDE solver with explicit history buffer.
    Mirrors the existing data_generators.py implementation but exposed as
    a standalone baseline for comparison.
    """
    v_n_steps = int(np.round(p_t_end / p_dt))
    v_t = np.linspace(0.0, p_t_end, v_n_steps + 1)
    v_delay_steps = int(np.round(p_tau / p_dt))
    l_x = [p_x0]

    def f_x_at(p_idx):
        if p_idx < 0:
            return p_x0
        return l_x[min(p_idx, len(l_x) - 1)]

    def f_rhs(p_x_t, p_x_tau):
        return p_beta * p_x_tau / (1.0 + abs(p_x_tau) ** p_n) - p_gamma * p_x_t

    for v_k in range(v_n_steps):
        v_k_tau = v_k - v_delay_steps
        v_xk = l_x[v_k]
        v_x_tau = f_x_at(v_k_tau)

        v_k1 = f_rhs(v_xk, v_x_tau)
        v_k2 = f_rhs(v_xk + 0.5 * p_dt * v_k1, v_x_tau)
        v_k3 = f_rhs(v_xk + 0.5 * p_dt * v_k2, v_x_tau)
        v_k4 = f_rhs(v_xk + p_dt * v_k3, v_x_tau)
        v_x_next = v_xk + (p_dt / 6.0) * (v_k1 + 2.0 * v_k2 + 2.0 * v_k3 + v_k4)
        l_x.append(v_x_next)

    return v_t, np.array(l_x)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: PyTorch PINN (GPU-accelerated via ROCm)
# ═══════════════════════════════════════════════════════════════════════════════

class SineActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MackeyGlassPINN(nn.Module):
    """Fully-connected PINN mapping t -> x(t) for the Mackey-Glass DDE.

    Supports Fourier features, SIREN init, and sine/tanh activations.
    """

    def __init__(
        self,
        hidden_layers: int = 6,
        hidden_size: int = 256,
        activation: str = "sine",
        siren_init: bool = True,
        siren_w0: float = 30.0,
        fourier_features: bool = False,
        fourier_dim: int = 64,
        fourier_scale: float = 5.0,
    ):
        super().__init__()
        self.activation_name = activation
        self.siren_init = siren_init
        self.siren_w0 = max(float(siren_w0), 1e-6)
        self.fourier_features = fourier_features

        input_dim = 1
        if fourier_features:
            self.fourier_dim = max(int(fourier_dim), 1)
            b = torch.randn(self.fourier_dim, 1) * float(fourier_scale)
            self.register_buffer("fourier_B", b)
            input_dim = 2 * self.fourier_dim + 1
        else:
            self.fourier_dim = 0
            self.register_buffer("fourier_B", torch.empty(0, 1))

        act_fn = SineActivation if activation == "sine" else nn.Tanh

        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_size), act_fn()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), act_fn()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linears = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        if self.activation_name == "sine" and self.siren_init and linears:
            with torch.no_grad():
                first = linears[0]
                first.weight.uniform_(-1.0 / max(first.in_features, 1),
                                       1.0 / max(first.in_features, 1))
                first.bias.zero_()
                for layer in linears[1:]:
                    bound = math.sqrt(6.0 / max(layer.in_features, 1)) / self.siren_w0
                    layer.weight.uniform_(-bound, bound)
                    layer.bias.zero_()
        else:
            for layer in linears:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _encode(self, t: torch.Tensor) -> torch.Tensor:
        if not self.fourier_features:
            return t
        proj = 2.0 * math.pi * (t @ self.fourier_B.T)
        return torch.cat([t, torch.sin(proj), torch.cos(proj)], dim=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self._encode(t))


def f_select_device() -> torch.device:
    """Pick the best available device (ROCm surfaces as CUDA)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        print(f"  [device] {backend} GPU: {name}")
        return dev
    print("  [device] CPU (no GPU detected)")
    return torch.device("cpu")


def f_apply_training_seed(p_seed: int) -> None:
    """Fix Python, NumPy, and PyTorch RNGs for reproducible PINN training."""
    s = int(p_seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── Physics residual helpers ──

def f_eval_delayed(
    model: MackeyGlassPINN,
    t: torch.Tensor,
    tau: float,
    history_val: float,
    prev_model: Optional["MackeyGlassPINN"] = None,
    win_t0: float = 0.0,
    detach_delayed_argument: bool = True,
) -> torch.Tensor:
    """Evaluate x(t - tau).

    For time-domain decomposition: if t-tau falls before the current window
    start (win_t0), use prev_model if available, else history constant.

    When ``detach_delayed_argument`` is True (default PINN recipe), gradients
    w.r.t. model parameters omit the pathway through delayed values in the
    current window (``model(t-delay)``). When False (``coupled`` ablation),
    gradients backpropagate through delayed arguments as well when autograd is
    active on ``t``.
    """
    t_delayed = t - tau
    result = torch.full_like(t_delayed, history_val)

    mask_prev_win = (t_delayed < win_t0).squeeze(1)
    mask_hist = (t_delayed <= 0.0).squeeze(1)
    mask_cur = (~mask_prev_win & ~mask_hist)

    if prev_model is not None:
        idx_prev = (mask_prev_win & ~mask_hist).nonzero(as_tuple=True)[0]
        if idx_prev.numel() > 0:
            with torch.no_grad():
                result[idx_prev, :] = prev_model(t_delayed[idx_prev, :])

    idx_cur = mask_cur.nonzero(as_tuple=True)[0]
    if idx_cur.numel() > 0:
        t_arg = (
            t_delayed[idx_cur, :].detach()
            if detach_delayed_argument
            else t_delayed[idx_cur, :]
        )
        result[idx_cur, :] = model(t_arg)

    return result


def f_compute_residual_abs(
    model, t_pts, beta, gamma, n_hill, tau, x0,
    prev_model=None, win_t0=0.0,
    detach_delayed_argument: bool = True,
):
    """Compute |dx/dt - rhs| for adaptive residual sampling."""
    t_eval = t_pts.detach().clone().requires_grad_(True)
    x_eval = model(t_eval)
    dx_dt = torch.autograd.grad(
        x_eval, t_eval,
        grad_outputs=torch.ones_like(x_eval),
        create_graph=False, retain_graph=False,
    )[0]
    x_tau = f_eval_delayed(
        model, t_eval.detach(), tau, x0, prev_model, win_t0,
        detach_delayed_argument=detach_delayed_argument,
    )
    rhs = beta * x_tau / (1.0 + torch.abs(x_tau) ** n_hill) - gamma * x_eval
    return torch.abs(dx_dt - rhs).detach().squeeze(1)


def f_sample_collocation(
    model, device, n_ode, t_lo, t_hi,
    beta, gamma, n_hill, tau, x0,
    adaptive, pool_mult, top_frac,
    prev_model=None, win_t0=0.0,
    detach_delayed_argument: bool = True,
):
    """Sample collocation points, optionally with adaptive residual focus."""
    if not adaptive:
        return torch.rand(n_ode, 1, device=device) * (t_hi - t_lo) + t_lo

    pool_n = n_ode * pool_mult
    t_pool = torch.rand(pool_n, 1, device=device) * (t_hi - t_lo) + t_lo
    res = f_compute_residual_abs(
        model, t_pool, beta, gamma, n_hill, tau, x0,
        prev_model, win_t0,
        detach_delayed_argument=detach_delayed_argument,
    )
    n_focus = max(1, int(round(n_ode * top_frac)))
    n_focus = min(n_focus, n_ode)
    top_idx = torch.topk(res, k=n_focus, largest=True).indices

    if n_focus >= n_ode:
        chosen = top_idx
    else:
        mask = torch.ones(pool_n, dtype=torch.bool, device=device)
        mask[top_idx] = False
        cands = mask.nonzero(as_tuple=True)[0]
        need = n_ode - n_focus
        perm = torch.randperm(cands.numel(), device=device)
        rand_idx = cands[perm[:need]]
        if rand_idx.numel() < need:
            extra = torch.randint(0, pool_n, (need - rand_idx.numel(),), device=device)
            rand_idx = torch.cat([rand_idx, extra])
        chosen = torch.cat([top_idx, rand_idx])

    return t_pool[chosen[torch.randperm(chosen.numel(), device=device)], :]


def f_mackey_glass_loss(
    model: MackeyGlassPINN,
    t_ode: torch.Tensor,
    t_hist: torch.Tensor,
    t_data: torch.Tensor,
    x_data: torch.Tensor,
    beta: float,
    gamma: float,
    n_hill: float,
    tau: float,
    x0: float,
    w_data: float,
    w_phy: float,
    w_hist: float,
    prev_model: Optional[MackeyGlassPINN] = None,
    win_t0: float = 0.0,
    w_ic: float = 10.0,
    ic_t: Optional[torch.Tensor] = None,
    ic_x: Optional[torch.Tensor] = None,
    detach_delayed_argument: bool = True,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Compute total PINN loss for Mackey-Glass DDE.

    Returns scalar unweighted MSEs::
        total,
        ``loss_data``, ``loss_phy``,
        ``loss_hist`` (mean over prescribed history samples on ``[-\\tau, 0]``),
        ``loss_ic`` (junction anchoring),
        ``loss_hist_far`` / ``mid`` / ``near0`` (means over equal index thirds
        of those history samples along ``-\\tau \\to 0^-`` --- the scalar analogue
        of plotting multiple spatial IC residuals in fluid PINNs).

    With no prescribed history tensor, history terms are zeros.
    """
    t_ode = t_ode.requires_grad_(True)
    x_pred = model(t_ode)
    dx_dt = torch.autograd.grad(
        x_pred, t_ode,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True, retain_graph=True,
    )[0]

    t_for_delay = (
        t_ode.detach() if detach_delayed_argument else t_ode
    )
    x_tau = f_eval_delayed(
        model, t_for_delay, tau, x0, prev_model, win_t0,
        detach_delayed_argument=detach_delayed_argument,
    )
    rhs = beta * x_tau / (1.0 + torch.abs(x_tau) ** n_hill) - gamma * x_pred
    loss_phy = torch.mean((dx_dt - rhs) ** 2)

    x_data_pred = model(t_data)
    loss_data = torch.mean((x_data_pred - x_data) ** 2)

    lh_far = torch.tensor(0.0, dtype=t_ode.dtype, device=t_ode.device)
    lh_mid = torch.tensor(0.0, dtype=t_ode.dtype, device=t_ode.device)
    lh_near0 = torch.tensor(0.0, dtype=t_ode.dtype, device=t_ode.device)
    loss_hist = torch.tensor(0.0, device=t_ode.device)

    if t_hist.numel() > 0:
        err_sq = ((model(t_hist) - x0) ** 2).reshape(-1)
        loss_hist = err_sq.mean()
        n_pts = int(err_sq.numel())
        if n_pts >= 3:
            i1 = n_pts // 3
            i2 = 2 * (n_pts // 3)
            lh_far = err_sq[:i1].mean()
            lh_mid = err_sq[i1:i2].mean()
            lh_near0 = err_sq[i2:].mean()
        else:
            lh_far = lh_mid = lh_near0 = loss_hist

    loss_ic = torch.tensor(0.0, device=t_ode.device)
    if ic_t is not None and ic_x is not None:
        loss_ic = torch.mean((model(ic_t) - ic_x) ** 2)

    total = (w_data * loss_data + w_phy * loss_phy
             + w_hist * loss_hist + w_ic * loss_ic)
    return (
        total, loss_data, loss_phy,
        loss_hist, loss_ic, lh_far, lh_mid, lh_near0,
    )


def _f_curriculum_frac(epoch, n_epochs, start_frac=0.1, power=2.0):
    """Fraction of time domain to use at given epoch (curriculum learning)."""
    if n_epochs <= 1:
        return 1.0
    progress = min(max((epoch - 1) / (n_epochs - 1), 0.0), 1.0)
    return start_frac + (1.0 - start_frac) * (progress ** power)


def f_build_time_windows(
    p_t_end: float,
    p_win_size: float,
    p_win_overlap: float,
) -> List[Tuple[float, float]]:
    """Build time intervals ``[t_0, t_1]`` for sequential PINN training.

    Windows have length at most ``p_win_size`` and advance by
    ``p_win_size - p_win_overlap`` until the right endpoint reaches
    ``p_t_end``. For example, with ``p_t_end = 200``, ``p_win_size = 25``,
    ``p_win_overlap = 5`` (default) this yields 10 windows:
    ``[0, 25], [20, 45], [40, 65], [60, 85], [80, 105], [100, 125],
    [120, 145], [140, 165], [160, 185], [180, 200]``.

    The expected count when the schedule closes exactly at ``p_t_end`` is
    ``ceil(p_t_end / (p_win_size - p_win_overlap))``.

    During inference, predictions on ``(t_0, t_1]`` from each window are
    stitched in order; overlapping regions are taken from the **later**
    window (last write wins).
    """
    v_step = float(p_win_size) - float(p_win_overlap)
    if v_step <= 0:
        raise ValueError("window_size must be greater than window_overlap")
    n_cap = max(1, int(math.ceil(float(p_t_end) / v_step)))
    windows: List[Tuple[float, float]] = []
    t0 = 0.0
    for _ in range(n_cap):
        t1 = min(t0 + float(p_win_size), float(p_t_end))
        windows.append((t0, t1))
        if t1 >= float(p_t_end):
            break
        t0 = t1 - float(p_win_overlap)
    return windows


def _f_training_plot_arrays_from_ckpt(
    ckpt: Dict[str, Any],
    n_rec: int,
) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    """Recover per-step phase + weighted-loss terms saved on a window checkpoint.

    Older checkpoints omit ``training_plot_trace``; synthesize optimizer phase from
    ``planned_{n_adam,n_lbfgs}`` when those sum to ``n_rec``, and emit NaNs for
    weighted-term series that were never logged.
    """
    tp_any = ckpt.get("training_plot_trace")
    nan4 = tuple([float("nan")] * n_rec for _ in range(4))

    def _phase_from_counts(pa_i: int, pl_i: int) -> Optional[List[str]]:
        if pa_i >= 0 and pl_i >= 0 and pa_i + pl_i == n_rec:
            return ["adam"] * pa_i + ["lbfgs"] * pl_i
        return None

    if isinstance(tp_any, dict):
        tp = tp_any
        lp = tp.get("optimizer_phase") or []
        wd = tp.get("weighted_data_term") or []
        wp = tp.get("weighted_phys_term") or []
        wh = tp.get("weighted_hist_term") or []
        wi = tp.get("weighted_ic_term") or []
        if (
            len(lp) == n_rec
            and len(wd) == n_rec
            and len(wp) == n_rec
            and len(wh) == n_rec
            and len(wi) == n_rec
        ):
            return (
                [str(x) for x in lp],
                [float(x) for x in wd],
                [float(x) for x in wp],
                [float(x) for x in wh],
                [float(x) for x in wi],
            )
        pa = int(tp.get("planned_n_adam", ckpt.get("planned_n_adam", -1)))
        pl = int(tp.get("planned_n_lbfgs", ckpt.get("planned_n_lbfgs", -1)))
        ph = _phase_from_counts(pa, pl)
        if ph is not None:
            c0, c1, c2, c3 = nan4
            return ph, list(c0), list(c1), list(c2), list(c3)

    pa = int(ckpt.get("planned_n_adam", -1))
    pl = int(ckpt.get("planned_n_lbfgs", -1))
    ph = _phase_from_counts(pa, pl)
    if ph is not None:
        c0, c1, c2, c3 = nan4
        return ph, list(c0), list(c1), list(c2), list(c3)
    c0, c1, c2, c3 = nan4
    return ["unknown_stage"] * n_rec, list(c0), list(c1), list(c2), list(c3)


def _f_trace_float_series_from_ckpt(
    ckpt: Dict[str, Any],
    key: str,
    n_rec: int,
) -> List[float]:
    """Load a numeric per-step trace from checkpoint (or NaNs if missing/old)."""
    tp = ckpt.get("training_plot_trace")
    if isinstance(tp, dict):
        xs = tp.get(key)
        if isinstance(xs, list) and len(xs) == n_rec:
            out: List[float] = []
            for z in xs:
                try:
                    out.append(float(z))
                except (TypeError, ValueError):
                    out.append(float("nan"))
            return out
    return [float("nan")] * int(n_rec)


_MERGE_EXTENDED_CKPT_FIELDS: Tuple[str, ...] = (
    "history_loss_history",
    "ic_loss_history",
    "hist_loss_far_history",
    "hist_loss_mid_history",
    "hist_loss_near0_history",
)


def _f_sorted_window_checkpoint_paths(p_ck_dir: str) -> List[str]:
    if not os.path.isdir(p_ck_dir):
        return []

    def _wid(pth: str) -> int:
        m = re.search(r"window_(\d+)\.pt$", os.path.basename(pth))
        return int(m.group(1)) if m else 10**9

    return sorted(glob.glob(os.path.join(p_ck_dir, "window_*.pt")), key=_wid)


def f_try_merge_extended_histories_from_checkpoints(
    p_output_dir: str,
    p_n: float,
    p_total_iterations: int,
) -> Dict[str, List[float]]:
    """Concatenate per-window ``checkpoint`` loss traces when pickles omit fields.

    Returns a dict keyed like ``history_loss_history`` for every field where **all**
    ``window_*.pt`` shards contain aligned lists matching that window's ``loss_history``
    length, and concatenated length equals ``p_total_iterations``. Empty dict if
    nothing can be reconstructed (common for checkpoints from older trainers).
    """
    ck_dir = os.path.join(os.path.expanduser(str(p_output_dir)), f"checkpoints_n{p_n:g}")
    out: Dict[str, List[float]] = {}
    if int(p_total_iterations) <= 0:
        return out

    paths = _f_sorted_window_checkpoint_paths(ck_dir)
    if not paths:
        return out

    def _flt_list(seq: Sequence[Any]) -> List[float]:
        z: List[float] = []
        for x in seq:
            try:
                z.append(float(x))
            except (TypeError, ValueError):
                z.append(float("nan"))
        return z

    for field in _MERGE_EXTENDED_CKPT_FIELDS:
        acc: List[float] = []
        ok_run = True
        for wp in paths:
            try:
                ck = torch.load(wp, map_location="cpu", weights_only=False)
            except Exception:
                ok_run = False
                break
            n_seg = len(ck.get("loss_history") or [])
            if n_seg <= 0:
                ok_run = False
                break
            chunk = ck.get(field)
            if not isinstance(chunk, list) or len(chunk) != n_seg:
                ok_run = False
                break
            acc.extend(_flt_list(chunk))
        if ok_run and len(acc) == int(p_total_iterations):
            out[field] = acc
    return out


def f_describe_extended_history_gap(
    p_output_dir: str,
    p_n: float,
) -> Optional[str]:
    """Explain why pickles / checkpoints may lack MG history + junction losses."""
    ck_dir = os.path.join(os.path.expanduser(str(p_output_dir)), f"checkpoints_n{p_n:g}")
    paths = _f_sorted_window_checkpoint_paths(ck_dir)
    if not paths:
        return (
            f"No folder ``checkpoints_n{p_n:g}/`` beside this pickle — extended "
            "loss traces cannot be reconstructed from checkpoints."
        )
    try:
        ck0 = torch.load(paths[0], map_location="cpu", weights_only=False)
    except Exception as exc:
        return f"Could not read {paths[0]}: {exc}"
    if ck0.get("history_loss_history") is None and ck0.get("ic_loss_history") is None:
        return (
            r"The first shard ``window_0.pt`` has no ``history_loss_history`` / "
            r"``ic_loss_history`` (saved with older code). Delete the folder ``"
            f"{ck_dir}`` or pass ``--ignore-pinn-checkpoints`` once during a fresh "
            "PINN pass, regenerate ``mglass_run.pkl``, then re-run this export."
        )
    return (
        r"Checkpoint shards declare extended histories but their concatenated length "
        "does not match ``loss_history`` in the pickle — retrain PINN into this "
        "output directory with the current script."
    )


def f_shard0_extended_hist_absent(p_output_dir: str, p_n: float) -> bool:
    """True iff ``window_0.pt`` has neither history nor junction loss lists."""
    ck_dir = os.path.join(os.path.expanduser(str(p_output_dir)), f"checkpoints_n{p_n:g}")
    paths = _f_sorted_window_checkpoint_paths(ck_dir)
    if not paths:
        return False
    try:
        ck0 = torch.load(paths[0], map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        ck0.get("history_loss_history") is None
        and ck0.get("ic_loss_history") is None
    )


def _f_loss_trace_usable(seq: Optional[Sequence[float]], expected: int) -> bool:
    if seq is None or int(expected) <= 0:
        return False
    try:
        return len(seq) == int(expected)
    except TypeError:
        return False


def _f_train_single_window(
    model, device, d_config,
    v_t_train_np, v_x_train_np,
    v_beta, v_gamma, v_n_hill, v_tau, v_x0,
    win_t0, win_t1,
    prev_model=None,
    ic_t_val=None, ic_x_val=None,
    window_label="",
):
    """Train one PINN on a single time window [win_t0, win_t1].

    Includes: curriculum learning, adaptive residual sampling, Fourier features,
    Adam + L-BFGS hybrid.
    """
    v_n_adam = int(d_config["training"]["iterations"]["main"])
    v_lr = float(d_config["training"]["optimizer"]["learning_rate"])
    v_wd = float(d_config["training"]["optimizer"].get("weight_decay", 0.0))
    v_log_every = int(d_config["training"].get("log_interval", 200))
    v_w_data = float(d_config["training"]["loss_weights"]["data_loss"])
    v_w_phy = float(d_config["training"]["loss_weights"]["physics_loss"])
    v_w_hist = float(d_config["training"]["loss_weights"].get("history_loss", 1.0))
    v_w_ic = float(d_config["training"]["loss_weights"].get("ic_loss", 10.0))
    v_n_ode = int(d_config["training"].get("n_collocation", 5000))
    v_n_hist_pts = 100
    v_ramp_iters = int(d_config["training"].get("physics_ramp_iters", 0))
    v_curriculum = bool(d_config["training"].get("curriculum", True))
    v_adaptive = bool(d_config["training"].get("adaptive_sampling", True))
    v_resample_every = int(d_config["training"].get("resample_every", 500))
    v_n_lbfgs = int(d_config["training"].get("n_lbfgs", 500))

    v_ablation = (
        ((d_config.get("training") or {}).get("ablation")) or {}
    )
    if "curriculum" in v_ablation and v_ablation["curriculum"] is not None:
        v_curriculum = bool(v_ablation["curriculum"])
    if v_ablation.get("n_lbfgs") is not None:
        v_n_lbfgs = int(v_ablation["n_lbfgs"])

    _dg = str(v_ablation.get("delay_grad", "detach")).strip().lower()
    v_detach_delay = _dg != "coupled"

    mask_tr = (v_t_train_np >= win_t0) & (v_t_train_np <= win_t1)
    t_data = torch.tensor(v_t_train_np[mask_tr], dtype=torch.float32, device=device).unsqueeze(1)
    x_data = torch.tensor(v_x_train_np[mask_tr], dtype=torch.float32, device=device).unsqueeze(1)

    t_hist = torch.empty(0, 1, device=device)
    if win_t0 <= 0.0:
        t_hist = torch.linspace(-v_tau, 0.0, v_n_hist_pts, device=device).unsqueeze(1)

    ic_t = None
    ic_x = None
    if ic_t_val is not None and ic_x_val is not None:
        ic_t = torch.tensor([[ic_t_val]], dtype=torch.float32, device=device)
        ic_x = torch.tensor([[ic_x_val]], dtype=torch.float32, device=device)

    opt = optim.Adam(model.parameters(), lr=v_lr, weight_decay=v_wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=v_n_adam, eta_min=1e-6)

    l_loss, l_data_l, l_phy_l = [], [], []
    l_hist_l: List[float] = []
    l_ic_l: List[float] = []
    l_h_far_l: List[float] = []
    l_h_mid_l: List[float] = []
    l_h_near_l: List[float] = []
    l_phase: List[str] = []
    l_w_data_term: List[float] = []
    l_w_phys_term: List[float] = []
    l_w_hist_term: List[float] = []
    l_w_hist_far_term: List[float] = []
    l_w_hist_mid_term: List[float] = []
    l_w_hist_near0_term: List[float] = []
    l_w_ic_term: List[float] = []
    l_phys_weight_fraction: List[float] = []
    l_curriculum_time_frac: List[float] = []
    start = time.time()

    use_adaptive = False

    for ep in range(1, v_n_adam + 1):
        model.train()
        opt.zero_grad()

        if v_curriculum:
            frac = _f_curriculum_frac(ep, v_n_adam, start_frac=0.15, power=2.0)
            t_hi_cur = win_t0 + (win_t1 - win_t0) * frac
        else:
            frac = 1.0
            t_hi_cur = win_t1

        if v_adaptive and ep > v_n_adam // 5 and (ep == 1 or ep % v_resample_every == 0):
            use_adaptive = True

        t_ode = f_sample_collocation(
            model, device, v_n_ode, win_t0, t_hi_cur,
            v_beta, v_gamma, v_n_hill, v_tau, v_x0,
            use_adaptive, pool_mult=3, top_frac=0.5,
            prev_model=prev_model, win_t0=win_t0,
            detach_delayed_argument=v_detach_delay,
        )

        w_phy_eff = v_w_phy
        if v_ramp_iters > 0 and ep < v_ramp_iters:
            w_phy_eff = v_w_phy * (ep / v_ramp_iters)

        data_cur = t_data
        xdata_cur = x_data
        if v_curriculum and frac < 0.99:
            m = t_data.squeeze(1) <= t_hi_cur
            if m.sum() >= 4:
                data_cur = t_data[m]
                xdata_cur = x_data[m]

        loss, ld, lp, lh, lic, lh_f, lh_m, lh_n = f_mackey_glass_loss(
            model, t_ode, t_hist, data_cur, xdata_cur,
            v_beta, v_gamma, v_n_hill, v_tau, v_x0,
            v_w_data, w_phy_eff, v_w_hist,
            prev_model, win_t0, v_w_ic, ic_t, ic_x,
            detach_delayed_argument=v_detach_delay,
        )

        with torch.no_grad():
            l_phase.append("adam")
            l_w_data_term.append(float(v_w_data * ld.detach()))
            l_w_phys_term.append(float(w_phy_eff * lp.detach()))
            l_w_hist_term.append(float(v_w_hist * lh.detach()))
            l_w_hist_far_term.append(float(v_w_hist * lh_f.detach()))
            l_w_hist_mid_term.append(float(v_w_hist * lh_m.detach()))
            l_w_hist_near0_term.append(float(v_w_hist * lh_n.detach()))
            l_w_ic_term.append(float(v_w_ic * lic.detach()))
            if v_w_phy > 1e-30:
                l_phys_weight_fraction.append(float(w_phy_eff / v_w_phy))
            else:
                l_phys_weight_fraction.append(1.0)
            l_curriculum_time_frac.append(float(frac))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        l_loss.append(float(loss.item()))
        l_data_l.append(float(ld.item()))
        l_phy_l.append(float(lp.item()))
        l_hist_l.append(float(lh.item()))
        l_ic_l.append(float(lic.item()))
        l_h_far_l.append(float(lh_f.item()))
        l_h_mid_l.append(float(lh_m.item()))
        l_h_near_l.append(float(lh_n.item()))

        if ep % v_log_every == 0 or ep == 1:
            el = time.time() - start
            cur_str = f" cur={frac:.2f}" if v_curriculum else ""
            print(
                f"  [{window_label}] {ep:5d}/{v_n_adam} | Loss {loss.item():.3e} | "
                f"data {ld.item():.3e} | phys {lp.item():.3e} | "
                f"hist {lh.item():.3e} | ic {lic.item():.3e}{cur_str} | {el:.1f}s"
            )

    # L-BFGS refinement
    if v_n_lbfgs > 0:
        print(f"  [{window_label}] L-BFGS ({v_n_lbfgs} steps)...")
        opt_lb = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                              history_size=50, line_search_fn="strong_wolfe")
        t_ode_fix = f_sample_collocation(
            model, device, v_n_ode, win_t0, win_t1,
            v_beta, v_gamma, v_n_hill, v_tau, v_x0,
            v_adaptive, pool_mult=3, top_frac=0.5,
            prev_model=prev_model, win_t0=win_t0,
            detach_delayed_argument=v_detach_delay,
        )
        _cache: Dict[str, float] = {}

        for s in range(1, v_n_lbfgs + 1):
            def closure():
                opt_lb.zero_grad()
                lo, dd, pp, lh, lic, lh_f, lh_m, lh_n = f_mackey_glass_loss(
                    model, t_ode_fix, t_hist, t_data, x_data,
                    v_beta, v_gamma, v_n_hill, v_tau, v_x0,
                    v_w_data, v_w_phy, v_w_hist,
                    prev_model, win_t0, v_w_ic, ic_t, ic_x,
                    detach_delayed_argument=v_detach_delay,
                )
                lo.backward()
                _cache.update(
                    loss=float(lo.item()), data=float(dd.item()),
                    phys=float(pp.item()), hist=float(lh.item()), ic=float(lic.item()),
                    hf=float(lh_f.item()), hm=float(lh_m.item()), hn=float(lh_n.item()),
                )
                return lo

            opt_lb.step(closure)
            l_loss.append(_cache.get("loss", float("nan")))
            l_data_l.append(_cache.get("data", float("nan")))
            l_phy_l.append(_cache.get("phys", float("nan")))
            l_hist_l.append(_cache.get("hist", float("nan")))
            l_ic_l.append(_cache.get("ic", float("nan")))
            l_h_far_l.append(_cache.get("hf", float("nan")))
            l_h_mid_l.append(_cache.get("hm", float("nan")))
            l_h_near_l.append(_cache.get("hn", float("nan")))
            l_phase.append("lbfgs")
            l_w_data_term.append(float(v_w_data * _cache.get("data", float("nan"))))
            l_w_phys_term.append(float(v_w_phy * _cache.get("phys", float("nan"))))
            l_w_hist_term.append(float(v_w_hist * _cache.get("hist", float("nan"))))
            l_w_hist_far_term.append(float(v_w_hist * _cache.get("hf", float("nan"))))
            l_w_hist_mid_term.append(float(v_w_hist * _cache.get("hm", float("nan"))))
            l_w_hist_near0_term.append(float(v_w_hist * _cache.get("hn", float("nan"))))
            l_w_ic_term.append(float(v_w_ic * _cache.get("ic", float("nan"))))
            l_phys_weight_fraction.append(1.0)
            l_curriculum_time_frac.append(1.0)

            if s % 100 == 0 or s == 1:
                el = time.time() - start
                print(f"  [{window_label} LB] {s:4d}/{v_n_lbfgs} | "
                      f"Loss {_cache.get('loss',0):.3e} | {el:.1f}s")

    wall = time.time() - start
    print(f"  [{window_label}] Done in {wall:.1f}s")
    d_training_trace: Dict[str, Any] = {
        "optimizer_phase": l_phase,
        "weighted_data_term": l_w_data_term,
        "weighted_phys_term": l_w_phys_term,
        "weighted_hist_term": l_w_hist_term,
        "weighted_hist_far_term": l_w_hist_far_term,
        "weighted_hist_mid_term": l_w_hist_mid_term,
        "weighted_hist_near0_term": l_w_hist_near0_term,
        "weighted_ic_term": l_w_ic_term,
        "physics_weight_fraction": l_phys_weight_fraction,
        "curriculum_time_fraction": l_curriculum_time_frac,
        "planned_n_adam": int(v_n_adam),
        "planned_n_lbfgs": int(v_n_lbfgs),
    }
    return (
        l_loss, l_data_l, l_phy_l, l_hist_l, l_ic_l,
        l_h_far_l, l_h_mid_l, l_h_near_l, wall,
        d_training_trace,
    )


def f_stitch_pinn_on_grid(
    models: List[Tuple[float, float, MackeyGlassPINN]],
    t_grid_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate stitched PINN on ``t_grid_np`` (later windows overwrite overlaps)."""
    v_x_out = np.zeros(t_grid_np.shape[0], dtype=np.float64)
    for wt0, wt1, mdl in models:
        mask = (t_grid_np >= wt0) & (t_grid_np <= wt1)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        with torch.no_grad():
            t_t = torch.tensor(
                t_grid_np[idx], dtype=torch.float32, device=device
            ).unsqueeze(1)
            v_x_out[idx] = mdl(t_t).cpu().numpy().ravel()
    return v_x_out


def f_train_pinn_mackey_glass(
    p_config: Dict[str, Any],
    p_n_value: float,
    p_metric_t_ref: Optional[np.ndarray] = None,
    p_junction_t_ref: Optional[np.ndarray] = None,
    p_junction_x_ref: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Train a PyTorch PINN on the Mackey-Glass DDE using time-domain decomposition.

    Features: windowed training, curriculum learning, adaptive residual sampling,
    Fourier features, 6x256 network, Adam + L-BFGS hybrid.

    Optional ``p_junction_*``: fine reference trajectory for junction loss at
    window starts (defaults to the training-supervisor RK4 grid).
    Optional ``p_metric_t_ref``: extra time grid for stitched evaluation
    (``u_pred_metric_ref`` in the return dict).
    """
    d_config = copy.deepcopy(p_config)

    if "_seed" in d_config:
        v_seed = int(d_config["_seed"])
    else:
        v_seed = int(d_config.get("training", {}).get("random_seed", 1234))
    f_apply_training_seed(v_seed)
    print(f"  [PINN] Random seed: {v_seed}")
    if d_config.get("_ignore_pinn_checkpoints"):
        print(
            "  [PINN] Ignoring resumed checkpoints (`--ignore-pinn-checkpoints`); "
            "retraining every window.",
        )

    v_beta = float(d_config["problem"]["beta_true"])
    v_gamma = float(d_config["problem"]["gamma_true"])
    v_n_hill = float(p_n_value)
    v_tau = float(d_config["problem"]["tau"])
    v_x0 = float(d_config["problem"].get("initial_x_history", 1.2))
    v_t_end = float(d_config["data"]["t_total"])
    v_dt = float(d_config["data"]["dt"])

    v_ablation: Dict[str, Any] = (
        ((d_config.get("training") or {}).get("ablation")) or {}
    )

    v_t_coarse_np, v_x_coarse_np = f_solve_mackey_glass_rk4_fixed(
        p_beta=v_beta, p_gamma=v_gamma, p_n=v_n_hill, p_tau=v_tau,
        p_x0=v_x0, p_t_end=v_t_end, p_dt=v_dt,
    )
    ref_interp_coarse = interp1d(
        v_t_coarse_np, v_x_coarse_np, kind="cubic", fill_value="extrapolate",
    )

    v_n_train = int(d_config["data"]["training"]["n_points"])

    sup_orbit = str(v_ablation.get("supervision_orbit", "coarse_dt")).strip().lower()
    if sup_orbit == "fine_subsample":
        if p_junction_t_ref is None or p_junction_x_ref is None:
            raise ValueError(
                "training.ablation.supervision_orbit=fine_subsample requires "
                "p_junction_t_ref and p_junction_x_ref on the metric fine grid.",
            )
        v_fline = len(np.asarray(p_junction_t_ref, dtype=np.float64).ravel())
        v_stride_fs = max(1, int(v_fline) // max(1, v_n_train))
        v_ft = np.asarray(p_junction_t_ref, dtype=np.float64).ravel()
        v_fx = np.asarray(p_junction_x_ref, dtype=np.float64).ravel()
        v_t_train_np = v_ft[::v_stride_fs]
        v_x_train_np = v_fx[::v_stride_fs]
    else:
        v_stride_cs = max(1, len(v_t_coarse_np) // max(1, v_n_train))
        v_t_train_np = v_t_coarse_np[::v_stride_cs]
        v_x_train_np = v_x_coarse_np[::v_stride_cs]

    device = f_select_device()

    # Network config
    v_layers_cfg = d_config["network"]["layers"]
    v_hidden_size = v_layers_cfg[1] if len(v_layers_cfg) > 2 else 256
    v_hidden_layers = len(v_layers_cfg) - 2
    v_activation = d_config["network"].get("activation", "sine")
    v_siren_init = d_config["network"].get("initialization", "siren") == "siren"
    v_fourier = bool(d_config["network"].get("fourier_features", True))
    v_fourier_dim = int(d_config["network"].get("fourier_dim", 64))
    v_fourier_scale = float(d_config["network"].get("fourier_scale", 5.0))

    # Time-domain decomposition (see f_build_time_windows)
    v_win_size = float(d_config["training"].get("window_size", 20.0))
    v_win_overlap = float(d_config["training"].get("window_overlap", 2.0))
    windows = f_build_time_windows(v_t_end, v_win_size, v_win_overlap)

    # Checkpoint directory
    v_ckpt_dir = os.path.join(
        d_config.get("_output_dir", "mglass_results"),
        f"checkpoints_n{p_n_value:g}",
    )
    os.makedirs(v_ckpt_dir, exist_ok=True)

    print(f"  [PINN] Time-domain decomposition: {len(windows)} windows")
    print(f"  [PINN] Checkpoints: {v_ckpt_dir}")
    for i, (a, b) in enumerate(windows):
        print(f"         Window {i}: [{a:.1f}, {b:.1f}]")

    v_start_train = time.perf_counter()
    all_loss, all_data_l, all_phy_l = [], [], []
    all_hist_l: List[float] = []
    all_ic_l: List[float] = []
    all_hist_far_l: List[float] = []
    all_hist_mid_l: List[float] = []
    all_hist_near0_l: List[float] = []
    all_segment_lengths: List[int] = []
    all_phase: List[str] = []
    all_win_ix: List[int] = []
    all_w_data_term: List[float] = []
    all_w_phys_term: List[float] = []
    all_w_hist_term: List[float] = []
    all_w_ic_term: List[float] = []
    all_w_hist_far_term: List[float] = []
    all_w_hist_mid_term: List[float] = []
    all_w_hist_near0_term: List[float] = []
    all_phys_weight_fraction: List[float] = []
    all_curriculum_time_frac: List[float] = []
    models: List[Tuple[float, float, MackeyGlassPINN]] = []
    prev_model = None

    if p_junction_t_ref is not None and p_junction_x_ref is not None:
        junction_interp_fine = interp1d(
            p_junction_t_ref, p_junction_x_ref, kind="cubic",
            fill_value="extrapolate",
        )
    else:
        junction_interp_fine = ref_interp_coarse

    junction_mode = str(v_ablation.get("junction", "oracle_fine")).strip().lower()

    v_lw_nom = (d_config.get("training") or {}).get("loss_weights") or {}
    v_plan_lbfgs = int((d_config.get("training") or {}).get("n_lbfgs", 500))
    if v_ablation.get("n_lbfgs") is not None:
        v_plan_lbfgs = int(v_ablation["n_lbfgs"])
    v_plan_adam = int(d_config["training"]["iterations"]["main"])
    v_ramp_iters = int((d_config.get("training") or {}).get("physics_ramp_iters", 0))

    for wi, (wt0, wt1) in enumerate(windows):
        label = f"W{wi} [{wt0:.0f}-{wt1:.0f}]"
        ckpt_path = os.path.join(v_ckpt_dir, f"window_{wi}.pt")

        # Resume from checkpoint if available
        if (not d_config.get("_ignore_pinn_checkpoints")
                and os.path.exists(ckpt_path)):
            print(f"\n  ── {label} (loading checkpoint) ──")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = MackeyGlassPINN(
                hidden_layers=v_hidden_layers,
                hidden_size=v_hidden_size,
                activation=v_activation,
                siren_init=v_siren_init,
                fourier_features=v_fourier,
                fourier_dim=v_fourier_dim,
                fourier_scale=v_fourier_scale,
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            models.append((wt0, wt1, model))
            prev_model = model
            n_rec = len(ckpt.get("loss_history", []))
            all_loss.extend(ckpt.get("loss_history", []))
            all_data_l.extend(ckpt.get("data_loss_history", []))
            all_phy_l.extend(ckpt.get("physics_loss_history", []))
            all_hist_l.extend(
                ckpt.get("history_loss_history")
                or [0.0] * n_rec
            )
            all_ic_l.extend(
                ckpt.get("ic_loss_history")
                or [0.0] * n_rec
            )
            all_hist_far_l.extend(
                ckpt.get("hist_loss_far_history")
                or [0.0] * n_rec
            )
            all_hist_mid_l.extend(
                ckpt.get("hist_loss_mid_history")
                or [0.0] * n_rec
            )
            all_hist_near0_l.extend(
                ckpt.get("hist_loss_near0_history")
                or [0.0] * n_rec
            )
            all_segment_lengths.append(n_rec)
            ph_ck, wd_ck, wp_ck, wh_ck, wi_ck = _f_training_plot_arrays_from_ckpt(
                ckpt, n_rec,
            )
            all_phase.extend(ph_ck)
            all_w_data_term.extend(wd_ck)
            all_w_phys_term.extend(wp_ck)
            all_w_hist_term.extend(wh_ck)
            all_w_ic_term.extend(wi_ck)
            all_w_hist_far_term.extend(
                _f_trace_float_series_from_ckpt(ckpt, "weighted_hist_far_term", n_rec),
            )
            all_w_hist_mid_term.extend(
                _f_trace_float_series_from_ckpt(ckpt, "weighted_hist_mid_term", n_rec),
            )
            all_w_hist_near0_term.extend(
                _f_trace_float_series_from_ckpt(ckpt, "weighted_hist_near0_term", n_rec),
            )
            all_phys_weight_fraction.extend(
                _f_trace_float_series_from_ckpt(ckpt, "physics_weight_fraction", n_rec),
            )
            all_curriculum_time_frac.extend(
                _f_trace_float_series_from_ckpt(ckpt, "curriculum_time_fraction", n_rec),
            )
            all_win_ix.extend([wi] * n_rec)
            print(f"  [{label}] Resumed from {ckpt_path} "
                  f"(wall_time={ckpt.get('wall_time', 0):.1f}s)")
            continue

        print(f"\n  ── {label} ──")

        model = MackeyGlassPINN(
            hidden_layers=v_hidden_layers,
            hidden_size=v_hidden_size,
            activation=v_activation,
            siren_init=v_siren_init,
            fourier_features=v_fourier,
            fourier_dim=v_fourier_dim,
            fourier_scale=v_fourier_scale,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [{label}] {n_params} params | "
              f"{v_hidden_layers}x{v_hidden_size} {v_activation} "
              f"fourier={v_fourier}")

        ic_t_val: Optional[float] = None
        ic_x_val: Optional[float] = None
        if wi > 0:
            if junction_mode == "none":
                pass
            elif junction_mode == "continuity":
                if prev_model is None:
                    raise RuntimeError("continuity junction requires previous window model")
                prev_model.eval()
                with torch.no_grad():
                    _t0 = torch.tensor(
                        [[float(wt0)]], dtype=torch.float32, device=device,
                    )
                    ic_x_val = float(prev_model(_t0).item())
                ic_t_val = float(wt0)
            elif junction_mode == "oracle_coarse":
                ic_x_val = float(ref_interp_coarse(float(wt0)))
                ic_t_val = float(wt0)
            else:
                ic_x_val = float(junction_interp_fine(float(wt0)))
                ic_t_val = float(wt0)

        (
            wl, wdl, wpl, whl, wil, whf, whm, whn, w_wall,
            d_training_trace_w,
        ) = _f_train_single_window(
            model, device, d_config,
            v_t_train_np, v_x_train_np,
            v_beta, v_gamma, v_n_hill, v_tau, v_x0,
            wt0, wt1,
            prev_model=prev_model,
            ic_t_val=ic_t_val, ic_x_val=ic_x_val,
            window_label=label,
        )
        all_loss.extend(wl)
        all_data_l.extend(wdl)
        all_phy_l.extend(wpl)
        all_hist_l.extend(whl)
        all_ic_l.extend(wil)
        all_hist_far_l.extend(whf)
        all_hist_mid_l.extend(whm)
        all_hist_near0_l.extend(whn)
        all_segment_lengths.append(len(wl))
        all_phase.extend(d_training_trace_w["optimizer_phase"])
        all_w_data_term.extend(d_training_trace_w["weighted_data_term"])
        all_w_phys_term.extend(d_training_trace_w["weighted_phys_term"])
        all_w_hist_term.extend(d_training_trace_w["weighted_hist_term"])
        all_w_ic_term.extend(d_training_trace_w["weighted_ic_term"])
        all_w_hist_far_term.extend(d_training_trace_w["weighted_hist_far_term"])
        all_w_hist_mid_term.extend(d_training_trace_w["weighted_hist_mid_term"])
        all_w_hist_near0_term.extend(d_training_trace_w["weighted_hist_near0_term"])
        all_phys_weight_fraction.extend(d_training_trace_w["physics_weight_fraction"])
        all_curriculum_time_frac.extend(d_training_trace_w["curriculum_time_fraction"])
        all_win_ix.extend([wi] * len(wl))

        model.eval()
        models.append((wt0, wt1, model))
        prev_model = model

        # Save checkpoint
        torch.save({
            "window_idx": wi,
            "win_t0": wt0,
            "win_t1": wt1,
            "model_state": model.state_dict(),
            "loss_history": wl,
            "data_loss_history": wdl,
            "physics_loss_history": wpl,
            "history_loss_history": whl,
            "ic_loss_history": wil,
            "hist_loss_far_history": whf,
            "hist_loss_mid_history": whm,
            "hist_loss_near0_history": whn,
            "wall_time": w_wall,
            "n_value": p_n_value,
            "planned_n_adam": d_training_trace_w["planned_n_adam"],
            "planned_n_lbfgs": d_training_trace_w["planned_n_lbfgs"],
            "training_plot_trace": d_training_trace_w,
        }, ckpt_path)
        print(f"  [{label}] Checkpoint saved: {ckpt_path}")

    v_wall_train = time.perf_counter() - v_start_train
    print(f"\n  [PINN] All windows complete in {v_wall_train:.1f}s (training only)")

    t_infer0 = time.perf_counter()
    # Evaluate: stitch predictions from all windows
    v_n_test = int(d_config["optimizer_comparison"]["visualization"]["test_points"])
    v_t_test_np = np.linspace(0.0, v_t_end, v_n_test)
    v_x_pred_np = f_stitch_pinn_on_grid(models, v_t_test_np, device)

    # For overlapping regions, later windows take priority (handled in stitch)

    out: Dict[str, Any] = {
        "t_train": v_t_train_np.reshape(-1, 1),
        "u_train": v_x_train_np.reshape(-1, 1),
        "t_test": v_t_test_np.reshape(-1, 1),
        "u_pred": v_x_pred_np.reshape(-1, 1),
        "f_pred": np.zeros((v_n_test, 1)),
        "params": {"beta": v_beta, "gamma": v_gamma, "n": v_n_hill, "tau": v_tau},
        "loss_history": all_loss,
        "data_loss_history": all_data_l,
        "physics_loss_history": all_phy_l,
        "history_loss_history": all_hist_l,
        "ic_loss_history": all_ic_l,
        "hist_loss_far_history": all_hist_far_l,
        "hist_loss_mid_history": all_hist_mid_l,
        "hist_loss_near0_history": all_hist_near0_l,
        "loss_segment_lengths": all_segment_lengths,
        "wall_time": v_wall_train,
        "wall_time_train": v_wall_train,
        "wall_time_infer": 0.0,
    }
    if p_metric_t_ref is not None:
        v_tm = np.asarray(p_metric_t_ref, dtype=np.float64).ravel()
        v_x_metric = f_stitch_pinn_on_grid(models, v_tm, device)
        out["t_metric_ref"] = v_tm.reshape(-1, 1)
        out["u_pred_metric_ref"] = v_x_metric.reshape(-1, 1)
    out["wall_time_infer"] = float(time.perf_counter() - t_infer0)
    out["ablation_config"] = copy.deepcopy(v_ablation)
    v_n_it_tot = len(all_loss)
    if v_n_it_tot and (
        len(all_phase) != v_n_it_tot
        or len(all_win_ix) != v_n_it_tot
        or len(all_w_data_term) != v_n_it_tot
        or len(all_w_phys_term) != v_n_it_tot
        or len(all_w_hist_term) != v_n_it_tot
        or len(all_w_ic_term) != v_n_it_tot
        or len(all_w_hist_far_term) != v_n_it_tot
        or len(all_w_hist_mid_term) != v_n_it_tot
        or len(all_w_hist_near0_term) != v_n_it_tot
        or len(all_phys_weight_fraction) != v_n_it_tot
        or len(all_curriculum_time_frac) != v_n_it_tot
    ):
        print(
            "  [PINN][warn] training_plot_bundle step arrays length mismatch vs "
            f"loss_history ({v_n_it_tot}); phase={len(all_phase)}, "
            f"win_idx={len(all_win_ix)}, w_terms="
            f"{len(all_w_data_term)}/{len(all_w_phys_term)}/"
            f"{len(all_w_hist_term)}/{len(all_w_ic_term)}, hist_ic_weighted="
            f"{len(all_w_hist_far_term)}/{len(all_w_hist_mid_term)}/"
            f"{len(all_w_hist_near0_term)}, modifiers="
            f"{len(all_phys_weight_fraction)}/{len(all_curriculum_time_frac)}",
        )
    out["training_plot_bundle"] = {
        "random_seed": int(v_seed),
        "junction_mode": junction_mode,
        "supervision_orbit": str(
            v_ablation.get("supervision_orbit", "coarse_dt"),
        ).strip().lower(),
        "delay_grad": str(v_ablation.get("delay_grad", "detach")).strip().lower(),
        "curriculum_default": bool(
            (d_config.get("training") or {}).get("curriculum", True),
        ),
        "curriculum_ablation_override": (
            None if v_ablation.get("curriculum") is None
            else bool(v_ablation["curriculum"])
        ),
        "loss_weights_nominal": {
            "data_loss": float(v_lw_nom.get("data_loss", 1.0)),
            "physics_loss": float(v_lw_nom.get("physics_loss", 1.0)),
            "history_loss": float(v_lw_nom.get("history_loss", 1.0)),
            "ic_loss": float(v_lw_nom.get("ic_loss", 10.0)),
        },
        "physics_ramp_iters": int(v_ramp_iters),
        "planned_n_adam_per_window": int(v_plan_adam),
        "planned_n_lbfgs_per_window": int(v_plan_lbfgs),
        "n_windows": int(len(windows)),
        "windows_physical": [
            {"t0": float(a), "t1": float(b)} for a, b in windows
        ],
        "window_size": float(v_win_size),
        "window_overlap": float(v_win_overlap),
        "optimizer_phase_per_step": all_phase,
        "window_idx_per_step": all_win_ix,
        "weighted_data_term_per_step": all_w_data_term,
        "weighted_physics_term_per_step": all_w_phys_term,
        "weighted_history_term_per_step": all_w_hist_term,
        "weighted_hist_far_term_per_step": all_w_hist_far_term,
        "weighted_hist_mid_term_per_step": all_w_hist_mid_term,
        "weighted_hist_near0_term_per_step": all_w_hist_near0_term,
        "weighted_ic_term_per_step": all_w_ic_term,
        "physics_weight_fraction_per_step": all_phys_weight_fraction,
        "curriculum_time_fraction_per_step": all_curriculum_time_frac,
    }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Metrics and extended diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def f_compute_metrics(
    p_x_ref: np.ndarray,
    p_x_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MSE and relative discrete l^2_N error on the evaluation grid."""
    v_mse = float(np.mean((p_x_ref - p_x_pred) ** 2))
    v_rel_l2 = float(np.sqrt(np.sum((p_x_ref - p_x_pred) ** 2) / np.sum(p_x_ref ** 2)))
    v_max_err = float(np.max(np.abs(p_x_ref - p_x_pred)))
    return {"mse": v_mse, "rel_l2": v_rel_l2, "max_abs_err": v_max_err}


def f_first_exceedance_times(
    p_t: np.ndarray,
    p_x_ref: np.ndarray,
    p_x_pred: np.ndarray,
    p_thresholds: Sequence[float],
) -> Dict[str, float]:
    """
    First time |x_pred - x_ref| strictly exceeds each threshold (same grid as p_t).
    If never exceeded, value is NaN.
    """
    v_t = np.asarray(p_t, dtype=np.float64).ravel()
    v_e = np.abs(np.asarray(p_x_ref, dtype=np.float64).ravel() - np.asarray(p_x_pred, dtype=np.float64).ravel())
    out: Dict[str, float] = {}
    for theta in p_thresholds:
        mask = v_e > float(theta)
        key = str(float(theta))
        if not np.any(mask):
            out[key] = float("nan")
        else:
            out[key] = float(v_t[np.argmax(mask)])
    return out


def f_segment_mse_table(
    p_t: np.ndarray,
    p_x_ref: np.ndarray,
    p_x_pred: np.ndarray,
    p_edges: Sequence[float],
) -> Dict[str, float]:
    """Piecewise MSE on half-open intervals [edges[i], edges[i+1]) in physical time."""
    v_t = np.asarray(p_t, dtype=np.float64).ravel()
    v_ref = np.asarray(p_x_ref, dtype=np.float64).ravel()
    v_pr = np.asarray(p_x_pred, dtype=np.float64).ravel()
    out: Dict[str, float] = {}
    for i in range(len(p_edges) - 1):
        a, b = float(p_edges[i]), float(p_edges[i + 1])
        mask = (v_t >= a) & (v_t < b if i < len(p_edges) - 2 else v_t <= b)
        if not np.any(mask):
            out[f"[{a:g},{b:g}]"] = float("nan")
        else:
            out[f"[{a:g},{b:g}]"] = float(np.mean((v_ref[mask] - v_pr[mask]) ** 2))
    return out


def f_normalized_acf(p_x: np.ndarray, p_max_lag: int) -> np.ndarray:
    """Unbiased ACF for lags 0..max_lag (normalized so lag-0 is 1)."""
    v_x = np.asarray(p_x, dtype=np.float64).ravel()
    v_x = v_x - np.mean(v_x)
    n = v_x.size
    var0 = float(np.dot(v_x, v_x)) / max(n, 1)
    if var0 <= 0:
        return np.zeros(p_max_lag + 1, dtype=np.float64)
    out = np.empty(p_max_lag + 1, dtype=np.float64)
    for k in range(p_max_lag + 1):
        if k == 0:
            out[k] = 1.0
        else:
            out[k] = float(np.dot(v_x[:-k], v_x[k:]) / (var0 * (n - k)))
    return out


def f_delay_embed_3d(p_x: np.ndarray, p_step: int) -> np.ndarray:
    """3D Takens embedding [x[i], x[i-step], x[i-2*step]] for i >= 2*step."""
    v_x = np.asarray(p_x, dtype=np.float64).ravel()
    k = max(int(p_step), 1)
    start = 2 * k
    if v_x.size <= start:
        return np.zeros((0, 3), dtype=np.float64)
    a = v_x[start:]
    b = v_x[start - k : -k]
    c = v_x[start - 2 * k : -2 * k]
    return np.column_stack([a, b, c])


def f_chamfer_symmetric(p_a: np.ndarray, p_b: np.ndarray, p_max_points: int = 2048) -> float:
    """Mean min distance A->B plus B->A (subsample large sets for speed)."""
    if p_a.shape[0] == 0 or p_b.shape[0] == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    a = p_a.astype(np.float64, copy=False)
    b = p_b.astype(np.float64, copy=False)
    if a.shape[0] > p_max_points:
        a = a[rng.choice(a.shape[0], size=p_max_points, replace=False)]
    if b.shape[0] > p_max_points:
        b = b[rng.choice(b.shape[0], size=p_max_points, replace=False)]
    d_ab = np.sqrt(np.min(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2), axis=1))
    d_ba = np.sqrt(np.min(np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=2), axis=1))
    return float(0.5 * (np.mean(d_ab) + np.mean(d_ba)))


def f_log_psd(p_x: np.ndarray, p_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided positive frequencies and log10(PSD) for real x (detrended)."""
    v_x = np.asarray(p_x, dtype=np.float64).ravel()
    v_x = v_x - np.mean(v_x)
    n = v_x.size
    if n < 4:
        return np.array([]), np.array([])
    fft = np.fft.rfft(v_x)
    psd = (np.abs(fft) ** 2) / (np.sum(np.hanning(n) ** 2) + 1e-30)
    freq = np.fft.rfftfreq(n, d=float(p_dt))
    psd = np.maximum(psd, 1e-30)
    return freq, np.log10(psd)


def f_psd_l2_distance(
    p_x_ref: np.ndarray,
    p_x_pred: np.ndarray,
    p_dt: float,
) -> float:
    """L2 distance of log-PSD on the intersection of positive frequency grids."""
    f0, s0 = f_log_psd(p_x_ref, p_dt)
    f1, s1 = f_log_psd(p_x_pred, p_dt)
    if f0.size < 2 or f1.size < 2:
        return float("nan")
    n = min(f0.size, f1.size)
    return float(np.sqrt(np.mean((s0[:n] - s1[:n]) ** 2)))


def f_attractor_metrics_bundle(
    p_t_ref: np.ndarray,
    p_x_ref: np.ndarray,
    p_x_pred: np.ndarray,
    p_tau: float,
    p_dt: float,
) -> Dict[str, Any]:
    """Phase-insensitive diagnostics: stats, ACF lag=tau, PSD gap, delay-embedding Chamfer."""
    v_t = np.asarray(p_t_ref, dtype=np.float64).ravel()
    xr = np.asarray(p_x_ref, dtype=np.float64).ravel()
    xp = np.asarray(p_x_pred, dtype=np.float64).ravel()
    delay_idx = int(max(1, round(float(p_tau) / float(p_dt))))
    acf_r = f_normalized_acf(xr, min(delay_idx * 3, 200))
    acf_p = f_normalized_acf(xp, min(delay_idx * 3, 200))
    lag_sel = min(delay_idx, acf_r.size - 1, acf_p.size - 1)
    emb_r = f_delay_embed_3d(xr, delay_idx)
    emb_p = f_delay_embed_3d(xp, delay_idx)
    n = min(emb_r.shape[0], emb_p.shape[0])
    if n > 0:
        emb_r = emb_r[-n:]
        emb_p = emb_p[-n:]
    return {
        "mean_ref": float(np.mean(xr)),
        "mean_pred": float(np.mean(xp)),
        "std_ref": float(np.std(xr, ddof=0)),
        "std_pred": float(np.std(xp, ddof=0)),
        "min_ref": float(np.min(xr)),
        "max_ref": float(np.max(xr)),
        "min_pred": float(np.min(xp)),
        "max_pred": float(np.max(xp)),
        "acf_lag_tau_ref": float(acf_r[lag_sel]) if lag_sel < acf_r.size else float("nan"),
        "acf_lag_tau_pred": float(acf_p[lag_sel]) if lag_sel < acf_p.size else float("nan"),
        "psd_log_l2": f_psd_l2_distance(xr, xp, float(p_dt)),
        "delay_embed_chamfer": f_chamfer_symmetric(emb_r, emb_p),
        "delay_embedding_step": int(delay_idx),
    }


def f_reference_rk4_convergence_rows(
    p_beta: float,
    p_gamma: float,
    p_n: float,
    p_tau: float,
    p_x0: float,
    p_t_end: float,
    p_dt_finest: float,
    p_dt_coarser: Sequence[float],
) -> List[Dict[str, Any]]:
    """Pair each RK4(dt) trajectory against the finest RK4 grid; report discrepancy."""
    t_f, x_f = f_solve_mackey_glass_rk4_fixed(
        p_beta, p_gamma, p_n, p_tau, p_x0, p_t_end, p_dt_finest,
    )
    rows: List[Dict[str, Any]] = []
    for dt in p_dt_coarser:
        if abs(float(dt) - float(p_dt_finest)) < 1e-15:
            rows.append(
                {
                    "n": float(p_n),
                    "dt": float(dt),
                    "mse_vs_finest": 0.0,
                    "rel_l2_vs_finest": 0.0,
                    "max_abs_vs_finest": 0.0,
                }
            )
            continue
        t_c, x_c = f_solve_mackey_glass_rk4_fixed(
            p_beta, p_gamma, p_n, p_tau, p_x0, p_t_end, float(dt),
        )
        interp_c = interp1d(t_c, x_c, kind="cubic", fill_value="extrapolate")
        x_on_f = interp_c(t_f)
        m = f_compute_metrics(x_f, x_on_f)
        rows.append(
            {
                "n": float(p_n),
                "dt": float(dt),
                "mse_vs_finest": m["mse"],
                "rel_l2_vs_finest": m["rel_l2"],
                "max_abs_vs_finest": m["max_abs_err"],
            }
        )
    return rows


def f_capture_runtime_env() -> Dict[str, Any]:
    """Versions and hardware for reproducibility notes (torch may be missing pieces on import)."""
    d: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "",
    }
    try:
        import scipy

        d["scipy_version"] = scipy.__version__
    except ImportError:
        d["scipy_version"] = None
    d["pytorch_version"] = torch.__version__
    d["torch_num_threads"] = torch.get_num_threads()
    d["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    d["mkl_num_threads"] = os.environ.get("MKL_NUM_THREADS")
    d["hip_version"] = getattr(torch.version, "hip", None)
    if torch.cuda.is_available():
        d["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        d["cuda_device"] = None
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Raissi-style Visualization + Individual Plot Export
# ═══════════════════════════════════════════════════════════════════════════════

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers projection
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _save(fig, path):
    """Save figure as PNG and optionally PDF, then close."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if _MGLASS_SAVE_PDF:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3D Delay-Embedding Attractor Plots (publication-quality, IEEE 2-col sizing)
# ═══════════════════════════════════════════════════════════════════════════════

_IEEE_COL_W = 3.5   # single-column width in inches
_IEEE_2COL_W = 7.16  # full-width (2-column) in inches

# 3D overlay (Fig.~5): black RK4 ref; indigo MoS (distinct from black + purple PINN)
_COLOR_3D_REF = "k"             # black — fine RK4 reference
_COLOR_3D_CLASSICAL = "#3F51B5" # indigo — MoS / classical (vs #780078 PINN)
_COLOR_3D_PINN = "#780078"      # purple — PINN (thesis Adam predictor style)

# Four cameras for multi-panel overlay exports (primary = first tuple).
_OVERLAY_3D_VIEW_QUAD = (
    (30.0, 45.0),
    (28.0, -120.0),
    (22.0, 135.0),
    (40.0, -40.0),
)


def _slug_overlay_view(elev: float, azim: float) -> str:
    """Stable filename fragment, e.g. e30_a45, e28_am120."""
    e = int(round(float(elev)))
    a = int(round(float(azim)))
    a = ((a + 180) % 360) - 180
    if a < 0:
        return f"e{e}_am{abs(a)}"
    return f"e{e}_a{a}"


def _f_embed_3d_branch(p_x: np.ndarray, ds: int):
    """Return (x(t), x(t-τ), x(t-2τ)) or None."""
    p_x = np.asarray(p_x).reshape(-1)
    if len(p_x) <= 2 * ds:
        return None
    return (
        p_x[2 * ds:].copy(),
        p_x[ds:-ds].copy(),
        p_x[:-2 * ds].copy(),
    )


def _f_bounds_xyz_from_branches(branches) -> Optional[Tuple]:
    """branches: iterable of (x,y,z) or None."""
    xs, ys, zs = [], [], []
    for b in branches:
        if b is None:
            continue
        x, y, z = b
        xs.extend([float(np.min(x)), float(np.max(x))])
        ys.extend([float(np.min(y)), float(np.max(y))])
        zs.extend([float(np.min(z)), float(np.max(z))])
    if not xs:
        return None
    pad = 0.02
    def _pad(lo, hi):
        d = max(hi - lo, 1e-9)
        return lo - pad * d, hi + pad * d
    return (
        _pad(min(xs), max(xs)),
        _pad(min(ys), max(ys)),
        _pad(min(zs), max(zs)),
    )


def _f_draw_3d_overlay_on_ax(
    ax,
    *,
    branch_ref,
    branch_cl,
    branch_pinn,
    p_z_floor_shadow: Optional[float],
    p_n_value: float,
    p_elev: float,
    p_azim: float,
    p_dist: float,
    p_font_title: int,
    p_font_label: int,
    p_font_tick: int,
    p_font_legend: int,
    p_lw_ref: float,
    p_lw_cl: float,
    p_lw_pinn: float,
    p_alpha_ref: float,
    p_alpha_cl: float,
    p_alpha_pinn: float,
    p_bounds: Optional[Tuple],
    p_show_title: bool,
    p_show_legend: bool,
    p_title_suffix: str = "",
):
    """Draw overlay trajectories on a 3D axis; optional equal bounds for grids."""
    if p_z_floor_shadow is not None and branch_ref is not None:
        xr, xr1, xr2 = branch_ref
        ax.plot(
            xr, xr1, np.full_like(xr, p_z_floor_shadow),
            color="gray", linewidth=1.0, alpha=0.25, zorder=1,
        )
    if branch_ref is not None:
        xr, xr1, xr2 = branch_ref
        ax.plot(
            xr, xr1, xr2,
            color=_COLOR_3D_REF,
            linewidth=p_lw_ref,
            alpha=p_alpha_ref,
            label=r"RK4 reference ($\Delta t = 10^{-3}$)",
            zorder=2,
        )
    if branch_cl is not None:
        xc, yc, zc = branch_cl
        ax.plot(
            xc, yc, zc,
            color=_COLOR_3D_CLASSICAL,
            linewidth=p_lw_cl,
            alpha=p_alpha_cl,
            label="Classical DDE solver",
            zorder=5,
        )
    if branch_pinn is not None:
        xp, yp, zp = branch_pinn
        ax.plot(
            xp, yp, zp,
            color=_COLOR_3D_PINN,
            linewidth=p_lw_pinn,
            alpha=p_alpha_pinn,
            linestyle="--",
            label="PINN",
            zorder=10,
        )

    ax.set_xlabel("$x(t)$", fontsize=p_font_label, labelpad=2)
    ax.set_ylabel("$x(t{-}\\tau)$", fontsize=p_font_label, labelpad=2)
    ax.set_zlabel("$x(t{-}2\\tau)$", fontsize=p_font_label, labelpad=2)
    if p_show_title:
        ax.set_title(
            f"Mackey-Glass ($n={p_n_value:g}$){p_title_suffix}",
            fontsize=p_font_title, fontweight="bold", pad=6,
        )
    ax.tick_params(axis="both", labelsize=p_font_tick, pad=1)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.view_init(elev=p_elev, azim=p_azim)
    ax.dist = p_dist
    if p_bounds is not None:
        ax.set_xlim(p_bounds[0])
        ax.set_ylim(p_bounds[1])
        ax.set_zlim(p_bounds[2])
    if p_show_legend:
        ax.legend(fontsize=p_font_legend, loc="upper left", framealpha=0.85)


def _delay_indices(t: np.ndarray, tau: float):
    """Return integer lag corresponding to delay tau on a uniform grid."""
    dt = t[1] - t[0]
    return max(1, int(round(tau / dt)))


def _color_segments_3d(x, y, z, cmap_name="viridis", linewidth=0.8, alpha=0.9):
    """Build a Line3DCollection coloured by index (proxy for time)."""
    pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(0, len(x) - 1)
    lc = Line3DCollection(segs, cmap=cmap_name, norm=norm, linewidths=linewidth,
                          alpha=alpha)
    lc.set_array(np.arange(len(x)))
    return lc


def f_export_3d_attractor_comparison(
    p_t_classical, p_x_classical,
    p_t_pinn_test, p_x_pinn,
    p_tau, p_n_value, p_output_dir,
):
    """
    Side-by-side 3D delay embedding: Classical DDE solver vs PINN.
    Axes: x(t), x(t-tau), x(t-2tau).  Time-gradient colouring.
    Sized for IEEE 2-column full width.
    """
    ds_cl = _delay_indices(p_t_classical, p_tau)
    ds_p = _delay_indices(p_t_pinn_test.flatten(), p_tau)

    fig = plt.figure(figsize=(_IEEE_2COL_W, 3.2))
    titles = ["Classical DDE solver", "PINN"]
    datasets = [
        (p_x_classical, ds_cl),
        (p_x_pinn[:, 0], ds_p),
    ]

    for col, (xarr, ds) in enumerate(datasets):
        if len(xarr) <= 2 * ds:
            continue
        xt = xarr[2 * ds:]
        xt1 = xarr[ds: -ds]
        xt2 = xarr[: -2 * ds]

        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        lc = _color_segments_3d(xt, xt1, xt2, cmap_name="inferno",
                                linewidth=0.6, alpha=0.85)
        ax.add_collection3d(lc)
        ax.auto_scale_xyz(xt, xt1, xt2)

        z_floor = xt2.min() - 0.05
        ax.plot(xt, xt1, np.full_like(xt, z_floor),
                color="gray", linewidth=0.3, alpha=0.15)

        ax.set_xlabel("$x(t)$", fontsize=7, labelpad=1)
        ax.set_ylabel("$x(t{-}\\tau)$", fontsize=7, labelpad=1)
        ax.set_zlabel("$x(t{-}2\\tau)$", fontsize=7, labelpad=1)
        ax.set_title(titles[col], fontsize=8, pad=2)
        ax.tick_params(axis="both", labelsize=5, pad=0)
        ax.view_init(elev=25, azim=-55)
        ax.dist = 11

    fig.suptitle(f"3D delay embedding — Mackey-Glass ($n={p_n_value:g}$)",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.5)
    _save(fig, os.path.join(p_output_dir, f"n{p_n_value:g}_3d_attractor_comparison.png"))


def f_export_3d_attractor_single(
    p_t, p_x, p_tau, p_n_value,
    p_label, p_cmap, p_output_path,
):
    """
    Single 3D delay embedding plot, sized for one IEEE column.
    """
    ds = _delay_indices(p_t, p_tau)
    if len(p_x) <= 2 * ds:
        return

    xt = p_x[2 * ds:]
    xt1 = p_x[ds: -ds]
    xt2 = p_x[: -2 * ds]

    fig = plt.figure(figsize=(_IEEE_COL_W, 3.0))
    ax = fig.add_subplot(111, projection="3d")

    lc = _color_segments_3d(xt, xt1, xt2, cmap_name=p_cmap,
                            linewidth=0.7, alpha=0.9)
    ax.add_collection3d(lc)
    ax.auto_scale_xyz(xt, xt1, xt2)

    z_floor = xt2.min() - 0.05
    ax.plot(xt, xt1, np.full_like(xt, z_floor),
            color="gray", linewidth=0.3, alpha=0.15)

    ax.set_xlabel("$x(t)$", fontsize=8, labelpad=2)
    ax.set_ylabel("$x(t{-}\\tau)$", fontsize=8, labelpad=2)
    ax.set_zlabel("$x(t{-}2\\tau)$", fontsize=8, labelpad=2)
    ax.set_title(f"{p_label} ($n={p_n_value:g}$)", fontsize=9, pad=4)
    ax.tick_params(axis="both", labelsize=6, pad=1)
    ax.view_init(elev=25, azim=-55)
    ax.dist = 11

    sm = plt.cm.ScalarMappable(cmap=p_cmap,
                                norm=plt.Normalize(0, len(xt) - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=18)
    cbar.set_label("time index", fontsize=7)
    cbar.ax.tick_params(labelsize=5)

    fig.tight_layout(pad=0.3)
    _save(fig, p_output_path)


def f_export_3d_overlay(
    p_t_ref: np.ndarray,
    p_x_ref: np.ndarray,
    p_t_classical: np.ndarray,
    p_x_classical: np.ndarray,
    p_t_pinn_test: np.ndarray,
    p_x_pinn: np.ndarray,
    p_tau: float,
    p_n_value: float,
    p_output_path: str,
):
    """
    Overlay fine RK4 reference, classical MoS, and PINN in 3D delay space.

    Black RK4 reference, indigo MoS trajectory, dashed purple PINN (#780078),
    transparent panes, no grid, and a floor shadow of the reference curve.

    Writes:

    1. ``p_output_path`` — primary camera (first entry in ``_OVERLAY_3D_VIEW_QUAD``).
    2. Three sibling files ``{stem}_pov_{elev_azim_tag}.png`` for the other cameras.
    3. A composite ``{stem}_2x2.png`` (shared limits; legend on the first panel only).
    """
    ds_ref = _delay_indices(p_t_ref, p_tau)
    ds_cl = _delay_indices(p_t_classical, p_tau)
    ds_p = _delay_indices(p_t_pinn_test.flatten(), p_tau)

    _FIG_W, _FIG_H = 5.0, 6.0
    _FONT_TITLE, _FONT_LABEL, _FONT_TICK = 11, 9, 7
    _FONT_LEGEND = 7
    _LW_REF, _LW_CL, _LW_PINN = 1.2, 1.5, 1.2
    _ALPHA_REF, _ALPHA_CL, _ALPHA_PINN = 0.88, 0.9, 0.8
    _DIST = 11

    xp = np.asarray(p_x_pinn[:, 0]).reshape(-1)
    branch_ref = (
        _f_embed_3d_branch(p_x_ref, ds_ref)
        if len(np.asarray(p_x_ref).reshape(-1)) > 2 * ds_ref else None
    )
    branch_cl = (
        _f_embed_3d_branch(p_x_classical, ds_cl)
        if len(np.asarray(p_x_classical).reshape(-1)) > 2 * ds_cl else None
    )
    branch_pn = _f_embed_3d_branch(xp, ds_p) if len(xp) > 2 * ds_p else None

    z_floor = None
    if branch_ref is not None:
        z_floor = float(np.min(branch_ref[2]) - 0.05)

    v_bounds = _f_bounds_xyz_from_branches([branch_ref, branch_cl, branch_pn])
    if v_bounds is None:
        return

    v_base = os.path.splitext(p_output_path)[0]

    def _emit_one(v_path: str, v_elev: float, v_azim: float,
                  v_show_title: bool, v_show_legend: bool, v_suffix: str,
                  v_use_bounds: Optional[Tuple],
                  v_ftitle: int, v_flabel: int, v_ftick: int, v_fleg: int):
        fig = plt.figure(figsize=(_FIG_W, _FIG_H))
        ax = fig.add_subplot(111, projection="3d")
        _f_draw_3d_overlay_on_ax(
            ax,
            branch_ref=branch_ref,
            branch_cl=branch_cl,
            branch_pinn=branch_pn,
            p_z_floor_shadow=z_floor,
            p_n_value=p_n_value,
            p_elev=v_elev,
            p_azim=v_azim,
            p_dist=_DIST,
            p_font_title=v_ftitle,
            p_font_label=v_flabel,
            p_font_tick=v_ftick,
            p_font_legend=v_fleg,
            p_lw_ref=_LW_REF,
            p_lw_cl=_LW_CL,
            p_lw_pinn=_LW_PINN,
            p_alpha_ref=_ALPHA_REF,
            p_alpha_cl=_ALPHA_CL,
            p_alpha_pinn=_ALPHA_PINN,
            p_bounds=v_use_bounds,
            p_show_title=v_show_title,
            p_show_legend=v_show_legend,
            p_title_suffix=v_suffix,
        )
        fig.tight_layout(pad=0.4)
        if not v_path.endswith(".png"):
            v_path = f"{os.path.splitext(v_path)[0]}.png"
        _save(fig, v_path)

    for v_i, (v_el, v_az) in enumerate(_OVERLAY_3D_VIEW_QUAD):
        v_slug = _slug_overlay_view(v_el, v_az)
        if v_i == 0:
            v_path_cur = (
                p_output_path if p_output_path.lower().endswith(".png")
                else f"{v_base}.png"
            )
        else:
            v_path_cur = f"{v_base}_pov_{v_slug}.png"
        _emit_one(
            v_path_cur, v_el, v_az,
            v_show_title=True, v_show_legend=True, v_suffix="",
            v_use_bounds=None,
            v_ftitle=_FONT_TITLE, v_flabel=_FONT_LABEL,
            v_ftick=_FONT_TICK, v_fleg=_FONT_LEGEND,
        )

    _GRID_W, _GRID_H = 10.0, 12.0
    _GTITLE, _GLABEL, _GTICK, _GLEG = 9, 8, 6, 6

    fig_g = plt.figure(figsize=(_GRID_W, _GRID_H))
    for v_i, (v_el, v_az) in enumerate(_OVERLAY_3D_VIEW_QUAD):
        ax_g = fig_g.add_subplot(2, 2, v_i + 1, projection="3d")
        v_panel = chr(ord("a") + v_i)
        _f_draw_3d_overlay_on_ax(
            ax_g,
            branch_ref=branch_ref,
            branch_cl=branch_cl,
            branch_pinn=branch_pn,
            p_z_floor_shadow=z_floor,
            p_n_value=p_n_value,
            p_elev=v_el,
            p_azim=v_az,
            p_dist=_DIST,
            p_font_title=_GTITLE,
            p_font_label=_GLABEL,
            p_font_tick=_GTICK,
            p_font_legend=_GLEG,
            p_lw_ref=_LW_REF,
            p_lw_cl=_LW_CL,
            p_lw_pinn=_LW_PINN,
            p_alpha_ref=_ALPHA_REF,
            p_alpha_cl=_ALPHA_CL,
            p_alpha_pinn=_ALPHA_PINN,
            p_bounds=v_bounds,
            p_show_title=True,
            p_show_legend=(v_i == 0),
            p_title_suffix=f" ({v_panel})",
        )
    fig_g.suptitle(
        f"Mackey-Glass ($n={p_n_value:g}$) — 3D delay embedding overlays",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig_g.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    v_grid_path = f"{v_base}_2x2.png"
    _save(fig_g, v_grid_path)


def f_export_3d_error_surface(
    p_t_ref, p_x_ref,
    p_t_pinn_test, p_x_pinn,
    p_tau, p_n_value, p_output_path,
):
    """
    3D surface: pointwise error magnitude in the delay-embedding space.
    Colour = |x_ref - x_PINN|. Single column width.
    """
    ref_interp = interp1d(p_t_ref, p_x_ref, kind="cubic",
                          fill_value="extrapolate")
    t_p = p_t_pinn_test.flatten()
    xp = p_x_pinn[:, 0]
    xr = ref_interp(t_p)

    ds = _delay_indices(t_p, p_tau)
    if len(xp) <= 2 * ds:
        return

    xt_p = xp[2 * ds:]
    xt1_p = xp[ds: -ds]
    err = np.abs(xr[2 * ds:] - xt_p)

    fig = plt.figure(figsize=(_IEEE_COL_W, 3.2))
    ax = fig.add_subplot(111, projection="3d")

    pts = np.column_stack([xt_p, xt1_p, err]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(err.min(), np.percentile(err, 97))
    lc = Line3DCollection(segs, cmap="hot", norm=norm, linewidths=0.6, alpha=0.9)
    lc.set_array(err[:-1])
    ax.add_collection3d(lc)
    ax.auto_scale_xyz(xt_p, xt1_p, err)

    ax.set_xlabel("$x_{\\mathrm{PINN}}(t)$", fontsize=7, labelpad=2)
    ax.set_ylabel("$x_{\\mathrm{PINN}}(t{-}\\tau)$", fontsize=7, labelpad=2)
    ax.set_zlabel("$|\\epsilon(t)|$", fontsize=7, labelpad=2)
    ax.set_title(f"PINN error in delay space ($n={p_n_value:g}$)",
                 fontsize=8, pad=4)
    ax.tick_params(axis="both", labelsize=5, pad=0)
    ax.view_init(elev=30, azim=-50)
    ax.dist = 11

    sm = plt.cm.ScalarMappable(cmap="hot", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=18)
    cbar.set_label("$|x_{\\mathrm{ref}} - x_{\\mathrm{PINN}}|$", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.tight_layout(pad=0.3)
    _save(fig, p_output_path)


def f_export_heatmap(
    p_t_ref: np.ndarray,
    p_x_ref: np.ndarray,
    p_t_pinn_test: np.ndarray,
    p_x_pinn: np.ndarray,
    p_snapshot_times: List[float],
    p_n_value: float,
    p_output_path: str,
    p_t_classical=None, p_x_classical=None,
):
    """
    Raissi-style rainbow heatmap for 1D DDE.

    Builds a 2D image [rows x time] where each row is a slightly
    time-shifted copy of x(t), giving a gradient colored band similar
    to the KdV u(t,x) plot.  White vertical lines mark snapshots.
    Plots three stacked strips: baseline trajectory (reference or classical
    DDE), PINN, and pointwise absolute error on the same (t, δ) grid.
    """
    n_rows = 100
    tau_vis = 2.0
    t_flat = p_t_pinn_test.flatten()
    x_flat = p_x_pinn[:, 0]

    shifts = np.linspace(-tau_vis, tau_vis, n_rows)
    if p_t_classical is not None and p_x_classical is not None:
        base_interp = interp1d(p_t_classical, p_x_classical, kind="cubic",
                               bounds_error=False, fill_value=np.nan)
        base_title = "Classical DDE $x(t+\\delta)$"
        err_title = r"$|x(t+\delta)-\hat{x}(t+\delta)|$ (Classical vs.\ PINN)"
    else:
        base_interp = interp1d(p_t_ref, p_x_ref, kind="cubic",
                               bounds_error=False, fill_value=np.nan)
        base_title = "Reference $x(t+\\delta)$"
        err_title = r"$|x^{\mathrm{ref}}(t+\delta)-\hat{x}(t+\delta)|$"
    pinn_interp = interp1d(t_flat, x_flat, kind="cubic",
                           bounds_error=False, fill_value=np.nan)

    t_grid = np.linspace(t_flat.min(), t_flat.max(), 2000)
    img_base = np.zeros((n_rows, len(t_grid)))
    img_pinn = np.zeros_like(img_base)
    for i, s in enumerate(shifts):
        img_base[i, :] = base_interp(t_grid + s)
        img_pinn[i, :] = pinn_interp(t_grid + s)

    img_err = np.abs(img_base - img_pinn)

    # Shorter figure height + tighter row spacing keeps each δ-strip visually thinner
    # without changing the interpolated (t, δ) resolution (`n_rows` above).
    fig = plt.figure(figsize=(13, 7.2))

    gs = gridspec.GridSpec(3, 1, hspace=0.40, left=0.08, right=0.92,
                           top=0.91, bottom=0.06)

    extent = [t_grid.min(), t_grid.max(), shifts.min(), shifts.max()]
    vmin = np.nanmin(np.minimum(img_base, img_pinn))
    vmax = np.nanmax(np.maximum(img_base, img_pinn))

    ax0 = fig.add_subplot(gs[0, 0])
    h0 = ax0.imshow(img_base, interpolation="nearest", cmap="rainbow",
                    extent=extent, origin="lower", aspect="auto",
                    vmin=vmin, vmax=vmax)
    for ts in p_snapshot_times:
        ax0.plot([ts, ts], [shifts.min(), shifts.max()], "w-", linewidth=1.0)
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="2.5%", pad=0.06)
    fig.colorbar(h0, cax=cax0)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("time shift $\\delta$")
    ax0.set_title(base_title, fontsize=11)

    ax1 = fig.add_subplot(gs[1, 0])
    h1 = ax1.imshow(img_pinn, interpolation="nearest", cmap="rainbow",
                    extent=extent, origin="lower", aspect="auto",
                    vmin=vmin, vmax=vmax)
    for ts in p_snapshot_times:
        ax1.plot([ts, ts], [shifts.min(), shifts.max()], "w-", linewidth=1.0)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2.5%", pad=0.06)
    fig.colorbar(h1, cax=cax1)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("time shift $\\delta$")
    ax1.set_title("PINN $\\hat{x}(t+\\delta)$", fontsize=11)

    fe = img_err[np.isfinite(img_err)]
    vmin_err = 0.0
    vmax_err = float(np.percentile(fe, 98.5)) if fe.size > 0 else 1.0
    vmax_err = max(vmax_err, 1e-9)
    if np.isnan(vmax_err) or np.isinf(vmax_err):
        vmax_err = 1e-6

    ax2 = fig.add_subplot(gs[2, 0])
    h2 = ax2.imshow(img_err, interpolation="nearest", cmap="hot",
                    extent=extent, origin="lower", aspect="auto",
                    vmin=vmin_err, vmax=vmax_err)
    for ts in p_snapshot_times:
        ax2.plot([ts, ts], [shifts.min(), shifts.max()], "c-", linewidth=0.9)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="2.5%", pad=0.06)
    cbar2 = fig.colorbar(h2, cax=cax2)
    cbar2.set_label("$|e|$ scale to 98.5\\%tile", fontsize=9)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("time shift $\\delta$")
    ax2.set_title(err_title, fontsize=11)

    fig.suptitle(f"Mackey-Glass ($n={p_n_value:g}$)", fontsize=13, fontweight="bold")
    _save(fig, p_output_path)


def f_export_heatmap_n_sweep(
    p_results: Dict[float, Dict[str, Any]],
    p_snapshot_times: List[float],
    p_output_path: str,
):
    """
    Raissi-style rainbow heatmap: rows = different n values, columns = time,
    colour = x(t).  One panel for reference, one for PINN.
    """
    l_n_vals = sorted(p_results.keys())
    if not l_n_vals:
        return

    t_common = p_results[l_n_vals[0]]["t_ref"]
    n_t = len(t_common)
    n_n = len(l_n_vals)

    img_ref = np.zeros((n_n, n_t))
    img_pinn = np.zeros((n_n, n_t))
    for i, n in enumerate(l_n_vals):
        r = p_results[n]
        img_ref[i, :] = r["x_ref"][:n_t]
        pinn_interp = interp1d(
            r["t_pinn_test"].flatten(), r["x_pinn"][:, 0],
            kind="cubic", bounds_error=False, fill_value=np.nan,
        )
        img_pinn[i, :] = pinn_interp(t_common[:n_t])

    vmin = np.nanmin(img_ref)
    vmax = np.nanmax(img_ref)
    extent = [t_common[0], t_common[n_t - 1], -0.5, n_n - 0.5]

    fig = plt.figure(figsize=(10, 3.0 + 1.2 * n_n))
    gs = gridspec.GridSpec(1, 2, wspace=0.3, left=0.10, right=0.90,
                           top=0.85, bottom=0.15)

    for col, (img, title) in enumerate([
        (img_ref, "Reference (RK4)"),
        (img_pinn, "PINN prediction"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        h = ax.imshow(img, interpolation="nearest", cmap="rainbow",
                       extent=extent, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax)
        for ts in p_snapshot_times:
            ax.axvline(ts, color="w", linewidth=1.0)
        ax.set_yticks(range(n_n))
        ax.set_yticklabels([f"$n={v:g}$" for v in l_n_vals])
        ax.set_xlabel("$t$")
        ax.set_title(title, fontsize=11)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

    fig.suptitle("Mackey-Glass $x(t)$ across Hill exponents",
                 fontsize=13, fontweight="bold")
    _save(fig, p_output_path)


def f_export_time_series(
    p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
    p_snapshot_times, p_n_value, p_params_true, p_output_path,
    p_t_classical=None, p_x_classical=None,
):
    """Full time series x(t): classical DDE solver + PINN (ref used only for metrics)."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if p_t_classical is not None and p_x_classical is not None:
        ax.plot(p_t_classical, p_x_classical, "b-",
                linewidth=1.5, label="Classical DDE solver")
    else:
        ax.plot(p_t_ref, p_x_ref, "b-", linewidth=1.5, label="Reference (RK4)")
    ax.plot(p_t_pinn_test.flatten(), p_x_pinn[:, 0], "r--",
            linewidth=1.2, alpha=0.85, label="PINN")
    for ts in p_snapshot_times:
        ax.axvline(ts, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$x(t)$", fontsize=12)
    ax.set_title(
        f"$n={p_n_value:g}$, $\\beta={p_params_true['beta']}$, "
        f"$\\gamma={p_params_true['gamma']}$, $\\tau={p_params_true['tau']}$",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
    _save(fig, p_output_path)


def f_export_snapshot(
    p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
    p_t_train, p_x_train,
    p_ts, p_window, p_output_path,
    p_t_classical=None, p_x_classical=None,
):
    """Single snapshot window: Exact (blue), Classical (green-.), PINN (red--), Data (rx)."""
    t_lo, t_hi = p_ts - p_window / 2, p_ts + p_window / 2
    m_ref = (p_t_ref >= t_lo) & (p_t_ref <= t_hi)
    t_p = p_t_pinn_test.flatten()
    m_pinn = (t_p >= t_lo) & (t_p <= t_hi)
    t_tr = p_t_train.flatten()
    m_tr = (t_tr >= t_lo) & (t_tr <= t_hi)
    n_data = int(m_tr.sum())

    fig, ax = plt.subplots(figsize=(5, 3.8))
    if p_t_classical is not None and p_x_classical is not None:
        m_cl = (p_t_classical >= t_lo) & (p_t_classical <= t_hi)
        ax.plot(p_t_classical[m_cl], p_x_classical[m_cl], "b",
                linewidth=2, label="Classical DDE solver")
    else:
        ax.plot(p_t_ref[m_ref], p_x_ref[m_ref], "b", linewidth=2, label="Reference")
    ax.plot(t_p[m_pinn], p_x_pinn[:, 0][m_pinn], "r--", linewidth=1.5, label="PINN")
    if n_data > 0:
        ax.plot(t_tr[m_tr], p_x_train[:, 0][m_tr], "rx",
                markersize=5, markeredgewidth=1.2, label="Data")
    ax.set_xlabel("$t$", fontsize=11)
    ax.set_ylabel("$x(t)$", fontsize=11)
    ax.set_title(f"$t = {p_ts:.2f}$\n{n_data} training data", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, p_output_path)
    return n_data


def f_export_table_latex(
    p_params_true, p_params_pinn,
    p_metrics_classical, p_metrics_pinn,
    p_n_value, p_output_path,
):
    """Raissi-style parameter / metrics table as matplotlib table."""
    beta_t = p_params_true["beta"]
    gamma_t = p_params_true["gamma"]
    n_t = p_params_true["n"]
    tau_t = p_params_true["tau"]
    beta_p = p_params_pinn.get("beta", beta_t)
    gamma_p = p_params_pinn.get("gamma", gamma_t)
    n_p = p_params_pinn.get("n", n_t)

    rows = [
        ["Correct DDE",
         f"$\\dot{{x}} = {beta_t}\\,"
         f"\\frac{{x(t-{tau_t})}}{{1+|x(t-{tau_t})|^{{{n_t:g}}}}} "
         f"- {gamma_t}\\,x$"],
        ["Identified (PINN)",
         f"$\\dot{{x}} = {beta_p:.4f}\\,"
         f"\\frac{{x(t-{tau_t})}}{{1+|x(t-{tau_t})|^{{{n_p:.4f}}}}} "
         f"- {gamma_p:.4f}\\,x$"],
        ["MSE  (Classical / PINN)",
         f"${p_metrics_classical['mse']:.2e}$  /  ${p_metrics_pinn['mse']:.2e}$"],
        ["Rel. $\\ell^2_N$  (Classical / PINN)",
         f"${p_metrics_classical['rel_l2']:.4f}$  /  ${p_metrics_pinn['rel_l2']:.4f}$"],
    ]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
                    bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for i in range(len(rows)):
        tbl[i, 0].set_facecolor("#eaf2f8")
        tbl[i, 0].set_text_props(fontweight="bold")
        for j in range(2):
            tbl[i, j].set_edgecolor("#bdc3c7")
            tbl[i, j].set_height(0.22)
    fig.tight_layout()
    _save(fig, p_output_path)


def f_export_delay_embedding(
    p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn, p_tau, p_output_path,
    p_t_classical=None, p_x_classical=None,
):
    """Delay-coordinate embedding x(t) vs x(t-tau)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))

    if p_t_classical is not None and p_x_classical is not None:
        dt_cl = p_t_classical[1] - p_t_classical[0]
        ds_cl = max(1, int(round(p_tau / dt_cl)))
        if ds_cl < len(p_x_classical):
            ax.plot(p_x_classical[:len(p_x_classical) - ds_cl],
                    p_x_classical[ds_cl:],
                    "b-", linewidth=0.5, alpha=0.7,
                    label="Classical DDE solver")
    else:
        dt_ref = p_t_ref[1] - p_t_ref[0]
        ds = max(1, int(round(p_tau / dt_ref)))
        if ds < len(p_x_ref):
            ax.plot(p_x_ref[:len(p_x_ref) - ds], p_x_ref[ds:],
                    "b-", linewidth=0.5, alpha=0.7, label="Reference")

    t_p = p_t_pinn_test.flatten()
    dt_p = t_p[1] - t_p[0] if len(t_p) > 1 else 1.0
    ds_p = max(1, int(round(p_tau / dt_p)))
    xp = p_x_pinn[:, 0]
    if ds_p < len(xp):
        ax.plot(xp[:len(xp) - ds_p], xp[ds_p:],
                "r--", linewidth=0.5, alpha=0.7, label="PINN")

    ax.set_xlabel("$x(t - \\tau)$", fontsize=12)
    ax.set_ylabel("$x(t)$", fontsize=12)
    ax.set_title(f"Delay-coordinate embedding ($\\tau = {p_tau}$)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
    _save(fig, p_output_path)


def _f_iteration_axis_scale(num_steps: int) -> Tuple[float, str]:
    """Return (scale, xlabel_suffix) matching common PINN convergence plots."""
    if num_steps <= 1:
        return 1.0, ""
    m = float(num_steps - 1)
    e = int(np.floor(np.log10(m)))
    if e >= 5:
        return 1e5, r" ($\times 10^{5}$)"
    if e >= 4:
        return 1e4, r" ($\times 10^{4}$)"
    if e >= 3:
        return 1e3, r" ($\times 10^{3}$)"
    return 1.0, ""


def f_export_loss_curves(
    p_loss: Sequence[float],
    p_data_loss: Optional[Sequence[float]],
    p_phy_loss: Optional[Sequence[float]],
    p_output_path: str,
    p_hist_loss: Optional[Sequence[float]] = None,
    p_ic_loss: Optional[Sequence[float]] = None,
    p_loss_segment_lengths: Optional[Sequence[int]] = None,
    p_known_missing_extended_hist_terms: bool = False,
) -> None:
    """Unweighted MSE terms and weighted ``\\mathcal{L}(\\theta)`` vs iteration (log scale).

    Legend matches the paper: ``\\mathcal{L}_{\\mathrm{data}}``,
    ``\\mathcal{L}_{\\mathrm{phys}}``, ``\\mathcal{L}_{\\mathrm{hist}}``,
    ``\\mathcal{L}_{\\mathrm{ic}}``, and composite ``\\mathcal{L}(\\theta)``.

    Older ``mglass_run.pkl`` files often omit ``history_loss_history`` /
    ``ic_loss_history``; those curves are drawn as zeros with a console note
    unless lengths mismatch ``loss_history`` (then the curve is skipped).

    Pass ``p_known_missing_extended_hist_terms=True`` after detecting legacy
    checkpoints so the exporter stays quiet once the CLI prints a fuller hint.
    """
    y_tot_full = np.asarray(p_loss, dtype=np.float64).ravel()
    n_it = int(y_tot_full.size)
    if n_it == 0:
        return
    xt = np.arange(n_it, dtype=np.float64)
    sx, sfx = _f_iteration_axis_scale(n_it)
    x_scaled = xt / sx

    def _series_ok(seq: Optional[Sequence[float]]) -> bool:
        if seq is None:
            return False
        a = np.asarray(seq, dtype=np.float64).ravel()
        return a.shape[0] == n_it

    def _hist_ic_series(p_ckpt_field: str, seq: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        """Return aligned series; zeros if missing so the legend stays complete."""
        if seq is None:
            if not p_known_missing_extended_hist_terms:
                print(
                    f"  [loss curves] `{p_ckpt_field}` missing ({n_it} iters): "
                    "drawing zeros — train again (or regenerate pickle) to log histories.",
                )
            return np.zeros(n_it, dtype=np.float64)
        a = np.asarray(seq, dtype=np.float64).ravel()
        if a.size == 0:
            print(
                f"  [loss curves] `{p_ckpt_field}` empty: zeros ({n_it} iters).",
            )
            return np.zeros(n_it, dtype=np.float64)
        if a.size != n_it:
            print(
                f"  [loss curves] `{p_ckpt_field}` length {a.size} != "
                f"{n_it}; skipping curve.",
            )
            return None
        return a.copy()

    l_for_floor: List[np.ndarray] = [y_tot_full]
    if _series_ok(p_data_loss):
        l_for_floor.append(np.asarray(p_data_loss, dtype=np.float64).ravel())
    if _series_ok(p_phy_loss):
        l_for_floor.append(np.asarray(p_phy_loss, dtype=np.float64).ravel())
    if _series_ok(p_hist_loss):
        l_for_floor.append(np.asarray(p_hist_loss, dtype=np.float64).ravel())
    if _series_ok(p_ic_loss):
        l_for_floor.append(np.asarray(p_ic_loss, dtype=np.float64).ravel())

    v_pos: List[float] = []
    for u in l_for_floor:
        pmask = np.isfinite(u) & (u > 0)
        if np.any(pmask):
            v_pos.append(float(np.min(u[pmask])))
    v_y_floor = max(1e-20, float(min(v_pos)) * 1e-6) if v_pos else 1e-12

    y_tot = np.maximum(y_tot_full, v_y_floor)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    parts: List[Tuple[np.ndarray, str, str]] = []

    if _series_ok(p_data_loss):
        parts.append((
            np.maximum(np.asarray(p_data_loss, dtype=np.float64).ravel(), v_y_floor),
            r"$\mathcal{L}_{\mathrm{data}}$",
            "#1f77b4",
        ))
    if _series_ok(p_phy_loss):
        parts.append((
            np.maximum(np.asarray(p_phy_loss, dtype=np.float64).ravel(), v_y_floor),
            r"$\mathcal{L}_{\mathrm{phys}}$",
            "#2ca02c",
        ))

    arr_hist = _hist_ic_series("history_loss_history", p_hist_loss)
    arr_ic = _hist_ic_series("ic_loss_history", p_ic_loss)
    if arr_hist is not None:
        parts.append((
            np.maximum(arr_hist, v_y_floor),
            r"$\mathcal{L}_{\mathrm{hist}}$",
            "#ff7f0e",
        ))
    if arr_ic is not None:
        parts.append((
            np.maximum(arr_ic, v_y_floor),
            r"$\mathcal{L}_{\mathrm{ic}}$",
            "#9467bd",
        ))
    for y_plt, lbl, clr in parts:
        ax.semilogy(x_scaled, y_plt, color=clr, linewidth=1.15, alpha=0.95, label=lbl)

    ax.semilogy(x_scaled, y_tot, color="black", linestyle="--", linewidth=1.45,
                alpha=0.9, label=r"$\mathcal{L}(\theta)$")

    if p_loss_segment_lengths:
        lengths = np.asarray(p_loss_segment_lengths, dtype=np.int64).tolist()
        if sum(lengths) == n_it and len(lengths) > 1:
            cuts = np.cumsum(lengths[:-1])
            for k in cuts:
                ax.axvline(
                    float(k) / sx, color="#555555",
                    linestyle=":", linewidth=0.9, alpha=0.5, zorder=0,
                )

    ax.set_yscale("log")
    ax.set_title("Loss convergence", fontsize=13, pad=10)
    xlbl = r"Iteration" + (sfx if sfx else "")
    ax.set_xlabel(xlbl, fontsize=12)
    ax.set_ylabel(r"MSE loss terms", fontsize=12)
    hL, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, hL))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=10, ncol=2, framealpha=0.92,
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.38)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.33, alpha=0.28)
    fig.subplots_adjust(bottom=0.11, left=0.09, top=0.92, right=0.98)
    _save(fig, p_output_path)


def f_export_junction_ic_loss_convergence(
    p_ic_loss: Sequence[float],
    p_output_path: str,
    p_loss_segment_lengths: Optional[Sequence[int]] = None,
    p_aligned_len: Optional[int] = None,
) -> None:
    """MSE convergence for the junction / initial-condition term $\\mathcal{L}_{\\mathrm{ic}}$."""
    v_y_floor = 1e-16
    y_ic = np.asarray(p_ic_loss, dtype=np.float64).ravel()
    n_ic = int(y_ic.size)
    if n_ic == 0:
        return
    if p_aligned_len is not None and int(p_aligned_len) != n_ic:
        print(
            f"  [skip] IC loss convergence: length mismatch "
            f"({n_ic} vs {int(p_aligned_len)} total iterations)",
        )
        return

    xt = np.arange(n_ic, dtype=np.float64)
    sx, sfx = _f_iteration_axis_scale(n_ic)
    x_scaled = xt / sx

    fig, ax = plt.subplots(figsize=(10, 4.25))
    y_plot = np.maximum(y_ic, v_y_floor)
    ax.semilogy(x_scaled, y_plot, color="#9467bd", linewidth=1.35, alpha=0.95,
                label=r"$\mathcal{L}_{\mathrm{ic}}$")

    if p_loss_segment_lengths:
        lengths = np.asarray(p_loss_segment_lengths, dtype=np.int64).tolist()
        if sum(lengths) == n_ic and len(lengths) > 1:
            cuts = np.cumsum(lengths[:-1])
            for k in cuts:
                ax.axvline(
                    float(k) / sx, color="#555555",
                    linestyle=":", linewidth=0.9, alpha=0.5, zorder=0,
                )

    ax.set_yscale("log")
    ax.set_title(r"$\mathcal{L}_{\mathrm{ic}}$ (junction IC MSE)", fontsize=13, pad=10)
    xlbl = r"Iteration" + (sfx if sfx else "")
    ax.set_xlabel(xlbl, fontsize=12)
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{ic}}$", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.92)
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.38)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.33, alpha=0.28)
    fig.subplots_adjust(bottom=0.13, left=0.10, top=0.90, right=0.98)
    _save(fig, p_output_path)


def f_export_mglass_prescribed_ic_breakdown_ns(
    p_aligned_len: int,
    p_hist_far: Optional[Sequence[float]],
    p_hist_mid: Optional[Sequence[float]],
    p_hist_near0: Optional[Sequence[float]],
    p_hist_agg: Optional[Sequence[float]],
    p_ic: Optional[Sequence[float]],
    p_output_path: str,
    p_loss_segment_lengths: Optional[Sequence[int]] = None,
) -> None:
    """Navier-Stokes-style panel: residuals on prescribed history slices + junction \\mathcal{L}_{ic}.

    MG is scalar; analogous “multiple ICs” split the enforced history curve on ``[-\\tau,0]``.
    """
    def _nz(seq: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if seq is None:
            return None
        a = np.asarray(seq, dtype=np.float64).ravel()
        if a.size != int(p_aligned_len):
            return None
        return np.maximum(a, 1e-16)

    n_it = int(p_aligned_len)
    if n_it <= 0:
        return

    xf = _nz(p_hist_far)
    xm = _nz(p_hist_mid)
    xn0 = _nz(p_hist_near0)
    xh = _nz(p_hist_agg)
    x_ic = _nz(p_ic)

    traces: List[Tuple[np.ndarray, str, str]] = []
    v_bins_ready = xf is not None and xm is not None and xn0 is not None
    if v_bins_ready:
        traces.extend([
            (xf,
             r"$\mathcal{L}_{\mathrm{hist}}\,(-\tau\ \mathrm{band})$", "#084594"),
            (xm,
             r"$\mathcal{L}_{\mathrm{hist}}\,(\mathrm{mid})$", "#4292c6"),
            (xn0,
             r"$\mathcal{L}_{\mathrm{hist}}\,(0^{-}\ \mathrm{band})$", "#9ecae1"),
        ])
    elif xh is not None:
        traces.append(
            (xh, r"$\mathcal{L}_{\mathrm{hist}}$", "#ff7f0e"),
        )
    if x_ic is not None:
        traces.append(
            (x_ic, r"$\mathcal{L}_{\mathrm{ic}}$", "#9467bd"),
        )
    if not traces:
        return

    xt = np.arange(n_it, dtype=np.float64)
    sx, sfx = _f_iteration_axis_scale(n_it)
    x_scaled = xt / sx

    fig, ax = plt.subplots(figsize=(10, 5.25))
    for y_plt, lbl, clr in traces:
        ax.semilogy(x_scaled, y_plt, linewidth=1.15, alpha=0.95, label=lbl,
                    color=clr)

    if p_loss_segment_lengths:
        lengths = np.asarray(p_loss_segment_lengths, dtype=np.int64).tolist()
        if sum(lengths) == n_it and len(lengths) > 1:
            cuts = np.cumsum(lengths[:-1])
            for k in cuts:
                ax.axvline(
                    float(k) / sx, color="#555555",
                    linestyle=":", linewidth=0.9, alpha=0.5, zorder=0,
                )

    ax.set_yscale("log")
    ax.set_title(
        "Prescribed history & junction (IC-style residuals; scalar Mackey-Glass PINN)",
        fontsize=13,
        pad=10,
    )
    xlbl = r"Iteration" + (sfx if sfx else "")
    ax.set_xlabel(xlbl, fontsize=12)
    ax.set_ylabel(r"MSE residual", fontsize=12)
    hL, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, hL))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=10, ncol=2, framealpha=0.92,
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.38)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.33, alpha=0.28)
    fig.subplots_adjust(bottom=0.13, left=0.10, top=0.90, right=0.98)
    _save(fig, p_output_path)


def f_export_mglass_ic_terms_ns_three_panel(
    p_aligned_len: int,
    p_hist_far: Optional[Sequence[float]],
    p_hist_mid: Optional[Sequence[float]],
    p_hist_near0: Optional[Sequence[float]],
    p_hist_agg: Optional[Sequence[float]],
    p_ic: Optional[Sequence[float]],
    p_training_plot_bundle: Optional[Dict[str, Any]],
    p_output_path: str,
    p_loss_segment_lengths: Optional[Sequence[int]] = None,
) -> None:
    """Navier-Stokes-style **stack** for MG: IC-type histories + junction, plus modifiers.

    * **Top**: unweighted MSE on prescribed history thirds on ``[-\\tau, 0]`` (when
      available) plus junction ``\\mathcal{L}_{\\mathrm{ic}}``.
    * **Middle**: weighted contributions logged during training when present
      (``weighted_*_term_per_step`` inside ``training_plot_bundle``); else
      nominal ``loss_weights_nominal`` times the unweighted MSE traces.
    * **Bottom**: physics-loss multiplier relative to YAML weight (Adam ramp),
      and curriculum fraction of the active time window — both ``\\in [0,1]``.
      If those series were never saved (old checkpoints/pickles), the panel shows
      a short note instead of empty axes.
    """
    def _as_pos(seq: Optional[Sequence[float]], n: int) -> Optional[np.ndarray]:
        if seq is None:
            return None
        a = np.asarray(seq, dtype=np.float64).ravel()
        if a.size != int(n):
            return None
        out = np.array(a, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            return None
        return np.maximum(out, 1e-16)

    n_it = int(p_aligned_len)
    if n_it <= 0:
        return

    tb_any = (
        p_training_plot_bundle if isinstance(p_training_plot_bundle, dict) else {}
    )
    lw = tb_any.get("loss_weights_nominal") or {}
    w_hist_nom = float(lw.get("history_loss", 1.0))
    w_ic_nom = float(lw.get("ic_loss", 10.0))

    xf = _as_pos(p_hist_far, n_it)
    xm = _as_pos(p_hist_mid, n_it)
    xn0 = _as_pos(p_hist_near0, n_it)
    xh = _as_pos(p_hist_agg, n_it)
    x_ic = _as_pos(p_ic, n_it)

    pan1_traces: List[Tuple[np.ndarray, str, str]] = []
    v_bins_ready = xf is not None and xm is not None and xn0 is not None
    if v_bins_ready:
        pan1_traces.extend([
            (xf,
             r"$\mathcal{L}_{\mathrm{hist}}\,(-\tau\ \mathrm{band})$", "#084594"),
            (xm,
             r"$\mathcal{L}_{\mathrm{hist}}\,(\mathrm{mid})$", "#4292c6"),
            (xn0,
             r"$\mathcal{L}_{\mathrm{hist}}\,(0^{-}\ \mathrm{band})$", "#9ecae1"),
        ])
    elif xh is not None:
        pan1_traces.append((xh, r"$\mathcal{L}_{\mathrm{hist}}$", "#ff7f0e"))
    if x_ic is not None:
        pan1_traces.append((x_ic, r"$\mathcal{L}_{\mathrm{ic}}$", "#9467bd"))

    if not pan1_traces:
        return

    def _bundle_series(keyx: str) -> Optional[np.ndarray]:
        xs = tb_any.get(keyx)
        if xs is None:
            return None
        if isinstance(xs, (list, tuple, np.ndarray)):
            a = np.asarray(xs, dtype=np.float64).ravel()
        else:
            return None
        if a.size != n_it:
            return None
        return a

    wf_pt = _bundle_series("weighted_hist_far_term_per_step")
    wm_pt = _bundle_series("weighted_hist_mid_term_per_step")
    wn0_pt = _bundle_series("weighted_hist_near0_term_per_step")
    wic_pt = _bundle_series("weighted_ic_term_per_step")
    wt_hist_agg = _bundle_series("weighted_history_term_per_step")

    pan2_traces: List[Tuple[np.ndarray, str, str]] = []
    if v_bins_ready:
        for arr_uw, lbl, clr, bk in [
            (xf, r"$\lambda_{\mathrm{hist}}\mathcal{L}_{\mathrm{hist}}\,(-\tau)$",
             "#084594", wf_pt),
            (xm, r"$\lambda_{\mathrm{hist}}\mathcal{L}_{\mathrm{hist}}\,(\mathrm{mid})$",
             "#4292c6", wm_pt),
            (xn0,
             r"$\lambda_{\mathrm{hist}}\mathcal{L}_{\mathrm{hist}}\,(0^{-})$",
             "#9ecae1", wn0_pt),
        ]:
            s = bk if bk is not None else (w_hist_nom * arr_uw)
            pan2_traces.append((np.maximum(s, 1e-16), lbl, clr))
    elif xh is not None:
        hist_w = wt_hist_agg if wt_hist_agg is not None else w_hist_nom * xh
        pan2_traces.append((np.maximum(hist_w, 1e-16),
                            r"$\lambda_{\mathrm{hist}}\mathcal{L}_{\mathrm{hist}}$",
                            "#ff7f0e"))

    if x_ic is not None:
        sic = (
            wic_pt if wic_pt is not None else w_ic_nom * x_ic
        )
        pan2_traces.append((
            np.maximum(sic, 1e-16),
            r"$\lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}$",
            "#9467bd",
        ))

    xt = np.arange(n_it, dtype=np.float64)
    sx, sfx = _f_iteration_axis_scale(n_it)
    x_scaled = xt / sx

    def _vlines(ax_plt) -> None:
        if not p_loss_segment_lengths:
            return
        lengths = np.asarray(p_loss_segment_lengths, dtype=np.int64).tolist()
        if sum(lengths) == n_it and len(lengths) > 1:
            for k in np.cumsum(lengths[:-1]):
                ax_plt.axvline(
                    float(k) / sx, color="#555555",
                    linestyle=":", linewidth=0.85, alpha=0.45, zorder=0,
                )

    phy_f = _bundle_series("physics_weight_fraction_per_step")
    cur_f = _bundle_series("curriculum_time_fraction_per_step")

    floor_vals: List[float] = []
    for arr, _, __ in pan1_traces:
        pv = arr[np.isfinite(arr) & (arr > 0)]
        if pv.size > 0:
            floor_vals.append(float(np.min(pv)))
    for arr, _, __ in pan2_traces:
        pv = arr[np.isfinite(arr) & (arr > 0)]
        if pv.size > 0:
            floor_vals.append(float(np.min(pv)))
    y_floor_log = (
        max(1e-20, min(floor_vals) * 1e-6) if floor_vals else 1e-12
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11.8), squeeze=False)
    ax1, ax2, ax3 = axes.flatten()

    for arr, lbl, clr in pan1_traces:
        yy = np.maximum(arr, y_floor_log)
        ax1.semilogy(x_scaled, yy, linewidth=1.12, alpha=0.95, label=lbl, color=clr)
    _vlines(ax1)
    ax1.set_yscale("log")
    ax1.set_title(
        "(a) IC-style residuals (unweighted MSE)",
        fontsize=13, pad=8,
    )
    ax1.set_ylabel(r"MSE term", fontsize=11)
    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.38)
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.26)
    hL, lbls = ax1.get_legend_handles_labels()
    by_li = dict(zip(lbls, hL))
    ax1.legend(
        by_li.values(), by_li.keys(),
        fontsize=9, ncol=2, framealpha=0.93, loc="upper right",
    )

    for arr, lbl, clr in pan2_traces:
        yy = np.maximum(arr, y_floor_log)
        ax2.semilogy(x_scaled, yy, linewidth=1.12, alpha=0.95, label=lbl, color=clr)
    _vlines(ax2)
    ax2.set_yscale("log")
    ax2.set_title(
        "(b) Weighted IC-style contributions "
        r"($\lambda_{\mathrm{hist}}$, $\lambda_{\mathrm{ic}}$ from config / trace)",
        fontsize=13, pad=8,
    )
    ax2.set_ylabel(r"Weighted term", fontsize=11)
    ax2.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.38)
    ax2.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.26)
    h2, lb2 = ax2.get_legend_handles_labels()
    by_2 = dict(zip(lb2, h2))
    ax2.legend(
        by_2.values(), by_2.keys(),
        fontsize=9, ncol=2, framealpha=0.93, loc="upper right",
    )

    xlbl_main = r"Iteration" + (sfx if sfx else "")
    v_phy_ok = phy_f is not None and np.any(np.isfinite(phy_f))
    v_cur_ok = cur_f is not None and np.any(np.isfinite(cur_f))
    v_show_sched = v_phy_ok or v_cur_ok

    if v_show_sched:
        ax3.set_ylim(-0.05, 1.08)
    if v_phy_ok:
        ax3.plot(
            x_scaled, np.clip(phy_f, 0.0, 1.05), color="#333333",
            linewidth=1.15, alpha=0.95,
            label=r"$\lambda_{\mathrm{phys}}(k) \,/\, "
                  r"\lambda_{\mathrm{phys}}^{(\mathrm{nom})}$",
        )
    if v_cur_ok:
        ax3.plot(
            x_scaled, np.clip(cur_f, 0.0, 1.05), color="#3182bd",
            linewidth=1.05, alpha=0.9,
            label=r"Curriculum frac.\ (active $t$-range)",
        )
    if not v_show_sched:
        ax3.text(
            0.5, 0.52,
            "No physics ramp / curriculum traces\n"
            "(retrain after upgrade, or use newer checkpoints)",
            ha="center", va="center", fontsize=10, transform=ax3.transAxes,
        )
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_frame_on(False)

    ax3.set_title(
        "(c) Training schedule (modifiers vs iteration)",
        fontsize=13, pad=8,
    )
    if v_show_sched:
        _vlines(ax3)
        ax3.set_xlabel(xlbl_main, fontsize=12)
        ax3.set_ylabel(r"Fraction in $[0,1]$", fontsize=11)
        ax3.grid(True, linestyle=":", linewidth=0.45, alpha=0.42)
        h3, lb3 = ax3.get_legend_handles_labels()
        by3 = dict(zip(lb3, h3))
        ax3.legend(
            by3.values(), by3.keys(),
            fontsize=9, ncol=1, framealpha=0.92, loc="upper right",
        )
    else:
        ax3.set_xlabel("")
        ax3.set_ylabel("")

    if v_show_sched:
        ax1.tick_params(axis="x", labelbottom=False)
        ax2.tick_params(axis="x", labelbottom=False)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.06, hspace=0.36)
    else:
        ax2.set_xlabel(xlbl_main, fontsize=12)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.08, hspace=0.38)
    _save(fig, p_output_path)


def f_export_pointwise_error(
    p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn, p_output_path,
    p_t_classical=None, p_x_classical=None,
):
    """Pointwise absolute error |x_ref - x_pred| for both classical and PINN."""
    ref_interp = interp1d(p_t_ref, p_x_ref, kind="cubic", fill_value="extrapolate")
    t_p = p_t_pinn_test.flatten()
    err_pinn = np.abs(ref_interp(t_p) - p_x_pinn[:, 0])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if p_t_classical is not None and p_x_classical is not None:
        cl_interp = interp1d(p_t_classical, p_x_classical, kind="cubic",
                             fill_value="extrapolate")
        err_cl = np.abs(ref_interp(t_p) - cl_interp(t_p))
        ax.semilogy(t_p, err_cl, "b-", linewidth=0.8,
                     label="Classical DDE solver")
    ax.semilogy(t_p, err_pinn, "r-", linewidth=0.8, label="PINN")
    ax.set_xlabel("$t$", fontsize=11)
    ax.set_ylabel("Pointwise absolute error", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
    _save(fig, p_output_path)


# ─────── Raissi-style combined figure ───────

def f_create_mglass_figure(
    p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
    p_t_train, p_x_train,
    p_params_true, p_params_pinn,
    p_metrics_pinn, p_metrics_classical,
    p_t_classical, p_x_classical,
    p_n_value, p_tau,
    p_snapshot_times, p_snapshot_window,
    p_output_path,
    p_loss_history=None, p_data_loss_history=None, p_physics_loss_history=None,
    p_hist_loss_history=None,
    p_ic_loss_history=None,
    p_loss_segment_lengths=None,
    p_hist_loss_far_history=None,
    p_hist_loss_mid_history=None,
    p_hist_loss_near0_history=None,
    p_training_plot_bundle: Optional[Dict[str, Any]] = None,
):
    """
    Raissi PINNs-style combined figure (gridspec layout).

    Row 0 : Rainbow heatmap of x(t+delta) with white snapshot lines
    Row 1 : Snapshot windows  (Exact blue + PINN red-- + Data rx)
    Row 2 : Legend  +  LaTeX parameter / metrics table
    """
    v_n_snap = len(p_snapshot_times)

    fig = plt.figure(figsize=(10, 11))
    fig.patch.set_facecolor("white")

    # ── Row 0: Heatmap ──
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3 + 0.05, left=0.10, right=0.92,
               wspace=0.30)

    n_rows = 80
    tau_vis = p_tau
    shifts = np.linspace(-tau_vis, tau_vis, n_rows)
    t_grid = np.linspace(p_t_ref.min(), p_t_ref.max(), 2000)
    cl_interp = interp1d(p_t_classical, p_x_classical, kind="cubic",
                         bounds_error=False, fill_value=np.nan)
    t_pinn = p_t_pinn_test.flatten()
    pinn_interp = interp1d(t_pinn, p_x_pinn[:, 0], kind="cubic",
                           bounds_error=False, fill_value=np.nan)

    img_cl = np.array([cl_interp(t_grid + s) for s in shifts])
    img_pinn = np.array([pinn_interp(t_grid + s) for s in shifts])
    extent = [t_grid.min(), t_grid.max(), shifts.min(), shifts.max()]
    vmin, vmax = np.nanmin(img_cl), np.nanmax(img_cl)

    for col, (img, ttl) in enumerate([
        (img_cl, "Classical DDE $x(t+\\delta)$"),
        (img_pinn, "PINN $\\hat{x}(t+\\delta)$"),
    ]):
        ax = plt.subplot(gs0[0, col])
        h = ax.imshow(img, interpolation="nearest", cmap="rainbow",
                       extent=extent, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax)
        for ts in p_snapshot_times:
            ax.plot([ts, ts], [shifts.min(), shifts.max()], "w-", linewidth=1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$\\delta$")
        ax.set_title(ttl, fontsize=10)

    # ── Row 1: Snapshot subplots ──
    gs1 = gridspec.GridSpec(1, v_n_snap)
    gs1.update(top=1 - 1 / 3 - 0.03, bottom=1 - 2 / 3 + 0.02, left=0.10,
               right=0.92, wspace=0.45)

    snap_n_data = []
    for i, ts in enumerate(p_snapshot_times):
        ax = plt.subplot(gs1[0, i])
        t_lo, t_hi = ts - p_snapshot_window / 2, ts + p_snapshot_window / 2
        m_ref = (p_t_ref >= t_lo) & (p_t_ref <= t_hi)
        m_pinn = (t_pinn >= t_lo) & (t_pinn <= t_hi)
        t_tr = p_t_train.flatten()
        m_tr = (t_tr >= t_lo) & (t_tr <= t_hi)
        nd = int(m_tr.sum())
        snap_n_data.append(nd)

        m_cl = (p_t_classical >= t_lo) & (p_t_classical <= t_hi)
        ax.plot(p_t_classical[m_cl], p_x_classical[m_cl], "b",
                linewidth=2, label="Classical DDE solver")
        ax.plot(t_pinn[m_pinn], p_x_pinn[:, 0][m_pinn], "r--",
                linewidth=1.5, label="PINN")
        if nd > 0:
            ax.plot(t_tr[m_tr], p_x_train[:, 0][m_tr], "rx",
                    markersize=5, markeredgewidth=1.2, label="Data")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x(t)$")
        ax.set_title(f"$t = {ts:.2f}$\n{nd} training data", fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 2 / 3 - 0.005), ncol=3, frameon=False,
               fontsize=10)

    # ── Row 2: Metrics table ──
    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(top=1 - 2 / 3 - 0.06, bottom=0.02, left=0.10, right=0.92)

    beta_t = p_params_true["beta"]
    gamma_t = p_params_true["gamma"]
    n_t = p_params_true["n"]
    tau_t = p_params_true["tau"]
    beta_p = p_params_pinn.get("beta", beta_t)
    gamma_p = p_params_pinn.get("gamma", gamma_t)
    n_p = p_params_pinn.get("n", n_t)

    rows = [
        ["Mackey-Glass DDE",
         f"$\\dot{{x}} = {beta_t}\\,"
         f"\\frac{{x(t-{tau_t})}}{{1+|x(t-{tau_t})|^{{{n_t:g}}}}} "
         f"- {gamma_t}\\,x$"],
        ["MSE  (Classical DDE / PINN)",
         f"${p_metrics_classical['mse']:.2e}$  /  ${p_metrics_pinn['mse']:.2e}$"],
        ["Rel. $\\ell^2_N$  (Classical DDE / PINN)",
         f"${p_metrics_classical['rel_l2']:.4f}$  /  ${p_metrics_pinn['rel_l2']:.4f}$"],
    ]
    ax_tab = plt.subplot(gs2[0, 0])
    ax_tab.axis("off")
    tbl = ax_tab.table(cellText=rows, cellLoc="center", loc="center",
                        bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for i in range(len(rows)):
        tbl[i, 0].set_facecolor("#eaf2f8")
        tbl[i, 0].set_text_props(fontweight="bold")
        for j in range(2):
            tbl[i, j].set_edgecolor("#bdc3c7")
            tbl[i, j].set_height(0.22)

    _save(fig, p_output_path)

    # ── Export all subplots individually ──
    v_dir = os.path.dirname(p_output_path)
    v_prefix = f"n{p_n_value:g}"
    print(f"  Exporting individual plots (prefix={v_prefix})...")

    f_export_heatmap(
        p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn, p_snapshot_times,
        p_n_value, os.path.join(v_dir, f"{v_prefix}_heatmap.png"),
        p_t_classical=p_t_classical, p_x_classical=p_x_classical,
    )
    f_export_time_series(
        p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn, p_snapshot_times,
        p_n_value, p_params_true,
        os.path.join(v_dir, f"{v_prefix}_timeseries.png"),
        p_t_classical=p_t_classical, p_x_classical=p_x_classical,
    )
    for i, ts in enumerate(p_snapshot_times):
        f_export_snapshot(
            p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
            p_t_train, p_x_train,
            ts, p_snapshot_window,
            os.path.join(v_dir, f"{v_prefix}_snapshot_t{ts:g}.png"),
            p_t_classical=p_t_classical, p_x_classical=p_x_classical,
        )
    f_export_table_latex(
        p_params_true, p_params_pinn,
        p_metrics_classical, p_metrics_pinn,
        p_n_value, os.path.join(v_dir, f"{v_prefix}_table.png"),
    )
    f_export_delay_embedding(
        p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn, p_tau,
        os.path.join(v_dir, f"{v_prefix}_delay_embedding.png"),
        p_t_classical=p_t_classical, p_x_classical=p_x_classical,
    )
    if p_loss_history and len(p_loss_history) > 0:
        f_export_loss_curves(
            p_loss_history, p_data_loss_history, p_physics_loss_history,
            os.path.join(v_dir, f"{v_prefix}_loss_curves.png"),
            p_hist_loss=p_hist_loss_history,
            p_ic_loss=p_ic_loss_history,
            p_loss_segment_lengths=p_loss_segment_lengths,
        )
        if (
            p_ic_loss_history is not None
            and len(p_ic_loss_history) > 0
            and len(p_ic_loss_history) == len(p_loss_history)
        ):
            f_export_junction_ic_loss_convergence(
                p_ic_loss_history,
                os.path.join(v_dir, f"{v_prefix}_lic_loss_convergence.png"),
                p_loss_segment_lengths=p_loss_segment_lengths,
                p_aligned_len=len(p_loss_history),
            )
        f_export_mglass_prescribed_ic_breakdown_ns(
            len(p_loss_history),
            p_hist_far=p_hist_loss_far_history,
            p_hist_mid=p_hist_loss_mid_history,
            p_hist_near0=p_hist_loss_near0_history,
            p_hist_agg=p_hist_loss_history,
            p_ic=p_ic_loss_history,
            p_output_path=os.path.join(v_dir,
                                       f"{v_prefix}_prescribed_ic_hist_breakdown.png"),
            p_loss_segment_lengths=p_loss_segment_lengths,
        )
        f_export_mglass_ic_terms_ns_three_panel(
            len(p_loss_history),
            p_hist_far=p_hist_loss_far_history,
            p_hist_mid=p_hist_loss_mid_history,
            p_hist_near0=p_hist_loss_near0_history,
            p_hist_agg=p_hist_loss_history,
            p_ic=p_ic_loss_history,
            p_training_plot_bundle=p_training_plot_bundle,
            p_output_path=os.path.join(
                v_dir, f"{v_prefix}_ic_terms_ns_three_panel.png",
            ),
            p_loss_segment_lengths=p_loss_segment_lengths,
        )
    f_export_pointwise_error(
        p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
        os.path.join(v_dir, f"{v_prefix}_pointwise_error.png"),
        p_t_classical=p_t_classical, p_x_classical=p_x_classical,
    )

    # ── 3D delay-embedding plots ──
    if p_t_classical is not None and p_x_classical is not None:
        f_export_3d_attractor_comparison(
            p_t_classical, p_x_classical,
            p_t_pinn_test, p_x_pinn,
            p_tau, p_n_value, v_dir,
        )
        f_export_3d_attractor_single(
            p_t_classical, p_x_classical, p_tau, p_n_value,
            "Classical DDE solver", "viridis",
            os.path.join(v_dir, f"{v_prefix}_3d_classical.png"),
        )
        f_export_3d_overlay(
            p_t_ref, p_x_ref,
            p_t_classical, p_x_classical,
            p_t_pinn_test, p_x_pinn,
            p_tau, p_n_value,
            os.path.join(v_dir, f"{v_prefix}_3d_overlay.png"),
        )
    f_export_3d_attractor_single(
        p_t_pinn_test.flatten(), p_x_pinn[:, 0], p_tau, p_n_value,
        "PINN", "inferno",
        os.path.join(v_dir, f"{v_prefix}_3d_pinn.png"),
    )
    f_export_3d_error_surface(
        p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
        p_tau, p_n_value,
        os.path.join(v_dir, f"{v_prefix}_3d_error_surface.png"),
    )


def f_create_multi_n_summary(
    p_results: Dict[float, Dict[str, Any]],
    p_output_path: str,
    p_snapshot_times: Optional[List[float]] = None,
):
    """Summary across n values: heatmap + per-n time series."""
    l_n_vals = sorted(p_results.keys())
    n_panels = len(l_n_vals)

    if p_snapshot_times is None:
        p_snapshot_times = []

    # Also export n-sweep heatmap
    v_dir = os.path.dirname(p_output_path)
    f_export_heatmap_n_sweep(
        p_results, p_snapshot_times,
        os.path.join(v_dir, "heatmap_n_sweep.png"),
    )

    fig, axes = plt.subplots(n_panels, 2, figsize=(12, 3.0 * n_panels), squeeze=False)
    for i, n in enumerate(l_n_vals):
        r = p_results[n]
        ax_ts = axes[i, 0]
        ax_ts.plot(r["t_ref"], r["x_ref"], "b-", linewidth=1.2, label="Reference")
        if len(r.get("x_pinn", [])) > 0:
            ax_ts.plot(r["t_pinn_test"].flatten(), r["x_pinn"][:, 0],
                       "r--", linewidth=0.9, alpha=0.85, label="PINN")
        ax_ts.set_ylabel("$x(t)$")
        ax_ts.set_title(f"$n={n:g}$  |  MSE$_{{\\mathrm{{PINN}}}}$"
                        f" = {r['metrics_pinn']['mse']:.2e}")
        ax_ts.grid(True, alpha=0.2, linestyle=":")
        if i == 0:
            ax_ts.legend(fontsize=9)
        if i == n_panels - 1:
            ax_ts.set_xlabel("$t$")

        ax_de = axes[i, 1]
        tau = r["tau"]
        dt_r = r["t_ref"][1] - r["t_ref"][0]
        ds = max(1, int(round(tau / dt_r)))
        xr = r["x_ref"]
        if ds < len(xr):
            ax_de.plot(xr[:len(xr) - ds], xr[ds:], "b-", linewidth=0.4, alpha=0.6)
        if len(r.get("x_pinn", [])) > 0:
            xp = r["x_pinn"][:, 0]
            t_p = r["t_pinn_test"].flatten()
            dt_p = t_p[1] - t_p[0] if len(t_p) > 1 else 1.0
            ds_p = max(1, int(round(tau / dt_p)))
            if ds_p < len(xp):
                ax_de.plot(xp[:len(xp) - ds_p], xp[ds_p:], "r--",
                           linewidth=0.4, alpha=0.6)
        ax_de.set_ylabel("$x(t)$")
        ax_de.set_title(f"Delay embedding $n={n:g}$")
        ax_de.grid(True, alpha=0.2, linestyle=":")
        if i == n_panels - 1:
            ax_de.set_xlabel("$x(t-\\tau)$")

    fig.tight_layout()
    _save(fig, p_output_path)


def f_create_metrics_table_figure(
    p_results: Dict[float, Dict[str, Any]],
    p_output_path: str,
):
    """Standalone metrics table for all n values."""
    l_n_vals = sorted(p_results.keys())
    l_rows = []
    for n in l_n_vals:
        r = p_results[n]
        l_rows.append([
            f"{n:g}",
            f"{r['metrics_classical']['mse']:.2e}",
            f"{r['metrics_pinn']['mse']:.2e}",
            f"{r['metrics_classical']['rel_l2']:.4f}",
            f"{r['metrics_pinn']['rel_l2']:.4f}",
            f"{r['wall_time_classical']:.1f}",
            f"{r['wall_time_pinn']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.5 * len(l_rows)))
    ax.axis("off")
    cols = ["$n$", "MSE\n(Classical)", "MSE\n(PINN)",
            "Rel. $\\ell^2_N$\n(Classical)", "Rel. $\\ell^2_N$\n(PINN)",
            "Time (s)\n(Classical)", "Time (s)\n(PINN)"]
    tbl = ax.table(cellText=l_rows, colLabels=cols, cellLoc="center",
                    loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#d4e6f1")
        tbl[0, j].set_text_props(fontweight="bold", fontsize=9)
    for i in range(1, len(l_rows) + 1):
        tbl[i, 0].set_facecolor("#eaf2f8")
    ax.set_title("Mackey-Glass: Classical DDE Solver vs PINN",
                 fontsize=13, fontweight="bold", pad=15)
    _save(fig, p_output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def f_load_config(p_path: str) -> Dict[str, Any]:
    """Load YAML config."""
    import yaml
    with open(p_path, "r") as fh:
        return yaml.safe_load(fh)


def f_parse_args():
    v_parser = argparse.ArgumentParser(description="Mackey-Glass DDE comparison: Classical DDE vs PINN")
    v_parser.add_argument(
        "--config",
        default=str(MGLASS_COMPARISON_ROOT / "configs" / "config_mackey_glass_t200_windowed.yaml"),
        help="PINN config YAML (relative paths resolve under this reproducibility bundle)",
    )
    v_parser.add_argument(
        "--n-values",
        default="7,10,20",
        help="Comma-separated Hill exponent values",
    )
    v_parser.add_argument(
        "--output-dir",
        default=str(MGLASS_COMPARISON_ROOT / "results"),
        help="Output directory for figures and data (relative paths are under mglass_comparison/)",
    )
    v_parser.add_argument(
        "--skip-pinn",
        action="store_true",
        help="Skip PINN training (classical solver only)",
    )
    v_parser.add_argument(
        "--ignore-pinn-checkpoints",
        action="store_true",
        help="Never resume PINN from checkpoints_n{N}/window_*.pt; retrain all windows "
             "fresh (needed when old checkpoints omit history / junction traces).",
    )
    v_parser.add_argument(
        "--snapshot-times",
        default=None,
        help="Comma-separated snapshot times for zoomed windows (auto if not set)",
    )
    v_parser.add_argument(
        "--snapshot-window",
        type=float,
        default=40.0,
        help="Width of each snapshot window in time units",
    )
    v_parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Override time horizon (default: from config)",
    )
    v_parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override config data.dt (training grid / YAML). For MoS output/max_step "
        "use --classical-dt when set; otherwise MoS uses min(data.dt, 0.01).",
    )
    v_parser.add_argument(
        "--classical-dt",
        type=float,
        default=None,
        help="MoS-RK45 output grid spacing and scipy solve_ivp max_step (overrides min(dt,0.01)).",
    )
    v_parser.add_argument(
        "--ref-dt",
        type=float,
        default=None,
        help="Fixed-step RK4 reference trajectory dt (default: 0.001).",
    )
    v_parser.add_argument(
        "--reference-convergence",
        action="store_true",
        help="Write reference_convergence.csv: RK4 self-discrepancy vs --ref-dt for --reference-convergence-dts.",
    )
    v_parser.add_argument(
        "--reference-convergence-dts",
        default="0.01,0.005,0.002,0.001",
        help="Comma-separated RK4 dts compared against --ref-dt (must include ref-dt for baseline row).",
    )
    v_parser.add_argument(
        "--valid-thresholds",
        default="0.05,0.1,0.2,0.5",
        help="Comma-separated |error| thresholds for first exceedance time T_valid.",
    )
    v_parser.add_argument(
        "--no-extended-metrics",
        action="store_true",
        help="Skip segment MSE, T_valid, and attractor bundles (faster).",
    )
    v_parser.add_argument(
        "--n-sweep-grid",
        action="store_true",
        help="Generate thesis-style n-sweep 3D attractor grid",
    )
    v_parser.add_argument(
        "--sweep-n-values",
        default="7,7.75,8.5,8.79,9.65,9.696,9.7056,9.7451,10,20",
        help="Comma-separated n values for the sweep grid",
    )
    v_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="PINN training seed (Python / NumPy / PyTorch). Overrides training.random_seed in YAML.",
    )
    v_parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Save PNG only (skip PDF; faster on large multi-panel figures).",
    )
    v_parser.add_argument(
        "--only-3d-overlay",
        action="store_true",
        help="Load mglass_run.pkl and regenerate n*_3d_overlay.{png,pdf} only "
             "(no classical solve, no reference RK4, no PINN).",
    )
    v_parser.add_argument(
        "--only-loss-convergence",
        action="store_true",
        help="Load mglass_run.pkl and regenerate n*_loss_curves.*, LIC plot, "
             "prescribed-history IC-breakdown, ``n*_ic_terms_ns_three_panel.*``, "
             "and (when checkpoints contain them but the pickle lacks them) splice "
             "extended histories from ``checkpoints_n{N}/window_*.pt``.",
    )
    v_parser.add_argument(
        "--only-heatmap",
        action="store_true",
        help="Load mglass_run.pkl and regenerate n*_heatmap.{png,pdf} only "
             "(classical/reference strip, PINN strip, absolute-error strip).",
    )
    v_parser.add_argument(
        "--from-pkl",
        default=None,
        help="Pickle path for export-only modes: ``--only-3d-overlay``, "
             "``--only-heatmap``, ``--only-loss-convergence`` "
             "(default: OUTPUT_DIR/mglass_run.pkl).",
    )
    return v_parser.parse_args()


def _f_result_key_for_n(p_results: dict, p_n: float):
    """Resolve dict key for Hill exponent (float / numpy scalar mismatches)."""
    if p_n in p_results:
        return p_n
    for k in p_results:
        try:
            if abs(float(k) - float(p_n)) < 1e-9:
                return k
        except (TypeError, ValueError):
            continue
    return None


def main():
    global _MGLASS_SAVE_PDF
    v_args = f_parse_args()
    _MGLASS_SAVE_PDF = not v_args.no_pdf

    v_args.output_dir = f_resolve_bundle_output_dir(v_args.output_dir)

    if v_args.only_3d_overlay:
        v_pkl, v_export_dir = f_resolve_mglass_run_pkl_for_export(
            v_args.output_dir, v_args.from_pkl, "--only-3d-overlay",
        )
        os.makedirs(v_export_dir, exist_ok=True)
        with open(v_pkl, "rb") as fh:
            d_all_results = pickle.load(fh)
        l_export_n = [float(x.strip()) for x in v_args.n_values.split(",")]
        print("=" * 60)
        print("ONLY 3D OVERLAY: loading", v_pkl)
        print("  output:", v_export_dir)
        print("  n values:", l_export_n)
        print("=" * 60)
        for v_n in l_export_n:
            v_key = _f_result_key_for_n(d_all_results, v_n)
            if v_key is None:
                print(f"  [skip] n={v_n:g} not in pickle (keys: {list(d_all_results.keys())})")
                continue
            d_r = d_all_results[v_key]
            v_xp = d_r.get("x_pinn")
            if v_xp is None or (hasattr(v_xp, "size") and v_xp.size == 0):
                print(f"  [skip] n={v_n:g}: no PINN trajectory in pickle")
                continue
            v_tau = float(d_r.get("tau", np.nan))
            if not np.isfinite(v_tau):
                print(f"  [skip] n={v_n:g}: missing tau in pickle")
                continue
            v_out = os.path.join(v_export_dir, f"n{v_n:g}_3d_overlay.png")
            f_export_3d_overlay(
                d_r["t_ref"], d_r["x_ref"],
                d_r["t_classical"], d_r["x_classical"],
                d_r["t_pinn_test"], v_xp,
                v_tau, v_n,
                v_out,
            )
        print(f"\nDone. Overlay(s) written under {v_export_dir}/")
        return

    if v_args.only_heatmap:
        v_pkl, v_export_dir = f_resolve_mglass_run_pkl_for_export(
            v_args.output_dir, v_args.from_pkl, "--only-heatmap",
        )
        os.makedirs(v_export_dir, exist_ok=True)
        with open(v_pkl, "rb") as fh:
            d_all_results = pickle.load(fh)
        l_export_n = [float(x.strip()) for x in v_args.n_values.split(",")]
        print("=" * 60)
        print("ONLY HEATMAP: loading", v_pkl)
        print("  output:", v_export_dir)
        print("  n values:", l_export_n)
        print("=" * 60)
        for v_n in l_export_n:
            v_key = _f_result_key_for_n(d_all_results, v_n)
            if v_key is None:
                print(f"  [skip] n={v_n:g} not in pickle (keys: {list(d_all_results.keys())})")
                continue
            d_r = d_all_results[v_key]
            xp = d_r.get("x_pinn")
            if xp is None or (hasattr(xp, "size") and xp.size == 0):
                print(f"  [skip] n={v_n:g}: no PINN trajectory in pickle")
                continue
            t_hi = float(np.nanmax(np.asarray(d_r["t_ref"]).ravel()))
            if v_args.snapshot_times:
                l_snaps = [float(x.strip()) for x in v_args.snapshot_times.split(",")]
            else:
                l_snaps = [t_hi * 0.1, t_hi * 0.4, t_hi * 0.8]
            v_ht = os.path.join(v_export_dir, f"n{v_n:g}_heatmap.png")
            f_export_heatmap(
                d_r["t_ref"], d_r["x_ref"],
                d_r["t_pinn_test"], xp,
                l_snaps, v_n, v_ht,
                p_t_classical=d_r.get("t_classical"),
                p_x_classical=d_r.get("x_classical"),
            )
        print(f"\nDone. Heatmap(s) written under {v_export_dir}/")
        return

    if v_args.only_loss_convergence:
        v_pkl, v_export_dir = f_resolve_mglass_run_pkl_for_export(
            v_args.output_dir, v_args.from_pkl, "--only-loss-convergence",
        )
        os.makedirs(v_export_dir, exist_ok=True)
        with open(v_pkl, "rb") as fh:
            d_all_results = pickle.load(fh)
        l_export_n = [float(x.strip()) for x in v_args.n_values.split(",")]
        print("=" * 60)
        print("ONLY LOSS CONVERGENCE: loading", v_pkl)
        print("  output:", v_export_dir)
        print("  n values:", l_export_n)
        print("=" * 60)
        for v_n in l_export_n:
            v_key = _f_result_key_for_n(d_all_results, v_n)
            if v_key is None:
                print(f"  [skip] n={v_n:g} not in pickle (keys: {list(d_all_results.keys())})")
                continue
            d_r = d_all_results[v_key]
            v_lh = d_r.get("loss_history") or []
            if len(v_lh) == 0:
                print(f"  [skip] n={v_n:g}: empty loss_history in pickle")
                continue
            n_it = len(v_lh)
            d_plot = dict(d_r)
            v_merge = f_try_merge_extended_histories_from_checkpoints(
                v_export_dir, v_n, n_it,
            )
            for fk, fv in v_merge.items():
                if not _f_loss_trace_usable(d_plot.get(fk), n_it):
                    d_plot[fk] = fv
            if v_merge:
                print(
                    f"  [info] n={v_n:g}: filled from checkpoints_n{v_n:g}/: "
                    f"{', '.join(sorted(v_merge.keys()))}",
                )
            v_quiet_miss = (
                len(v_merge) == 0
                and f_shard0_extended_hist_absent(v_export_dir, v_n)
            )
            v_out = os.path.join(v_export_dir, f"n{v_n:g}_loss_curves.png")
            f_export_loss_curves(
                v_lh,
                d_plot.get("data_loss_history"),
                d_plot.get("physics_loss_history"),
                v_out,
                p_hist_loss=d_plot.get("history_loss_history"),
                p_ic_loss=d_plot.get("ic_loss_history"),
                p_loss_segment_lengths=d_plot.get("loss_segment_lengths"),
                p_known_missing_extended_hist_terms=v_quiet_miss,
            )
            f_export_mglass_prescribed_ic_breakdown_ns(
                n_it,
                p_hist_far=d_plot.get("hist_loss_far_history"),
                p_hist_mid=d_plot.get("hist_loss_mid_history"),
                p_hist_near0=d_plot.get("hist_loss_near0_history"),
                p_hist_agg=d_plot.get("history_loss_history"),
                p_ic=d_plot.get("ic_loss_history"),
                p_output_path=os.path.join(
                    v_export_dir, f"n{v_n:g}_prescribed_ic_hist_breakdown.png",
                ),
                p_loss_segment_lengths=d_plot.get("loss_segment_lengths"),
            )
            f_export_mglass_ic_terms_ns_three_panel(
                n_it,
                p_hist_far=d_plot.get("hist_loss_far_history"),
                p_hist_mid=d_plot.get("hist_loss_mid_history"),
                p_hist_near0=d_plot.get("hist_loss_near0_history"),
                p_hist_agg=d_plot.get("history_loss_history"),
                p_ic=d_plot.get("ic_loss_history"),
                p_training_plot_bundle=d_plot.get("training_plot_bundle"),
                p_output_path=os.path.join(
                    v_export_dir, f"n{v_n:g}_ic_terms_ns_three_panel.png",
                ),
                p_loss_segment_lengths=d_plot.get("loss_segment_lengths"),
            )
            v_ic = d_plot.get("ic_loss_history") or []
            if len(v_ic) > 0 and len(v_ic) == n_it:
                v_ic_out = os.path.join(
                    v_export_dir, f"n{v_n:g}_lic_loss_convergence.png",
                )
                f_export_junction_ic_loss_convergence(
                    v_ic,
                    v_ic_out,
                    p_loss_segment_lengths=d_plot.get("loss_segment_lengths"),
                    p_aligned_len=n_it,
                )
            v_miss_hist = not _f_loss_trace_usable(
                d_plot.get("history_loss_history"), n_it,
            )
            v_miss_ic = not _f_loss_trace_usable(
                d_plot.get("ic_loss_history"), n_it,
            )
            if v_miss_hist or v_miss_ic:
                v_expl = f_describe_extended_history_gap(v_export_dir, v_n)
                if v_expl:
                    v_which = []
                    if v_miss_hist:
                        v_which.append("`history_loss_history`")
                    if v_miss_ic:
                        v_which.append("`ic_loss_history`")
                    print(
                        f"  [info] n={v_n:g}: missing or misaligned "
                        f"{', '.join(v_which)}. {v_expl}",
                    )
        print(f"\nDone. Loss curve(s) written under {v_export_dir}/")
        return

    v_args.config = f_resolve_bundle_config_path(v_args.config)

    l_n_values = [float(x.strip()) for x in v_args.n_values.split(",")]
    os.makedirs(v_args.output_dir, exist_ok=True)

    d_config = f_load_config(v_args.config)
    if v_args.ignore_pinn_checkpoints:
        d_config["_ignore_pinn_checkpoints"] = True
        print(
            "[config] --ignore-pinn-checkpoints: PINN windows will not resume "
            "from checkpoints_n*/window_*.pt on disk.",
        )
    if v_args.seed is not None:
        d_config["_seed"] = int(v_args.seed)
    if v_args.dt is not None:
        d_config.setdefault("data", {})["dt"] = float(v_args.dt)

    v_beta = float(d_config["problem"]["beta_true"])
    v_gamma = float(d_config["problem"]["gamma_true"])
    v_tau = float(d_config["problem"]["tau"])
    v_x0 = float(d_config["problem"].get("initial_x_history", 1.2))
    v_t_end = v_args.t_end or float(d_config["data"]["t_total"])
    v_dt_cfg = v_args.dt if v_args.dt is not None else float(d_config["data"]["dt"])
    if v_args.classical_dt is not None:
        v_dt_fine = float(v_args.classical_dt)
    else:
        v_dt_fine = min(v_dt_cfg, 0.01)
    v_dt_ref = float(v_args.ref_dt) if v_args.ref_dt is not None else 0.001
    l_valid_thr = [
        float(x.strip())
        for x in v_args.valid_thresholds.split(",")
        if x.strip()
    ]
    l_ref_conv_dts = [
        float(x.strip())
        for x in v_args.reference_convergence_dts.split(",")
        if x.strip()
    ]
    if v_args.reference_convergence:
        l_ref_conv_dts = sorted(set(l_ref_conv_dts + [v_dt_ref]), reverse=True)

    if v_args.snapshot_times:
        l_snapshot_times = [float(x.strip()) for x in v_args.snapshot_times.split(",")]
    else:
        l_snapshot_times = [
            v_t_end * 0.1,
            v_t_end * 0.4,
            v_t_end * 0.8,
        ]

    d_all_results = {}
    l_ref_convergence_rows: List[Dict[str, Any]] = []
    v_runtime_env = f_capture_runtime_env()
    t_plot_total = 0.0

    print("=" * 80)
    print("MACKEY-GLASS COMPARISON: Classical DDE Solver vs PINN (Mackey-Glass)")
    print("=" * 80)
    print(f"  beta={v_beta}, gamma={v_gamma}, tau={v_tau}, x0={v_x0}")
    print(f"  t_end={v_t_end}, classical_dt (MoS)={v_dt_fine}, ref_dt (RK4)={v_dt_ref}, config data.dt={v_dt_cfg}")
    print(f"  n values: {l_n_values}")
    print(f"  Snapshot times: {l_snapshot_times}")
    print(f"  Output: {v_args.output_dir}")
    print("=" * 80)

    for v_n in l_n_values:
        print(f"\n{'─' * 60}")
        print(f"  Hill exponent n = {v_n}")
        print(f"{'─' * 60}")

        if v_args.reference_convergence and l_ref_conv_dts:
            l_ref_convergence_rows.extend(
                f_reference_rk4_convergence_rows(
                    v_beta, v_gamma, v_n, v_tau, v_x0, v_t_end,
                    v_dt_ref,
                    sorted(set(l_ref_conv_dts), reverse=True),
                )
            )

        # ── Classical solver ──
        print(f"  [Classical] Running method-of-steps RK45 solver (max_step={v_dt_fine})...")
        t0 = time.perf_counter()
        v_t_classical, v_x_classical = f_solve_mackey_glass_classical(
            p_beta=v_beta, p_gamma=v_gamma, p_n=v_n, p_tau=v_tau,
            p_x0=v_x0, p_t_end=v_t_end, p_dt=v_dt_fine,
        )
        v_wall_classical = float(time.perf_counter() - t0)
        print(f"  [Classical] Done in {v_wall_classical:.2f}s, {len(v_t_classical)} points")

        # High-accuracy reference (fixed-step RK4)
        print(f"  [Reference] Generating fixed-step RK4 reference (dt={v_dt_ref})...")
        t0 = time.perf_counter()
        v_t_ref, v_x_ref = f_solve_mackey_glass_rk4_fixed(
            p_beta=v_beta, p_gamma=v_gamma, p_n=v_n, p_tau=v_tau,
            p_x0=v_x0, p_t_end=v_t_end, p_dt=v_dt_ref,
        )
        v_wall_ref_rk4 = float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        v_interp_classical = interp1d(
            v_t_classical, v_x_classical, kind="cubic", fill_value="extrapolate",
        )
        v_x_classical_at_ref = v_interp_classical(v_t_ref)
        v_wall_interp_cl = float(time.perf_counter() - t0)

        d_metrics_classical = f_compute_metrics(v_x_ref, v_x_classical_at_ref)
        print(f"  [Classical] MSE={d_metrics_classical['mse']:.2e}, "
              f"rel $\\ell^2_N$={d_metrics_classical['rel_l2']:.6f}")

        # ── PINN ──
        d_pinn_result = None
        d_metrics_pinn = {"mse": np.nan, "rel_l2": np.nan, "max_abs_err": np.nan}
        d_params_pinn = {}
        v_wall_pinn_train = 0.0
        v_wall_pinn_infer = 0.0

        if not v_args.skip_pinn:
            print(f"  [PINN] Training PINN (n={v_n})...")
            d_config["_output_dir"] = v_args.output_dir
            try:
                d_pinn_result = f_train_pinn_mackey_glass(
                    d_config,
                    v_n,
                    p_metric_t_ref=v_t_ref,
                    p_junction_t_ref=v_t_ref,
                    p_junction_x_ref=v_x_ref,
                )
                v_wall_pinn_train = float(d_pinn_result.get("wall_time_train", d_pinn_result["wall_time"]))
                v_wall_pinn_infer = float(d_pinn_result.get("wall_time_infer", 0.0))

                v_x_pinn_fine = d_pinn_result.get("u_pred_metric_ref")
                if v_x_pinn_fine is not None:
                    d_metrics_pinn = f_compute_metrics(
                        v_x_ref, v_x_pinn_fine[:, 0]
                    )
                else:
                    v_interp_ref_at_pinn = interp1d(
                        v_t_ref, v_x_ref, kind="cubic", fill_value="extrapolate",
                    )
                    v_x_ref_at_pinn_test = v_interp_ref_at_pinn(
                        d_pinn_result["t_test"].flatten()
                    )
                    d_metrics_pinn = f_compute_metrics(
                        v_x_ref_at_pinn_test, d_pinn_result["u_pred"][:, 0]
                    )
                d_params_pinn = d_pinn_result["params"]

                print(
                    f"  [PINN] MSE={d_metrics_pinn['mse']:.2e}, "
                    f"rel $\\ell^2_N$={d_metrics_pinn['rel_l2']:.6f}, "
                    f"train={v_wall_pinn_train:.1f}s, infer(stitch)={v_wall_pinn_infer:.3f}s",
                )
                print(f"  [PINN] Identified params: {d_params_pinn}")
            except Exception as e:
                print(f"  [PINN] Training failed: {e}")
                import traceback
                traceback.print_exc()

        d_params_true = {"beta": v_beta, "gamma": v_gamma, "n": v_n, "tau": v_tau}

        v_seg_edges = [0.0]
        while v_seg_edges[-1] + 50.0 < v_t_end - 1e-9:
            v_seg_edges.append(v_seg_edges[-1] + 50.0)
        v_seg_edges.append(float(v_t_end))
        d_timing = {
            "ref_rk4_gen_s": v_wall_ref_rk4,
            "classical_solve_s": v_wall_classical,
            "classical_interp_to_ref_s": v_wall_interp_cl,
            "pinn_train_s": v_wall_pinn_train,
            "pinn_infer_stitch_s": v_wall_pinn_infer,
        }

        d_valid_cl: Dict[str, float] = {}
        d_valid_pn: Dict[str, float] = {}
        d_seg_cl: Dict[str, float] = {}
        d_seg_pn: Dict[str, float] = {}
        d_attr_cl: Dict[str, Any] = {}
        d_attr_pn: Dict[str, Any] = {}

        if not v_args.no_extended_metrics:
            d_valid_cl = f_first_exceedance_times(
                v_t_ref, v_x_ref, v_x_classical_at_ref, l_valid_thr,
            )
            d_seg_cl = f_segment_mse_table(v_t_ref, v_x_ref, v_x_classical_at_ref, v_seg_edges)
            d_attr_cl = f_attractor_metrics_bundle(
                v_t_ref, v_x_ref, v_x_classical_at_ref, v_tau, v_dt_ref,
            )
            v_x_pinn_on_ref = None
            if d_pinn_result and d_pinn_result.get("u_pred_metric_ref") is not None:
                v_x_pinn_on_ref = np.asarray(d_pinn_result["u_pred_metric_ref"])[:, 0]
            elif d_pinn_result:
                v_x_pinn_on_ref = np.full_like(v_x_ref, np.nan)
            if v_x_pinn_on_ref is not None and np.any(np.isfinite(v_x_pinn_on_ref)):
                d_valid_pn = f_first_exceedance_times(
                    v_t_ref, v_x_ref, v_x_pinn_on_ref, l_valid_thr,
                )
                d_seg_pn = f_segment_mse_table(v_t_ref, v_x_ref, v_x_pinn_on_ref, v_seg_edges)
                d_attr_pn = f_attractor_metrics_bundle(
                    v_t_ref, v_x_ref, v_x_pinn_on_ref, v_tau, v_dt_ref,
                )

        d_all_results[v_n] = {
            "t_ref": v_t_ref,
            "x_ref": v_x_ref,
            "ref_dt": v_dt_ref,
            "classical_dt": v_dt_fine,
            "t_classical": v_t_classical,
            "x_classical": v_x_classical,
            "metrics_classical": d_metrics_classical,
            "wall_time_classical": v_wall_classical,
            "t_pinn_test": d_pinn_result["t_test"] if d_pinn_result else np.array([]),
            "x_pinn": d_pinn_result["u_pred"] if d_pinn_result else np.array([]),
            "t_train": d_pinn_result["t_train"] if d_pinn_result else np.array([]),
            "x_train": d_pinn_result["u_train"] if d_pinn_result else np.array([]),
            "metrics_pinn": d_metrics_pinn,
            "params_pinn": d_params_pinn,
            "params_true": d_params_true,
            "wall_time_pinn": v_wall_pinn_train,
            "wall_time_pinn_train": v_wall_pinn_train,
            "wall_time_pinn_infer": v_wall_pinn_infer,
            "tau": v_tau,
            "loss_history": d_pinn_result["loss_history"] if d_pinn_result else [],
            "data_loss_history": d_pinn_result["data_loss_history"] if d_pinn_result else [],
            "physics_loss_history": d_pinn_result["physics_loss_history"] if d_pinn_result else [],
            "history_loss_history": d_pinn_result.get("history_loss_history", []) if d_pinn_result else [],
            "ic_loss_history": d_pinn_result.get("ic_loss_history", []) if d_pinn_result else [],
            "hist_loss_far_history": d_pinn_result.get("hist_loss_far_history", []) if d_pinn_result else [],
            "hist_loss_mid_history": d_pinn_result.get("hist_loss_mid_history", []) if d_pinn_result else [],
            "hist_loss_near0_history": d_pinn_result.get("hist_loss_near0_history", []) if d_pinn_result else [],
            "loss_segment_lengths": d_pinn_result.get("loss_segment_lengths", []) if d_pinn_result else [],
            "training_plot_bundle": (
                d_pinn_result.get("training_plot_bundle") if d_pinn_result else {}
            ),
            "timing": d_timing,
            "runtime_env": v_runtime_env,
            "valid_prediction_time_classical": d_valid_cl,
            "valid_prediction_time_pinn": d_valid_pn,
            "segment_mse_classical": d_seg_cl,
            "segment_mse_pinn": d_seg_pn,
            "attractor_metrics_classical": d_attr_cl,
            "attractor_metrics_pinn": d_attr_pn,
        }

        # Per-n figure
        if d_pinn_result is not None:
            v_fig_path = os.path.join(v_args.output_dir, f"mackey_glass_n{v_n:g}_comparison.png")
            t_fig0 = time.perf_counter()
            f_create_mglass_figure(
                p_t_ref=v_t_ref,
                p_x_ref=v_x_ref,
                p_t_pinn_test=d_pinn_result["t_test"],
                p_x_pinn=d_pinn_result["u_pred"],
                p_t_train=d_pinn_result["t_train"],
                p_x_train=d_pinn_result["u_train"],
                p_params_true=d_params_true,
                p_params_pinn=d_params_pinn,
                p_metrics_pinn=d_metrics_pinn,
                p_metrics_classical=d_metrics_classical,
                p_t_classical=v_t_classical,
                p_x_classical=v_x_classical,
                p_n_value=v_n,
                p_tau=v_tau,
                p_snapshot_times=l_snapshot_times,
                p_snapshot_window=v_args.snapshot_window,
                p_output_path=v_fig_path,
                p_loss_history=d_pinn_result["loss_history"],
                p_data_loss_history=d_pinn_result["data_loss_history"],
                p_physics_loss_history=d_pinn_result["physics_loss_history"],
                p_hist_loss_history=d_pinn_result.get("history_loss_history"),
                p_ic_loss_history=d_pinn_result.get("ic_loss_history"),
                p_loss_segment_lengths=d_pinn_result.get("loss_segment_lengths"),
                p_hist_loss_far_history=d_pinn_result.get("hist_loss_far_history"),
                p_hist_loss_mid_history=d_pinn_result.get("hist_loss_mid_history"),
                p_hist_loss_near0_history=d_pinn_result.get("hist_loss_near0_history"),
                p_training_plot_bundle=d_pinn_result.get("training_plot_bundle"),
            )
            t_plot_total += time.perf_counter() - t_fig0

    if v_args.reference_convergence and l_ref_convergence_rows:
        v_csv_conv = os.path.join(v_args.output_dir, "reference_convergence.csv")
        with open(v_csv_conv, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["n", "dt", "mse_vs_finest", "rel_l2_vs_finest", "max_abs_vs_finest"],
            )
            w.writeheader()
            for row in l_ref_convergence_rows:
                w.writerow(row)
        print(f"\nWrote {v_csv_conv}")

    # Multi-n summary
    if len(d_all_results) > 1 and any(
        d_all_results[n].get("x_pinn") is not None and len(d_all_results[n]["x_pinn"]) > 0
        for n in d_all_results
    ):
        f_create_multi_n_summary(
            d_all_results,
            os.path.join(v_args.output_dir, "multi_n_summary.png"),
            p_snapshot_times=l_snapshot_times,
        )

    # Metrics table
    f_create_metrics_table_figure(
        d_all_results,
        os.path.join(v_args.output_dir, "metrics_comparison_table.png"),
    )

    # Save raw results
    v_pkl_path = os.path.join(v_args.output_dir, "mglass_run.pkl")
    with open(v_pkl_path, "wb") as fh:
        pickle.dump(d_all_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nRaw results saved to: {v_pkl_path}")

    v_manifest = {
        "output_dir": v_args.output_dir,
        "classical_dt": v_dt_fine,
        "ref_dt": v_dt_ref,
        "config_data_dt": v_dt_cfg,
        "plotting_s": t_plot_total,
        "runtime_env": v_runtime_env,
        "argv": sys.argv,
    }
    with open(os.path.join(v_args.output_dir, "run_manifest.json"), "w") as fh:
        json.dump(v_manifest, fh, indent=2)
    print(f"Wrote {os.path.join(v_args.output_dir, 'run_manifest.json')}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'n':>8} | {'Cl MSE':>14} | {'PINN MSE':>14} | {'Cl s':>8} | {'train s':>10} | {'infer s':>9}")
    print("-" * 80)
    for v_n in sorted(d_all_results.keys()):
        d_r = d_all_results[v_n]
        print(
            f"{v_n:8g} | {d_r['metrics_classical']['mse']:14.2e} | "
            f"{d_r['metrics_pinn']['mse']:14.2e} | "
            f"{d_r['wall_time_classical']:8.2f} | "
            f"{d_r['wall_time_pinn']:10.1f} | "
            f"{d_r.get('wall_time_pinn_infer', 0.0):9.4f}"
        )
    print("=" * 80)
    print(f"\nAll outputs in: {v_args.output_dir}/")

    # Optional: thesis-style n-sweep 3D grid
    if v_args.n_sweep_grid:
        print("\n" + "=" * 80)
        print("Generating n-sweep 3D attractor grid...")
        print("=" * 80)
        l_sweep_n = [float(x.strip()) for x in v_args.sweep_n_values.split(",")]
        f_generate_n_sweep_3d_grid(
            p_n_values=l_sweep_n,
            p_beta=v_beta, p_gamma=v_gamma, p_tau=v_tau, p_x0=v_x0,
            p_t_end=v_t_end, p_dt=v_dt_fine,
            p_output_dir=v_args.output_dir,
            p_pinn_checkpoint_dir=v_args.output_dir,
            p_pinn_config=d_config,
        )


def f_generate_n_sweep_3d_grid(
    p_n_values: List[float],
    p_beta: float = 2.0,
    p_gamma: float = 1.0,
    p_tau: float = 2.0,
    p_x0: float = 1.2,
    p_t_end: float = 100.0,
    p_dt: float = 0.01,
    p_output_dir: str = "mglass_results",
    p_pinn_checkpoint_dir: Optional[str] = None,
    p_pinn_config: Optional[Dict[str, Any]] = None,
):
    """
    Generate a thesis-style grid of 3D delay-embedding attractors for varied n.

    Row 0: Classical DDE solver (blue) for each n
    Row 1: Classical (blue) + PINN overlay (red) where checkpoint exists

    If no PINN checkpoint is available for a given n, the bottom row shows
    only the Classical solution.
    """
    os.makedirs(p_output_dir, exist_ok=True)
    n_cols = min(5, len(p_n_values))
    n_rows_per = int(np.ceil(len(p_n_values) / n_cols))

    # Thesis-matching plot settings (from config visualization.plot_settings)
    _FIG_W, _FIG_H = 25, 12
    _DPI = 300
    _HSPACE, _WSPACE = 0.4, 0.4
    _ELEV, _AZIM = 30, 45
    _DIST = 11
    _LW_CL = 1.5
    _LW_PINN = 1.2
    _ALPHA_CL = 0.9
    _ALPHA_PINN = 0.8
    _FONT_TITLE = 11
    _FONT_LABEL = 9
    _FONT_TICK = 7
    _FONT_SUPTITLE = 20
    _FONT_LEGEND = 7
    _COLOR_CL = "k"          # black, matching thesis "True Solution"
    _COLOR_PINN = "#780078"   # thesis magenta for PINN predictions
    _COLOR_SHADOW = "gray"

    # --- Figure 1: Classical DDE solver attractor grid (time-gradient) ---
    fig_cl = plt.figure(figsize=(_FIG_W, _FIG_H))
    fig_cl.suptitle(
        "Mackey-Glass 3D Delay Embedding Attractor for varied n\n"
        f"Classical DDE solver ($\\beta={p_beta}$, $\\gamma={p_gamma}$, $\\tau={p_tau}$)",
        fontsize=_FONT_SUPTITLE, fontweight="bold",
    )

    # --- Figure 2: Comparison grid ---
    fig_ov = plt.figure(figsize=(_FIG_W, _FIG_H))
    fig_ov.suptitle(
        "Mackey-Glass 3D Delay Embedding Attractor for varied n\n"
        f"Classical DDE solver vs PINN ($\\beta={p_beta}$, $\\gamma={p_gamma}$, $\\tau={p_tau}$)",
        fontsize=_FONT_SUPTITLE, fontweight="bold",
    )

    device = None
    for idx, n_val in enumerate(p_n_values):

        print(f"  [n={n_val:g}] Solving classical DDE...")
        t_cl, x_cl = f_solve_mackey_glass_classical(
            p_beta=p_beta, p_gamma=p_gamma, p_n=n_val, p_tau=p_tau,
            p_x0=p_x0, p_t_end=p_t_end, p_dt=p_dt,
        )

        ds = _delay_indices(t_cl, p_tau)
        if len(x_cl) <= 2 * ds:
            continue
        xt_cl = x_cl[2 * ds:]
        xt1_cl = x_cl[ds: -ds]
        xt2_cl = x_cl[: -2 * ds]
        z_floor = xt2_cl.min() - 0.05

        # --- Classical grid (time-gradient colouring) ---
        ax_cl = fig_cl.add_subplot(n_rows_per, n_cols, idx + 1, projection="3d")
        lc = _color_segments_3d(xt_cl, xt1_cl, xt2_cl, cmap_name="viridis",
                                linewidth=_LW_CL * 0.5, alpha=_ALPHA_CL)
        ax_cl.add_collection3d(lc)
        ax_cl.auto_scale_xyz(xt_cl, xt1_cl, xt2_cl)
        ax_cl.plot(xt_cl, xt1_cl, np.full_like(xt_cl, z_floor),
                   color=_COLOR_SHADOW, linewidth=1.0, alpha=0.25)
        ax_cl.set_xlabel("$x(t)$", fontsize=_FONT_LABEL)
        ax_cl.set_ylabel("$x(t{-}\\tau)$", fontsize=_FONT_LABEL)
        ax_cl.set_zlabel("$x(t{-}2\\tau)$", fontsize=_FONT_LABEL)
        ax_cl.set_title(f"$n={n_val:g}$", fontsize=_FONT_TITLE, fontweight="bold")
        ax_cl.tick_params(axis="both", labelsize=_FONT_TICK)
        ax_cl.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_cl.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_cl.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_cl.grid(False)
        ax_cl.view_init(elev=_ELEV, azim=_AZIM)
        ax_cl.dist = _DIST

        # --- Overlay grid ---
        ax_ov = fig_ov.add_subplot(n_rows_per, n_cols, idx + 1, projection="3d")
        ax_ov.plot(xt_cl, xt1_cl, xt2_cl, color=_COLOR_CL,
                   linewidth=_LW_CL, alpha=_ALPHA_CL, label="Classical DDE solver",
                   zorder=5)
        ax_ov.plot(xt_cl, xt1_cl, np.full_like(xt_cl, z_floor),
                   color=_COLOR_SHADOW, linewidth=1.0, alpha=0.25, zorder=1)

        # Try loading PINN checkpoint
        pinn_loaded = False
        if p_pinn_checkpoint_dir is not None:
            ckpt_path = os.path.join(p_pinn_checkpoint_dir,
                                     f"checkpoints_n{n_val:g}", "window_0.pt")
            if os.path.exists(ckpt_path) and p_pinn_config is not None:
                if device is None:
                    device = f_select_device()
                v_layers_cfg = p_pinn_config["network"]["layers"]
                v_hidden_size = v_layers_cfg[1] if len(v_layers_cfg) > 2 else 256
                v_hidden_layers = len(v_layers_cfg) - 2
                model = MackeyGlassPINN(
                    hidden_layers=v_hidden_layers,
                    hidden_size=v_hidden_size,
                    activation=p_pinn_config["network"].get("activation", "sine"),
                    siren_init=p_pinn_config["network"].get("initialization", "siren") == "siren",
                    fourier_features=bool(p_pinn_config["network"].get("fourier_features", True)),
                    fourier_dim=int(p_pinn_config["network"].get("fourier_dim", 64)),
                    fourier_scale=float(p_pinn_config["network"].get("fourier_scale", 5.0)),
                ).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                model.eval()

                t_test = np.linspace(0.0, p_t_end, 5000)
                with torch.no_grad():
                    t_t = torch.tensor(t_test, dtype=torch.float32,
                                       device=device).unsqueeze(1)
                    x_pinn = model(t_t).cpu().numpy().flatten()

                ds_p = _delay_indices(t_test, p_tau)
                if len(x_pinn) > 2 * ds_p:
                    xt_p = x_pinn[2 * ds_p:]
                    xt1_p = x_pinn[ds_p: -ds_p]
                    xt2_p = x_pinn[: -2 * ds_p]
                    ax_ov.plot(xt_p, xt1_p, xt2_p, color=_COLOR_PINN,
                               linewidth=_LW_PINN, alpha=_ALPHA_PINN,
                               linestyle="--", label="PINN", zorder=10)
                    pinn_loaded = True
                    print(f"  [n={n_val:g}] PINN checkpoint loaded")

        ax_ov.set_xlabel("$x(t)$", fontsize=_FONT_LABEL)
        ax_ov.set_ylabel("$x(t{-}\\tau)$", fontsize=_FONT_LABEL)
        ax_ov.set_zlabel("$x(t{-}2\\tau)$", fontsize=_FONT_LABEL)
        ax_ov.set_title(f"$n={n_val:g}$", fontsize=_FONT_TITLE, fontweight="bold")
        ax_ov.tick_params(axis="both", labelsize=_FONT_TICK)
        ax_ov.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_ov.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_ov.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_ov.grid(False)
        ax_ov.view_init(elev=_ELEV, azim=_AZIM)
        ax_ov.dist = _DIST
        ax_ov.legend(fontsize=_FONT_LEGEND, loc="upper left", framealpha=0.7)

    fig_cl.subplots_adjust(hspace=_HSPACE, wspace=_WSPACE)
    _save(fig_cl, os.path.join(p_output_dir, "n_sweep_3d_classical_grid.png"))

    fig_ov.subplots_adjust(hspace=_HSPACE, wspace=_WSPACE)
    _save(fig_ov, os.path.join(p_output_dir, "n_sweep_3d_comparison_grid.png"))


if __name__ == "__main__":
    main()
