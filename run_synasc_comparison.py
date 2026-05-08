#!/usr/bin/env python3
"""
SYNASC Paper: Classical DDE Solver vs PINN for Mackey-Glass Equation.

PyTorch PINN with ROCm GPU acceleration (AMD RX 6700S / gfx1032).

Produces KdV-style comparison figures:
  - Solution heatmap u(t) with marked training snapshot locations
  - Training data overlays at selected time windows
  - Quantitative comparison table (MSE, relative L2, wall time)
  - Delay-coordinate embedding visualizations

Usage (from the ``synasc/`` directory, or with absolute paths):

    python run_synasc_comparison.py \\
        [--config configs/config_mackey_glass_synasc_t100_windowed.yaml] \\
        [--n-values 10] [--output-dir results/my_run]

YAML recipes live in ``synasc/configs/``. By default, figures and ``synasc_results.pkl``
are written under ``synasc/results/`` (see ``--output-dir``).

    The ``config_mackey_glass_synasc.yaml`` file keeps a shorter ``[0,100]``
    single-window setup; ``config_mackey_glass_synasc_t100_windowed.yaml`` is
    the windowed recipe on ``[0,100]`` (horizon matches ``data.t_total`` /
    ``problem.time_span``) with ``L=25``, ``O=5``.
    Window construction is in ``f_build_time_windows``.
"""

import os
import random

# ROCm override MUST be set before importing torch
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import sys
import math
import time
import copy
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Root of the SYNASC reproducibility bundle (this directory).
SYNASC_ROOT = Path(__file__).resolve().parent

# If False (see ``--no-pdf``), skip PDF export — ``bbox_inches="tight"`` PDFs are slow.
_SYNASC_SAVE_PDF = True


def f_resolve_synasc_config_path(raw: str) -> str:
    """Resolve YAML path: CWD, then ``synasc/`` bundle (``SYNASC_ROOT``)."""
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())
    alt = SYNASC_ROOT / raw
    if alt.is_file():
        return str(alt.resolve())
    return str(p.resolve())


def f_resolve_synasc_output_dir(raw: str) -> str:
    """Place relative output paths under ``synasc/`` unless absolute."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((SYNASC_ROOT / p).resolve())


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
) -> torch.Tensor:
    """Evaluate x(t - tau).

    For time-domain decomposition: if t-tau falls before the current window
    start (win_t0), use prev_model if available, else history constant.
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
        result[idx_cur, :] = model(t_delayed[idx_cur, :].detach())

    return result


def f_compute_residual_abs(
    model, t_pts, beta, gamma, n_hill, tau, x0,
    prev_model=None, win_t0=0.0,
):
    """Compute |dx/dt - rhs| for adaptive residual sampling."""
    t_eval = t_pts.detach().clone().requires_grad_(True)
    x_eval = model(t_eval)
    dx_dt = torch.autograd.grad(
        x_eval, t_eval,
        grad_outputs=torch.ones_like(x_eval),
        create_graph=False, retain_graph=False,
    )[0]
    x_tau = f_eval_delayed(model, t_eval.detach(), tau, x0, prev_model, win_t0)
    rhs = beta * x_tau / (1.0 + torch.abs(x_tau) ** n_hill) - gamma * x_eval
    return torch.abs(dx_dt - rhs).detach().squeeze(1)


def f_sample_collocation(
    model, device, n_ode, t_lo, t_hi,
    beta, gamma, n_hill, tau, x0,
    adaptive, pool_mult, top_frac,
    prev_model=None, win_t0=0.0,
):
    """Sample collocation points, optionally with adaptive residual focus."""
    if not adaptive:
        return torch.rand(n_ode, 1, device=device) * (t_hi - t_lo) + t_lo

    pool_n = n_ode * pool_mult
    t_pool = torch.rand(pool_n, 1, device=device) * (t_hi - t_lo) + t_lo
    res = f_compute_residual_abs(model, t_pool, beta, gamma, n_hill, tau, x0,
                                  prev_model, win_t0)
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total PINN loss for Mackey-Glass DDE."""
    t_ode = t_ode.requires_grad_(True)
    x_pred = model(t_ode)
    dx_dt = torch.autograd.grad(
        x_pred, t_ode,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True, retain_graph=True,
    )[0]

    x_tau = f_eval_delayed(model, t_ode.detach(), tau, x0, prev_model, win_t0)
    rhs = beta * x_tau / (1.0 + torch.abs(x_tau) ** n_hill) - gamma * x_pred
    loss_phy = torch.mean((dx_dt - rhs) ** 2)

    x_data_pred = model(t_data)
    loss_data = torch.mean((x_data_pred - x_data) ** 2)

    loss_hist = torch.mean((model(t_hist) - x0) ** 2) if t_hist.numel() > 0 else \
        torch.tensor(0.0, device=t_ode.device)

    loss_ic = torch.tensor(0.0, device=t_ode.device)
    if ic_t is not None and ic_x is not None:
        loss_ic = torch.mean((model(ic_t) - ic_x) ** 2)

    total = (w_data * loss_data + w_phy * loss_phy
             + w_hist * loss_hist + w_ic * loss_ic)
    return total, loss_data, loss_phy, loss_hist


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
    ``p_win_overlap = 5`` (SYNASC default) this yields 10 windows:
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
    start = time.time()

    use_adaptive = False

    for ep in range(1, v_n_adam + 1):
        model.train()
        opt.zero_grad()

        if v_curriculum:
            frac = _f_curriculum_frac(ep, v_n_adam, start_frac=0.15, power=2.0)
            t_hi_cur = win_t0 + (win_t1 - win_t0) * frac
        else:
            t_hi_cur = win_t1

        if v_adaptive and ep > v_n_adam // 5 and (ep == 1 or ep % v_resample_every == 0):
            use_adaptive = True

        t_ode = f_sample_collocation(
            model, device, v_n_ode, win_t0, t_hi_cur,
            v_beta, v_gamma, v_n_hill, v_tau, v_x0,
            use_adaptive, pool_mult=3, top_frac=0.5,
            prev_model=prev_model, win_t0=win_t0,
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

        loss, ld, lp, lh = f_mackey_glass_loss(
            model, t_ode, t_hist, data_cur, xdata_cur,
            v_beta, v_gamma, v_n_hill, v_tau, v_x0,
            v_w_data, w_phy_eff, v_w_hist,
            prev_model, win_t0, v_w_ic, ic_t, ic_x,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        l_loss.append(float(loss.item()))
        l_data_l.append(float(ld.item()))
        l_phy_l.append(float(lp.item()))

        if ep % v_log_every == 0 or ep == 1:
            el = time.time() - start
            cur_str = f" cur={frac:.2f}" if v_curriculum else ""
            print(
                f"  [{window_label}] {ep:5d}/{v_n_adam} | Loss {loss.item():.3e} | "
                f"data {ld.item():.3e} | phys {lp.item():.3e} | "
                f"lh {lh.item():.3e}{cur_str} | {el:.1f}s"
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
        )
        _cache: Dict[str, float] = {}

        for s in range(1, v_n_lbfgs + 1):
            def closure():
                opt_lb.zero_grad()
                lo, dd, pp, hh = f_mackey_glass_loss(
                    model, t_ode_fix, t_hist, t_data, x_data,
                    v_beta, v_gamma, v_n_hill, v_tau, v_x0,
                    v_w_data, v_w_phy, v_w_hist,
                    prev_model, win_t0, v_w_ic, ic_t, ic_x,
                )
                lo.backward()
                _cache.update(loss=float(lo.item()), data=float(dd.item()),
                              phys=float(pp.item()))
                return lo

            opt_lb.step(closure)
            l_loss.append(_cache.get("loss", float("nan")))
            l_data_l.append(_cache.get("data", float("nan")))
            l_phy_l.append(_cache.get("phys", float("nan")))

            if s % 100 == 0 or s == 1:
                el = time.time() - start
                print(f"  [{window_label} LB] {s:4d}/{v_n_lbfgs} | "
                      f"Loss {_cache.get('loss',0):.3e} | {el:.1f}s")

    wall = time.time() - start
    print(f"  [{window_label}] Done in {wall:.1f}s")
    return l_loss, l_data_l, l_phy_l, wall


def f_train_pinn_mackey_glass(
    p_config: Dict[str, Any],
    p_n_value: float,
) -> Dict[str, Any]:
    """
    Train a PyTorch PINN on the Mackey-Glass DDE using time-domain decomposition.

    Features: windowed training, curriculum learning, adaptive residual sampling,
    Fourier features, 6x256 network, Adam + L-BFGS hybrid.
    """
    d_config = copy.deepcopy(p_config)

    if "_seed" in d_config:
        v_seed = int(d_config["_seed"])
    else:
        v_seed = int(d_config.get("training", {}).get("random_seed", 1234))
    f_apply_training_seed(v_seed)
    print(f"  [PINN] Random seed: {v_seed}")

    v_beta = float(d_config["problem"]["beta_true"])
    v_gamma = float(d_config["problem"]["gamma_true"])
    v_n_hill = float(p_n_value)
    v_tau = float(d_config["problem"]["tau"])
    v_x0 = float(d_config["problem"].get("initial_x_history", 1.2))
    v_t_end = float(d_config["data"]["t_total"])
    v_dt = float(d_config["data"]["dt"])

    v_t_ref_np, v_x_ref_np = f_solve_mackey_glass_rk4_fixed(
        p_beta=v_beta, p_gamma=v_gamma, p_n=v_n_hill, p_tau=v_tau,
        p_x0=v_x0, p_t_end=v_t_end, p_dt=v_dt,
    )
    v_n_train = int(d_config["data"]["training"]["n_points"])
    v_stride = max(1, len(v_t_ref_np) // v_n_train)
    v_t_train_np = v_t_ref_np[::v_stride]
    v_x_train_np = v_x_ref_np[::v_stride]

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
        d_config.get("_output_dir", "SYNASC_results"),
        f"checkpoints_n{p_n_value:g}",
    )
    os.makedirs(v_ckpt_dir, exist_ok=True)

    print(f"  [PINN] Time-domain decomposition: {len(windows)} windows")
    print(f"  [PINN] Checkpoints: {v_ckpt_dir}")
    for i, (a, b) in enumerate(windows):
        print(f"         Window {i}: [{a:.1f}, {b:.1f}]")

    v_start = time.time()
    all_loss, all_data_l, all_phy_l = [], [], []
    models: List[Tuple[float, float, MackeyGlassPINN]] = []
    prev_model = None

    ref_interp = interp1d(v_t_ref_np, v_x_ref_np, kind="cubic",
                          fill_value="extrapolate")

    for wi, (wt0, wt1) in enumerate(windows):
        label = f"W{wi} [{wt0:.0f}-{wt1:.0f}]"
        ckpt_path = os.path.join(v_ckpt_dir, f"window_{wi}.pt")

        # Resume from checkpoint if available
        if os.path.exists(ckpt_path):
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
            all_loss.extend(ckpt.get("loss_history", []))
            all_data_l.extend(ckpt.get("data_loss_history", []))
            all_phy_l.extend(ckpt.get("physics_loss_history", []))
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

        ic_t_val = wt0 if wi > 0 else None
        ic_x_val = float(ref_interp(wt0)) if wi > 0 else None

        wl, wdl, wpl, w_wall = _f_train_single_window(
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
            "wall_time": w_wall,
            "n_value": p_n_value,
        }, ckpt_path)
        print(f"  [{label}] Checkpoint saved: {ckpt_path}")

    v_wall_time = time.time() - v_start
    print(f"\n  [PINN] All windows complete in {v_wall_time:.1f}s")

    # Evaluate: stitch predictions from all windows
    v_n_test = int(d_config["optimizer_comparison"]["visualization"]["test_points"])
    v_t_test_np = np.linspace(0.0, v_t_end, v_n_test)
    v_x_pred_np = np.zeros(v_n_test)

    for wt0, wt1, mdl in models:
        mask = (v_t_test_np >= wt0) & (v_t_test_np <= wt1)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        with torch.no_grad():
            t_t = torch.tensor(v_t_test_np[idx], dtype=torch.float32,
                               device=device).unsqueeze(1)
            v_x_pred_np[idx] = mdl(t_t).cpu().numpy().flatten()

    # For overlapping regions, later windows take priority (already overwritten)

    return {
        "t_train": v_t_train_np.reshape(-1, 1),
        "u_train": v_x_train_np.reshape(-1, 1),
        "t_test": v_t_test_np.reshape(-1, 1),
        "u_pred": v_x_pred_np.reshape(-1, 1),
        "f_pred": np.zeros((v_n_test, 1)),
        "params": {"beta": v_beta, "gamma": v_gamma, "n": v_n_hill, "tau": v_tau},
        "loss_history": all_loss,
        "data_loss_history": all_data_l,
        "physics_loss_history": all_phy_l,
        "wall_time": v_wall_time,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def f_compute_metrics(
    p_x_ref: np.ndarray,
    p_x_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MSE and relative L2 error."""
    v_mse = float(np.mean((p_x_ref - p_x_pred) ** 2))
    v_rel_l2 = float(np.sqrt(np.sum((p_x_ref - p_x_pred) ** 2) / np.sum(p_x_ref ** 2)))
    v_max_err = float(np.max(np.abs(p_x_ref - p_x_pred)))
    return {"mse": v_mse, "rel_l2": v_rel_l2, "max_abs_err": v_max_err}


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
    if _SYNASC_SAVE_PDF:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3D Delay-Embedding Attractor Plots (publication-quality, IEEE 2-col sizing)
# ═══════════════════════════════════════════════════════════════════════════════

_IEEE_COL_W = 3.5   # single-column width in inches
_IEEE_2COL_W = 7.16  # full-width (2-column) in inches


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
    p_t_classical, p_x_classical,
    p_t_pinn_test, p_x_pinn,
    p_tau, p_n_value, p_output_path,
):
    """
    Overlay: Classical (blue) + PINN (red) on the same 3D axes.
    Single IEEE column width.
    """
    ds_cl = _delay_indices(p_t_classical, p_tau)
    ds_p = _delay_indices(p_t_pinn_test.flatten(), p_tau)

    fig = plt.figure(figsize=(_IEEE_COL_W, 3.2))
    ax = fig.add_subplot(111, projection="3d")

    if len(p_x_classical) > 2 * ds_cl:
        xt_cl = p_x_classical[2 * ds_cl:]
        xt1_cl = p_x_classical[ds_cl: -ds_cl]
        xt2_cl = p_x_classical[: -2 * ds_cl]
        ax.plot(xt_cl, xt1_cl, xt2_cl, color="#1f77b4",
                linewidth=0.5, alpha=0.7, label="Classical DDE solver")

    xp = p_x_pinn[:, 0]
    if len(xp) > 2 * ds_p:
        xt_p = xp[2 * ds_p:]
        xt1_p = xp[ds_p: -ds_p]
        xt2_p = xp[: -2 * ds_p]
        ax.plot(xt_p, xt1_p, xt2_p, color="#d62728",
                linewidth=0.5, alpha=0.7, linestyle="--", label="PINN")

    ax.set_xlabel("$x(t)$", fontsize=8, labelpad=2)
    ax.set_ylabel("$x(t{-}\\tau)$", fontsize=8, labelpad=2)
    ax.set_zlabel("$x(t{-}2\\tau)$", fontsize=8, labelpad=2)
    ax.set_title(f"Mackey-Glass ($n={p_n_value:g}$)", fontsize=9, pad=4)
    ax.tick_params(axis="both", labelsize=6, pad=1)
    ax.view_init(elev=25, azim=-55)
    ax.dist = 11
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout(pad=0.3)
    _save(fig, p_output_path)


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
    Classical DDE solver and PINN solutions are shown side-by-side.
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
    else:
        base_interp = interp1d(p_t_ref, p_x_ref, kind="cubic",
                               bounds_error=False, fill_value=np.nan)
        base_title = "Reference $x(t+\\delta)$"
    pinn_interp = interp1d(t_flat, x_flat, kind="cubic",
                           bounds_error=False, fill_value=np.nan)

    t_grid = np.linspace(t_flat.min(), t_flat.max(), 2000)
    img_base = np.zeros((n_rows, len(t_grid)))
    img_pinn = np.zeros_like(img_base)
    for i, s in enumerate(shifts):
        img_base[i, :] = base_interp(t_grid + s)
        img_pinn[i, :] = pinn_interp(t_grid + s)

    fig = plt.figure(figsize=(10, 7))

    gs = gridspec.GridSpec(1, 2, wspace=0.35, left=0.08, right=0.92,
                           top=0.88, bottom=0.12)

    extent = [t_grid.min(), t_grid.max(), shifts.min(), shifts.max()]
    vmin = np.nanmin(img_base)
    vmax = np.nanmax(img_base)

    ax0 = fig.add_subplot(gs[0, 0])
    h0 = ax0.imshow(img_base, interpolation="nearest", cmap="rainbow",
                     extent=extent, origin="lower", aspect="auto",
                     vmin=vmin, vmax=vmax)
    for ts in p_snapshot_times:
        ax0.plot([ts, ts], [shifts.min(), shifts.max()], "w-", linewidth=1.0)
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h0, cax=cax0)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("time shift $\\delta$")
    ax0.set_title(base_title, fontsize=11)

    ax1 = fig.add_subplot(gs[0, 1])
    h1 = ax1.imshow(img_pinn, interpolation="nearest", cmap="rainbow",
                     extent=extent, origin="lower", aspect="auto",
                     vmin=vmin, vmax=vmax)
    for ts in p_snapshot_times:
        ax1.plot([ts, ts], [shifts.min(), shifts.max()], "w-", linewidth=1.0)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h1, cax=cax1)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("time shift $\\delta$")
    ax1.set_title("PINN $\\hat{x}(t+\\delta)$", fontsize=11)

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
        ["Rel. $L^2$  (Classical / PINN)",
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


def f_export_loss_curves(p_loss, p_data_loss, p_phy_loss, p_output_path):
    """Training loss curves (log scale)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    iters = np.arange(len(p_loss))
    ax.semilogy(iters, p_loss, "k-", linewidth=1.0, label="Total")
    if p_data_loss and len(p_data_loss) == len(p_loss):
        ax.semilogy(iters, p_data_loss, "b-", linewidth=0.8, alpha=0.7, label="Data")
    if p_phy_loss and len(p_phy_loss) == len(p_loss):
        ax.semilogy(iters, p_phy_loss, "g-", linewidth=0.8, alpha=0.7, label="Physics")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
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

def f_create_synasc_figure(
    p_t_ref, p_x_ref, p_t_pinn_test, p_x_pinn,
    p_t_train, p_x_train,
    p_params_true, p_params_pinn,
    p_metrics_pinn, p_metrics_classical,
    p_t_classical, p_x_classical,
    p_n_value, p_tau,
    p_snapshot_times, p_snapshot_window,
    p_output_path,
    p_loss_history=None, p_data_loss_history=None, p_physics_loss_history=None,
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
        ["Rel. $L^2$  (Classical DDE / PINN)",
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
            "Rel. $L^2$\n(Classical)", "Rel. $L^2$\n(PINN)",
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
    v_parser = argparse.ArgumentParser(description="SYNASC comparison: Classical DDE vs PINN")
    v_parser.add_argument(
        "--config",
        default=str(SYNASC_ROOT / "configs" / "config_mackey_glass_synasc_t100_windowed.yaml"),
        help="PINN config YAML (relative paths resolve under this synasc bundle)",
    )
    v_parser.add_argument(
        "--n-values",
        default="7,10,20",
        help="Comma-separated Hill exponent values",
    )
    v_parser.add_argument(
        "--output-dir",
        default=str(SYNASC_ROOT / "results"),
        help="Output directory for figures and data (relative paths are under synasc/)",
    )
    v_parser.add_argument(
        "--skip-pinn",
        action="store_true",
        help="Skip PINN training (classical solver only)",
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
        help="Override time step for classical solver",
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
    return v_parser.parse_args()


def main():
    global _SYNASC_SAVE_PDF
    v_args = f_parse_args()
    _SYNASC_SAVE_PDF = not v_args.no_pdf

    v_args.config = f_resolve_synasc_config_path(v_args.config)
    v_args.output_dir = f_resolve_synasc_output_dir(v_args.output_dir)

    l_n_values = [float(x.strip()) for x in v_args.n_values.split(",")]
    os.makedirs(v_args.output_dir, exist_ok=True)

    d_config = f_load_config(v_args.config)
    if v_args.seed is not None:
        d_config["_seed"] = int(v_args.seed)

    v_beta = float(d_config["problem"]["beta_true"])
    v_gamma = float(d_config["problem"]["gamma_true"])
    v_tau = float(d_config["problem"]["tau"])
    v_x0 = float(d_config["problem"].get("initial_x_history", 1.2))
    v_t_end = v_args.t_end or float(d_config["data"]["t_total"])
    v_dt = v_args.dt or float(d_config["data"]["dt"])
    v_dt_fine = min(v_dt, 0.01)

    if v_args.snapshot_times:
        l_snapshot_times = [float(x.strip()) for x in v_args.snapshot_times.split(",")]
    else:
        l_snapshot_times = [
            v_t_end * 0.1,
            v_t_end * 0.4,
            v_t_end * 0.8,
        ]

    d_all_results = {}

    print("=" * 80)
    print("SYNASC COMPARISON: Classical DDE Solver vs PINN (Mackey-Glass)")
    print("=" * 80)
    print(f"  beta={v_beta}, gamma={v_gamma}, tau={v_tau}, x0={v_x0}")
    print(f"  t_end={v_t_end}, dt_fine={v_dt_fine}")
    print(f"  n values: {l_n_values}")
    print(f"  Snapshot times: {l_snapshot_times}")
    print(f"  Output: {v_args.output_dir}")
    print("=" * 80)

    for v_n in l_n_values:
        print(f"\n{'─' * 60}")
        print(f"  Hill exponent n = {v_n}")
        print(f"{'─' * 60}")

        # ── Classical solver ──
        print(f"  [Classical] Running method-of-steps RK45 solver...")
        v_start_classical = time.time()
        v_t_classical, v_x_classical = f_solve_mackey_glass_classical(
            p_beta=v_beta, p_gamma=v_gamma, p_n=v_n, p_tau=v_tau,
            p_x0=v_x0, p_t_end=v_t_end, p_dt=v_dt_fine,
        )
        v_wall_classical = time.time() - v_start_classical
        print(f"  [Classical] Done in {v_wall_classical:.2f}s, {len(v_t_classical)} points")

        # High-accuracy reference (fixed-step RK4 at very fine dt)
        v_dt_ref = 0.001
        print(f"  [Reference] Generating high-accuracy RK4 reference (dt={v_dt_ref})...")
        v_t_ref, v_x_ref = f_solve_mackey_glass_rk4_fixed(
            p_beta=v_beta, p_gamma=v_gamma, p_n=v_n, p_tau=v_tau,
            p_x0=v_x0, p_t_end=v_t_end, p_dt=v_dt_ref,
        )

        # Metrics for classical solver vs fine reference
        v_interp_classical = interp1d(
            v_t_classical, v_x_classical, kind="cubic", fill_value="extrapolate",
        )
        v_x_classical_at_ref = v_interp_classical(v_t_ref)
        d_metrics_classical = f_compute_metrics(v_x_ref, v_x_classical_at_ref)
        print(f"  [Classical] MSE={d_metrics_classical['mse']:.2e}, "
              f"Rel L2={d_metrics_classical['rel_l2']:.6f}")

        # ── PINN ──
        d_pinn_result = None
        d_metrics_pinn = {"mse": np.nan, "rel_l2": np.nan, "max_abs_err": np.nan}
        d_params_pinn = {}
        v_wall_pinn = 0.0

        if not v_args.skip_pinn:
            print(f"  [PINN] Training PINN (n={v_n})...")
            d_config["_output_dir"] = v_args.output_dir
            try:
                d_pinn_result = f_train_pinn_mackey_glass(d_config, v_n)
                v_wall_pinn = d_pinn_result["wall_time"]

                v_interp_ref_at_pinn = interp1d(
                    v_t_ref, v_x_ref, kind="cubic", fill_value="extrapolate",
                )
                v_x_ref_at_pinn_test = v_interp_ref_at_pinn(d_pinn_result["t_test"].flatten())
                d_metrics_pinn = f_compute_metrics(v_x_ref_at_pinn_test, d_pinn_result["u_pred"][:, 0])
                d_params_pinn = d_pinn_result["params"]

                print(f"  [PINN] MSE={d_metrics_pinn['mse']:.2e}, "
                      f"Rel L2={d_metrics_pinn['rel_l2']:.6f}, "
                      f"Wall time={v_wall_pinn:.1f}s")
                print(f"  [PINN] Identified params: {d_params_pinn}")
            except Exception as e:
                print(f"  [PINN] Training failed: {e}")
                import traceback
                traceback.print_exc()

        d_params_true = {"beta": v_beta, "gamma": v_gamma, "n": v_n, "tau": v_tau}

        d_all_results[v_n] = {
            "t_ref": v_t_ref,
            "x_ref": v_x_ref,
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
            "wall_time_pinn": v_wall_pinn,
            "tau": v_tau,
            "loss_history": d_pinn_result["loss_history"] if d_pinn_result else [],
            "data_loss_history": d_pinn_result["data_loss_history"] if d_pinn_result else [],
            "physics_loss_history": d_pinn_result["physics_loss_history"] if d_pinn_result else [],
        }

        # Per-n figure
        if d_pinn_result is not None:
            v_fig_path = os.path.join(v_args.output_dir, f"mackey_glass_n{v_n:g}_comparison.png")
            f_create_synasc_figure(
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
            )

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
    v_pkl_path = os.path.join(v_args.output_dir, "synasc_results.pkl")
    with open(v_pkl_path, "wb") as fh:
        pickle.dump(d_all_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nRaw results saved to: {v_pkl_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'n':>8} | {'Classical MSE':>14} | {'PINN MSE':>14} | {'Classical Time':>14} | {'PINN Time':>14}")
    print("-" * 80)
    for v_n in sorted(d_all_results.keys()):
        d_r = d_all_results[v_n]
        print(
            f"{v_n:8g} | {d_r['metrics_classical']['mse']:14.2e} | "
            f"{d_r['metrics_pinn']['mse']:14.2e} | "
            f"{d_r['wall_time_classical']:13.1f}s | "
            f"{d_r['wall_time_pinn']:13.1f}s"
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
    p_output_dir: str = "SYNASC_results",
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
