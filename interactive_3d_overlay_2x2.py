#!/usr/bin/env python3
"""
Interactive 2×2 3D delay-embedding overlay (RK4 + MoS + PINN).

Uses a GUI backend (default TkAgg). Import run_synasc_comparison only after
SYNASC_MPL_INTERACTIVE is set so matplotlib is not forced to Agg.

  cd synasc && SYNASC_MPL_INTERACTIVE=1 python interactive_3d_overlay_2x2.py \\
      --pkl ../SYNASC_results_t200_windowed/synasc_results.pkl --n 10

Rotate / zoom each subplot with the mouse (standard mpl3d toolbar). When the
framing looks right, focus the figure window and press **s** to write PDF and PNG.

Qt backend (if Tk is unavailable)::

  MPLBACKEND=QtAgg SYNASC_MPL_INTERACTIVE=1 python interactive_3d_overlay_2x2.py ...
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle
from pathlib import Path

os.environ.setdefault("SYNASC_MPL_INTERACTIVE", "1")

import numpy as np

# Must run after SYNASC_MPL_INTERACTIVE (see run_synasc_comparison).
from run_synasc_comparison import (  # noqa: E402
    SYNASC_ROOT,
    _OVERLAY_3D_VIEW_QUAD,
    _delay_indices,
    _f_bounds_xyz_from_branches,
    _f_draw_3d_overlay_on_ax,
    _f_embed_3d_branch,
    _f_result_key_for_n,
)

import matplotlib.pyplot as plt  # noqa: E402


def _load_record(p_pkl: Path, p_n: float) -> dict:
    with open(p_pkl, "rb") as fh:
        d_all = pickle.load(fh)
    v_key = _f_result_key_for_n(d_all, p_n)
    if v_key is None:
        raise SystemExit(f"n={p_n:g} not in pickle; keys: {list(d_all.keys())!r}")
    return d_all[v_key]


def main() -> None:
    v_ap = argparse.ArgumentParser(description="Interactive 2x2 3D overlay editor")
    v_ap.add_argument(
        "--pkl",
        type=Path,
        default=SYNASC_ROOT.parent / "SYNASC_results_t200_windowed" / "synasc_results.pkl",
        help="synasc_results.pkl",
    )
    v_ap.add_argument("--n", type=float, default=10.0)
    v_ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output stem (no extension); writes .pdf and .png on 's'",
    )
    v_args = v_ap.parse_args()

    if not v_args.pkl.is_file():
        raise SystemExit(f"Pickle not found: {v_args.pkl}")

    v_rec = _load_record(v_args.pkl, v_args.n)
    v_tau = float(v_rec["tau"])
    v_xp = v_rec["x_pinn"]
    if v_xp is None or (hasattr(v_xp, "size") and v_xp.size == 0):
        raise SystemExit("No PINN trajectory in record")

    p_t_ref = v_rec["t_ref"]
    p_x_ref = v_rec["x_ref"]
    p_t_cl = v_rec["t_classical"]
    p_x_cl = v_rec["x_classical"]
    p_t_p = v_rec["t_pinn_test"]

    ds_ref = _delay_indices(p_t_ref, v_tau)
    ds_cl = _delay_indices(p_t_cl, v_tau)
    ds_p = _delay_indices(np.asarray(p_t_p).reshape(-1), v_tau)

    xp = np.asarray(v_xp[:, 0]).reshape(-1)
    branch_ref = (
        _f_embed_3d_branch(p_x_ref, ds_ref)
        if len(np.asarray(p_x_ref).reshape(-1)) > 2 * ds_ref
        else None
    )
    branch_cl = (
        _f_embed_3d_branch(p_x_cl, ds_cl)
        if len(np.asarray(p_x_cl).reshape(-1)) > 2 * ds_cl
        else None
    )
    branch_pn = _f_embed_3d_branch(xp, ds_p) if len(xp) > 2 * ds_p else None

    z_floor = None
    if branch_ref is not None:
        z_floor = float(np.min(branch_ref[2]) - 0.05)

    v_bounds = _f_bounds_xyz_from_branches([branch_ref, branch_cl, branch_pn])
    if v_bounds is None:
        raise SystemExit("Could not build embedding branches (grid too short?)")

    _GRID_W, _GRID_H = 10.0, 12.0
    _GTITLE, _GLABEL, _GTICK, _GLEG = 9, 8, 6, 6
    _LW_REF, _LW_CL, _LW_PINN = 1.2, 1.5, 1.2
    _ALPHA_REF, _ALPHA_CL, _ALPHA_PINN = 0.88, 0.9, 0.8
    _DIST = 11

    fig = plt.figure(figsize=(_GRID_W, _GRID_H))
    for v_i, (v_el, v_az) in enumerate(_OVERLAY_3D_VIEW_QUAD):
        ax_g = fig.add_subplot(2, 2, v_i + 1, projection="3d")
        v_panel = chr(ord("a") + v_i)
        _f_draw_3d_overlay_on_ax(
            ax_g,
            branch_ref=branch_ref,
            branch_cl=branch_cl,
            branch_pinn=branch_pn,
            p_z_floor_shadow=z_floor,
            p_n_value=float(v_args.n),
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

    fig.suptitle(
        f"Mackey-Glass ($n={float(v_args.n):g}$) — 3D delay embedding — press **s** to save",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    v_out = v_args.output
    if v_out is None:
        v_out = Path.cwd() / f"n{v_args.n:g}_3d_overlay_2x2_interactive"

    def _on_key(event):
        if event.key != "s":
            return
        v_pdf = Path(str(v_out) + ".pdf")
        v_png = Path(str(v_out) + ".png")
        fig.savefig(v_pdf, bbox_inches="tight")
        fig.savefig(v_png, dpi=300, bbox_inches="tight")
        print(f"Saved {v_pdf.resolve()}")
        print(f"Saved {v_png.resolve()}")

    fig.canvas.mpl_connect("key_press_event", _on_key)

    print(
        "Figure open. Rotate/zoom each 3D axes with the mouse.\n"
        "Press **s** (with the figure focused) to save PDF + PNG.\n"
        f"Output stem: {v_out}",
        file=sys.stderr,
    )
    plt.show()


if __name__ == "__main__":
    main()
