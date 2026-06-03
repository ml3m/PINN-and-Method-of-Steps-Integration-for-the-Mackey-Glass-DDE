# Table S2 — Largest Lyapunov Exponent λ₁

Estimated via Benettin's algorithm (Benettin et al., 1980) applied to the
Jacobian of the frozen-delay explicit RK4 map used in the reference generator
(delay-line state of dimension τ/Δ*t*_ref + 1).

**Parameters:** burn-in 15 %, 170 s of the orbit evaluated.
PINN values: mean ± std (*K* = 5) with common tangent initializer
(`-lyapunov-seed 42`).

| Orbit sampled on the fine grid | λ₁ (s⁻¹)        |
|--------------------------------|:----------------:|
| RK4 reference                  | 0.0404           |
| Stitched PINN                  | 0.0533 ± 0.0023 |

**Range over seeds:** [0.0504, 0.0558] (same five seeds as the global errors
table in the paper: 42, 1234, 2025, 31415, 99999).

The reference orbit yields λ₁ > 0, indicating average exponential separation
of tiny initial offsets in tangent space under this map. The ensemble mean λ₁
along PINN surrogates exceeds the reference value — a diagnostic of local
stretching when the Mackey–Glass tangent dynamics are evaluated on surrogate
samples. This is not interpreted as a Lyapunov exponent of a true DDE orbit,
because the PINN trajectory does not exactly satisfy the Mackey–Glass equation.

## Reproducing this table

```bash
python lyapunov_mackey_glass_benettin.py -multi-seed-root <results_dir>
```

See `lyapunov_mackey_glass_benettin.py` (`-multi-seed-root`) for per-seed CSV
output.
