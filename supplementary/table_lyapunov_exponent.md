# Table S2 — Largest Lyapunov Exponent $\lambda_1$

Estimated via Benettin's algorithm [Benettin et al., 1980] applied to the Jacobian of the frozen-delay explicit RK4 map used in the reference generator (delay-line state of dimension $\tau/\Delta t \approx 1000$).

**Parameters:** burn-in 15%, 170 s of the orbit evaluated. PINN values: mean ± std ($K = 5$) with common tangent initializer (`-lyapunov-seed 42`).

| Orbit sampled on the fine grid | $\lambda_1$ (s$^{-1}$) |
|--------------------------------|:--------------------:|
| RK4 reference                  | 0.0404               |
| Stitched PINN                  | 0.0533 ± 0.0023      |

**Range over seeds:** [0.0504, 0.0558] (same five seeds as the global errors table in the paper: 42, 1234, 2025, 31415, 99999).

The reference orbit yields $\lambda_1 > 0$, indicating average exponential separation of tiny initial offsets in tangent space under this map. The ensemble mean $\lambda_1$ along PINN surrogates exhibits slightly higher sensitivity, consistent with localized approximation errors compounding along chaotic trajectories.

## Reproducing this table

```bash
python lyapunov_mackey_glass_benettin.py -multi-seed-root <results_dir>
```

See `lyapunov_mackey_glass_benettin.py` (`-multi-seed-root`) for per-seed CSV output.

## References

- Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory. *Meccanica*, 15(1), 9–20.
