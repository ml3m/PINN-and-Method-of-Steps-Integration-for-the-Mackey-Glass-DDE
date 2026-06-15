# Table S3: Method-of-Steps Solver Configuration

Method-of-steps solver configuration used in this study.

| Setting | Value |
|---|---|
| ODE method | RK45 (Dormand–Prince) |
| Relative tolerance (rtol) | 10⁻⁹ |
| Absolute tolerance (atol) | 10⁻¹¹ |
| Maximum step (Δt_max) | 0.01 |
| Output grid spacing | Δt = 0.01 |
| History interpolation | cubic spline |
| Number of delay sub-intervals K | ⌈T/τ⌉ = 100 |
| Output points | 20,001 |
