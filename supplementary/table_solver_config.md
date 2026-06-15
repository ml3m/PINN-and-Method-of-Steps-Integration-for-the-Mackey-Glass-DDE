# Table S3: Method-of-Steps Solver Configuration

Method-of-steps solver configuration used in this study.

| Setting | Value |
|---|---|
| ODE method | RK45 (Dormand–Prince) |
| Relative tolerance (rtol) | $10^{-9}$ |
| Absolute tolerance (atol) | $10^{-11}$ |
| Maximum step ($\Delta t_{\max}$) | 0.01 |
| Output grid spacing | $\Delta t = 0.01$ |
| History interpolation | cubic spline |
| Number of delay sub-intervals $K$ | $\lceil T/\tau \rceil = 100$ |
| Output points | 20,001 |
