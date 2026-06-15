# Algorithm S1: Method-of-Steps Solver for the Mackey-Glass DDE

Method-of-steps solver for Eq. (1) in the paper.

## Pseudocode

```
Input: β, γ, n, τ, φ₀, T, Δt, rtol, atol

1.  Initialize history      H ← {(−τ, φ₀), (0, φ₀)}
2.  Build interpolant       x̂(·) ← interp(H, linear)
3.  K ← ⌈T/τ⌉;  y ← φ₀
4.  FOR k = 0, 1, …, K−1 DO
5.      tₐ ← kτ,  t_b ← min((k+1)τ, T)
6.      Define g_k as in Eq. (3), using x̂
7.      Solve ẋ = g_k(t, x),  x(tₐ) = y,  on [tₐ, t_b]
            with adaptive RK45 (rtol, atol, Δt)
8.      Append solution to H;  y ← x(t_b)
9.      Rebuild x̂ as a cubic spline of H
10. ENDFOR
11. Return (t, x) on the requested uniform grid.
```

## Description

On each interval $[k\tau, (k+1)\tau]$ the delayed argument $t - \tau$ falls in $[(k-1)\tau, k\tau]$, where $x$ has already been computed (or, for $k = 0$, evaluated from the constant history). The Mackey–Glass equation is integrated as:

$$\dot{x}(t) = g_k(t, x(t)),$$

where

$$g_k(t, y) = \beta \cdot \frac{\hat{x}(t-\tau)}{1 + |\hat{x}(t-\tau)|^n} - \gamma \cdot y$$

Here, $\hat{x}(\cdot)$ is a continuous interpolant of the previously stored trajectory. Step 7 dispatches the local initial-value problem (IVP) to an embedded fourth/fifth-order Runge–Kutta pair (Dormand–Prince), with adaptive step-size control governed by the prescribed tolerances `rtol` and `atol`.

### Key Features

- **History management:** The history buffer $H$ maintains all previously computed solution points.
- **Interpolation:** At step 2 and 9, we rebuild $\hat{x}$ as a cubic spline to ensure smooth evaluation of delayed terms.
- **Step intervals:** The solver progresses through $K$ intervals of width $\tau$, adapting to partial intervals near the terminal time $T$.
- **Adaptive integration:** RK45 with user-specified tolerances ensures accuracy while minimizing computational overhead.
