# Algorithm S1: Method-of-Steps Solver for the Mackey-Glass DDE

Method-of-steps solver for Eq. (1) in the paper.

## Pseudocode

```
Input: β, γ, n, τ, φ₀, T, Δt, rtol, atol

1.  Initialize history  H ← {(−τ, φ₀), (0, φ₀)}
2.  Build interpolant   x̂(·) ← interp(H, linear)
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

On each interval [kτ, (k+1)τ] the delayed argument t − τ falls in [(k−1)τ, kτ], where x has already been computed (or, for k = 0, evaluated from the constant history). The Mackey–Glass equation therefore reduces on each interval to a non-autonomous ordinary differential equation

  ẋ(t) = g_k(t, x(t)),    g_k(t, y) = β · x̂(t−τ) / (1 + |x̂(t−τ)|ⁿ) − γ · y

where x̂ is a continuous interpolant of the previously stored trajectory. Step 7 dispatches the local IVP to an embedded fourth/fifth order Runge–Kutta pair (Dormand–Prince), with adaptive step-size control governed by relative and absolute tolerances rtol, atol and a maximum step Δt_max. After each interval the trajectory buffer is extended and the interpolant is rebuilt as a cubic spline.
