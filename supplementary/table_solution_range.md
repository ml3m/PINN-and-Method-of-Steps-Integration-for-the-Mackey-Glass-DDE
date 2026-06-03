# Table S1 — Range of Solution Values on [0, 200] (*n* = 10)

Minimum and maximum values of *x*(*t*) on the integration horizon [0, 200]
for each method, evaluated on the respective output grids (RK4 reference at
Δ*t* = 10⁻³, MoS-RK45 interpolated to the same grid, PINN stitched
forward evaluation).

| Method               | min *x*(*t*) | max *x*(*t*) |
|----------------------|:------------:|:------------:|
| RK4 reference        | 0.3289       | 1.362        |
| Method-of-steps RK45 | 0.335        | 1.351        |
| PINN                 | 0.322        | 1.363        |

The classical solver remains close to the reference extrema, whereas the
stitched PINN undershoots the minimum slightly while nearly recovering the
observed maximum — a pattern consistent with spectral bias smoothing sharp
turning points (Rahaman et al., 2019; Xu et al., 2020).
