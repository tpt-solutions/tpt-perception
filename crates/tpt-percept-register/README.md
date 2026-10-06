# tpt-percept-register

Point cloud registration for `tpt-perception`.

- **ICP** — point-to-point (closed-form Kabsch per iteration) and
  point-to-plane (Chen–Medioni linearised solve), shared convergence
  contract with fitness/RMSE reporting.
- **NDT** — voxel-Gaussian target map, Gauss–Newton likelihood ascent with
  backtracking line search (monotone improvement).
- **Feature-based** — descriptor ratio matching + RANSAC/Kabsch initial
  pose, deterministic (seeded xorshift64*).
- **Global registration** — branch-and-bound over (x, y, yaw) with a valid
  lower bound on trimmed residuals (levelled-LiDAR setup).
- **Loop closure** — yaw Hough voting over normal-azimuth pairs with
  translation RANSAC.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.
