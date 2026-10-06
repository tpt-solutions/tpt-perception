# tpt-percept-fusion

Multi-sensor fusion for `tpt-perception`.

- **Temporal alignment** — pose timelines with bounded extrapolation and
  orthonormalised interpolation; rolling latency estimation with
  mean/std-dev tracking (unit-checked timestamps via `tpt-math-units`).
- **Filtering** — self-contained dense EKF (Joseph-form update, NIS
  diagnostics) and UKF (scaled sigma-point transform, Cholesky-based
  sampling). These are the crate's own substrate per the resolved
  `tpt-control` checkpoint in `todo.md`.
- **Extrinsic calibration** — Tsai–Lenz hand–eye from synchronized motion
  pairs (modified Rodrigues rotation + linear translation).
- **Robust fusion** — covariance intersection (consistent for unknown
  cross-correlation), chi-square innovation gating with consecutive-failure
  sensor falloff, healthy-sensor fallback.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.
