# tpt-percept-verify

Verification harnesses for tpt-perception (std-only; the algorithm crates
remain `no_std`).

- **proptest strategies** — valid points, exactly-orthonormal rotations,
  rigid transforms, point clouds, bounded noise profiles, PSD covariances.
- **Invariant checkers** — distance preservation, inverse roundtrips,
  monotone residual histories, covariance symmetry/PSD.
- **Kani harnesses** — `#[cfg(kani)]` bounded model checks: voxel-key
  safety and roundtrips, log-odds clamping bounds, Kabsch no-panic on
  degenerate sets. Run with `cargo kani -p tpt-percept-verify`.
- **Property tests** — the spec's statistical invariants: rigid transforms
  preserve distances, ICP residual monotonicity, plane detection on
  synthetic clouds (phase-1 integration), normal robustness, EKF
  covariance validity, pose graph cycle consistency.

Part of [tpt-perception](../../). Dual-licensed MIT OR Apache-2.0.
