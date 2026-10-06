# tpt-percept-core

Foundation layer of `tpt-perception`: the frame-safe geometric substrate
every other perception crate builds on.

- **Coordinate frames as types** — `Point3D<Lidar<0>>`, `Isometry3<Lidar<0>, World>`:
  frame mismatches are compile errors, not runtime bugs.
- **Rigid body transforms** — `Rotation3<From, To>`, `Isometry3<From, To>`,
  `Twist3<F>` with exact SE(3) exponential/logarithm maps.
- **Small dense linear algebra** — symmetric 3×3 eigendecomposition (cyclic
  Jacobi), Cramer solver, skew/outer products. Heap-free.
- **Closed-form alignment** — weighted Kabsch and Umeyama (rigid + similarity).
- **Unit-safe quantities** — SI types via `tpt-math-units` at API boundaries.

`no_std` compatible (pure core allocates nothing). Conventions: active
rotations, column vectors, right-handed frames, metres/radians/seconds.

Part of [tpt-perception](../../). Dual-licensed MIT OR Apache-2.0.
