# tpt-percept-fusion

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-fusion.svg)](https://crates.io/crates/tpt-percept-fusion)
[![Documentation](https://docs.rs/tpt-percept-fusion/badge.svg)](https://docs.rs/tpt-percept-fusion)
[![License](https://img.shields.io/crates/l/tpt-percept-fusion.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Multi-sensor fusion for `tpt-perception`.

## Features

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

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-fusion = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-fusion = "0.1"
```

## Usage Examples

### Temporal Alignment & Latency Estimation

```rust
use tpt_percept_fusion::prelude::*;
use tpt_percept_core::prelude::*;
use tpt_math_units::Time;

// Build a pose timeline for a sensor
let mut timeline = PoseTimeline::<Lidar<0>, World>::new();

// Add timestamped poses (from odometry)
timeline.push(TimedPose {
    timestamp: Time::from_seconds(0.0),
    pose: Isometry3::identity(),
});
timeline.push(TimedPose {
    timestamp: Time::from_seconds(0.1),
    pose: Isometry3::from_translation([0.1, 0.0, 0.0]),
});

// Interpolate pose at arbitrary time (bounded extrapolation)
let query_time = Time::from_seconds(0.05);
let interpolated = timeline.interpolate(query_time).unwrap();

// Estimate latency from timestamp offsets
let mut latency_est = LatencyEstimator::new();
latency_est.update(Time::from_seconds(0.05), Time::from_seconds(0.048));
let (mean, std) = latency_est.stats();
```

### Extended Kalman Filter (EKF)

```rust
use tpt_percept_fusion::prelude::*;
use tpt_percept_core::prelude::*;
use tpt_math_linalg_dense::*;

// State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
type State = Vec<f64, 12>;

// Process model: constant velocity
let mut ekf = Ekf::new(State::zeros(), Mat::<12,12>::identity() * 0.1);

// Predict step
ekf.predict(|x| {
    let dt = 0.1;
    let mut x_next = x.clone();
    x_next[0] += x[6] * dt;
    x_next[1] += x[7] * dt;
    x_next[2] += x[8] * dt;
    x_next
}, |x| {
    let mut f = Mat::<12,12>::identity();
    let dt = 0.1;
    f[(0,6)] = dt; f[(1,7)] = dt; f[(2,8)] = dt;
    f
}, Mat::<12,12>::identity() * 0.01);

// Update with measurement (e.g., GPS position)
let h = |x: &State| vec![x[0], x[1], x[2]];
let H = |x: &State| {
    let mut h = Mat::<3,12>::zeros();
    h[(0,0)] = 1.0; h[(1,1)] = 1.0; h[(2,2)] = 1.0;
    h
};
let z = vec![10.1, 0.05, 0.0];
let R = Mat::<3,3>::identity() * 0.1;

let (state, nis) = ekf.update(&z, &h, &H, &R);
```

### Unscented Kalman Filter (UKF)

```rust
use tpt_percept_fusion::prelude::*;

let mut ukf = Ukf::new(
    State::zeros(),
    Mat::<12,12>::identity() * 0.1,
    1e-3, // alpha
    2.0,  // kappa
    0.0,  // beta
);

ukf.predict(|x| { /* ... */ }, Mat::<12,12>::identity() * 0.01);
ukf.update(&z, |x| vec![x[0], x[1], x[2]], Mat::<3,3>::identity() * 0.1);
```

### Extrinsic Calibration (Hand-Eye)

```rust
use tpt_percept_fusion::prelude::*;
use tpt_percept_core::prelude::*;

let pairs = vec![
    MotionPair {
        a: Isometry3::<Camera, Camera>::from_rotation_translation(/* ... */),
        b: Isometry3::<Lidar<0>, Lidar<0>>::from_rotation_translation(/* ... */),
    },
];

let Extrinsic { rotation, translation } = calibrate_extrinsic(&pairs).unwrap();
```

### Robust Fusion with Covariance Intersection

```rust
use tpt_percept_fusion::prelude::*;
use tpt_percept_core::prelude::*;

let gauss1 = Gaussian { mean: vec![1.0, 0.0], cov: Mat::identity() * 0.1 };
let gauss2 = Gaussian { mean: vec![1.1, 0.1], cov: Mat::identity() * 0.2 };

let fused = covariance_intersection(&gauss1, &gauss2, 0.5);

let gate = InnovationGate::new(0.95);
let is_valid = gate.check(&innovation, &covariance);

let mut fallback = RobustFuse::new()
    .add_sensor(gauss1)
    .add_sensor(gauss2)
    .with_falloff_threshold(3);
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default) |
| `alloc` | Enable `alloc` only (no `std`) |

## Modules

| Module | Description |
|--------|-------------|
| [`calibration`] | Tsai–Lenz hand–eye calibration from motion pairs |
| [`ekf`] | Extended Kalman Filter (Joseph form) and Unscented KF |
| [`error`] | `FusionError` error types |
| [`robust`] | Covariance intersection, innovation gating, sensor fallback |
| [`temporal`] | Pose timelines, interpolation, latency estimation |

## Conventions

- **Timestamps**: `tpt-math-units::Time` (SI seconds)
- **State vectors**: Column vectors, dense matrices via `tpt-math-linalg-dense`
- **EKF**: Joseph-form covariance update for numerical stability
- **UKF**: Scaled sigma-point transform (Julier parameters)
- **Covariance intersection**: Consistent for unknown correlations
- **Calibration**: Tsai–Lenz with modified Rodrigues parameterization

## no_std Support

Build without `std`:

```bash
cargo build -p tpt-percept-fusion --no-default-features --features alloc
```

Note: EKF/UKF require `tpt-math-linalg-dense` which needs `alloc`.

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
