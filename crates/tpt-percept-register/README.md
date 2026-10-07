# tpt-percept-register

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-register.svg)](https://crates.io/crates/tpt-percept-register)
[![Documentation](https://docs.rs/tpt-percept-register/badge.svg)](https://docs.rs/tpt-percept-register)
[![License](https://img.shields.io/crates/l/tpt-percept-register.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Point cloud registration for `tpt-perception`.

## Features

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

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-register = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-register = "0.1"
```

## Usage Examples

### ICP (Iterative Closest Point)

```rust
use tpt_percept_register::prelude::*;
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

let source = PointCloud::from_points(/* ... */);
let target = PointCloud::from_points(/* ... */);

// Point-to-point ICP
let params = IcpParams {
    max_correspondence_distance: 1.0,
    min_correspondences: 100,
    max_iterations: 50,
    fitness_epsilon: 1e-4,
    rmse_epsilon: 1e-4,
    ..Default::default()
};

let initial_guess = Isometry3::<Lidar<0>, World>::identity();
let result = icp(&source, &target, Some(&initial_guess), IcpVariant::PointToPoint, params);

if result.converged {
    println!("Fitness: {}, RMSE: {}", result.fitness, result.rmse);
    let aligned = source.transformed(&result.transform);
}

// Point-to-plane ICP (requires normals on target)
let target_normals = estimate_normals(&target, &KdTree::new(&target), 10);
let result = icp(&source, &target, Some(&initial_guess), IcpVariant::PointToPlane, params);
```

### NDT (Normal Distributions Transform)

```rust
use tpt_percept_register::prelude::*;
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

let target = PointCloud::from_points(/* ... */);

// Build NDT map (voxel-Gaussian)
let map_params = NdtMapParams {
    voxel_size: 1.0,
    min_points_per_voxel: 5,
};
let ndt_map = build_ndt_map(&target, &map_params);

// Register source to NDT map
let params = NdtParams {
    max_iterations: 30,
    step_size: 1.0,
    ..Default::default()
};

let initial_guess = Isometry3::<Lidar<0>, World>::identity();
let result = ndt_register(&source, &ndt_map, Some(&initial_guess), params);
```

### Feature-Based Registration

```rust
use tpt_percept_register::prelude::*;
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

// Compute descriptors on both clouds
let kdtree_s = KdTree::new(&source);
let kdtree_t = KdTree::new(&target);
let normals_s = estimate_normals(&source, &kdtree_s, 10);
let normals_t = estimate_normals(&target, &kdtree_t, 10);

let fpfh_s = compute_fpfh(&source, &kdtree_s, &normals_s, 0.5);
let fpfh_t = compute_fpfh(&target, &kdtree_t, &normals_t, 0.5);

// Match descriptors + RANSAC/Kabsch
let match_params = FeatureMatchParams {
    ratio_threshold: 0.8,
    ransac_iterations: 1000,
    inlier_threshold: 0.1,
    seed: 42,
};

let matches = match_descriptors(&fpfh_s, &fpfh_t, &match_params);
let FeatureRegistration { transform, inliers } = register_features(&source, &target, &matches, &match_params);
```

### Global Registration (Branch-and-Bound)

```rust
use tpt_percept_register::prelude::*;
use tpt_percept_cloud::prelude::*;

// For levelled LiDAR: search over (x, y, yaw)
let params = BnbParams {
    x_range: (-50.0, 50.0),
    y_range: (-50.0, 50.0),
    yaw_range: (-std::f64::consts::PI, std::f64::consts::PI),
    voxel_size: 2.0,
    trimmed_ratio: 0.3,
    max_iterations: 10000,
};

let result = register_bnb(&source, &target, params);
// result: BnbResult { transform, score, iterations }
```

### Loop Closure (Hough Voting)

```rust
use tpt_percept_register::prelude::*;
use tpt_percept_cloud::prelude::*;
use tpt_percept_features::prelude::*;

// Detect landmarks (keypoints with normals)
let kdtree = KdTree::new(&cloud);
let normals = estimate_normals(&cloud, &kdtree, 10);
let landmarks = extract_landmarks(&cloud, &normals, 0.5, 100);

// Hough voting for loop closure
let params = HoughParams {
    yaw_bins: 180,
    translation_bin: 1.0,
    min_votes: 10,
};

let candidates = hough_loop_closure(&landmarks, &params);
// candidates: Vec<LoopClosureCandidate> with relative transform
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default) |
| `alloc` | Enable `alloc` only (no `std`) |

## Modules

| Module | Description |
|--------|-------------|
| [`icp`] | Point-to-point and point-to-plane ICP with convergence reporting |
| [`ndt`] | NDT map building and Gauss–Newton registration |
| [`feature_match`] | Descriptor ratio matching + RANSAC/Kabsch |
| [`bnb`] | Branch-and-bound global registration over (x, y, yaw) |
| [`hough`] | Yaw Hough voting loop closure with translation RANSAC |
| [`error`] | `RegistrationError` error types |

## Conventions

- **ICP**: Returns `IcpResult` with `converged`, `fitness`, `rmse`, `transform`
- **NDT**: Voxel-Gaussian target map, squared Mahalanobis objective
- **Feature matching**: Lowe's ratio test + deterministic RANSAC
- **Global**: Valid lower bound on trimmed residuals for branch-and-bound
- **Loop closure**: Normal-azimuth pairs → yaw Hough → translation RANSAC
- **Determinism**: Seeded Xorshift64* for reproducible results

## no_std Support

Build without `std`:

```bash
cargo build -p tpt-percept-register --no-default-features --features alloc
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
