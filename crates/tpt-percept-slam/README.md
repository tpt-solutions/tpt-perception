# tpt-percept-slam

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-slam.svg)](https://crates.io/crates/tpt-percept-slam)
[![Documentation](https://docs.rs/tpt-percept-slam/badge.svg)](https://docs.rs/tpt-percept-slam)
[![License](https://img.shields.io/crates/l/tpt-percept-slam.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

SLAM for `tpt-perception`: odometry, place recognition, and pose graph
optimization.

## Features

- **Monocular visual odometry** — normalized 8-point essential matrix
  (9×9 cyclic-Jacobi smallest eigenvector), singular-value projection onto
  the essential manifold, four-fold decomposition with exact-depth
  cheirality selection, RANSAC over correspondences. Motion up to scale.
- **Stereo visual odometry** — disparity triangulation (`Z = f·b/d`) and
  metric 3-D-3-D RANSAC/Kabsch alignment.
- **LiDAR odometry** — point-to-plane scan matching with constant-velocity
  prediction, poses accumulated in the odometry frame.
- **Loop closure** — Scan Context descriptors (polar grid, ring-key
  screening, azimuth-shift matching).
- **Pose graph** — dense Gauss–Newton over SE(3) with numeric Jacobians,
  Huber robustification, fixed anchor node. Intentionally small-scale; the
  sparse large-scale solver is the open `tpt-systems-optimisation`
  checkpoint in `todo.md`.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-slam = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-slam = "0.1"
```

## Usage Examples

### Monocular Visual Odometry

```rust
use tpt_percept_slam::prelude::*;
use tpt_percept_core::prelude::*;

// Normalized correspondences (from feature tracking)
let correspondences = vec![
    NormalizedCorrespondence {
        p1: [0.1, 0.2, 1.0],
        p2: [0.12, 0.19, 1.0],
    },
];

let params = MonoParams {
    ransac_iterations: 1000,
    ransac_threshold: 1e-3,
    min_inliers: 50,
    seed: 42,
};

let MonoMotion { essential, inliers, scale } = estimate_motion_mono(&correspondences, params);
```

### Stereo Visual Odometry

```rust
use tpt_percept_slam::prelude::*;
use tpt_percept_core::prelude::*;

let matches = vec![
    StereoMatch {
        left: [100.0, 200.0],
        disparity: 15.0,
    },
];

let params = StereoParams {
    focal_length: 500.0,
    baseline: 0.12,
    ransac_iterations: 1000,
    inlier_threshold: 0.05,
    seed: 42,
};

let StereoMotion { transform, inliers } = estimate_motion_stereo(&matches, params);
```

### LiDAR Odometry

```rust
use tpt_percept_slam::prelude::*;
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

let mut odom = LidarOdometry::new(LidarOdometryParams {
    voxel_size: 0.5,
    max_correspondence_distance: 1.0,
    max_iterations: 10,
    ..Default::default()
});

for scan in lidar_stream {
    let OdometryStep { transform, fitness, rmse } = odom.step(&scan);
}
```

### Loop Closure Detection (Scan Context)

```rust
use tpt_percept_slam::prelude::*;
use tpt_percept_cloud::prelude::*;

let params = ScanContextParams {
    num_rings: 20,
    num_sectors: 60,
    max_range: 80.0,
    ring_key_threshold: 0.1,
};

let descriptor = ScanContext::from_cloud(&scan, &params);
let match_result = detect_loop(&descriptor, &descriptor_database, &params);
```

### Pose Graph Optimization

```rust
use tpt_percept_slam::prelude::*;
use tpt_percept_core::prelude::*;

let mut graph = PoseGraph::new(PoseGraphParams {
    max_iterations: 50,
    huber_delta: 1.0,
    anchor_node: 0,
});

for i in 1..num_poses {
    let rel_pose = compute_odometry(i-1, i);
    graph.add_edge(PoseEdge {
        from: i-1,
        to: i,
        transform: rel_pose,
        information: Mat::<6,6>::identity() * 100.0,
    });
}

for candidate in loop_candidates {
    graph.add_edge(PoseEdge {
        from: candidate.from,
        to: candidate.to,
        transform: candidate.transform,
        information: Mat::<6,6>::identity() * 10.0,
    });
}

let optimized_poses = graph.optimize();
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default) |
| `alloc` | Enable `alloc` only (no `std`) |

## Modules

| Module | Description |
|--------|-------------|
| [`vo_mono`] | Monocular visual odometry (essential matrix, RANSAC) |
| [`vo_stereo`] | Stereo visual odometry (disparity triangulation, metric alignment) |
| [`lidar_odometry`] | LiDAR scan-matching odometry (point-to-plane ICP) |
| [`loop_closure`] | Scan Context place recognition, loop detection |
| [`pose_graph`] | Dense Gauss–Newton SE(3) pose graph optimization |
| [`error`] | `SlamError` error types |

## Conventions

- **Monocular VO**: Motion up to scale, essential matrix from normalized correspondences
- **Stereo VO**: Metric scale via disparity triangulation (`Z = f·b/d`)
- **LiDAR odometry**: Point-to-plane ICP with constant-velocity prediction
- **Scan Context**: Polar grid descriptor (rings × sectors), ring-key screening
- **Pose graph**: Dense Gauss–Newton over SE(3), numeric Jacobians, Huber loss
- **Frames**: Odometry frame for VO, World frame for optimized poses

## no_std Support

Build without `std`:

```bash
cargo build -p tpt-percept-slam --no-default-features --features alloc
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
