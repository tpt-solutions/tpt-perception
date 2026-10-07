# tpt-perception

[![Crates.io](https://img.shields.io/crates/v/tpt-perception.svg)](https://crates.io/crates/tpt-perception)
[![Documentation](https://docs.rs/tpt-perception/badge.svg)](https://docs.rs/tpt-perception)
[![License](https://img.shields.io/crates/l/tpt-perception.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Umbrella crate: feature-gated re-exports of the full perception stack.

## Feature Matrix

| Feature | Sub-crate | Contents |
|---------|-----------|----------|
| `core` | tpt-percept-core | frame-typed geometry, rigid transforms, twists, Kabsch/Umeyama |
| `cloud` | tpt-percept-cloud | point clouds, voxel grids, filters, k-d/octree/R-tree, streaming |
| `features` | tpt-percept-features | normals, Harris 3D, edges, RANSAC planes, FPFH, SHOT, learned hooks |
| `register` | tpt-percept-register | ICP, NDT, feature matching, branch-and-bound, Hough loop closure |
| `slam` | tpt-percept-slam | visual/LiDAR odometry, Scan Context loop closure, pose graph |
| `fusion` | tpt-percept-fusion | temporal alignment, EKF/UKF, hand–eye calibration, covariance intersection |
| `map` | tpt-percept-map | occupancy grids, TSDF, SDF, semantic clouds |
| `verify` | tpt-percept-verify | proptest strategies, invariant checkers, Kani harnesses |

`default` = everything except `verify`; `full` = everything. Each feature
can be toggled independently for minimal builds.

## Quick Start

```toml
[dependencies]
# Full stack (default features)
tpt-perception = "0.1"

# Minimal: only registration and SLAM
tpt-perception = { version = "0.1", default-features = false, features = ["register", "slam"] }

# With verification harnesses
tpt-perception = { version = "0.1", features = ["full"] }
```

```rust
use tpt_perception::prelude::*;

// Frames are types, not comments: this does not compile —
// let p_world: Point3D<World> = sensor_point.transform_with(world_to_camera);
```

## Usage Example: End-to-End Pipeline

```rust
use tpt_perception::prelude::*;

// 1. Capture a scan in lidar frame
let scan = PointCloud::from_points(vec![
    [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
]);
let lidar_to_world = Isometry3::<Lidar<0>, World>::from_translation([1.0, 2.0, 0.5]).unwrap();

// 2. Transform to world frame (frame-safe!)
let world_scan: PointCloud = {
    use tpt_percept_cloud::cloud::FrameCloud;
    FrameCloud::<Lidar<0>>::from_cloud(scan)
        .transform_into(&lidar_to_world)
        .into_cloud()
};

// 3. Register against reference (ICP)
let reference = world_scan.clone();
let result = register::icp(
    &world_scan,
    &reference,
    None,
    register::icp::IcpVariant::PointToPoint,
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [0.0; 3],
    &register::icp::IcpParams {
        max_correspondence_distance: 1.0,
        min_correspondences: 3,
        ..Default::default()
    }
).unwrap();
assert!(result.converged);

// 4. Fuse into occupancy grid
let mut grid = map::occupancy::OccupancyGrid2D::new(
    [0.0; 2],
    map::occupancy::OccupancyParams::default(),
);
for p in world_scan.points() {
    grid.insert_ray([0.0, 0.0], [p[0], p[1]]);
}
assert!(!grid.is_empty());
```

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `core` | Frame-typed geometry (always included) | tpt-percept-core |
| `cloud` | Point cloud processing | tpt-percept-cloud, tpt-percept-core |
| `features` | Feature extraction | tpt-percept-features, tpt-percept-cloud, tpt-percept-core |
| `register` | Point cloud registration | tpt-percept-register, tpt-percept-features, tpt-percept-cloud, tpt-percept-core |
| `slam` | SLAM (odometry, loop closure, pose graph) | tpt-percept-slam, tpt-percept-register, tpt-percept-features, tpt-percept-cloud, tpt-percept-core |
| `fusion` | Multi-sensor fusion | tpt-percept-fusion, tpt-percept-core |
| `map` | Spatial mapping | tpt-percept-map, tpt-percept-cloud, tpt-percept-core |
| `verify` | Verification harnesses (dev) | tpt-percept-verify |
| `default` | `core`, `cloud`, `features`, `register`, `slam`, `fusion`, `map` | — |
| `full` | Everything including `verify` | — |

## Re-exports

The `prelude` module re-exports the most commonly used types from all enabled features:

```rust
use tpt_perception::prelude::*;
// Core types
Point3D, Vector3D, Isometry3, Rotation3, Twist3, World, Lidar, Camera, ...
// Cloud types (if `cloud`)
PointCloud, FrameCloud, KdTree, Octree, RTree, voxelize, ...
// Feature types (if `features`)
estimate_normals, harris3d, detect_boundaries, compute_fpfh, ...
// Registration types (if `register`)
icp, ndt_register, register_features, register_bnb, hough_loop_closure, ...
// SLAM types (if `slam`)
LidarOdometry, ScanContext, detect_loop, PoseGraph, ...
// Fusion types (if `fusion`)
PoseTimeline, Ekf, Ukf, calibrate_extrinsic, covariance_intersection, ...
// Map types (if `map`)
OccupancyGrid2D, TsdfMap, SignedDistanceField, SemanticCloud, ...
// Verify types (if `verify`)
rigid_transform, finite_point, check_distance_preserving, ...
```

## Building and Testing

```bash
# Build with default features
cargo build -p tpt-perception

# Build minimal (register + slam only)
cargo build -p tpt-perception --no-default-features --features register,slam

# Build with all features
cargo build -p tpt-perception --features full

# Run tests
cargo test -p tpt-perception

# no_std check (core only)
cargo build -p tpt-percept-core --no-default-features --features alloc
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.