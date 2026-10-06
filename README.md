# tpt-perception

A pure-Rust, AI-native 3D spatial perception and sensor fusion library:
point cloud processing, SLAM, multi-sensor fusion, geometric feature
extraction, and spatial registration — with **no C/C++ FFI** and no
external algorithm libraries.

Part of the `tpt-*` family. See [spec.txt](spec.txt) for the full design
specification and [todo.md](todo.md) for the phased build tracker.

## Design principles

- **Zero external dependencies for algorithms.** ICP, NDT, voxelization,
  feature detection, filters — all implemented from scratch in pure Rust,
  built on the in-house [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
  numeric substrate.
- **Strict licensing.** MIT OR Apache-2.0 throughout; `deny.toml` blocks
  Apache-2.0-only wrap targets.
- **Type-state spatial safety.** Coordinate frames are phantom type
  parameters: transforming a `Point3D<Sensor<Lidar>>` with a
  `Sensor<Lidar> → World` isometry compiles; mixing frames does not.
- **Native formal verification.** Kani bounded model checking harnesses for
  critical spatial algorithms, property-based tests (`proptest`) for
  statistical geometric invariants, documented mathematical contracts.
- **`no_std` + `alloc` core.** All algorithm crates compile without `std`.
- **AI-native.** Feature descriptors and cost functions expose hooks for
  learned (autodiff-ready) perception.

## Crates

| Crate | Purpose |
|-------|---------|
| [`tpt-percept-core`](crates/tpt-percept-core) | 3D geometry, frame-typed points/transforms, twists, small dense linear algebra |
| [`tpt-percept-cloud`](crates/tpt-percept-cloud) | Point clouds: voxelization, filtering, k-d tree, octree, R-tree, streaming |
| [`tpt-percept-features`](crates/tpt-percept-features) | Normals, curvature, PCA, Harris 3D, edges, RANSAC planes, FPFH, SHOT, learned-descriptor hooks |
| [`tpt-percept-register`](crates/tpt-percept-register) | ICP (point-to-point / point-to-plane), NDT, feature + RANSAC, branch-and-bound, 3D Hough |
| [`tpt-percept-slam`](crates/tpt-percept-slam) | Visual & LiDAR odometry, loop closure, pose graph optimization |
| [`tpt-percept-fusion`](crates/tpt-percept-fusion) | Temporal alignment, EKF/UKF, extrinsic calibration, covariance intersection, robust fusion |
| [`tpt-percept-map`](crates/tpt-percept-map) | Occupancy grids (2D/3D log-odds), TSDF, signed distance fields, semantic maps |
| [`tpt-percept-verify`](crates/tpt-percept-verify) | proptest strategies, geometric invariant checks, Kani proof harnesses |
| [`tpt-perception`](crates/tpt-perception) | Feature-gated umbrella crate re-exporting everything above |

## Quick start

```toml
[dependencies]
tpt-perception = "0.1"          # umbrella, all features
# or pick crates à la carte:
tpt-percept-register = "0.1"
```

```rust
use tpt_perception::prelude::*;

// Frames are types, not comments: this does not compile —
// let p_world: Point3D<World> = sensor_point.transform_with(world_to_camera);
```

## Building and verifying

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo deny check            # license/advisory hygiene
cargo build -p tpt-percept-core --no-default-features --features alloc   # no_std gate
# Kani proofs (requires cargo-kani): cargo kani -p tpt-percept-verify
```

## Status

Phase 4 complete — see [todo.md](todo.md) for the itemized checklist.
Open integration checkpoints (external `tpt-systems-optimisation`,
`tpt-gis`, `tpt-control` repos) are tracked at the top of [todo.md](todo.md).

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).
© TPT Solutions.
