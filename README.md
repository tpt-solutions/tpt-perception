# tpt-perception

A pure-Rust, AI-native 3D spatial perception and sensor fusion library:
point cloud processing, SLAM, multi-sensor fusion, geometric feature
extraction, and spatial registration, with **no C/C++ FFI** and no
external algorithm libraries.

Part of the `tpt-*` family. See [spec.txt](spec.txt) for the design
specification, [todo.md](todo.md) for the phased build tracker, and
[docs/](docs/) for guides.

## Design principles

- **Zero external dependencies for algorithms.** ICP, NDT, voxelization,
  feature detection, filters are all implemented from scratch in pure Rust,
  on top of the in-house [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
  numeric substrate.
- **Strict licensing.** MIT OR Apache-2.0 throughout; `deny.toml` blocks
  Apache-2.0-only wrap targets.
- **Type-state spatial safety.** Coordinate frames are phantom type
  parameters: `icp_frames` takes `FrameCloud<From>` and `FrameCloud<To>` and
  returns an `Isometry3<From, To>`. Mixing frames does not compile.
- **Native formal verification.** Kani bounded model checking harnesses for
  critical spatial algorithms, property-based tests (`proptest`) for
  geometric invariants, documented mathematical contracts.
- **`no_std` + `alloc` core.** The algorithm crates compile without `std`
  and for `wasm32-unknown-unknown`.

## Crates

| Crate | Purpose |
|-------|---------|
| [`tpt-percept-core`](crates/tpt-percept-core) | 3D geometry, frame-typed points/transforms, twists, small dense linear algebra |
| [`tpt-percept-cloud`](crates/tpt-percept-cloud) | Point clouds: voxelization, filtering, k-d tree, octree, R-tree, streaming, per-point attributes |
| [`tpt-percept-features`](crates/tpt-percept-features) | Normals, curvature, PCA, Harris 3D, edges, RANSAC planes, FPFH, SHOT, learned-descriptor hooks |
| [`tpt-percept-register`](crates/tpt-percept-register) | ICP (point-to-point / point-to-plane, robust kernel, covariance, degeneracy), GICP, NDT, feature + RANSAC, branch-and-bound, 3D Hough |
| [`tpt-percept-slam`](crates/tpt-percept-slam) | Visual and LiDAR odometry (scan-to-map), motion deskew, IMU preintegration, loop closure, Levenberg–Marquardt pose graph |
| [`tpt-percept-fusion`](crates/tpt-percept-fusion) | Temporal alignment, EKF/UKF, extrinsic calibration, covariance intersection, robust fusion |
| [`tpt-percept-map`](crates/tpt-percept-map) | Occupancy grids (2D/3D log-odds), TSDF with mesh extraction, signed distance fields, semantic maps |
| [`tpt-percept-verify`](crates/tpt-percept-verify) | proptest strategies, geometric invariant checks, Kani proof harnesses |
| [`tpt-percept-io`](crates/tpt-percept-io) | PCD, PLY, KITTI `.bin` and TUM trajectory readers and writers; sensor presets (std) |
| [`tpt-percept-cli`](crates/tpt-percept-cli) | `tpt-percept` command: `convert`, `downsample`, `register`, `odometry` |
| [`tpt-percept-py`](crates/tpt-percept-py) | Python bindings (PyO3, the only FFI crate): ICP, odometry, planes, file readers |
| [`tpt-percept-ros`](crates/tpt-percept-ros) | ROS 2 `PointCloud2` byte-layout conversion (pure Rust, no ROS dependency) |
| [`tpt-percept-wasm`](crates/tpt-percept-wasm) | WebAssembly bindings and a browser ICP demo |
| [`tpt-perception`](crates/tpt-perception) | Feature-gated umbrella crate re-exporting everything above, plus examples |

## Quick start

Add the umbrella crate, then run an example to see ICP align two scans:

```sh
cargo run -p tpt-perception --example icp_two_scans
```

The example builds a synthetic surface, transforms it, and recovers the
transform with both ICP variants. The core call looks like this:

```rust
use tpt_perception::cloud::PointCloud;
use tpt_perception::register::icp::{icp, IcpParams, IcpVariant};

fn align(source: &PointCloud, target: &PointCloud) {
    let params = IcpParams::auto(source, target).expect("non-empty clouds");
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let result = icp(
        source,
        target,
        None,                       // target normals: needed only for PointToPlane
        IcpVariant::PointToPoint,
        identity,
        [0.0; 3],
        &params,
    )
    .expect("ICP runs");
    println!("rmse {:.4} m, fitness {:.2}", result.rmse, result.fitness);
}
```

Read the file formats and the command-line tool directly:

```sh
cargo install --path crates/tpt-percept-cli
tpt-percept convert scan.pcd scan.ply
tpt-percept odometry ./kitti/sequence00 --out est.tum
```

## Examples

All run offline on synthetic data.

| Example | Shows |
|---------|-------|
| `icp_two_scans` | Point-to-point and point-to-plane ICP, auto gate, weak-direction check |
| `plane_segmentation` | Seeded RANSAC over a room with a floor and two walls |
| `kitti_lidar_odometry` | Scan-to-map odometry; pass a KITTI directory to run on real data |
| `pose_graph_loop_closure` | Drift correction around a loop |
| `ekf_imu_gps_fusion` | Extended Kalman filter with IMU control and GPS updates |
| `occupancy_from_scans` | 2-D occupancy grid from sensor sweeps |
| `tsdf_to_mesh` | TSDF fusion and marching-tetrahedra mesh extraction |

## Guides

- [Which algorithm should I use?](docs/which-algorithm.md): choosing a
  registration or odometry method from what you know about the scans.
- [Cookbook](docs/cookbook/README.md): task-oriented recipes for building a
  mesh, localisation, loop closure, odometry and checking a registration.
- [PCL and Open3D mapping](docs/pcl-open3d-mapping.md): mapping concepts from
  those libraries onto this one.
- [Browser demo](crates/tpt-percept-wasm/README.md): ICP running in
  WebAssembly.

## What is proven

Each claim names the mechanism that checks it.

| Property | Mechanism | Where |
|----------|-----------|-------|
| Voxel index safety and centre round-trip | Kani harness | `tpt-percept-verify/src/kani_proofs.rs` |
| Log-odds clamping stays in bounds | Kani harness | same |
| Kabsch alignment does not panic on degenerate input | Kani harness | same |
| Rigid transforms preserve pairwise distance | proptest | `tpt-percept-verify/src/properties.rs` |
| ICP residual never increases | proptest | same |
| EKF covariance stays positive semi-definite | proptest | same |
| Plane detection recovers synthetic planes | proptest | same |
| Pose graph is cycle-consistent after optimisation | proptest and unit test | `slam/src/pose_graph.rs` |
| Pose graph cost decreases under Levenberg–Marquardt | unit test | same |
| Degenerate and singular inputs return errors, not panics | unit tests | each crate |
| File readers reject malformed input | unit tests; fuzz targets in `fuzz/` (not yet run in CI) | `tpt-percept-io` |

Kani runs weekly in CI (`.github/workflows/kani.yml`). Run it locally with
`cargo kani -p tpt-percept-verify` after installing `cargo-kani`.

## Building and verifying

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo deny check
cargo build -p tpt-percept-core --no-default-features --features alloc   # no_std gate
cargo check --target wasm32-unknown-unknown --no-default-features --features alloc \
    -p tpt-percept-core -p tpt-percept-cloud -p tpt-percept-register -p tpt-percept-slam
cargo bench -p tpt-percept-register    # criterion benchmarks
```

The numeric substrate comes from the published `tpt-math-*` crates on
crates.io, so no sibling checkout is needed.

## Status

Phases 0 to 4 are complete. Phase 5 (stabilisation, ergonomics and adoption)
is in progress; see [todo.md](todo.md) for the itemised checklist and what
remains open.

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).
© TPT Solutions.
