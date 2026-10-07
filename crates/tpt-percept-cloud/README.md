# tpt-percept-cloud

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-cloud.svg)](https://crates.io/crates/tpt-percept-cloud)
[![Documentation](https://docs.rs/tpt-percept-cloud/badge.svg)](https://docs.rs/tpt-percept-cloud)
[![License](https://img.shields.io/crates/l/tpt-percept-cloud.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Point cloud data structures and processing for `tpt-perception`.

## Features

- **Containers** — `PointCloud` (raw) and `FrameCloud<F>` (frame-tagged so
  clouds from different sensors cannot be mixed without an explicit
  `Isometry3<F1, F2>`).
- **Voxel grids** — sparse lattice hashing with saturating, panic-free key
  computation; voxel-grid downsampling by centroid.
- **Filters** — statistical outlier removal (kNN mean-distance threshold),
  radius outlier removal, pass-through cropping. All return survivor
  indices so per-point attributes filter alongside.
- **Spatial indexing** — balanced k-d tree (exact k-NN and radius queries,
  squared-distance API), region octree (ball and box queries), static
  R-tree (STR-style bulk load over AABBs).
- **Streaming** — sliding scan windows and composable filter stages for
  real-time LiDAR pipelines.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-cloud = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-cloud = "0.1"
```

## Usage Examples

### Frame-Tagged Point Clouds

```rust
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::frame::Lidar;

// Raw point cloud (no frame)
let mut cloud = PointCloud::new();
cloud.push([1.0, 2.0, 3.0]);
cloud.push([4.0, 5.0, 6.0]);

// Frame-tagged cloud — prevents accidental mixing
let mut frame_cloud = FrameCloud::<Lidar<0>>::new();
frame_cloud.push([1.0, 2.0, 3.0]);

// Transform to world frame using a core isometry
use tpt_percept_core::prelude::*;
use tpt_percept_core::frame::World;

let lidar_to_world = Isometry3::<Lidar<0>, World>::identity();
let world_cloud: PointCloud = frame_cloud.transform_into(&lidar_to_world).into_cloud();
```

### Voxel Grid Downsampling

```rust
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

let cloud = PointCloud::from_points(vec![
    [0.0, 0.0, 0.0],
    [0.01, 0.0, 0.0],
    [0.02, 0.0, 0.0],
    [1.0, 1.0, 1.0],
]);

// Downsample to 0.05m voxels (centroid of points in each voxel)
let downsampled = voxel_downsample(&cloud, 0.05);
assert_eq!(downsampled.len(), 2); // two voxel centroids
```

### Statistical Outlier Removal

```rust
use tpt_percept_cloud::prelude::*;

let mut cloud = PointCloud::from_points(vec![
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [100.0, 0.0, 0.0], // outlier
]);

// Remove points whose mean kNN distance > 1.0 std devs from mean
let (filtered, indices) = statistical_outlier_removal(&cloud, 3, 1.0);
assert_eq!(filtered.len(), 3); // outlier removed
assert_eq!(indices, vec![0, 1, 2]); // survivor indices
```

### k-d Tree Nearest Neighbor Search

```rust
use tpt_percept_cloud::prelude::*;

let cloud = PointCloud::from_points(vec![
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [10.0, 10.0, 10.0],
]);

let kdtree = KdTree::new(&cloud);

// k-NN search
let (indices, sq_dists) = kdtree.knn(&[0.1, 0.1, 0.0], 2);
assert_eq!(indices, vec![0, 1]);

// Radius search
let indices = kdtree.radius(&[0.0, 0.0, 0.0], 1.5);
assert_eq!(indices.len(), 3);
```

### Streaming Pipeline

```rust
use tpt_percept_cloud::prelude::*;

// Build a processing pipeline
let pipeline = Pipeline::new()
    .add_stage(VoxelStage::new(0.1))     // voxel downsample
    .add_stage(CropStage::new(Axis::Z, -1.0, 5.0)); // keep Z in [-1, 5]

// Process a stream of scans
let mut window = WindowAggregator::new(5); // sliding window of 5 scans

for scan in lidar_stream {
    window.push(scan);
    if let Some(aggregated) = window.get() {
        let processed = pipeline.run(aggregated);
        // ... use processed cloud
    }
}
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default) |
| `alloc` | Enable `alloc` only (no `std`) |

## Modules

| Module | Description |
|--------|-------------|
| [`cloud`] | `PointCloud`, `FrameCloud<F>` containers with transform/AABB/centroid helpers |
| [`filter`] | Statistical outlier, radius, and pass-through filters |
| [`kdtree`] | Balanced k-d tree for exact k-NN and radius queries |
| [`octree`] | Region octree for hierarchical volume queries |
| [`rtree`] | Static R-tree (STR bulk load) for AABB indexing |
| [`stream`] | Sliding windows and composable pipeline stages |
| [`voxel`] | Sparse voxel hashing and voxel-grid downsampling |

## Conventions

- **Coordinates**: Right-handed, metres
- **Filtering**: Returns survivor indices for attribute preservation
- **Spatial indices**: Squared distances for performance
- **Streaming**: Zero-copy where possible, `alloc` only

## no_std Support

Build without `std`:

```bash
cargo build -p tpt-percept-cloud --no-default-features --features alloc
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
