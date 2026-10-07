# tpt-percept-map

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-map.svg)](https://crates.io/crates/tpt-percept-map)
[![Documentation](https://docs.rs/tpt-percept-map/badge.svg)](https://docs.rs/tpt-percept-map)
[![License](https://img.shields.io/crates/l/tpt-percept-map.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Spatial mapping for `tpt-perception`.

## Features

- **Occupancy grids** — 2-D log-odds (Bresenham) and 3-D log-odds
  (Amanatides–Woo) with clamped, bounded updates; probability queries.
- **TSDF** — truncated signed distance fusion with inverse-range weighting,
  weighted merging, trilinear sampling and zero-crossing surface extraction
  (13-neighbour, robust to diagonal ray steps).
- **SDF** — dense trilinear signed distance grids over bounded boxes,
  numeric gradients for trajectory optimization, margin collision checks.
- **Semantic maps** — labeled point clouds with per-class counts, centroids
  and bounding boxes; class/confidence filters.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-map = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-map = "0.1"
```

## Usage Examples

### 2D Occupancy Grid

```rust
use tpt_percept_map::prelude::*;
use tpt_percept_core::prelude::*;

let params = OccupancyParams {
    resolution: 0.05,      // 5cm cells
    prob_hit: 0.7,
    prob_miss: 0.4,
    clamp_min: 0.12,
    clamp_max: 0.97,
};

let mut grid = OccupancyGrid2D::new([0.0, 0.0], params);

// Insert a ray from origin to point (Bresenham)
grid.insert_ray([0.0, 0.0], [5.0, 3.0]);

// Query occupancy probability
let prob = grid.probability_at([2.5, 1.5]);

// Convert to probability grid for visualization
let prob_grid = grid.to_probability_grid();
```

### 3D Occupancy Grid

```rust
use tpt_percept_map::prelude::*;
use tpt_percept_core::prelude::*;

let params = OccupancyParams {
    resolution: 0.1,
    prob_hit: 0.65,
    prob_miss: 0.35,
    clamp_min: 0.1,
    clamp_max: 0.9,
};

let mut grid = OccupancyGrid3D::new([0.0, 0.0, 0.0], params);

// Amanatides-Woo 3D ray casting
grid.insert_ray([0.0, 0.0, 0.0], [10.0, 5.0, 2.0]);

// Batch insert from point cloud
use tpt_percept_cloud::prelude::*;
let cloud = PointCloud::from_points(/* ... */);
for p in cloud.points() {
    grid.insert_ray([0.0, 0.0, 0.0], p);
}
```

### TSDF Fusion

```rust
use tpt_percept_map::prelude::*;
use tpt_percept_core::prelude::*;

let params = TsdfParams {
    voxel_size: 0.05,
    truncation: 0.15,  // 3x voxel size
    max_weight: 100,
};

let mut tsdf = TsdfMap::new(params);

// Fuse a depth frame (point cloud + pose)
let pose = Isometry3::<Camera, World>::from_translation([0.0, 0.0, 0.0]);
tsdf.fuse_cloud(&cloud, &pose, 0.5); // weight = 0.5

// Trilinear sample at world position
let sdf_value = tsdf.sample(&[1.0, 2.0, 0.5]);

// Extract zero-crossing surface (marching cubes style)
let mesh = tsdf.extract_surface(/* bounds */);
```

### Signed Distance Field (SDF)

```rust
use tpt_percept_map::prelude::*;
use tpt_percept_core::prelude::*;

let bounds = [[-10.0, 10.0], [-10.0, 10.0], [-2.0, 2.0]];
let resolution = 0.1;
let mut sdf = SignedDistanceField::new(bounds, resolution);

// Populate from TSDF or point cloud
sdf.populate_from_tsdf(&tsdf);

// Query distance and gradient
let (dist, grad) = sdf.distance_and_gradient(&[1.5, -0.5, 0.0]);

// Margin collision check (for trajectory optimization)
let is_clear = sdf.check_margin(&trajectory_points, 0.5); // 0.5m margin
```

### Semantic Maps

```rust
use tpt_percept_map::prelude::*;
use tpt_percept_cloud::prelude::*;

let mut semantic = SemanticCloud::new();

// Add labeled points (class_id, confidence)
semantic.push(LabeledPoint {
    point: [1.0, 2.0, 0.0],
    class_id: 1,      // e.g., "car"
    confidence: 0.95,
});

// Get per-class statistics
let stats = semantic.class_statistics(1);
// stats: ClassStatistics { count, centroid, bbox_min, bbox_max }

// Filter by class
let cars = semantic.filter_by_class(1);

// Filter by confidence
let high_conf = semantic.filter_by_confidence(0.8);
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default) |
| `alloc` | Enable `alloc` only (no `std`) |

## Modules

| Module | Description |
|--------|-------------|
| [`occupancy`] | 2D/3D log-odds grids, Bresenham/Amanatides-Woo ray casting |
| [`tsdf`] | Truncated signed distance fusion, surface extraction |
| [`sdf`] | Dense trilinear SDF, gradients, margin collision checks |
| [`semantic`] | Labeled point clouds, class statistics, filtering |
| [`error`] | `MapError` error types |

## Conventions

- **Log-odds**: Clamped to prevent numerical saturation
- **TSDF**: Truncation = 3× voxel size (standard)
- **SDF**: Positive outside, negative inside (standard)
- **Semantic**: Class IDs are application-defined `u32`
- **Coordinates**: Right-handed, metres, world frame

## no_std Support

Build without `std`:

```bash
cargo build -p tpt-percept-map --no-default-features --features alloc
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
