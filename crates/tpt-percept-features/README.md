# tpt-percept-features

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-features.svg)](https://crates.io/crates/tpt-percept-features)
[![Documentation](https://docs.rs/tpt-percept-features/badge.svg)](https://docs.rs/tpt-percept-features)
[![License](https://img.shields.io/crates/l/tpt-percept-features.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Geometric feature extraction for `tpt-perception`.

## Features

- **Normals & curvature** — local-PCA over kNN neighbourhoods; viewpoint
  orientation. Curvature = λ_min/Σλ (0 on planes, ⅓ isotropic).
- **Harris 3D corners** — normal-variation second-moment matrix + classical
  Harris response with radius non-max suppression.
- **Edges** — tangent-plane angle-gap boundary detection; principal-curvature
  linear-structure (thin rod/wire) detection.
- **Plane segmentation** — deterministic RANSAC (seeded xorshift64*) with
  least-squares refinement, greedy multi-plane extraction.
- **Descriptors** — FPFH (3×11 histogram) and SHOT-352 (4×4×2 spatial × 11
  normal-angle bins, covariant local reference frame).
- **AI-native hooks** — `LearnedDescriptor` and `ScalarCost` traits with an
  optional analytic-gradient hook for learning-agent integration.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-features = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-features = "0.1"
```

## Usage Examples

### Normal Estimation

```rust
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

let cloud = PointCloud::from_points(vec![
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1],
]);

let kdtree = KdTree::new(&cloud);

// Estimate normals with 10 nearest neighbors
let normals = estimate_normals(&cloud, &kdtree, 10);

// Orient normals towards a viewpoint (e.g., sensor origin)
let oriented = orient_normals_towards(&cloud, &normals, &Point3D::origin());

// Access curvature (λ_min / Σλ)
for (normal, curvature) in normals.iter().zip(oriented.curvatures()) {
    println!("Normal: {:?}, Curvature: {}", normal, curvature);
}
```

### Harris 3D Corner Detection

```rust
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;

let cloud = PointCloud::from_points(/* ... */);
let kdtree = KdTree::new(&cloud);

let params = HarrisParams {
    radius: 0.5,
    k: 0.04,
    nms_radius: 0.3,
    min_response: 1e-4,
};

let corners = harris3d(&cloud, &kdtree, &params);
// corners: Vec<Corner> with position, response, scale
```

### Edge Detection

```rust
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;

let cloud = PointCloud::from_points(/* ... */);
let kdtree = KdTree::new(&cloud);

// Boundary edges (depth discontinuities)
let boundaries = detect_boundaries(&cloud, &kdtree, 0.5, 0.3);

// Crease edges (high curvature ridges)
let creases = detect_crease_edges(&cloud, &kdtree, 0.5, 0.1);
```

### Plane Segmentation (RANSAC)

```rust
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;

let cloud = PointCloud::from_points(/* ... */);
let kdtree = KdTree::new(&cloud);

let params = RansacParams {
    max_iterations: 1000,
    distance_threshold: 0.02,
    min_inliers: 100,
    seed: 42, // deterministic
};

let planes = segment_planes(&cloud, &kdtree, &params);
// planes: Vec<PlaneSegment> with plane equation, inliers, centroid
```

### FPFH Descriptors

```rust
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;

let cloud = PointCloud::from_points(/* ... */);
let kdtree = KdTree::new(&cloud);
let normals = estimate_normals(&cloud, &kdtree, 10);

let fpfh = compute_fpfh(&cloud, &kdtree, &normals, 0.5);
// fpfh: Vec<[f32; FPFH_DIM]> — 33-bin histogram per point
```

### SHOT Descriptors

```rust
use tpt_percept_features::prelude::*;
use tpt_percept_cloud::prelude::*;

let cloud = PointCloud::from_points(/* ... */);
let kdtree = KdTree::new(&cloud);
let normals = estimate_normals(&cloud, &kdtree, 10);

let shot = compute_shot(&cloud, &kdtree, &normals, 0.5);
// shot: Vec<[f32; SHOT_DIM]> — 352-bin descriptor per point
```

### Learned Descriptor Hooks (AI-Native)

```rust
use tpt_percept_features::learned::{LearnedDescriptor, ScalarCost, DescriptorContext};
use tpt_percept_cloud::prelude::*;
use tpt_percept_core::prelude::*;

// Implement for your learned model
struct MyDescriptor;
impl LearnedDescriptor for MyDescriptor {
    type Descriptor = Vec<f32>;
    
    fn compute(&self, ctx: &DescriptorContext, cloud: &PointCloud, idx: usize) -> Self::Descriptor {
        // Your neural network inference here
        vec![0.0; 128]
    }
    
    fn dim(&self) -> usize { 128 }
}

// Optional: analytic gradient for end-to-end learning
impl ScalarCost for MyDescriptor {
    fn cost(&self, desc1: &Self::Descriptor, desc2: &Self::Descriptor) -> f32 {
        // Differentiable cost (e.g., L2, cosine)
        desc1.iter().zip(desc2).map(|(a,b)| (a-b).powi(2)).sum()
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
| [`normal`] | PCA-based normal estimation, curvature, viewpoint orientation |
| [`harris3d`] | 3D Harris corner detection with NMS |
| [`edge`] | Boundary and crease edge detection |
| [`planes`] | Deterministic RANSAC plane segmentation |
| [`fpfh`] | Fast Point Feature Histograms (33-dim) |
| [`shot`] | Signature of Histograms of Orientations (352-dim) |
| [`learned`] | `LearnedDescriptor` and `ScalarCost` traits for AI integration |

## Conventions

- **Normals**: Oriented consistently (viewpoint or MST propagation)
- **Curvature**: Normalized λ_min/Σλ ∈ [0, 1/3]
- **RANSAC**: Seeded Xorshift64* for determinism
- **Descriptors**: Float32 histograms, L2-normalized where applicable
- **AI hooks**: Zero-cost traits, no_std compatible

## no_std Support

Build without `std`:

```bash
cargo build -p tpt-percept-features --no-default-features --features alloc
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
