# tpt-percept-core

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-core.svg)](https://crates.io/crates/tpt-percept-core)
[![Documentation](https://docs.rs/tpt-percept-core/badge.svg)](https://docs.rs/tpt-percept-core)
[![License](https://img.shields.io/crates/l/tpt-percept-core.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Foundation layer of `tpt-perception`: the frame-safe geometric substrate
every other perception crate builds on.

## Features

- **Coordinate frames as types** — `Point3D<Lidar<0>>`, `Isometry3<Lidar<0>, World>`:
  frame mismatches are compile errors, not runtime bugs.
- **Rigid body transforms** — `Rotation3<From, To>`, `Isometry3<From, To>`,
  `Twist3<F>` with exact SE(3) exponential/logarithm maps.
- **Small dense linear algebra** — symmetric 3×3 eigendecomposition (cyclic
  Jacobi), Cramer solver, skew/outer products.
- **Closed-form alignment** — weighted Kabsch and Umeyama (rigid + similarity).
- **Unit-safe quantities** — SI types via `tpt-math-units` at API boundaries.

`no_std` compatible (pure core allocates nothing). Conventions: active
rotations, column vectors, right-handed frames, metres/radians/seconds.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tpt-percept-core = { version = "0.1", default-features = false, features = ["alloc"] }
# or with std:
tpt-percept-core = "0.1"
```

## Usage Examples

### Frame-Typed Points and Transforms

```rust
use tpt_percept_core::prelude::*;

// Define frames (zero-cost phantom types)
struct Camera;
struct World;

// A point in the camera frame
let p_cam: Point3D<Camera> = Point3D::new([1.0, 2.0, 3.0]);

// Transform from camera to world
let cam_to_world = Isometry3::<Camera, World>::from_rotation_translation(
    Rotation3::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
    [10.0, 0.0, 0.0]
);

// Transform point to world frame — frame-safe!
let p_world: Point3D<World> = cam_to_world.transform_point(&p_cam);

// This would be a COMPILE ERROR:
// let p_wrong: Point3D<World> = p_cam; // mismatched frames
```

### SE(3) Exponential and Logarithm Maps

```rust
use tpt_percept_core::prelude::*;

let twist = Twist3::<World>::new([0.1, 0.0, 0.0], [0.0, 0.0, 0.5]); // v, ω
let transform = twist.exp(); // Isometry3<World, World>
let recovered = Twist3::from_transform(&transform);
assert!((twist.v - recovered.v).norm() < 1e-10);
```

### Closed-Form Alignment (Kabsch / Umeyama)

```rust
use tpt_percept_core::prelude::*;

let src = vec![Point3D::new([0.0, 0.0, 0.0]), Point3D::new([1.0, 0.0, 0.0])];
let dst = vec![Point3D::new([1.0, 2.0, 0.0]), Point3D::new([1.0, 3.0, 0.0])];

let RigidAlignment { rotation, translation } = kabsch(&src, &dst).unwrap();
// rotation: Rotation3, translation: Vector3D
```

### Linear Algebra Utilities

```rust
use tpt_percept_core::linalg3::{sym_eigen3, cramer_solve};

// Symmetric eigendecomposition (cyclic Jacobi)
let a = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
let (eigenvalues, eigenvectors) = sym_eigen3(a);

// Cramer's rule for 3×3
let a = [[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]];
let b = [6.0, 6.0, 6.0];
let x = cramer_solve(a, b); // [1.0, 1.0, 1.0]
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default) |
| `alloc` | Enable `alloc` only (no `std`) — heap-free core |

## Modules

| Module | Description |
|--------|-------------|
| [`align`] | Kabsch and Umeyama alignment algorithms |
| [`error`] | Error types (`CoreError`, `CoreResult`) |
| [`frame`] | Frame marker types (`World`, `Lidar<N>`, `Camera`, `Imu`, etc.) |
| [`iso`] | `Rotation3<F, T>`, `Isometry3<F, T>` — frame-typed transforms |
| [`linalg3`] | 3×3 linear algebra (eigendecomp, Cramer, skew/outer) |
| [`point`] | `Point3D<F>`, `Vector3D<F>` — frame-typed points/vectors |
| [`rng`] | Deterministic Xorshift64* RNG for reproducible algorithms |
| [`twist`] | `Twist3<F>` with SE(3) exp/log maps |
| [`units`] | SI unit wrappers via `tpt-math-units` |

## Conventions

- **Rotations**: Active (rotate vector in fixed frame), column-vector convention
- **Coordinate system**: Right-handed, Z-up (common in robotics)
- **Units**: Metres, radians, seconds at API boundaries
- **Frame composition**: `Isometry3<A, B> * Isometry3<B, C> = Isometry3<A, C>`

## no_std Support

The geometric core is completely allocation-free. Build with:

```bash
cargo build -p tpt-percept-core --no-default-features --features alloc
```

Only the `alloc` feature enables `Vec`, `Box`, etc. for convenience types.

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
