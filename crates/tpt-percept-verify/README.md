# tpt-percept-verify

[![Crates.io](https://img.shields.io/crates/v/tpt-percept-verify.svg)](https://crates.io/crates/tpt-percept-verify)
[![Documentation](https://docs.rs/tpt-percept-verify/badge.svg)](https://docs.rs/tpt-percept-verify)
[![License](https://img.shields.io/crates/l/tpt-percept-verify.svg)](https://github.com/tpt-solutions/tpt-perception/blob/main/LICENSE-MIT)

Verification harnesses for tpt-perception (std-only; the algorithm crates
remain `no_std`).

## Features

- **proptest strategies** — valid points, exactly-orthonormal rotations,
  rigid transforms, point clouds, bounded noise profiles, PSD covariances.
- **Invariant checkers** — distance preservation, inverse roundtrips,
  monotone residual histories, covariance symmetry/PSD.
- **Kani harnesses** — `#[cfg(kani)]` bounded model checks: voxel-key
  safety and roundtrips, log-odds clamping bounds, Kabsch no-panic on
  degenerate sets. Run with `cargo kani -p tpt-percept-verify`.
- **Property tests** — the spec's statistical invariants: rigid transforms
  preserve distances, ICP residual monotonicity, plane detection on
  synthetic clouds (phase-1 integration), normal robustness, EKF
  covariance validity, pose graph cycle consistency.

Part of [tpt-perception](../../). Dual-licensed MIT OR Apache-2.0.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
tpt-percept-verify = "0.1"
```

## Usage Examples

### Proptest Strategies

```rust
use tpt_percept_verify::prelude::*;
use proptest::prelude::*;

// Generate valid rigid transforms
proptest! {
    #[test]
    fn test_transform_inverse_roundtrip(
        transform in rigid_transform(),
        point in finite_point()
    ) {
        let transformed = transform.transform_point(&point);
        let recovered = transform.inverse().transform_point(&transformed);
        prop_assert!((point.coords - recovered.coords).norm() < 1e-10);
    }
}

// Generate point clouds with bounded noise
proptest! {
    #[test]
    fn test_icp_monotonicity(
        cloud in point_cloud(10..100),
        noise in noise_profile(0.0..0.1)
    ) {
        let noisy = apply_noise(&cloud, &noise);
        let result = icp(&noisy, &cloud, None, IcpVariant::PointToPoint, Default::default());
        prop_assert!(result.converged);
        prop_assert!(result.residual_history.is_monotone_decreasing());
    }
}

// Generate PSD covariance matrices
proptest! {
    #[test]
    fn test_covariance_validity(cov in covariance3()) {
        prop_assert!(cov.is_symmetric());
        prop_assert!(cov.is_positive_semidefinite());
    }
}
```

### Invariant Checkers

```rust
use tpt_percept_verify::prelude::*;
use tpt_percept_core::prelude::*;

// Check distance preservation of a transform
let transform = Isometry3::<World, World>::from_translation([1.0, 2.0, 3.0]);
let points = vec![Point3D::new([0.0, 0.0, 0.0]), Point3D::new([1.0, 0.0, 0.0])];
let result = check_distance_preserving(&transform, &points);
assert!(result.is_ok());

// Check inverse roundtrip
let result = check_inverse_roundtrip(&transform);
assert!(result.is_ok());

// Check monotone decreasing residuals
let residuals = vec![1.0, 0.8, 0.6, 0.5, 0.5];
assert!(check_monotone_decreasing(&residuals).is_ok());

// Check covariance validity
let cov = Mat::<3,3>::identity() * 0.1;
assert!(check_covariance_valid(&cov).is_ok());
```

### Kani Proofs (Bounded Model Checking)

```rust
// Run with: cargo kani -p tpt-percept-verify

#[cfg(kani)]
mod kani_proofs {
    use tpt_percept_verify::kani_proofs::*;
    
    #[kani::proof]
    fn voxel_key_safety() { /* kani harness */ }
    
    #[kani::proof]
    fn log_odds_clamping_bounds() { /* ... */ }
    
    #[kani::proof]
    fn kabsch_degenerate_no_panic() { /* ... */ }
    
    #[kani::proof]
    fn frame_transform_roundtrip() { /* ... */ }
}
```

### Property Tests (Integration)

```rust
// These run with `cargo test -p tpt-percept-verify`

#[test]
fn rigid_transforms_preserve_distances() { /* ... */ }

#[test]
fn icp_residual_monotonicity() { /* ... */ }

#[test]
fn plane_detection_on_synthetic_clouds() { /* ... */ }

#[test]
fn normal_robustness() { /* ... */ }

#[test]
fn ekf_covariance_validity() { /* ... */ }

#[test]
fn pose_graph_cycle_consistency() { /* ... */ }
```

## Crate Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Enable `std` support (default, required) |

Note: This crate requires `std` for proptest and Kani support.

## Modules

| Module | Description |
|--------|-------------|
| [`strategy`] | Proptest strategies for geometric types |
| [`invariants`] | Invariant checker functions |
| [`kani_proofs`] | Kani bounded model checking harnesses |
| [`properties`] | Property-based integration tests |

## Conventions

- **Proptest**: Strategies generate valid geometric objects by construction
- **Kani**: `#[cfg(kani)]` gated, requires `cargo kani` toolchain
- **Invariants**: Pure functions returning `Result` for easy testing
- **Properties**: Full integration tests in `tests/` module

## Running Verification

```bash
# Proptest property tests
cargo test -p tpt-percept-verify

# Kani bounded model checking (requires cargo-kani)
cargo kani -p tpt-percept-verify

# All workspace tests
cargo test --workspace
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) OR [Apache-2.0](../LICENSE-APACHE).
© TPT Solutions.
