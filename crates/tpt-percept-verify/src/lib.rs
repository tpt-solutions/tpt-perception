//! `tpt-percept-verify` — verification harnesses for tpt-perception.
//!
//! * **proptest strategies** — valid points, exact-rotation transforms,
//!   point clouds, noise profiles and PSD covariances ([`strategy`]).
//! * **Invariant checkers** — distance preservation, inverse roundtrips,
//!   monotone residual histories, covariance validity ([`invariants`]).
//! * **Kani harnesses** — bounded model checking of voxel indexing, log-odds
//!   clamping, frame-transform roundtrips and Kabsch robustness
//!   ([`kani_proofs`]; run with `cargo kani`).
//! * **Property tests** — the phase integration checks from `todo.md`
//!   (plane detection on synthetic clouds, ICP residual monotonicity, pose
//!   graph consistency, EKF covariance contraction) live in this crate's
//!   test modules.

#![forbid(unsafe_code)]

pub mod invariants;
pub mod kani_proofs;
pub mod properties;
pub mod strategy;

/// The strategies and checkers you almost always want in scope.
pub mod prelude {
    pub use crate::invariants::{
        check_covariance_valid, check_distance_preserving, check_inverse_roundtrip,
        check_monotone_decreasing, residual_summary,
    };
    pub use crate::strategy::{
        apply_rigid, axis_angle_matrix, covariance3, finite_point, noise_profile, point_cloud,
        rigid_transform, rotation,
    };
}
