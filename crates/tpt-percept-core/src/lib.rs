//! `tpt-percept-core` — the foundation layer of `tpt-perception`.
//!
//! This crate provides the frame-safe geometric substrate every other
//! perception crate builds on:
//!
//! * **Coordinate frames as types** — [`frame::Frame`] markers
//!   ([`frame::World`], [`frame::Lidar<0>`], …) parameterise all geometry,
//!   so frame mismatches are compile errors ([`point::Point3D`],
//!   [`point::Vector3D`], [`iso::Isometry3`], [`iso::Rotation3`],
//!   [`twist::Twist3`]).
//! * **Rigid body transforms** — frame-typed rotations, isometries and
//!   twists with exact SE(3) exponential/logarithm maps
//!   ([`twist::Twist3::exp`], [`twist::Twist3::from_transform`]).
//! * **Small dense linear algebra** — symmetric 3×3 eigendecomposition
//!   ([`linalg3::sym_eigen3`]), Cramer solver, skew/outer products.
//! * **Closed-form alignment** — weighted Kabsch and Umeyama
//!   ([`align::kabsch`], [`align::umeyama`]).
//! * **Unit-safe quantities** — SI quantities at API boundaries via
//!   [`units`].
//!
//! # Conventions
//!
//! Active rotations, column vectors, right-handed frames, metres/radians/
//! seconds — matching `tpt-math-geometry` (see its crate documentation).
//!
//! # `no_std`
//!
//! The crate is `no_std + alloc` compatible: build with
//! `--no-default-features --features alloc`. Nothing in the geometric core
//! allocates.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod align;
pub mod error;
pub mod frame;
pub mod iso;
pub mod linalg3;
pub mod point;
pub mod rng;
pub mod twist;
pub mod units;

/// The traits and types you almost always want in scope.
pub mod prelude {
    pub use crate::align::{kabsch, kabsch_weighted, umeyama, RigidAlignment, SimilarityAlignment};
    pub use crate::error::{CoreError, CoreResult};
    pub use crate::frame::{Body, Camera, Frame, Imu, Lidar, Odometry, Radar, Sensor, World};
    pub use crate::iso::{Isometry3, Rotation3};
    pub use crate::point::{Point3D, Vector3D};
    pub use crate::twist::Twist3;
}
