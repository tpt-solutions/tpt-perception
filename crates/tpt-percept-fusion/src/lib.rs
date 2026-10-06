//! `tpt-percept-fusion` — multi-sensor fusion.
//!
//! * **Temporal alignment** — time-ordered pose timelines with bounded
//!   extrapolation and orthonormalised rotation interpolation; rolling
//!   latency estimation ([`temporal`]).
//! * **Filtering** — self-contained dense EKF (Joseph-form update, NIS
//!   output) and UKF (sigma-point transform, scaled weights) ([`ekf`]).
//!   These are the fusion crate's own substrate per the resolved
//!   `tpt-control` checkpoint in `todo.md`.
//! * **Extrinsic calibration** — Tsai–Lenz hand–eye from synchronized
//!   motion pairs ([`calibration`]).
//! * **Robust fusion** — covariance intersection (correlation-agnostic,
//!   consistent) with chi-square innovation gating and healthy-sensor
//!   fallback ([`robust`]).
//!
//! `no_std + alloc`.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod calibration;
pub mod ekf;
pub mod error;
pub mod robust;
pub mod temporal;

pub use error::FusionError;

/// The types you almost always want in scope.
pub mod prelude {
    pub use crate::calibration::{calibrate_extrinsic, Extrinsic, MotionPair};
    pub use crate::ekf::{Ekf, Ukf};
    pub use crate::error::FusionError;
    pub use crate::robust::{covariance_intersection, robust_fuse, Gaussian, InnovationGate};
    pub use crate::temporal::{LatencyEstimator, PoseTimeline, TimedPose};
}
