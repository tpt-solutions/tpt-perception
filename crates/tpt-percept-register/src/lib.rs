//! `tpt-percept-register` — point cloud registration and alignment.
//!
//! * **ICP** — point-to-point (Kabsch) and point-to-plane (Chen & Medioni)
//!   variants with shared convergence contract ([`icp`]).
//! * **NDT** — voxel-Gaussian map of the target + Gauss–Newton registration
//!   on squared Mahalanobis residuals ([`ndt`]).
//! * **Feature-based** — descriptor ratio matching + RANSAC/Kabsch initial
//!   pose estimation ([`feature_match`]).
//! * **Global registration** — branch-and-bound over translation + yaw with
//!   a valid lower bound on trimmed residuals ([`bnb`]).
//! * **Loop closure** — yaw Hough voting over normal-azimuth pairs with
//!   translation RANSAC ([`hough`]).
//!
//! `no_std + alloc`.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod bnb;
pub mod error;
pub use error::RegistrationError;
pub mod feature_match;
pub mod hough;
pub mod icp;
pub mod ndt;

/// The types you almost always want in scope.
pub mod prelude {
    pub use crate::bnb::{register_bnb, BnbParams, BnbResult};
    pub use crate::error::RegistrationError;
    pub use crate::feature_match::{
        match_descriptors, register_features, FeatureMatchParams, FeatureRegistration, Match,
    };
    pub use crate::hough::{
        extract_landmarks, hough_loop_closure, HoughParams, Landmark, LoopClosureCandidate,
    };
    pub use crate::icp::{icp, IcpParams, IcpResult, IcpVariant};
    pub use crate::ndt::{build_ndt_map, ndt_register, NdtMap, NdtMapParams, NdtParams, NdtResult};
}
