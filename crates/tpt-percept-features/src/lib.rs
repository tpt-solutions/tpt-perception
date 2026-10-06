//! `tpt-percept-features` — geometric feature extraction.
//!
//! * **Normals & curvature** — local PCA over kNN neighbourhoods
//!   ([`normal::estimate_normals`], [`normal::pca`]), orientation towards a
//!   viewpoint.
//! * **Corners** — 3-D Harris response over normal-variation second moments
//!   with non-max suppression ([`harris3d::harris3d`]).
//! * **Edges** — tangent-plane angle-gap boundary detection and
//!   principal-curvature crease edges ([`edge`]).
//! * **Planes** — deterministic RANSAC segmentation with least-squares
//!   refinement ([`planes::segment_planes`]).
//! * **Descriptors** — FPFH (33-bin histogram, [`fpfh::compute_fpfh`]) and
//!   SHOT-352 with a covariant local reference frame ([`shot::compute_shot`]).
//! * **AI-native hooks** — [`learned::LearnedDescriptor`] and
//!   [`learned::ScalarCost`] for learning-agent integration with optional
//!   analytic gradients.
//!
//! `no_std + alloc`.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod edge;
pub mod fpfh;
pub mod harris3d;
pub mod learned;
pub mod normal;
pub mod planes;
pub mod shot;

/// Feature extraction errors shared by all modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureError {
    /// A parameter violated its documented constraint.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FeatureError::InvalidParameter(what) => {
                write!(f, "invalid feature parameter: {what}")
            }
        }
    }
}

impl core::error::Error for FeatureError {}

/// The types you almost always want in scope.
pub mod prelude {
    pub use crate::edge::{detect_boundaries, detect_crease_edges};
    pub use crate::fpfh::{compute_fpfh, FPFH_DIM};
    pub use crate::harris3d::{harris3d, Corner, HarrisParams};
    pub use crate::learned::{contexts, DescriptorContext, LearnedDescriptor, ScalarCost};
    pub use crate::normal::{estimate_normals, orient_normals_towards, pca, Normal};
    pub use crate::planes::{segment_planes, Plane, PlaneSegment, RansacParams};
    pub use crate::shot::{compute_shot, SHOT_DIM};
    pub use crate::FeatureError;
}
