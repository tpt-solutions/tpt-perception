//! `tpt-percept-slam` — SLAM: odometry, loop closure and pose graph
//! optimization.
//!
//! * **Visual odometry (monocular)** — normalized 8-point essential matrix
//!   with RANSAC and cheirality-checked decomposition; motion up to scale
//!   ([`vo_mono`]).
//! * **Visual odometry (stereo)** — disparity triangulation + metric
//!   3-D-3-D RANSAC/Kabsch alignment ([`vo_stereo`]).
//! * **LiDAR odometry** — point-to-plane ICP scan matching with
//!   constant-velocity prediction ([`lidar_odometry`]).
//! * **Loop closure** — Scan Context place recognition (ring-key screening,
//!   azimuth-shift matching) ([`loop_closure`]).
//! * **Pose graph** — dense Gauss–Newton over SE(3) with numeric Jacobians
//!   and a Huber robustifier ([`pose_graph`]; large-scale sparse solving is
//!   the open `tpt-systems-optimisation` checkpoint).
//!
//! `no_std + alloc`.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod error;
pub mod lidar_odometry;
pub mod loop_closure;
pub mod pose_graph;
pub mod vo_mono;
pub mod vo_stereo;

pub use error::SlamError;

/// The types you almost always want in scope.
pub mod prelude {
    pub use crate::error::SlamError;
    pub use crate::lidar_odometry::{LidarOdometry, LidarOdometryParams, OdometryStep};
    pub use crate::loop_closure::{detect_loop, LoopCandidate, ScanContext, ScanContextParams};
    pub use crate::pose_graph::{Pose, PoseEdge, PoseGraph, PoseGraphParams};
    pub use crate::vo_mono::{
        estimate_motion_mono, MonoMotion, MonoParams, NormalizedCorrespondence,
    };
    pub use crate::vo_stereo::{
        estimate_motion_stereo, triangulate_stereo, StereoMatch, StereoMotion, StereoParams,
    };
}
