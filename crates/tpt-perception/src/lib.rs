//! `tpt-perception` — the umbrella crate for tpt-perception.
//!
//! Feature-gated re-exports of the full perception stack. Each cargo
//! feature maps to one sub-crate; `default` enables everything except the
//! verification harnesses, `full` includes those too.
//!
//! ```toml
//! [dependencies]
//! tpt-perception = { version = "0.1", default-features = false, features = ["register", "slam"] }
//! ```

#![forbid(unsafe_code)]

#[cfg(feature = "core")]
pub use tpt_percept_core;

/// Point cloud containers, voxel grids, filters, spatial indices, streaming.
#[cfg(feature = "cloud")]
pub mod cloud {
    pub use tpt_percept_cloud::prelude::*;
    pub use tpt_percept_cloud::{filter, kdtree, octree, rtree, stream, voxel};
}

/// Normals, corners, edges, plane segmentation, FPFH/SHOT descriptors,
/// learned-descriptor hooks.
#[cfg(feature = "features")]
pub mod features {
    pub use tpt_percept_features::prelude::*;
    pub use tpt_percept_features::{edge, fpfh, harris3d, learned, normal, planes, shot};
}

/// ICP, NDT, feature-based and global registration, loop closure Hough.
#[cfg(feature = "register")]
pub mod register {
    pub use tpt_percept_register::prelude::*;
    pub use tpt_percept_register::{bnb, error, feature_match, hough, icp, ndt};
}

/// Visual/LiDAR odometry, place recognition, pose graph optimization.
#[cfg(feature = "slam")]
pub mod slam {
    pub use tpt_percept_slam::prelude::*;
    pub use tpt_percept_slam::{
        error, lidar_odometry, loop_closure, pose_graph, vo_mono, vo_stereo,
    };
}

/// Temporal alignment, EKF/UKF, extrinsic calibration, robust fusion.
#[cfg(feature = "fusion")]
pub mod fusion {
    pub use tpt_percept_fusion::prelude::*;
    pub use tpt_percept_fusion::{calibration, ekf, error, robust, temporal};
}

/// Occupancy grids, TSDF, SDF, semantic maps.
#[cfg(feature = "map")]
pub mod map {
    pub use tpt_percept_map::prelude::*;
    pub use tpt_percept_map::{error, occupancy, sdf, semantic, tsdf};
}

/// Verification harnesses (opt-in via the `verify` or `full` feature).
#[cfg(feature = "verify")]
pub mod verify {
    pub use tpt_percept_verify::prelude::*;
    pub use tpt_percept_verify::{invariants, kani_proofs, properties, strategy};
}

/// Everything the default feature set exposes, re-exported flat.
pub mod prelude {
    #[cfg(feature = "cloud")]
    pub use tpt_percept_cloud::prelude::*;
    #[cfg(feature = "core")]
    pub use tpt_percept_core::prelude::*;
    #[cfg(feature = "features")]
    pub use tpt_percept_features::prelude::*;
    #[cfg(feature = "fusion")]
    pub use tpt_percept_fusion::prelude::*;
    #[cfg(feature = "map")]
    pub use tpt_percept_map::prelude::*;
    #[cfg(feature = "register")]
    pub use tpt_percept_register::prelude::*;
    #[cfg(feature = "slam")]
    pub use tpt_percept_slam::prelude::*;
    #[cfg(feature = "verify")]
    pub use tpt_percept_verify::prelude::*;
}

#[cfg(all(test, feature = "cloud", feature = "register", feature = "map"))]
mod tests {
    /// The default feature set must expose the frame-typed API end to end:
    /// a scan registered into a map and written into an occupancy grid.
    #[test]
    fn stack_composes() {
        use crate::prelude::*;

        // Frames: a scan captured by lidar 0, placed into the world.
        let scan = PointCloud::from_points(vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]);
        let lidar_to_world =
            Isometry3::<Lidar<0>, World>::from_translation([1.0, 2.0, 0.5]).unwrap();
        let world_scan: PointCloud = {
            use tpt_percept_cloud::cloud::FrameCloud;
            FrameCloud::<Lidar<0>>::from_cloud(scan)
                .transform_into(&lidar_to_world)
                .into_cloud()
        };

        // Register the moved scan against a synthetic reference (identity
        // motion is the truth).
        let reference = world_scan.clone();
        let result = crate::register::icp::icp(
            &world_scan,
            &reference,
            None,
            crate::register::icp::IcpVariant::PointToPoint,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &crate::register::icp::IcpParams {
                max_correspondence_distance: 1.0,
                min_correspondences: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);

        // Fuse the hits into an occupancy grid.
        let mut grid = crate::map::occupancy::OccupancyGrid2D::new(
            [0.0; 2],
            crate::map::occupancy::OccupancyParams::default(),
        );
        for p in world_scan.points() {
            grid.insert_ray([0.0, 0.0], [p[0], p[1]]);
        }
        assert!(!grid.is_empty());
    }
}
