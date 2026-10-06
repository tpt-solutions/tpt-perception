//! `tpt-percept-cloud` — point cloud data structures and processing.
//!
//! * **Container** — [`PointCloud`](cloud::PointCloud) (raw) and
//!   [`FrameCloud`](cloud::FrameCloud) (frame-tagged),
//!   with aabb/centroid/transform helpers.
//! * **Voxel grids** — sparse lattice hashing ([`voxel::voxelize`]) and
//!   memory-efficient downsampling ([`voxel::voxel_downsample`]).
//! * **Filters** — statistical outlier removal, radius filtering and
//!   pass-through cropping ([`filter`]).
//! * **Spatial indexing** — [`kdtree::KdTree`] (exact k-NN / radius),
//!   [`octree::Octree`] (hierarchical volume queries), [`rtree::RTree`]
//!   (static AABB index).
//! * **Streaming** — sliding windows and stage pipelines for real-time
//!   LiDAR ([`stream`]).
//!
//! The crate is `no_std + alloc`.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod cloud;
pub mod filter;
pub mod kdtree;
pub mod octree;
pub mod rtree;
pub mod stream;
pub mod voxel;

/// The types you almost always want in scope.
pub mod prelude {
    pub use crate::cloud::{FrameCloud, PointCloud};
    pub use crate::filter::{pass_through, radius_filter, statistical_outlier_removal, Axis};
    pub use crate::kdtree::KdTree;
    pub use crate::octree::Octree;
    pub use crate::rtree::{Aabb, RTree};
    pub use crate::stream::{CropStage, Pipeline, StreamStage, VoxelStage, WindowAggregator};
    pub use crate::voxel::{voxel_downsample, voxelize};
}
