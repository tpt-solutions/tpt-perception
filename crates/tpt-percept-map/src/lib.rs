//! `tpt-percept-map` — spatial mapping.
//!
//! * **Occupancy grids** — 2-D (Bresenham) and 3-D (Amanatides–Woo) log-odds
//!   grids with clamped, bounded updates ([`occupancy`]).
//! * **TSDF** — truncated signed distance fusion with weighted merging and
//!   zero-crossing surface extraction ([`tsdf`]).
//! * **SDF** — dense trilinear distance grids over bounded boxes with
//!   gradient queries and margin collision checks ([`sdf`]).
//! * **Semantic maps** — labeled point clouds with per-class statistics
//!   ([`semantic`]).
//!
//! `no_std + alloc`.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod error;
pub mod occupancy;
pub mod sdf;
pub mod semantic;
pub mod tsdf;

pub use error::MapError;

/// The types you almost always want in scope.
pub mod prelude {
    pub use crate::error::MapError;
    pub use crate::occupancy::{
        clamp_log_odds, log_odds_to_probability, OccupancyGrid2D, OccupancyGrid3D, OccupancyParams,
    };
    pub use crate::sdf::SignedDistanceField;
    pub use crate::semantic::{ClassStatistics, LabeledPoint, SemanticCloud};
    pub use crate::tsdf::{TsdfMap, TsdfParams, TsdfVoxel};
}
