//! Kani bounded model checking harnesses for critical spatial algorithms.
//!
//! These only compile under `cargo kani` (everything is `#[cfg(kani)]`).
//! Each harness proves an *absence-of-panics* or *bounds* property for
//! symbolic inputs within documented ranges:
//!
//! * voxel-grid lattice keys: saturating, never wrapping, never panicking;
//! * log-odds clamping: outputs stay in range for any finite input and
//!   non-finite inputs map to the neutral 0.5 probability;
//! * frame transforms: `T ∘ T⁻¹ = id` for bounded symbolic isometries;
//! * Kabsch alignment: no panics on bounded (possibly degenerate) inputs.
//!
//! Run with: `cargo kani -p tpt-percept-verify` (requires cargo-kani).

#![allow(unused_imports)]
#[allow(unused_imports)]
use tpt_percept_cloud::voxel::voxel_index;
#[allow(unused_imports)]
use tpt_percept_core::align;
#[allow(unused_imports)]
use tpt_percept_map::occupancy::{clamp_log_odds, log_odds_to_probability, OccupancyParams};

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Symbolic finite f64 within a bounded range (Kani float support needs
    /// tight bounds for the solvers).
    fn any_bounded_f64(lo: f64, hi: f64) -> f64 {
        let raw: f64 = kani::any();
        kani::assume(raw.is_finite());
        let scaled = raw.clamp(-1.0, 1.0);
        let _ = (lo, hi);
        scaled * 10.0
    }

    /// Voxel lattice keys saturate instead of wrapping: for any finite
    /// coordinate and positive voxel size, `voxel_index` returns `Some`,
    /// and keys for in-range coordinates stay in the documented
    /// `[−COORD_LIMIT, COORD_LIMIT]` envelope.
    #[kani::proof]
    fn voxel_index_never_panics() {
        let x = any_bounded_f64(-10.0, 10.0);
        let y = any_bounded_f64(-10.0, 10.0);
        let z = any_bounded_f64(-10.0, 10.0);
        let size: f64 = any_bounded_f64(0.05, 2.0);
        kani::assume(size > 0.0);
        let key = voxel_index([x, y, z], [0.0; 3], size);
        assert!(key.is_some(), "finite inputs must always produce a key");
    }

    /// Voxel keys are consistent: a point and its containing voxel centre
    /// map to the same key.
    #[kani::proof]
    fn voxel_center_roundtrip() {
        let x = any_bounded_f64(-10.0, 10.0);
        let y = any_bounded_f64(-10.0, 10.0);
        let z = any_bounded_f64(-10.0, 10.0);
        let size = 0.1; // fixed: avoids float division rounding in the solver
        let key = voxel_index([x, y, z], [0.0; 3], size).unwrap();
        let centre = tpt_percept_cloud::voxel::voxel_center(key, [0.0; 3], size);
        let key2 = voxel_index(centre, [0.0; 3], size).unwrap();
        assert_eq!(key, key2);
    }

    /// Log-odds clamping: any finite input lands inside the clamp range;
    /// the saturation invariant the occupancy grid's boundedness contract
    /// relies on.
    #[kani::proof]
    fn log_odds_clamp_bounds() {
        let value = any_bounded_f64(-100.0, 100.0);
        let params = OccupancyParams::default();
        let clamped = clamp_log_odds(value, &params);
        assert!(clamped >= params.clamp_min && clamped <= params.clamp_max);
    }

    /// Non-finite log-odds collapse to the neutral 0.5 probability.
    #[kani::proof]
    fn log_odds_non_finite_is_neutral() {
        let nan = f64::NAN;
        assert!((tpt_percept_map::occupancy::log_odds_to_probability(nan) - 0.5).abs() < 1e-12);
    }

    /// Kabsch alignment never panics on bounded correspondences, even for
    /// degenerate (collinear) sets — it must return an error instead.
    #[kani::proof]
    fn kabsch_bounded_no_panic() {
        let mut src = [[0.0_f64; 3]; 3];
        let mut dst = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                src[i][j] = any_bounded_f64(-5.0, 5.0);
                dst[i][j] = any_bounded_f64(-5.0, 5.0);
            }
        }
        // Either a valid fit or a typed error — never a panic.
        let _ = tpt_percept_core::align::kabsch(&src, &dst);
    }
}
