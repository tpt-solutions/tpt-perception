//! Loop closure detection: Scan Context-style place recognition.
//!
//! Each scan is summarised as a *scan context*: a polar-grid descriptor
//! (azimuth sectors × radial rings, cell values = maximum height in the
//! sector). The descriptor's row means form a *ring key* used for fast
//! candidate retrieval; candidate matches are refined with a circular shift
//! search over azimuth (which resolves the scanner's yaw ambiguity) and
//! finally verified geometrically through yaw-Hough loop closure.

use alloc::vec;
use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::SlamError;
use tpt_percept_cloud::cloud::PointCloud;

/// Scan Context parameters.
#[derive(Clone, Copy, Debug)]
pub struct ScanContextParams {
    /// Azimuth sectors (columns of the descriptor).
    pub sectors: usize,
    /// Radial rings (rows of the descriptor).
    pub rings: usize,
    /// Maximum radius encoded (metres).
    pub max_radius: f64,
    /// Height encoding: cells store the maximum z inside, clamped to
    /// `[-z_min, z_max]` and normalised to [0, 1].
    pub z_min: f64,
    /// Upper height clamp for cell encoding (metres).
    pub z_max: f64,
}

impl Default for ScanContextParams {
    fn default() -> Self {
        ScanContextParams {
            sectors: 60,
            rings: 20,
            max_radius: 20.0,
            z_min: -2.0,
            z_max: 2.0,
        }
    }
}

/// A computed scan context descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanContext {
    descriptor: Vec<Vec<f64>>,
    ring_key: Vec<f64>,
}

impl ScanContext {
    /// Encodes a scan (scan-frame points, metres).
    pub fn encode(cloud: &PointCloud, params: &ScanContextParams) -> Result<Self, SlamError> {
        if params.sectors == 0 || params.rings == 0 {
            return Err(SlamError::InvalidParameter("sectors and rings must be > 0"));
        }
        if !(params.max_radius.is_finite() && params.max_radius > 0.0) {
            return Err(SlamError::InvalidParameter("max_radius must be > 0"));
        }
        let mut descriptor = vec![vec![f64::NEG_INFINITY; params.sectors]; params.rings];
        let ring_width = params.max_radius / params.rings as f64;
        for &p in cloud.points() {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            if !(r.is_finite() && r <= params.max_radius) {
                continue;
            }
            let ring = ((r / ring_width) as usize).min(params.rings - 1);
            let az = p[1].atan2(p[0]); // (-π, π]
            let sector = (((az + core::f64::consts::PI) / (2.0 * core::f64::consts::PI)
                * params.sectors as f64) as usize)
                .min(params.sectors - 1);
            let z = p[2].clamp(params.z_min, params.z_max);
            let norm = (z - params.z_min) / (params.z_max - params.z_min);
            if norm > descriptor[ring][sector] {
                descriptor[ring][sector] = norm;
            }
        }
        // Empty cells → 0.
        for row in &mut descriptor {
            for v in row.iter_mut() {
                if *v == f64::NEG_INFINITY {
                    *v = 0.0;
                }
            }
        }
        let ring_key = descriptor
            .iter()
            .map(|row| row.iter().sum::<f64>() / row.len() as f64)
            .collect();
        Ok(ScanContext {
            descriptor,
            ring_key,
        })
    }

    /// Fast retrieval key (row means).
    pub fn ring_key(&self) -> &[f64] {
        &self.ring_key
    }

    /// L1 distance between ring keys (fast screening).
    pub fn ring_key_distance(&self, other: &ScanContext) -> Result<f64, SlamError> {
        if self.ring_key.len() != other.ring_key.len() {
            return Err(SlamError::InvalidParameter(
                "descriptors built with different params",
            ));
        }
        Ok(self
            .ring_key
            .iter()
            .zip(&other.ring_key)
            .map(|(a, b)| (a - b).abs())
            .sum())
    }

    /// Best distance over all azimuth shifts (returns `(distance,
    /// shift_sectors)`).
    pub fn context_distance(&self, other: &ScanContext) -> Result<(f64, usize), SlamError> {
        if self.descriptor.len() != other.descriptor.len()
            || self
                .descriptor
                .first()
                .is_none_or(|r| r.len() != other.descriptor.first().map_or(0, |r2| r2.len()))
        {
            return Err(SlamError::InvalidParameter(
                "descriptors built with different params",
            ));
        }
        let rings = self.descriptor.len();
        let sectors = self.descriptor[0].len();
        let mut best = (f64::INFINITY, 0usize);
        for shift in 0..sectors {
            let mut d = 0.0;
            for r in 0..rings {
                for c in 0..sectors {
                    d += (self.descriptor[r][c] - other.descriptor[r][(c + shift) % sectors]).abs();
                }
            }
            if d < best.0 {
                best = (d, shift);
            }
        }
        Ok(best)
    }
}

/// A detected loop: which previous keyframe and how well it matches.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopCandidate {
    /// Index of the matching previous keyframe.
    pub keyframe: usize,
    /// Scan-context distance (L1 over the aligned descriptors; lower is
    /// better).
    pub distance: f64,
    /// Yaw alignment in sector shifts (for diagnostics).
    pub sector_shift: usize,
}

/// Place recognition over a keyframe history: finds the best previous
/// keyframe matching `query` among candidates whose ring-key distance is
/// below `ring_key_threshold`.
pub fn detect_loop(
    query: &ScanContext,
    keyframes: &[(usize, ScanContext)],
    ring_key_threshold: f64,
) -> Result<Option<LoopCandidate>, SlamError> {
    let mut best: Option<LoopCandidate> = None;
    for (idx, kf) in keyframes {
        let rk = query.ring_key_distance(kf)?;
        if rk > ring_key_threshold {
            continue;
        }
        let (d, shift) = query.context_distance(kf)?;
        if best.as_ref().is_none_or(|b: &LoopCandidate| d < b.distance) {
            best = Some(LoopCandidate {
                keyframe: *idx,
                distance: d,
                sector_shift: shift,
            });
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic "room": ground plane plus a wall at one azimuth.
    fn room(yaw_offset: f64) -> PointCloud {
        let mut c = PointCloud::new();
        let (s, co) = yaw_offset.sin_cos();
        let map = |p: [f64; 3]| [co * p[0] - s * p[1], s * p[0] + co * p[1], p[2]];
        // Ground.
        for x in -20..=20i32 {
            for y in -20..=20i32 {
                if (x * x + y * y) as f64 <= 400.0 {
                    c.push(map([x as f64 * 0.5, y as f64 * 0.5, 0.0]));
                }
            }
        }
        // North wall (azimuth ≈ +y direction pre-rotation).
        for y in 15..=20i32 {
            for z in 0..4i32 {
                for x in -10..=10i32 {
                    c.push(map([x as f64 * 0.5, y as f64 * 0.5 + 7.5, z as f64 * 0.5]));
                }
            }
        }
        c
    }

    #[test]
    fn same_place_different_yaw_matches_with_shift() {
        let params = ScanContextParams::default();
        let kf = ScanContext::encode(&room(0.0), &params).unwrap();
        let query = ScanContext::encode(&room(0.35), &params).unwrap();

        // Ring keys are yaw-tolerant.
        assert!(query.ring_key_distance(&kf).unwrap() < 1.0);

        let (distance, shift) = query.context_distance(&kf).unwrap();
        // Sector size = 6°; yaw 0.35 rad ≈ 20° ≈ 3-4 sectors (in either
        // shift direction, depending on the alignment convention).
        let sector_angle = 2.0 * core::f64::consts::PI / params.sectors as f64;
        let shifted_yaw =
            (shift as f64 * sector_angle).min((params.sectors - shift) as f64 * sector_angle);
        assert!(
            (shifted_yaw - 0.35).abs() < 0.15,
            "shift {shift} → {shifted_yaw} rad"
        );
        assert!(distance < 30.0, "distance {distance}");

        let detection = detect_loop(&query, &[(0usize, kf)], 1.0).unwrap();
        let d = detection.expect("loop should be detected");
        assert_eq!(d.keyframe, 0);
    }

    #[test]
    fn different_places_do_not_match() {
        // Scan Context with shift search is yaw-invariant *by design*, so a
        // merely rotated copy of the same room must NOT be used as a
        // negative. The discriminator is structural difference: a room with
        // an extra wall matches clearly worse than the same room.
        let params = ScanContextParams::default();
        let kf = ScanContext::encode(&room(0.0), &params).unwrap();

        let mut two_walls = room(0.0);
        for y in -20..=20i32 {
            for z in 0..4i32 {
                for x in 15..=20i32 {
                    two_walls.push([x as f64 * 0.5 + 7.5, y as f64 * 0.5, z as f64 * 0.5]);
                }
            }
        }
        let q = ScanContext::encode(&two_walls, &params).unwrap();
        let same = ScanContext::encode(&room(0.0), &params).unwrap();
        let (d_other, _) = q.context_distance(&kf).unwrap();
        let (d_same, _) = same.context_distance(&kf).unwrap();
        assert!(
            d_other > d_same * 5.0 + 10.0,
            "structurally different rooms too similar: {d_other} vs {d_same}"
        );
    }

    #[test]
    fn empty_scan_encodes_to_zeros() {
        let params = ScanContextParams::default();
        let sc = ScanContext::encode(&PointCloud::new(), &params).unwrap();
        assert!(sc.descriptor.iter().all(|r| r.iter().all(|&v| v == 0.0)));
    }
}
