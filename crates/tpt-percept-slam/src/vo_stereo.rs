//! Stereo visual odometry: inter-frame motion from triangulated stereo
//! points.
//!
//! Each stereo frame yields 3-D points in the camera frame (depth from
//! disparity: `Z = f·b/disparity`); the inter-frame motion is then a rigid
//! 3-D-3-D alignment solved with RANSAC + weighted Kabsch — well-posed and
//! metric (unlike the monocular case).
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use crate::error::SlamError;
use tpt_percept_core::align::kabsch_weighted;
use tpt_percept_core::rng::XorShift64Star;

/// A triangulated 3-D point with a match in the next frame's point set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoMatch {
    /// Point in the previous frame's camera frame (metres).
    pub from: [f64; 3],
    /// Point in the current frame's camera frame (metres).
    pub to: [f64; 3],
}

/// Converts pixel correspondences with disparities into 3-D points.
///
/// `Z = f·b/d` with focal length `f` (pixels), baseline `b` (metres) and
/// disparity `d` (pixels). Returns `None` for non-positive disparities.
pub fn triangulate_stereo(
    left_points: &[[f64; 2]],
    disparities: &[f64],
    focal_px: f64,
    baseline_m: f64,
) -> Result<Vec<[f64; 3]>, SlamError> {
    if left_points.len() != disparities.len() {
        return Err(SlamError::InvalidParameter(
            "one disparity per point required",
        ));
    }
    if !(focal_px.is_finite() && focal_px > 0.0 && baseline_m.is_finite() && baseline_m > 0.0) {
        return Err(SlamError::InvalidParameter(
            "focal and baseline must be > 0",
        ));
    }
    let mut out = Vec::with_capacity(left_points.len());
    for (p, &d) in left_points.iter().zip(disparities) {
        if !(d.is_finite() && d > 0.0) {
            continue;
        }
        let z = focal_px * baseline_m / d;
        // Pinhole back-projection with unit principal point offset removed.
        out.push([p[0] * z, p[1] * z, z]);
    }
    Ok(out)
}

/// Stereo odometry parameters.
#[derive(Clone, Copy, Debug)]
pub struct StereoParams {
    /// RANSAC iterations over match triples.
    pub ransac_iterations: u32,
    /// Inlier distance (metres).
    pub inlier_threshold: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for StereoParams {
    fn default() -> Self {
        StereoParams {
            ransac_iterations: 500,
            inlier_threshold: 0.05,
            seed: 0xD1B54A32D192ED03,
        }
    }
}

/// Estimated stereo motion: `to ≈ R · from + t`.
#[derive(Clone, Debug, PartialEq)]
pub struct StereoMotion {
    /// Rotation.
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
    /// Inlier count.
    pub inliers: usize,
}

/// Estimates the inter-frame motion from matched 3-D stereo points.
pub fn estimate_motion_stereo(
    matches: &[StereoMatch],
    params: &StereoParams,
) -> Result<StereoMotion, SlamError> {
    if matches.len() < 3 {
        return Err(SlamError::InsufficientData(
            "at least 3 stereo matches required",
        ));
    }
    if !(params.inlier_threshold.is_finite() && params.inlier_threshold > 0.0) {
        return Err(SlamError::InvalidParameter("inlier_threshold must be > 0"));
    }
    let mut rng = XorShift64Star::new(params.seed);
    let t2 = params.inlier_threshold * params.inlier_threshold;
    let mut best: Option<(StereoMotion, usize)> = None;

    for _ in 0..params.ransac_iterations {
        let i0 = rng.below(matches.len());
        let i1 = loop {
            let j = rng.below(matches.len());
            if j != i0 {
                break j;
            }
        };
        let i2 = loop {
            let j = rng.below(matches.len());
            if j != i0 && j != i1 {
                break j;
            }
        };
        let trip = [matches[i0], matches[i1], matches[i2]];
        let src: Vec<[f64; 3]> = trip.iter().map(|m| m.from).collect();
        let dst: Vec<[f64; 3]> = trip.iter().map(|m| m.to).collect();
        let Ok(fit) = kabsch_weighted(&src, &dst, &[1.0; 3]) else {
            continue;
        };

        let inliers = matches
            .iter()
            .filter(|m| {
                let t = apply3(&fit.rotation, &fit.translation, m.from);
                let d = [t[0] - m.to[0], t[1] - m.to[1], t[2] - m.to[2]];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= t2
            })
            .count();
        if best.as_ref().is_none_or(|(_, n)| inliers > *n) {
            // Refine over inliers.
            let inlier_set: Vec<&StereoMatch> = matches
                .iter()
                .filter(|m| {
                    let t = apply3(&fit.rotation, &fit.translation, m.from);
                    let d = [t[0] - m.to[0], t[1] - m.to[1], t[2] - m.to[2]];
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= t2
                })
                .collect();
            if inlier_set.len() >= 3 {
                let src: Vec<[f64; 3]> = inlier_set.iter().map(|m| m.from).collect();
                let dst: Vec<[f64; 3]> = inlier_set.iter().map(|m| m.to).collect();
                if let Ok(refined) = kabsch_weighted(&src, &dst, &vec![1.0; src.len()]) {
                    let n = matches
                        .iter()
                        .filter(|m| {
                            let t = apply3(&refined.rotation, &refined.translation, m.from);
                            let d = [t[0] - m.to[0], t[1] - m.to[1], t[2] - m.to[2]];
                            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= t2
                        })
                        .count();
                    best = Some((
                        StereoMotion {
                            rotation: refined.rotation,
                            translation: refined.translation,
                            inliers: n,
                        },
                        n,
                    ));
                }
            }
        }
    }

    best.map(|(m, _)| m).ok_or(SlamError::NoSolution)
}

fn apply3(r: &[[f64; 3]; 3], t: &[f64; 3], p: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulates_depth_from_disparity() {
        // f = 700 px, b = 0.5 m, disparity 35 px → Z = 10 m.
        let pts =
            triangulate_stereo(&[[0.1, -0.2], [0.0, 0.0]], &[35.0, 70.0], 700.0, 0.5).unwrap();
        assert!((pts[0][2] - 10.0).abs() < 1e-9);
        assert!((pts[0][0] - 1.0).abs() < 1e-9);
        assert!((pts[1][2] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_disparity_skipped() {
        let pts = triangulate_stereo(&[[0.0, 0.0]; 2], &[0.0, 35.0], 700.0, 0.5).unwrap();
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn recovers_stereo_motion() {
        use tpt_percept_core::iso::Rotation3;
        let mut prev: Vec<[f64; 3]> = Vec::new();
        for i in 0..40 {
            let f = i as f64;
            prev.push([
                (f * 0.13).cos() * 4.0,
                (f * 0.29).sin() * 4.0,
                1.0 + (f % 7.0),
            ]);
        }
        let r = Rotation3::<tpt_percept_core::frame::World, tpt_percept_core::frame::World>::from_axis_angle(
            [0.1, 1.0, 0.2], 0.08).unwrap().matrix();
        let t = [0.4, -0.1, 0.2];
        let matches: Vec<StereoMatch> = prev
            .iter()
            .map(|&p| {
                let next = apply3(&r, &t, p);
                StereoMatch { from: p, to: next }
            })
            .collect();

        let motion = estimate_motion_stereo(&matches, &StereoParams::default()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((motion.rotation[i][j] - r[i][j]).abs() < 1e-9);
            }
            assert!((motion.translation[i] - t[i]).abs() < 1e-9);
        }
        assert_eq!(motion.inliers, 40);
    }

    #[test]
    fn too_few_matches_rejected() {
        let m = StereoMatch {
            from: [0.0; 3],
            to: [0.0; 3],
        };
        assert!(matches!(
            estimate_motion_stereo(&[m], &StereoParams::default()),
            Err(SlamError::InsufficientData(_))
        ));
    }
}
