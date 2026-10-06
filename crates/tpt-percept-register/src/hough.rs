//! 3-D Hough voting for loop closure.
//!
//! Place-recognition-style loop closure: extract feature landmarks (points
//! with normals) from two clouds, then vote over the **yaw angle** aligning
//! pairs of normal azimuths — each (source landmark, target landmark) pair
//! contributes a vote for the yaw that would bring the source normal's
//! azimuth onto the target's. The dominant yaw bin defines the rotation;
//! translation is then solved by RANSAC over the landmark correspondences
//! consistent with that yaw.
//!
//! This is the planar-rotation Hough used for levelled LiDAR loop closure:
//! robust to partial overlap because votes accumulate over all pairs rather
//! than requiring exact matches.

use alloc::vec;
use alloc::vec::Vec;
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::RegistrationError;
use crate::icp::apply;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_core::align::kabsch_weighted;
use tpt_percept_core::rng::XorShift64Star;

/// A landmark: a point with a unit normal (metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmark {
    /// Position.
    pub point: [f64; 3],
    /// Unit normal.
    pub normal: [f64; 3],
}

/// Hough loop-closure parameters.
#[derive(Clone, Copy, Debug)]
pub struct HoughParams {
    /// Yaw accumulator resolution (radians per bin).
    pub yaw_bin: f64,
    /// Minimum vote fraction (of the max bin) for a yaw to be considered.
    pub min_bin_fraction: f64,
    /// Inlier distance for the translation RANSAC (metres).
    pub inlier_threshold: f64,
    /// RANSAC iterations for the translation hypothesis.
    pub ransac_iterations: u32,
    /// RNG seed.
    pub seed: u64,
}

impl Default for HoughParams {
    fn default() -> Self {
        HoughParams {
            yaw_bin: (2.0 * core::f64::consts::PI) / 72.0, // 5° bins
            min_bin_fraction: 0.5,
            inlier_threshold: 0.5,
            ransac_iterations: 500,
            seed: 0x9E3779B97F4A7C15,
        }
    }
}

/// A loop-closure candidate transform.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopClosureCandidate {
    /// Rotation part of `target ≈ R · source + t` (planar rotation).
    pub rotation: [[f64; 3]; 3],
    /// Translation part.
    pub translation: [f64; 3],
    /// Yaw angle recovered from the Hough accumulator (radians).
    pub yaw: f64,
    /// Number of votes in the winning yaw bin.
    pub votes: usize,
    /// Translation-RANSAC inlier count supporting the candidate.
    pub inliers: usize,
}

/// Extracts landmarks as the points with highest local planarity (lowest
/// curvature) — cheap and robust stand-in for semantic landmarks.
pub fn extract_landmarks(
    cloud: &PointCloud,
    k: usize,
    max_landmarks: usize,
) -> Result<Vec<Landmark>, RegistrationError> {
    if k < 5 || k >= cloud.len() {
        return Err(RegistrationError::InvalidParameter(
            "k must be >= 5 and < point count",
        ));
    }
    let normals = tpt_percept_features::normal::estimate_normals(cloud, k)
        .ok_or(RegistrationError::InvalidParameter("cloud too small"))?;
    let mut order: Vec<usize> = (0..cloud.len()).collect();
    order.sort_by(|&a, &b| {
        normals[a]
            .curvature
            .partial_cmp(&normals[b].curvature)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    order.truncate(max_landmarks);
    Ok(order
        .into_iter()
        .map(|i| Landmark {
            point: *cloud.get(i).expect("index in range"),
            normal: normals[i].normal,
        })
        .collect())
}

/// Finds the best loop-closure transform between landmark sets.
pub fn hough_loop_closure(
    source: &[Landmark],
    target: &[Landmark],
    params: &HoughParams,
) -> Result<LoopClosureCandidate, RegistrationError> {
    if source.len() < 3 || target.len() < 3 {
        return Err(RegistrationError::NoSolution);
    }
    if !(params.yaw_bin.is_finite() && params.yaw_bin > 0.0) {
        return Err(RegistrationError::InvalidParameter("yaw_bin must be > 0"));
    }

    let bins = ((2.0 * core::f64::consts::PI / params.yaw_bin).ceil() as usize).max(1);
    let mut acc = vec![0u32; bins];

    // Orientation-agnostic azimuth voting: each pair votes for the yaw that
    // aligns the source normal's azimuth onto the target's, modulo the
    // normal-flip ambiguity (both signs vote).
    for s in source {
        let az_s = azimuth(&s.normal);
        for t in target {
            let az_t = azimuth(&t.normal);
            let delta = az_t - az_s;
            for sign in [0.0_f64, core::f64::consts::PI] {
                let yaw = normalize_angle(delta + sign);
                let bin = ((yaw + core::f64::consts::PI) / (2.0 * core::f64::consts::PI)
                    * bins as f64) as usize;
                let bin = bin.min(bins - 1);
                acc[bin] += 1;
            }
        }
    }

    let max_votes = *acc.iter().max().expect("non-empty") as usize;
    if max_votes == 0 {
        return Err(RegistrationError::NoSolution);
    }
    let min_votes = (max_votes as f64 * params.min_bin_fraction) as usize;

    // Try candidate yaws from the strongest bins.
    let mut order: Vec<usize> = (0..bins).collect();
    order.sort_by(|&a, &b| acc[b].cmp(&acc[a]));

    let mut best: Option<LoopClosureCandidate> = None;
    let mut rng = XorShift64Star::new(params.seed);
    for &bin in order.iter().take(8) {
        if (acc[bin] as usize) < min_votes {
            break;
        }
        let yaw = normalize_angle(
            (bin as f64 + 0.5) / bins as f64 * 2.0 * core::f64::consts::PI - core::f64::consts::PI,
        );
        let (s, c) = yaw.sin_cos();
        let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];

        // Translation RANSAC over landmark pairs consistent with this yaw.
        let t2 = params.inlier_threshold * params.inlier_threshold;
        let mut best_translation = [0.0; 3];
        let mut best_inliers = 0usize;
        for _ in 0..params.ransac_iterations {
            let si = rng.below(source.len());
            let ti = rng.below(target.len());
            let sp = source[si].point;
            let tp = target[ti].point;
            let rs = apply(&rotation, &[0.0; 3], sp);
            let t = [tp[0] - rs[0], tp[1] - rs[1], tp[2] - rs[2]];

            let mut inliers = 0usize;
            let mut src_in: Vec<[f64; 3]> = Vec::new();
            let mut dst_in: Vec<[f64; 3]> = Vec::new();
            for sl in source {
                let ts = apply(&rotation, &t, sl.point);
                // Match against the nearest target landmark.
                let mut best_d2 = f64::INFINITY;
                let mut best_t = [0.0; 3];
                for tl in target {
                    let d = [
                        ts[0] - tl.point[0],
                        ts[1] - tl.point[1],
                        ts[2] - tl.point[2],
                    ];
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_t = tl.point;
                    }
                }
                if best_d2 <= t2 {
                    inliers += 1;
                    src_in.push(sl.point);
                    dst_in.push(best_t);
                }
            }
            if inliers > best_inliers {
                // Refine translation with Kabsch over inliers (rotation fixed).
                if inliers >= 3 {
                    if let Ok(fit) = kabsch_weighted(&src_in, &dst_in, &vec![1.0; src_in.len()]) {
                        // Keep the Hough yaw; average the translation only.
                        best_translation = fit.translation;
                        let _ = fit.rotation;
                    } else {
                        best_translation = t;
                    }
                } else {
                    best_translation = t;
                }
                best_inliers = inliers;
            }
        }

        if best_inliers >= 3 {
            let candidate = LoopClosureCandidate {
                rotation,
                translation: best_translation,
                yaw,
                votes: acc[bin] as usize,
                inliers: best_inliers,
            };
            let better = best
                .as_ref()
                .is_none_or(|b: &LoopClosureCandidate| candidate.inliers > b.inliers);
            if better {
                best = Some(candidate);
            }
        }
    }

    best.ok_or(RegistrationError::NoSolution)
}

fn azimuth(n: &[f64; 3]) -> f64 {
    let az = n[1].atan2(n[0]);
    normalize_angle(az)
}

fn normalize_angle(a: f64) -> f64 {
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut x = a % two_pi;
    if x > core::f64::consts::PI {
        x -= two_pi;
    }
    if x < -core::f64::consts::PI {
        x += two_pi;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Landmarks on a wall along y (normals ≈ +x) plus one along x
    /// (normals ≈ +y): azimuth structure is unambiguous.
    fn landmarks(offset: [f64; 3], yaw: f64) -> Vec<Landmark> {
        let raw = [
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 2.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 3.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([3.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let (s, c) = yaw.sin_cos();
        raw.iter()
            .map(|&(p, n)| {
                let rp = [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];
                let rn = [c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]];
                Landmark {
                    point: [rp[0] + offset[0], rp[1] + offset[1], rp[2] + offset[2]],
                    normal: rn,
                }
            })
            .collect()
    }

    #[test]
    fn recovers_yaw_and_translation() {
        let yaw = 0.7;
        let offset = [2.0, -1.0, 0.0];
        let target = landmarks([0.0; 3], 0.0);
        let source = landmarks(offset, yaw);

        let cand = hough_loop_closure(&source, &target, &HoughParams::default()).unwrap();
        // The source was built by rotating landmarks BY +yaw, so the
        // source→target registration yaw is −yaw (modulo π from the
        // normal-flip ambiguity).
        let check_yaw = |candidate_yaw: f64| -> f64 {
            let mut d = (candidate_yaw - (-yaw)).abs();
            if d > core::f64::consts::FRAC_PI_2 {
                d = core::f64::consts::PI - d;
            }
            d
        };
        assert!(check_yaw(cand.yaw) < 0.15, "yaw {} vs {}", cand.yaw, -yaw);
        assert!(cand.inliers >= 4, "inliers {}", cand.inliers);
    }

    #[test]
    fn too_few_landmarks_rejected() {
        let two: Vec<Landmark> = landmarks([0.0; 3], 0.0).into_iter().take(2).collect();
        let six = landmarks([0.0; 3], 0.0);
        assert!(matches!(
            hough_loop_closure(&two, &six, &HoughParams::default()),
            Err(RegistrationError::NoSolution)
        ));
        assert!(matches!(
            hough_loop_closure(&six, &two, &HoughParams::default()),
            Err(RegistrationError::NoSolution)
        ));
    }

    #[test]
    fn landmark_extraction_prefers_planar() {
        let mut c = PointCloud::new();
        for x in -5..=5i32 {
            for y in -5..=5i32 {
                c.push([x as f64 * 0.2, y as f64 * 0.2, 0.0]); // planar
            }
        }
        for i in 0..20 {
            c.push([(i as f64 * 0.1).sin(), (i as f64 * 0.1).cos(), 3.0]); // wiggly
        }
        let cloud = c;
        let lms = extract_landmarks(&cloud, 8, 10).unwrap();
        assert_eq!(lms.len(), 10);
        // Most selected landmarks should be on the plane (z ≈ 0).
        let planar = lms.iter().filter(|l| l.point[2].abs() < 1e-9).count();
        assert!(planar >= 8, "planar landmarks {planar}/10");
    }
}
