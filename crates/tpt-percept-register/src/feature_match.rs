//! Feature-based registration: descriptor matching + RANSAC pose hypotheses.
//!
//! Match FPFH (or any `FPFH_DIM`-style) descriptors of two clouds by
//! nearest-neighbour in descriptor space, then run RANSAC over
//! correspondences: a hypothesis is the Kabsch fit of three random matches,
//! scored by how many matches agree within a Euclidean threshold. The
//! winning hypothesis is refined with full RANSAC-inlier Kabsch and handed
//! back as the initial pose for fine registration (e.g. [`crate::icp::icp`]).
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::RegistrationError;
use crate::icp::apply;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_core::align::kabsch_weighted;
use tpt_percept_core::linalg3::det3;

/// A correspondence between source index and target index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match {
    /// Index into the source cloud / descriptors.
    pub source: usize,
    /// Index into the target cloud / descriptors.
    pub target: usize,
}

/// Parameters for feature-based matching.
#[derive(Clone, Copy, Debug)]
pub struct FeatureMatchParams {
    /// RANSAC iterations over correspondence triples.
    pub ransac_iterations: u32,
    /// A matched pair is an inlier when the transformed source point lands
    /// within this distance (metres) of its target point.
    pub inlier_threshold: f64,
    /// RNG seed (deterministic).
    pub seed: u64,
    /// Reject mutual matches whose descriptor distance ratio is worse than
    /// this (nearest vs second-nearest; the classic Lowe ratio).
    pub ratio_threshold: f64,
}

impl Default for FeatureMatchParams {
    fn default() -> Self {
        FeatureMatchParams {
            ransac_iterations: 1000,
            inlier_threshold: 1.0,
            seed: 0x5851F42D4C957F2D,
            ratio_threshold: 0.9,
        }
    }
}

/// The result of feature-based registration.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureRegistration {
    /// Rotation part of `target ≈ R · source + t`.
    pub rotation: [[f64; 3]; 3],
    /// Translation part.
    pub translation: [f64; 3],
    /// RANSAC inlier matches supporting the final pose.
    pub inliers: Vec<Match>,
}

/// Matches descriptors by ratio test and registers with RANSAC + Kabsch.
///
/// `source_descriptors`/`target_descriptors` must have one entry per cloud
/// point, each component in a comparable range (FPFH and SHOT are both
/// normalised).
pub fn register_features(
    source: &PointCloud,
    target: &PointCloud,
    source_descriptors: &[&[f64]],
    target_descriptors: &[&[f64]],
    params: &FeatureMatchParams,
) -> Result<FeatureRegistration, RegistrationError> {
    if source.is_empty() || target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if source_descriptors.len() != source.len() || target_descriptors.len() != target.len() {
        return Err(RegistrationError::DimensionMismatch {
            what: "descriptors must be one-per-point",
        });
    }
    if source_descriptors.first().is_some_and(|d| d.is_empty()) {
        return Err(RegistrationError::InvalidParameter("descriptors are empty"));
    }
    if !(params.ratio_threshold > 0.0 && params.ratio_threshold <= 1.0) {
        return Err(RegistrationError::InvalidParameter(
            "ratio_threshold must be in (0, 1]",
        ));
    }

    let matches = match_descriptors(
        source_descriptors,
        target_descriptors,
        params.ratio_threshold,
    );
    if matches.len() < 3 {
        return Err(RegistrationError::NoSolution);
    }

    let mut rng = tpt_percept_core::rng::XorShift64Star::new(params.seed);
    let t2 = params.inlier_threshold * params.inlier_threshold;

    let mut best: Option<(FeatureRegistration, usize)> = None;
    for _ in 0..params.ransac_iterations {
        // Sample 3 distinct matches.
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
        let triple = [matches[i0], matches[i1], matches[i2]];

        let src: Vec<[f64; 3]> = triple
            .iter()
            .map(|m| *source.get(m.source).expect("match index in range"))
            .collect();
        let dst: Vec<[f64; 3]> = triple
            .iter()
            .map(|m| *target.get(m.target).expect("match index in range"))
            .collect();
        let Ok(fit) = kabsch_weighted(&src, &dst, &[1.0; 3]) else {
            continue; // collinear triple
        };
        if det3(&fit.rotation) <= 0.0 {
            continue;
        }

        // Inliers of the raw triple hypothesis.
        let hypothesis_inliers = inliers_of(
            &fit.rotation,
            &fit.translation,
            source,
            target,
            &matches,
            t2,
        );
        if hypothesis_inliers.len() < 3 {
            continue;
        }
        let (_cand_rotation, _cand_translation) = (fit.rotation, fit.translation);
        // Refine over the hypothesis inliers; keep the refined pose only if
        // it does not shrink support.
        let src: Vec<[f64; 3]> = hypothesis_inliers
            .iter()
            .map(|m| *source.get(m.source).expect("in range"))
            .collect();
        let dst: Vec<[f64; 3]> = hypothesis_inliers
            .iter()
            .map(|m| *target.get(m.target).expect("in range"))
            .collect();
        let mut refined_pose = None;
        if let Ok(refined) = kabsch_weighted(&src, &dst, &vec![1.0; src.len()]) {
            let refined_inliers = inliers_of(
                &refined.rotation,
                &refined.translation,
                source,
                target,
                &matches,
                t2,
            );
            if refined_inliers.len() >= hypothesis_inliers.len() {
                refined_pose = Some((refined.rotation, refined.translation, refined_inliers));
            }
        }
        let (rot, trans, inlier_list) = match refined_pose {
            Some((r, t, i)) => (r, t, i),
            None => (fit.rotation, fit.translation, hypothesis_inliers),
        };
        if inlier_list.len() >= 3 && best.as_ref().is_none_or(|(_, n)| inlier_list.len() > *n) {
            let n = inlier_list.len();
            best = Some((
                FeatureRegistration {
                    rotation: rot,
                    translation: trans,
                    inliers: inlier_list,
                },
                n,
            ));
        }
    }

    best.map(|(reg, _)| reg)
        .ok_or(RegistrationError::NoSolution)
}

/// All matches consistent with the given pose.
fn inliers_of(
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    source: &PointCloud,
    target: &PointCloud,
    matches: &[Match],
    t2: f64,
) -> Vec<Match> {
    matches
        .iter()
        .copied()
        .filter(|m| {
            let s = source.get(m.source).expect("in range");
            let t = target.get(m.target).expect("in range");
            let ts = apply(rotation, translation, *s);
            let d = [ts[0] - t[0], ts[1] - t[1], ts[2] - t[2]];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= t2
        })
        .collect()
}

/// Ratio-test matching: for each source descriptor take the nearest target
/// descriptor; accept when `d1 <= ratio · d2`.
pub fn match_descriptors(source: &[&[f64]], target: &[&[f64]], ratio: f64) -> Vec<Match> {
    let mut out = Vec::new();
    for (si, sd) in source.iter().enumerate() {
        let mut best1 = f64::INFINITY;
        let mut best2 = f64::INFINITY;
        let mut best_ti = usize::MAX;
        for (ti, td) in target.iter().enumerate() {
            let d: f64 = sd
                .iter()
                .zip(td.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best1 {
                best2 = best1;
                best1 = d;
                best_ti = ti;
            } else if d < best2 {
                best2 = d;
            }
        }
        if best_ti != usize::MAX && best1 <= ratio * ratio * best2 {
            out.push(Match {
                source: si,
                target: best_ti,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Distinct "descriptor" patterns: descriptor i is a one-hot vector at
    /// position i, so matching is unambiguous.
    fn one_hot(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let mut d = vec![0.0; dim];
                d[i] = 1.0;
                d
            })
            .collect()
    }

    fn view(v: &[Vec<f64>]) -> Vec<&[f64]> {
        v.iter().map(|d| d.as_slice()).collect()
    }

    #[test]
    fn ratio_matching_is_unambiguous() {
        let src_desc = one_hot(10, 12);
        let tgt_desc = one_hot(10, 12);
        let matches = match_descriptors(&view(&src_desc), &view(&tgt_desc), 0.9);
        assert_eq!(matches.len(), 10);
        for m in &matches {
            assert_eq!(m.source, m.target);
        }
    }

    #[test]
    fn registration_recovers_pose() {
        use tpt_percept_core::iso::Rotation3;

        // Fibonacci sphere: well-spread points (no near-collinear triples).
        let mut target = PointCloud::new();
        let n = 12;
        let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let rr = (1.0 - z * z).sqrt();
            let a = golden * i as f64;
            target.push([rr * a.cos(), rr * a.sin(), z]);
        }
        let src_desc = one_hot(12, 16);
        let tgt_desc = one_hot(12, 16);

        let r = Rotation3::<tpt_percept_core::frame::World, tpt_percept_core::frame::World>::from_axis_angle(
            [0.0, 0.0, 1.0],
            1.0,
        )
        .unwrap()
        .matrix();
        let t = [2.0, -1.0, 0.5];
        let source = target.transformed_rigid(r, t);

        let reg = register_features(
            &source,
            &target,
            &view(&src_desc),
            &view(&tgt_desc),
            &FeatureMatchParams {
                inlier_threshold: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(reg.inliers.len() >= 10, "inliers {}", reg.inliers.len());
        // Registration maps source back onto target: (r, t)⁻¹.
        let inv = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        let inv_t = [
            -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2]),
            -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2]),
            -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2]),
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((reg.rotation[i][j] - inv[i][j]).abs() < 1e-6);
            }
            assert!((reg.translation[i] - inv_t[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ambiguous_descriptors_give_no_matches() {
        // All target descriptors identical → the ratio test rejects every
        // source match (best and second-best tie) → NoSolution.
        let target = PointCloud::from_points(vec![[0.0; 3], [1.0; 3], [2.0; 3]]);
        let source = target.clone();
        let src_desc = one_hot(3, 4);
        let same: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 0.0, 0.0]; 3];
        assert!(matches!(
            register_features(
                &source,
                &target,
                &view(&src_desc),
                &view(&same),
                &FeatureMatchParams {
                    ratio_threshold: 0.9,
                    ..Default::default()
                }
            ),
            Err(RegistrationError::NoSolution)
        ));
    }

    #[test]
    fn dimension_mismatch_detected() {
        let target = PointCloud::from_points(vec![[0.0; 3]; 4]);
        let source = target.clone();
        let d4 = one_hot(4, 4);
        let d3 = one_hot(3, 4);
        assert!(matches!(
            register_features(
                &source,
                &target,
                &view(&d3),
                &view(&d4),
                &FeatureMatchParams::default()
            ),
            Err(RegistrationError::DimensionMismatch { .. })
        ));
    }
}
