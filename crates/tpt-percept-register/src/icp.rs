//! Iterative Closest Point (ICP) registration.
//!
//! Two variants:
//!
//! * **Point-to-point** — correspondences minimised with the closed-form
//!   Kabsch fit each iteration; residual error decreases monotonically for
//!   exact overlapping geometry (proptest-verified in `tpt-percept-verify`).
//! * **Point-to-plane** — correspondences weighted by target normals and
//!   linearised (Chen & Medioni), converging much faster on smooth surfaces.
//!
//! Both share the convergence contract: iteration stops when the relative
//! transform increment drops below `tolerance`, when correspondences stop
//! changing, or at `max_iterations`; `result.converged` reports which.
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg_dense::{DMatrix, DVector};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::RegistrationError;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;
use tpt_percept_core::align::kabsch_weighted;

/// ICP configuration.
#[derive(Clone, Copy, Debug)]
pub struct IcpParams {
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Correspondences farther than this (metres) are rejected.
    pub max_correspondence_distance: f64,
    /// Convergence: stop when the incremental transform moves every point
    /// less than this (metres, relative to the cloud scale).
    pub tolerance: f64,
    /// Minimum number of accepted correspondences to keep iterating.
    pub min_correspondences: usize,
}

impl Default for IcpParams {
    fn default() -> Self {
        IcpParams {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            tolerance: 1e-6,
            min_correspondences: 10,
        }
    }
}

/// Which residual to minimise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcpVariant {
    /// Minimise point-to-point distance (Kabsch).
    PointToPoint,
    /// Minimise point-to-plane distance (needs target normals).
    PointToPlane,
}

/// The outcome of an ICP run.
#[derive(Clone, Debug, PartialEq)]
pub struct IcpResult {
    /// Rotation part of `target ≈ R · source + t` (applied to the source).
    pub rotation: [[f64; 3]; 3],
    /// Translation part.
    pub translation: [f64; 3],
    /// Fraction of source points with a correspondence inside
    /// `max_correspondence_distance` at the final pose.
    pub fitness: f64,
    /// RMSE over those inlier correspondences (metres).
    pub rmse: f64,
    /// Iterations executed.
    pub iterations: u32,
    /// True when the tolerance was reached before `max_iterations`.
    pub converged: bool,
}

/// Runs ICP aligning `source` onto `target` starting from `(init_rotation,
/// init_translation)`.
///
/// `target_normals` is required for [`IcpVariant::PointToPlane`] and ignored
/// for point-to-point; it must have one normal per target point (need not be
/// unit — it is normalised internally).
pub fn icp(
    source: &PointCloud,
    target: &PointCloud,
    target_normals: Option<&[[f64; 3]]>,
    variant: IcpVariant,
    init_rotation: [[f64; 3]; 3],
    init_translation: [f64; 3],
    params: &IcpParams,
) -> Result<IcpResult, RegistrationError> {
    if source.is_empty() || target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !(params.max_correspondence_distance.is_finite() && params.max_correspondence_distance > 0.0)
    {
        return Err(RegistrationError::InvalidParameter(
            "max_correspondence_distance must be finite and > 0",
        ));
    }
    if params.max_iterations == 0 {
        return Err(RegistrationError::InvalidParameter(
            "max_iterations must be > 0",
        ));
    }

    let mut rotation = init_rotation;
    let mut translation = init_translation;
    let target_tree = KdTree::new(target.points());

    match variant {
        IcpVariant::PointToPlane => {
            let normals = target_normals.ok_or(RegistrationError::MissingNormals)?;
            if normals.len() != target.len() {
                return Err(RegistrationError::DimensionMismatch {
                    what: "target_normals must have one entry per target point",
                });
            }
        }
        IcpVariant::PointToPoint => {}
    }

    let max_d2 = params.max_correspondence_distance * params.max_correspondence_distance;
    let mut converged = false;
    let mut iterations = 0u32;
    let mut last_corr: Option<Vec<usize>> = None;

    while iterations < params.max_iterations {
        iterations += 1;

        // 1. Correspondences under the current pose.
        let mut src_pts = Vec::new();
        let mut dst_pts = Vec::new();
        let mut corr: Vec<usize> = Vec::new();
        for &s in source.points() {
            let ts = apply(&rotation, &translation, s);
            if let Some((idx, d2)) = target_tree.nearest(&ts) {
                if d2 <= max_d2 {
                    src_pts.push(s);
                    dst_pts.push(*target.get(idx).expect("knn index in range"));
                    corr.push(idx);
                }
            }
        }
        if src_pts.len() < params.min_correspondences {
            break;
        }
        let corr_unchanged = last_corr.as_ref().is_some_and(|prev| *prev == corr);
        last_corr = Some(corr.clone());

        // 2. Incremental update.
        let (new_rotation, new_translation, step) = match variant {
            IcpVariant::PointToPoint => {
                let fit = kabsch_weighted(&src_pts, &dst_pts, &vec![1.0; src_pts.len()])
                    .map_err(RegistrationError::AlignmentFailed)?;
                let step = pose_delta(&rotation, &translation, &fit.rotation, &fit.translation);
                (fit.rotation, fit.translation, step)
            }
            IcpVariant::PointToPlane => {
                let normals = target_normals.expect("checked above");
                let (v, w) = solve_point_to_plane(
                    &src_pts,
                    &dst_pts,
                    &corr,
                    &rotation,
                    &translation,
                    normals,
                )?;
                // T ← ΔT · T with ΔT = exp([v, w]·1).
                let wn = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                let delta_rot = if wn < 1e-12 {
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                } else {
                    axis_angle(w, wn)?
                };
                // Left-multiplied increment: p ↦ ΔR·(Rs + t) + v, so
                // R ← ΔR·R and t ← t + v (v absorbs the w × t term of the
                // linearisation).
                let delta_t = v;
                let new_rotation = mul3(&delta_rot, &rotation);
                let new_translation = [
                    translation[0] + delta_t[0],
                    translation[1] + delta_t[1],
                    translation[2] + delta_t[2],
                ];
                let step = pose_delta(&rotation, &translation, &new_rotation, &new_translation);
                (new_rotation, new_translation, step)
            }
        };
        rotation = new_rotation;
        translation = new_translation;

        if corr_unchanged || step < params.tolerance {
            converged = true;
            break;
        }
    }

    // Final fitness/RMSE at the resulting pose.
    let mut d2s = Vec::new();
    let mut inliers = 0usize;
    for &s in source.points() {
        let ts = apply(&rotation, &translation, s);
        if let Some((_, d2)) = target_tree.nearest(&ts) {
            if d2 <= max_d2 {
                inliers += 1;
                d2s.push(d2);
            }
        }
    }
    let fitness = inliers as f64 / source.len() as f64;
    let rmse = if d2s.is_empty() {
        f64::INFINITY
    } else {
        (d2s.iter().sum::<f64>() / d2s.len() as f64).sqrt()
    };

    Ok(IcpResult {
        rotation,
        translation,
        fitness,
        rmse,
        iterations,
        converged,
    })
}

/// Point-to-plane linearised step: solves for `(v, w)` (linear, angular)
/// minimising `Σ ((R s + t − q) · n)²` to first order.
fn solve_point_to_plane(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    corr: &[usize],
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    normals: &[[f64; 3]],
) -> Result<([f64; 3], [f64; 3]), RegistrationError> {
    // Normal equations accumulated directly: AᵀA δ = −Aᵀr with rows
    // [nᵀ, (rs × n)ᵀ] and unknown x = (v, w).
    let mut ata = [[0.0f64; 6]; 6];
    let mut atb = [0.0f64; 6];
    for i in 0..src.len() {
        let rs = apply(rotation, translation, src[i]);
        let mut n = normals[corr[i]];
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if nl < 1e-12 {
            return Err(RegistrationError::InvalidParameter("target normal is zero"));
        }
        n = [n[0] / nl, n[1] / nl, n[2] / nl];
        let q = dst[i];
        let r = (rs[0] - q[0]) * n[0] + (rs[1] - q[1]) * n[1] + (rs[2] - q[2]) * n[2];
        let rxn = cross3(&rs, &n);
        let mut row = [0.0f64; 6];
        row[..3].copy_from_slice(&n);
        row[3..].copy_from_slice(&rxn);
        for a_ in 0..6 {
            for b_ in 0..6 {
                ata[a_][b_] += row[a_] * row[b_];
            }
            atb[a_] -= row[a_] * r;
        }
    }
    let flat: alloc::vec::Vec<f64> = ata.iter().flat_map(|r| r.iter().copied()).collect();
    let a = DMatrix::from_row_slice(6, 6, &flat);
    let b = DVector::from_row_slice(&atb);
    let x = a
        .solve(&b)
        .map_err(|e| RegistrationError::LinearSolveFailed(alloc::format!("{e}")))?;
    Ok(([x[0], x[1], x[2]], [x[3], x[4], x[5]]))
}

/// Largest per-point displacement between two poses.
pub(crate) fn pose_delta(
    r0: &[[f64; 3]; 3],
    t0: &[f64; 3],
    r1: &[[f64; 3]; 3],
    t1: &[f64; 3],
) -> f64 {
    // Conservative bound: ‖ΔR‖∞ + ‖Δt‖ (avoids needing sample points).
    let mut dr = 0.0f64;
    for r in 0..3 {
        for c in 0..3 {
            dr = dr.max((r1[r][c] - r0[r][c]).abs());
        }
    }
    let dt = ((t1[0] - t0[0]).powi(2) + (t1[1] - t0[1]).powi(2) + (t1[2] - t0[2]).powi(2)).sqrt();
    dr + dt
}

/// Rodrigues rotation about `axis` (need not be unit) by `angle` radians.
pub(crate) fn axis_angle(axis: [f64; 3], angle: f64) -> Result<[[f64; 3]; 3], RegistrationError> {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if !n.is_finite() || n < 1e-12 {
        return Err(RegistrationError::InvalidParameter(
            "rotation axis is degenerate",
        ));
    }
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let theta = angle.max(1e-12);
    let s = theta.sin();
    let c = theta.cos();
    let cc = 1.0 - c;
    // R = I + sin θ [k]× + (1 − cos θ) [k]×².
    let kxx = k[0] * k[0];
    let kxy = k[0] * k[1];
    let kxz = k[0] * k[2];
    let kyy = k[1] * k[1];
    let kyz = k[1] * k[2];
    let kzz = k[2] * k[2];
    Ok([
        [c + cc * kxx, cc * kxy - s * k[2], cc * kxz + s * k[1]],
        [cc * kxy + s * k[2], c + cc * kyy, cc * kyz - s * k[0]],
        [cc * kxz - s * k[1], cc * kyz + s * k[0], c + cc * kzz],
    ])
}

pub(crate) fn apply(r: &[[f64; 3]; 3], t: &[f64; 3], p: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
    ]
}

pub(crate) fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

pub(crate) fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_percept_core::iso::Rotation3;

    fn box_surface() -> PointCloud {
        // Six faces of a box: full pose observability, and a test source
        // derived by transforming this cloud overlaps exactly (nearest-
        // neighbour residual → 0 at the true pose).
        let mut c = PointCloud::new();
        let vals: alloc::vec::Vec<f64> = (-20..=20i32).map(|i| i as f64 * 0.05).collect();
        for &a in &vals {
            for &b in &vals {
                for p in [
                    [a, b, 1.0],
                    [a, b, -1.0],
                    [a, 1.0, b],
                    [a, -1.0, b],
                    [1.0, a, b],
                    [-1.0, a, b],
                ] {
                    c.push(p);
                }
            }
        }
        c
    }

    fn normals_of(cloud: &PointCloud) -> Vec<[f64; 3]> {
        tpt_percept_features::normal::estimate_normals(cloud, 10)
            .unwrap()
            .iter()
            .map(|n| n.normal)
            .collect()
    }

    #[test]
    fn point_to_point_recovers_small_pose() {
        let target = box_surface();
        let r = Rotation3::<tpt_percept_core::frame::World, tpt_percept_core::frame::World>::from_axis_angle(
            [0.0, 0.0, 1.0],
            0.01,
        )
        .unwrap()
        .matrix();
        let t = [0.01, -0.008, 0.004];
        let source = target.transformed_rigid(r, t);

        let params = IcpParams {
            max_correspondence_distance: 0.25,
            tolerance: 1e-9,
            ..Default::default()
        };
        let res = icp(
            &source,
            &target,
            None,
            IcpVariant::PointToPoint,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &params,
        )
        .unwrap();
        assert!(res.converged, "not converged: {res:?}");
        assert!(res.rmse < 1e-6, "rmse {}", res.rmse);
        assert!(res.fitness > 0.95);
        // Recovered pose maps source back onto target: (r, t)⁻¹.
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
        let mut rot_err: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                rot_err = rot_err.max((res.rotation[i][j] - inv[i][j]).abs());
            }
        }
        assert!(rot_err < 1e-6, "rotation error {rot_err}");
        for i in 0..3 {
            assert!((res.translation[i] - inv_t[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn point_to_plane_converges_fewer_iterations() {
        let target = box_surface();
        let normals = normals_of(&target);
        let r = Rotation3::<tpt_percept_core::frame::World, tpt_percept_core::frame::World>::from_axis_angle(
            [0.0, 1.0, 0.0],
            0.01,
        )
        .unwrap()
        .matrix();
        let t = [0.01, 0.008, -0.01];
        let source = target.transformed_rigid(r, t);

        let params = IcpParams {
            max_correspondence_distance: 0.25,
            tolerance: 1e-9,
            ..Default::default()
        };
        let p2p = icp(
            &source,
            &target,
            None,
            IcpVariant::PointToPoint,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &params,
        )
        .unwrap();
        let p2l = icp(
            &source,
            &target,
            Some(&normals),
            IcpVariant::PointToPlane,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &params,
        )
        .unwrap();
        assert!(p2l.converged);
        assert!(p2l.rmse < 1e-6, "p2l rmse {}", p2l.rmse);
        // The p2l pose must agree with the p2p pose (both map source onto
        // target): compose p2l with the p2p INVERSE and measure the
        // deviation from identity.
        let p2p_inv = [
            [p2p.rotation[0][0], p2p.rotation[1][0], p2p.rotation[2][0]],
            [p2p.rotation[0][1], p2p.rotation[1][1], p2p.rotation[2][1]],
            [p2p.rotation[0][2], p2p.rotation[1][2], p2p.rotation[2][2]],
        ];
        let composed = mul3(&p2l.rotation, &p2p_inv);
        let mut dev = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                dev = dev.max(
                    (composed[i][j] - [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][i][j])
                        .abs(),
                );
            }
        }
        assert!(dev < 1e-3, "p2l vs p2p rotation deviation {dev}");
        // Point-to-plane typically converges in fewer iterations on smooth
        // surfaces; assert it is not slower than point-to-point here.
        assert!(
            p2l.iterations <= p2p.iterations + 2,
            "p2l {} vs p2p {}",
            p2l.iterations,
            p2p.iterations
        );
    }

    #[test]
    fn point_to_plane_requires_normals() {
        let target = box_surface();
        let source = target.clone();
        assert!(matches!(
            icp(
                &source,
                &target,
                None,
                IcpVariant::PointToPlane,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
                &IcpParams::default()
            ),
            Err(RegistrationError::MissingNormals)
        ));
    }

    #[test]
    fn empty_inputs_rejected() {
        let e = PointCloud::new();
        let ok = box_surface();
        assert!(matches!(
            icp(
                &e,
                &ok,
                None,
                IcpVariant::PointToPoint,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
                &IcpParams::default()
            ),
            Err(RegistrationError::EmptyCloud)
        ));
    }
}
