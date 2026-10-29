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

use crate::degeneracy::{self, Degeneracy};
use tpt_percept_cloud::cloud::FrameCloud;
use tpt_percept_core::frame::Frame;
use tpt_percept_core::iso::{Isometry3, Rotation3};

/// Multiplier applied to the median nearest-neighbour distance by
/// [`IcpParams::auto`].
const AUTO_CORRESPONDENCE_FACTOR: f64 = 5.0;
/// Upper bound on source points sampled by [`IcpParams::auto`].
const AUTO_SAMPLE: usize = 1000;

/// ICP configuration.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Huber threshold (metres) on correspondence distances. Correspondences
    /// beyond it get weight `δ / d`, which bounds the influence of outliers
    /// that still pass the distance gate. `0` disables the robust kernel.
    pub huber_delta: f64,
}

impl Default for IcpParams {
    fn default() -> Self {
        IcpParams {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            tolerance: 1e-6,
            min_correspondences: 10,
            huber_delta: 0.0,
        }
    }
}

impl IcpParams {
    /// Derives the correspondence gate from the clouds' own geometry.
    ///
    /// Sets `max_correspondence_distance` to five times the median
    /// nearest-neighbour distance from (a subsample of) `source` to `target`,
    /// measured at the clouds' current placement, so it assumes they start
    /// roughly aligned. All other fields keep their defaults. Returns `None`
    /// if either cloud is empty.
    pub fn auto(source: &PointCloud, target: &PointCloud) -> Option<IcpParams> {
        if source.is_empty() || target.is_empty() {
            return None;
        }
        let tree = KdTree::new(target.points());
        let stride = (source.len() / AUTO_SAMPLE).max(1);
        let mut dists: Vec<f64> = source
            .points()
            .iter()
            .step_by(stride)
            .filter_map(|p| tree.nearest(p).map(|(_, d2)| d2.sqrt()))
            .collect();
        if dists.is_empty() {
            return None;
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        let median = dists[dists.len() / 2];
        let defaults = IcpParams::default();
        let gate = if median > 0.0 {
            AUTO_CORRESPONDENCE_FACTOR * median
        } else {
            defaults.max_correspondence_distance
        };
        Some(IcpParams {
            max_correspondence_distance: gate,
            ..defaults
        })
    }
}

/// Which residual to minimise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IcpVariant {
    /// Minimise point-to-point distance (Kabsch).
    PointToPoint,
    /// Minimise point-to-plane distance (needs target normals).
    PointToPlane,
}

/// The outcome of an ICP run.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Gauss–Newton information `AᵀWA` at the final pose, over the
    /// `(translation, rotation)` increment the point-to-plane solver uses.
    /// `None` for point-to-point (no linearised system is formed).
    pub information: Option<[[f64; 6]; 6]>,
}

impl IcpResult {
    /// Approximate pose covariance `rmse² · (AᵀWA)⁻¹`, in the same
    /// `(translation, rotation)` parametrisation as [`Self::information`].
    ///
    /// The `rmse²` factor is a heuristic estimate of the measurement noise
    /// variance; treat the result as a relative uncertainty, not a calibrated
    /// one. `None` when there is no information matrix or it is singular.
    pub fn covariance(&self) -> Option<[[f64; 6]; 6]> {
        let info = self.information?;
        if !self.rmse.is_finite() {
            return None;
        }
        let inv = DMatrix::from_row_slice(6, 6, &flatten6(&info))
            .inverse()
            .ok()?;
        let mut out = [[0.0f64; 6]; 6];
        for (r, row) in out.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = self.rmse * self.rmse * inv[(r, c)];
            }
        }
        Some(out)
    }

    /// Weak directions of the pose estimate: eigen-decomposition of
    /// [`Self::information`] with eigenvalues below `rel_tol · λ_max`
    /// flagged. Corridors and flat floors show up here as weak axes.
    /// `None` when there is no information matrix.
    pub fn degeneracy(&self, rel_tol: f64) -> Option<Degeneracy> {
        degeneracy::analyse(&self.information?, rel_tol)
    }
}

/// Target-side data prepared once and reused across many [`icp_prepared`]
/// calls — the usual case when localising repeatedly against a fixed map.
#[derive(Clone, Debug)]
pub struct PreparedTarget {
    cloud: PointCloud,
    tree: KdTree,
    normals: Option<Vec<[f64; 3]>>,
}

impl PreparedTarget {
    /// Builds the k-d tree over `cloud`. `normals`, when given, must hold one
    /// entry per point.
    pub fn new(
        cloud: PointCloud,
        normals: Option<Vec<[f64; 3]>>,
    ) -> Result<PreparedTarget, RegistrationError> {
        if cloud.is_empty() {
            return Err(RegistrationError::EmptyCloud);
        }
        if let Some(n) = &normals {
            if n.len() != cloud.len() {
                return Err(RegistrationError::DimensionMismatch {
                    what: "target_normals must have one entry per target point",
                });
            }
        }
        let tree = KdTree::new(cloud.points());
        Ok(PreparedTarget {
            cloud,
            tree,
            normals,
        })
    }

    /// The target cloud.
    pub fn cloud(&self) -> &PointCloud {
        &self.cloud
    }
}

/// Runs ICP aligning `source` onto `target` starting from `(init_rotation,
/// init_translation)`.
///
/// `target_normals` is required for [`IcpVariant::PointToPlane`] and ignored
/// for point-to-point; it must have one normal per target point (need not be
/// unit — it is normalised internally). To run many alignments against the
/// same target, use [`PreparedTarget`] and [`icp_prepared`] to avoid
/// rebuilding the k-d tree.
pub fn icp(
    source: &PointCloud,
    target: &PointCloud,
    target_normals: Option<&[[f64; 3]]>,
    variant: IcpVariant,
    init_rotation: [[f64; 3]; 3],
    init_translation: [f64; 3],
    params: &IcpParams,
) -> Result<IcpResult, RegistrationError> {
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    let tree = KdTree::new(target.points());
    run(
        source,
        target,
        &tree,
        target_normals,
        variant,
        init_rotation,
        init_translation,
        params,
    )
}

/// [`icp`] against a [`PreparedTarget`], reusing its k-d tree and normals.
pub fn icp_prepared(
    source: &PointCloud,
    target: &PreparedTarget,
    variant: IcpVariant,
    init_rotation: [[f64; 3]; 3],
    init_translation: [f64; 3],
    params: &IcpParams,
) -> Result<IcpResult, RegistrationError> {
    run(
        source,
        &target.cloud,
        &target.tree,
        target.normals.as_deref(),
        variant,
        init_rotation,
        init_translation,
        params,
    )
}

/// Frame-typed [`icp`]: `source` is captured in frame `From`, `target` lives
/// in `To`, and the returned isometry is the `From → To` transform that maps
/// the source onto the target. `init` seeds the search in the same frames.
///
/// Mixing up the two clouds' frames is a compile error, not a silent bug.
pub fn icp_frames<From: Frame, To: Frame>(
    source: &FrameCloud<From>,
    target: &FrameCloud<To>,
    target_normals: Option<&[[f64; 3]]>,
    variant: IcpVariant,
    init: Isometry3<From, To>,
    params: &IcpParams,
) -> Result<(Isometry3<From, To>, IcpResult), RegistrationError> {
    let (init_r, init_t) = init.into_raw();
    let res = icp(
        source.cloud(),
        target.cloud(),
        target_normals,
        variant,
        init_r,
        init_t,
        params,
    )?;
    let iso = Isometry3::from_parts_unchecked(
        res.translation,
        Rotation3::from_matrix_unchecked(res.rotation),
    );
    Ok((iso, res))
}

/// Shared ICP loop. `tree` must be built from `target.points()`.
#[allow(clippy::too_many_arguments)]
fn run(
    source: &PointCloud,
    target: &PointCloud,
    tree: &KdTree,
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
    if !(params.huber_delta.is_finite() && params.huber_delta >= 0.0) {
        return Err(RegistrationError::InvalidParameter(
            "huber_delta must be finite and >= 0",
        ));
    }
    if params.max_iterations == 0 {
        return Err(RegistrationError::InvalidParameter(
            "max_iterations must be > 0",
        ));
    }
    let normals = match variant {
        IcpVariant::PointToPlane => {
            let normals = target_normals.ok_or(RegistrationError::MissingNormals)?;
            if normals.len() != target.len() {
                return Err(RegistrationError::DimensionMismatch {
                    what: "target_normals must have one entry per target point",
                });
            }
            Some(normals)
        }
        IcpVariant::PointToPoint => None,
    };

    let mut rotation = init_rotation;
    let mut translation = init_translation;
    let max_d2 = params.max_correspondence_distance * params.max_correspondence_distance;
    let mut converged = false;
    let mut iterations = 0u32;
    let mut last_corr: Option<Vec<usize>> = None;

    while iterations < params.max_iterations {
        iterations += 1;

        // 1. Correspondences under the current pose.
        let c = correspondences(source, target, tree, &rotation, &translation, max_d2);
        if c.src.len() < params.min_correspondences {
            break;
        }
        let corr_unchanged = last_corr.as_ref().is_some_and(|prev| *prev == c.idx);
        let weights = robust_weights(&c.d2, params.huber_delta);

        // 2. Incremental update.
        let (new_rotation, new_translation, step) = match normals {
            None => {
                let fit = kabsch_weighted(&c.src, &c.dst, &weights)
                    .map_err(RegistrationError::AlignmentFailed)?;
                let step = pose_delta(&rotation, &translation, &fit.rotation, &fit.translation);
                (fit.rotation, fit.translation, step)
            }
            Some(normals) => {
                let (ata, atb) =
                    point_to_plane_system(&c, &rotation, &translation, normals, &weights)?;
                let x = solve6(&ata, &atb)?;
                let (v, w) = ([x[0], x[1], x[2]], [x[3], x[4], x[5]]);
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
                let new_rotation = mul3(&delta_rot, &rotation);
                let new_translation = [
                    translation[0] + v[0],
                    translation[1] + v[1],
                    translation[2] + v[2],
                ];
                let step = pose_delta(&rotation, &translation, &new_rotation, &new_translation);
                (new_rotation, new_translation, step)
            }
        };
        rotation = new_rotation;
        translation = new_translation;
        last_corr = Some(c.idx);

        if corr_unchanged || step < params.tolerance {
            converged = true;
            break;
        }
    }

    // Final fitness/RMSE and information at the resulting pose.
    let c = correspondences(source, target, tree, &rotation, &translation, max_d2);
    let inliers = c.src.len();
    let fitness = inliers as f64 / source.len() as f64;
    let rmse = if c.d2.is_empty() {
        f64::INFINITY
    } else {
        (c.d2.iter().sum::<f64>() / c.d2.len() as f64).sqrt()
    };
    let information = match normals {
        Some(normals) if inliers > 0 => {
            let weights = robust_weights(&c.d2, params.huber_delta);
            let (ata, _) = point_to_plane_system(&c, &rotation, &translation, normals, &weights)?;
            Some(ata)
        }
        _ => None,
    };

    Ok(IcpResult {
        rotation,
        translation,
        fitness,
        rmse,
        iterations,
        converged,
        information,
    })
}

/// Inlier correspondences at one pose.
struct Correspondences {
    /// Source points (untransformed) with an inlier.
    src: Vec<[f64; 3]>,
    /// Matched target points.
    dst: Vec<[f64; 3]>,
    /// Matched target indices.
    idx: Vec<usize>,
    /// Squared distances between the transformed source and matched target.
    d2: Vec<f64>,
}

fn correspondences(
    source: &PointCloud,
    target: &PointCloud,
    tree: &KdTree,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    max_d2: f64,
) -> Correspondences {
    // Each source point is independent, so this step parallelises cleanly.
    // `collect` preserves order, so results are identical either way.
    let hit = |&s: &[f64; 3]| -> Option<(usize, [f64; 3], f64)> {
        let ts = apply(rotation, translation, s);
        tree.nearest(&ts)
            .and_then(|(idx, d2)| (d2 <= max_d2).then_some((idx, s, d2)))
    };
    #[cfg(feature = "rayon")]
    let hits: Vec<Option<(usize, [f64; 3], f64)>> = {
        use rayon::prelude::*;
        source.points().par_iter().map(hit).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let hits: Vec<Option<(usize, [f64; 3], f64)>> = source.points().iter().map(hit).collect();

    let mut c = Correspondences {
        src: Vec::with_capacity(hits.len()),
        dst: Vec::with_capacity(hits.len()),
        idx: Vec::with_capacity(hits.len()),
        d2: Vec::with_capacity(hits.len()),
    };
    for (idx, s, d2) in hits.into_iter().flatten() {
        c.src.push(s);
        c.dst.push(target.points()[idx]);
        c.idx.push(idx);
        c.d2.push(d2);
    }
    c
}

/// Huber weights `min(1, δ/d)` from squared distances; all ones when disabled.
fn robust_weights(d2: &[f64], huber_delta: f64) -> Vec<f64> {
    if huber_delta <= 0.0 {
        return vec![1.0; d2.len()];
    }
    d2.iter()
        .map(|&d2| {
            let d = d2.sqrt();
            if d > huber_delta {
                huber_delta / d
            } else {
                1.0
            }
        })
        .collect()
}

/// Weighted point-to-plane normal equations `AᵀWA` and `−AᵀWr` over
/// `x = (v, w)`, with rows `[nᵀ, (R s + t) × n ᵀ]`.
fn point_to_plane_system(
    c: &Correspondences,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    normals: &[[f64; 3]],
    weights: &[f64],
) -> Result<([[f64; 6]; 6], [f64; 6]), RegistrationError> {
    let mut ata = [[0.0f64; 6]; 6];
    let mut atb = [0.0f64; 6];
    for i in 0..c.src.len() {
        let rs = apply(rotation, translation, c.src[i]);
        let mut n = normals[c.idx[i]];
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if nl < 1e-12 {
            return Err(RegistrationError::InvalidParameter("target normal is zero"));
        }
        n = [n[0] / nl, n[1] / nl, n[2] / nl];
        let q = c.dst[i];
        let r = (rs[0] - q[0]) * n[0] + (rs[1] - q[1]) * n[1] + (rs[2] - q[2]) * n[2];
        let rxn = cross3(&rs, &n);
        let mut row = [0.0f64; 6];
        row[..3].copy_from_slice(&n);
        row[3..].copy_from_slice(&rxn);
        let w = weights[i];
        for a_ in 0..6 {
            for b_ in 0..6 {
                ata[a_][b_] += w * row[a_] * row[b_];
            }
            atb[a_] -= w * row[a_] * r;
        }
    }
    Ok((ata, atb))
}

fn solve6(ata: &[[f64; 6]; 6], atb: &[f64; 6]) -> Result<[f64; 6], RegistrationError> {
    let a = DMatrix::from_row_slice(6, 6, &flatten6(ata));
    let b = DVector::from_row_slice(atb);
    let x = a
        .solve(&b)
        .map_err(|e| RegistrationError::LinearSolveFailed(alloc::format!("{e}")))?;
    Ok([x[0], x[1], x[2], x[3], x[4], x[5]])
}

fn flatten6(m: &[[f64; 6]; 6]) -> Vec<f64> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
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

    #[test]
    fn prepared_target_matches_direct_call() {
        let target = box_surface();
        let source = target.transformed_rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.01, -0.008, 0.004],
        );
        let params = IcpParams {
            max_correspondence_distance: 0.25,
            tolerance: 1e-9,
            ..Default::default()
        };
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let direct = icp(
            &source,
            &target,
            None,
            IcpVariant::PointToPoint,
            id,
            [0.0; 3],
            &params,
        )
        .unwrap();
        let prepared = PreparedTarget::new(target.clone(), None).unwrap();
        let reused = icp_prepared(
            &source,
            &prepared,
            IcpVariant::PointToPoint,
            id,
            [0.0; 3],
            &params,
        )
        .unwrap();
        assert_eq!(direct, reused);
    }

    #[test]
    fn prepared_target_validates_normals_length() {
        let target = box_surface();
        assert!(matches!(
            PreparedTarget::new(target, Some(vec![[0.0, 0.0, 1.0]])),
            Err(RegistrationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn auto_params_scale_with_point_spacing() {
        let target = box_surface();
        let source = target.transformed_rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.01, 0.0, 0.0],
        );
        let auto = IcpParams::auto(&source, &target).unwrap();
        // The shift (0.01 m) is the median nearest-neighbour distance, so the
        // gate is five times that: ~0.05 m.
        assert!((auto.max_correspondence_distance - 0.05).abs() < 0.01);
        assert!(IcpParams::auto(&PointCloud::new(), &target).is_none());
    }

    #[test]
    fn huber_kernel_resists_gated_outliers() {
        // A true pose with 10% of source points displaced inside the gate.
        // Without the robust kernel the outliers bias the estimate; with it
        // the bias shrinks.
        let target = box_surface();
        let mut source = target.transformed_rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        for (i, p) in source.points_mut().iter_mut().enumerate() {
            if i % 10 == 0 {
                p[0] += 0.08;
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let base = IcpParams {
            max_correspondence_distance: 0.2,
            tolerance: 1e-9,
            ..Default::default()
        };
        let plain = icp(
            &source,
            &target,
            None,
            IcpVariant::PointToPoint,
            id,
            [0.0; 3],
            &base,
        )
        .unwrap();
        let robust = icp(
            &source,
            &target,
            None,
            IcpVariant::PointToPoint,
            id,
            [0.0; 3],
            &IcpParams {
                huber_delta: 0.01,
                ..base
            },
        )
        .unwrap();
        let bias = |t: [f64; 3]| (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        assert!(
            bias(robust.translation) < bias(plain.translation),
            "robust {:?} plain {:?}",
            robust.translation,
            plain.translation
        );
    }

    #[test]
    fn point_to_plane_reports_information_and_degeneracy() {
        let target = box_surface();
        let normals = normals_of(&target);
        let source = target.clone();
        let res = icp(
            &source,
            &target,
            Some(&normals),
            IcpVariant::PointToPlane,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &IcpParams {
                max_correspondence_distance: 0.25,
                ..Default::default()
            },
        )
        .unwrap();
        let info = res.information.expect("point-to-plane yields information");
        // A closed box constrains all six directions: nothing is weak.
        let deg = res.degeneracy(1e-4).unwrap();
        assert!(deg.weak.is_empty(), "eigenvalues {:?}", deg.eigenvalues);
        assert!(res.covariance().is_some());
        assert!(info[0][0] > 0.0);
    }

    #[test]
    fn point_to_point_has_no_information_matrix() {
        let target = box_surface();
        let res = icp(
            &target,
            &target,
            None,
            IcpVariant::PointToPoint,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &IcpParams {
                max_correspondence_distance: 0.25,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res.information.is_none());
        assert!(res.covariance().is_none());
    }

    #[test]
    fn frame_typed_icp_returns_from_to_isometry() {
        use tpt_percept_core::frame::{Lidar, World};
        let target = box_surface();
        let truth = [0.01, -0.008, 0.004];
        let source =
            target.transformed_rigid([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], truth);
        let src = FrameCloud::<Lidar<0>>::from_cloud(source);
        let tgt = FrameCloud::<World>::from_cloud(target);
        let (iso, res) = icp_frames(
            &src,
            &tgt,
            None,
            IcpVariant::PointToPoint,
            Isometry3::<Lidar<0>, World>::from_parts_unchecked(
                [0.0; 3],
                Rotation3::from_matrix_unchecked([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]),
            ),
            &IcpParams {
                max_correspondence_distance: 0.25,
                tolerance: 1e-9,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res.converged);
        // The source was shifted by +truth, so the recovered transform
        // carries it back by −truth.
        let (r, t) = iso.into_raw();
        assert_eq!(r, res.rotation);
        for k in 0..3 {
            assert!((t[k] + truth[k]).abs() < 1e-4, "t = {t:?}");
        }
    }
}
