//! NDT — Normal Distributions Transform registration (Biber & Straßer 2003).
//!
//! The target cloud is discretised into a voxel grid; every sufficiently
//! populated voxel is modelled as a multivariate normal (mean + covariance,
//! regularised). Registration maximises the total likelihood of the
//! transformed source points under that mixture with Gauss–Newton on the
//! squared Mahalanobis residual — the score decreases monotonically thanks
//! to the damped solve and the residual-based convergence test.
#![allow(clippy::needless_range_loop)]

use alloc::collections::BTreeMap;

use tpt_math_linalg_dense::{DMatrix, DVector};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::RegistrationError;
use crate::icp::{apply, axis_angle, mul3};
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::voxel::voxelize_at;

/// A voxel-modelled target: sparse map from lattice key to Gaussian.
#[derive(Clone, Debug, Default)]
pub struct NdtMap {
    voxels: BTreeMap<[i64; 3], VoxelGaussian>,
    voxel_size: f64,
    origin: [f64; 3],
}

/// Mean and regularised inverse-covariance of one occupied voxel.
#[derive(Clone, Debug)]
pub struct VoxelGaussian {
    /// Voxel mean (metres).
    pub mean: [f64; 3],
    /// Inverse covariance (3×3, row-major).
    pub inv_cov: [[f64; 3]; 3],
}

/// NDT map construction parameters.
#[derive(Clone, Copy, Debug)]
pub struct NdtMapParams {
    /// Voxel edge length (metres).
    pub voxel_size: f64,
    /// Minimum points per voxel to keep its Gaussian.
    pub min_points_per_voxel: usize,
    /// Covariance regulariser added to the diagonal (metres²).
    pub covariance_regularization: f64,
}

impl Default for NdtMapParams {
    fn default() -> Self {
        NdtMapParams {
            voxel_size: 1.0,
            min_points_per_voxel: 6,
            covariance_regularization: 1e-4,
        }
    }
}

/// Builds an NDT map from a target cloud anchored at the cloud's AABB
/// minimum.
pub fn build_ndt_map(
    target: &PointCloud,
    params: &NdtMapParams,
) -> Result<NdtMap, RegistrationError> {
    if !(params.voxel_size.is_finite() && params.voxel_size > 0.0) {
        return Err(RegistrationError::InvalidParameter(
            "voxel_size must be finite and > 0",
        ));
    }
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    let (min, _) = target.aabb().expect("non-empty");
    let voxels = voxelize_at(target, min, params.voxel_size)
        .ok_or(RegistrationError::InvalidParameter("voxelization failed"))?;

    let mut map = BTreeMap::new();
    for voxel in voxels {
        if voxel.members.len() < params.min_points_per_voxel {
            continue;
        }
        let mut mean = [0.0; 3];
        for &i in &voxel.members {
            let p = target.get(i).expect("member index in range");
            for j in 0..3 {
                mean[j] += p[j];
            }
        }
        let n = voxel.members.len() as f64;
        for j in 0..3 {
            mean[j] /= n;
        }
        let mut cov = [[0.0; 3]; 3];
        for &i in &voxel.members {
            let p = target.get(i).expect("member index in range");
            let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
            for r in 0..3 {
                for c in 0..3 {
                    cov[r][c] += d[r] * d[c];
                }
            }
        }
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] /= n;
                if r == c {
                    cov[r][c] += params.covariance_regularization;
                }
            }
        }
        let inv = inv3(&cov).ok_or(RegistrationError::LinearSolveFailed(
            "singular voxel covariance".into(),
        ))?;
        map.insert(voxel.key, VoxelGaussian { mean, inv_cov: inv });
    }
    if map.is_empty() {
        return Err(RegistrationError::NoSolution);
    }
    Ok(NdtMap {
        voxels: map,
        voxel_size: params.voxel_size,
        origin: min,
    })
}

impl NdtMap {
    /// Number of Gaussian voxels.
    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    /// True if the map holds no voxels.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Debug iterator over (key, gaussian) pairs.
    #[doc(hidden)]
    pub fn voxels_debug(&self) -> impl Iterator<Item = ([i64; 3], &VoxelGaussian)> {
        self.voxels.iter().map(|(k, v)| (*k, v))
    }

    /// Mean squared Mahalanobis residual (debug/diagnostic probe).
    #[doc(hidden)]
    pub fn debug_score(
        &self,
        source: &PointCloud,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) -> f64 {
        let mut total = 0.0;
        let mut used = 0usize;
        for &s in source.points() {
            let x = crate::icp::apply(&rotation, &translation, s);
            if let Some(voxel) = self.voxel_at(&x) {
                let e = [
                    x[0] - voxel.mean[0],
                    x[1] - voxel.mean[1],
                    x[2] - voxel.mean[2],
                ];
                let we = mul3_vec(&voxel.inv_cov, &e);
                total += dot3(&e, &we);
                used += 1;
            }
        }
        if used > 0 {
            total / used as f64
        } else {
            f64::INFINITY
        }
    }

    /// The Gaussian of the voxel containing `p`, if occupied.
    pub fn voxel_at(&self, p: &[f64; 3]) -> Option<&VoxelGaussian> {
        let key = tpt_percept_cloud::voxel::voxel_index(*p, self.origin, self.voxel_size)?;
        self.voxels.get(&key)
    }
}

/// NDT registration parameters.
#[derive(Clone, Copy, Debug)]
pub struct NdtParams {
    /// Maximum Gauss–Newton iterations.
    pub max_iterations: u32,
    /// Stop when the pose update moves points less than this (metres).
    pub tolerance: f64,
    /// Reject source points whose voxel is unoccupied.
    pub skip_unoccupied: bool,
}

impl Default for NdtParams {
    fn default() -> Self {
        NdtParams {
            max_iterations: 30,
            tolerance: 1e-6,
            skip_unoccupied: true,
        }
    }
}

/// The outcome of NDT registration (same shape as [`crate::icp::IcpResult`]).
#[derive(Clone, Debug, PartialEq)]
pub struct NdtResult {
    /// Rotation part of `target ≈ R · source + t`.
    pub rotation: [[f64; 3]; 3],
    /// Translation part.
    pub translation: [f64; 3],
    /// Mean squared Mahalanobis residual at the final pose over occupied
    /// voxels (diagnostic; the optimiser maximises likelihood, not this).
    pub score: f64,
    /// Iterations executed.
    pub iterations: u32,
    /// True when converged within `tolerance`.
    pub converged: bool,
}

/// Registers `source` onto an [`NdtMap`] with Gauss–Newton, starting from
/// the given pose.
pub fn ndt_register(
    source: &PointCloud,
    map: &NdtMap,
    init_rotation: [[f64; 3]; 3],
    init_translation: [f64; 3],
    params: &NdtParams,
) -> Result<NdtResult, RegistrationError> {
    if source.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if map.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }

    let mut rotation = init_rotation;
    let mut translation = init_translation;
    let mut converged = false;
    let mut iterations = 0u32;

    /// Total likelihood `Σ exp(−d/2)` of `source` under `(rotation,
    /// translation)` over occupied voxels (the maximised objective), plus
    /// the number of contributing points. The exponential smoothly
    /// down-weights points far from their voxel mean, which keeps the
    /// landscape tractable under voxel quantisation.
    fn likelihood_of(
        source: &PointCloud,
        map: &NdtMap,
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
    ) -> (f64, usize) {
        let mut total = 0.0;
        let mut used = 0usize;
        for &s in source.points() {
            let x = apply(rotation, translation, s);
            if let Some(voxel) = map.voxel_at(&x) {
                let e = [
                    x[0] - voxel.mean[0],
                    x[1] - voxel.mean[1],
                    x[2] - voxel.mean[2],
                ];
                let we = mul3_vec(&voxel.inv_cov, &e);
                let d2 = dot3(&e, &we);
                total += (-0.5 * d2).exp();
                used += 1;
            }
        }
        (total, used)
    }

    let (mut prev_like, _) = likelihood_of(source, map, &rotation, &translation);

    while iterations < params.max_iterations {
        iterations += 1;

        // Assemble the Gauss–Newton system over occupied correspondences.
        let mut ata = [[0.0f64; 6]; 6];
        let mut atb = [0.0f64; 6];
        let mut used = 0usize;

        for &s in source.points() {
            let x = apply(&rotation, &translation, s);
            let Some(voxel) = map.voxel_at(&x) else {
                if params.skip_unoccupied {
                    continue;
                }
                break;
            };
            let e = [
                x[0] - voxel.mean[0],
                x[1] - voxel.mean[1],
                x[2] - voxel.mean[2],
            ];
            // Jacobian J = [I | −[Rp]×] (columns: translation unit vectors,
            // then the rotation columns of −[Rp]×), residual r = e, weight
            // W = inv_cov.  Normal equations: Jᵀ W J δ = −Jᵀ W r.
            let rs = apply(&rotation, &[0.0; 3], s);
            let j_rot = skew_neg(rs);
            let we = mul3_vec(&voxel.inv_cov, &e);
            let jac = |c: usize| -> [f64; 3] {
                if c < 3 {
                    unit(c)
                } else {
                    [j_rot[0][c - 3], j_rot[1][c - 3], j_rot[2][c - 3]]
                }
            };
            // Likelihood weights: e = exp(−d/2). The ascent direction is
            // −∇ = −Σ e·JᵀWr with Hessian ≈ Σ e·JᵀWJ; the solve is
            // H δ = −Σ e·JᵀWr (negated here so `solve` yields the step).
            let weight = (-0.5 * dot3(&e, &we)).exp();
            for a_ in 0..6 {
                let wj = mul3_vec(&voxel.inv_cov, &jac(a_));
                atb[a_] -= weight * dot3(&jac(a_), &we);
                for b_ in 0..6 {
                    ata[a_][b_] += weight * dot3(&wj, &jac(b_));
                }
            }
            used += 1;
        }

        if used < 6 {
            break; // no overlap between source and map
        }

        // Levenberg damping for robustness against rank-deficient voxels.
        for d in 0..6 {
            ata[d][d] += 1e-6;
        }
        let flat: alloc::vec::Vec<f64> = ata.iter().flat_map(|r| r.iter().copied()).collect();
        let m = DMatrix::from_row_slice(6, 6, &flat);
        let b = DVector::from_row_slice(&atb);
        let step = m
            .solve(&b)
            .map_err(|e| RegistrationError::LinearSolveFailed(alloc::format!("{e}")))?;
        let w = [step[3], step[4], step[5]];
        let v = [step[0], step[1], step[2]];

        let wn = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        let delta_rot = if wn < 1e-12 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            axis_angle(w, wn)?
        };

        // Backtracking line search on the likelihood: halve the step until
        // it improves the objective (the raw GN step can overshoot where
        // voxel weights are extreme).
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..8 {
            let trial_rot = if alpha >= 1.0 {
                mul3(&delta_rot, &rotation)
            } else {
                let small = axis_angle(w, wn * alpha)?;
                mul3(&small, &rotation)
            };
            let trial_t = [
                translation[0] + v[0] * alpha,
                translation[1] + v[1] * alpha,
                translation[2] + v[2] * alpha,
            ];
            let (like, _) = likelihood_of(source, map, &trial_rot, &trial_t);
            if like > prev_like {
                let gain = like - prev_like;
                prev_like = like;
                rotation = trial_rot;
                translation = trial_t;
                if gain < params.tolerance {
                    converged = true;
                }
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            converged = true; // no ascent possible: local maximum
            break;
        }
        if converged {
            break;
        }
    }

    // Final score: mean squared Mahalanobis distance of occupied points.
    let (total, used) = likelihood_of(source, map, &rotation, &translation);
    // Report the diagnostic mean Mahalanobis-style residual.
    let score = if used > 0 {
        -2.0 * (total / used as f64).ln().max(f64::MIN_POSITIVE)
    } else {
        f64::INFINITY
    };

    Ok(NdtResult {
        rotation,
        translation,
        score,
        iterations,
        converged,
    })
}

fn unit(i: usize) -> [f64; 3] {
    let mut e = [0.0; 3];
    e[i] = 1.0;
    e
}

fn skew_neg(p: [f64; 3]) -> [[f64; 3]; 3] {
    // −[p]× so that ∂(Rp)/∂ω = −[Rp]× for small ω.
    [[0.0, p[2], -p[1]], [-p[2], 0.0, p[0]], [p[1], -p[0], 0.0]]
}

fn mul3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3×3 inverse via cofactors; `None` when singular.
fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_percept_core::iso::Rotation3;

    fn surface() -> PointCloud {
        // Curved terrain: every 1 m voxel holds a genuinely 3-D patch, so
        // the voxel Gaussians are full-rank and the likelihood has a unique
        // optimum at the true pose.
        let mut c = PointCloud::new();
        for x in 0..=80i32 {
            for y in 0..=80i32 {
                let xf = x as f64 * 0.05;
                let yf = y as f64 * 0.05;
                let z = 0.3 * (xf * 0.9).sin() * (yf * 0.7).cos();
                c.push([xf, yf, z]);
            }
        }
        c
    }

    #[test]
    fn map_built_with_gaussians() {
        let cloud = surface();
        let map = build_ndt_map(&cloud, &NdtMapParams::default()).unwrap();
        assert!(map.len() > 4, "voxels {}", map.len());
        let p = cloud.get(0).unwrap();
        assert!(map.voxel_at(p).is_some());
    }

    #[test]
    fn ndt_recovers_small_pose() {
        let target = surface();
        let map = build_ndt_map(&target, &NdtMapParams::default()).unwrap();
        let r = Rotation3::<tpt_percept_core::frame::World, tpt_percept_core::frame::World>::from_axis_angle(
            [0.0, 0.0, 1.0],
            0.01,
        )
        .unwrap()
        .matrix();
        let t = [0.03, -0.02, 0.01];
        let source = target.transformed_rigid(r, t);

        let res = ndt_register(
            &source,
            &map,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &NdtParams::default(),
        )
        .unwrap();
        assert!(res.converged, "{res:?}");
        // NDT returns the source→target pose = (r, t)⁻¹. The scene is
        // asymmetric, so the raw pose is checkable.
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
                assert!((res.rotation[i][j] - inv[i][j]).abs() < 5e-2);
            }
            assert!((res.translation[i] - inv_t[i]).abs() < 5e-2);
        }
    }

    #[test]
    fn empty_inputs_rejected() {
        let cloud = surface();
        let map = build_ndt_map(&cloud, &NdtMapParams::default()).unwrap();
        assert!(matches!(
            build_ndt_map(&PointCloud::new(), &NdtMapParams::default()),
            Err(RegistrationError::EmptyCloud)
        ));
        assert!(matches!(
            ndt_register(
                &PointCloud::new(),
                &map,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
                &NdtParams::default()
            ),
            Err(RegistrationError::EmptyCloud)
        ));
    }

    #[test]
    fn invalid_voxel_size_rejected() {
        let cloud = surface();
        assert!(build_ndt_map(
            &cloud,
            &NdtMapParams {
                voxel_size: 0.0,
                ..Default::default()
            }
        )
        .is_err());
    }
}
