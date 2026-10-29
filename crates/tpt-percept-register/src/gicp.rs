//! Generalised ICP (Segal, Haehnel & Thrun, 2009).
//!
//! Each point carries a local covariance that describes the surface around
//! it: large in the tangent plane, small along the normal. For a
//! correspondence `(s, q)` the residual `d = q − (R s + t)` is weighted by the
//! inverse of the combined covariance `C = R Cₛ Rᵀ + C_q`, so points on
//! flat patches constrain the pose along the normal and points on edges
//! constrain it across the edge.
//!
//! Covariances are built from estimated normals as
//! `C = (I − n nᵀ) + ε · n nᵀ`, which avoids a per-point eigen-decomposition
//! and gives the same surface-aligned shape. `ε` (default `1e-3`) sets how
//! sharply the normal direction is constrained.

#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::RegistrationError;
use crate::icp::{apply, axis_angle, cross3, mul3, pose_delta, IcpParams, IcpResult};
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;
use tpt_percept_features::normal::estimate_normals;

/// G-ICP configuration.
#[derive(Clone, Copy, Debug)]
pub struct GicpParams {
    /// Iteration, gate and convergence settings shared with ICP.
    pub icp: IcpParams,
    /// Neighbourhood size for normal estimation (per cloud).
    pub normal_k: usize,
    /// Variance along the normal, relative to the tangent plane (`ε`).
    pub epsilon: f64,
}

impl Default for GicpParams {
    fn default() -> Self {
        GicpParams {
            icp: IcpParams::default(),
            normal_k: 10,
            epsilon: 1e-3,
        }
    }
}

/// Aligns `source` onto `target` with generalised ICP, starting from
/// `(init_rotation, init_translation)`.
///
/// The returned [`IcpResult`] follows the ICP conventions: `information` is
/// the Gauss–Newton Hessian at the final pose over `(translation, rotation)`,
/// so [`IcpResult::covariance`] and [`IcpResult::degeneracy`] apply.
pub fn gicp(
    source: &PointCloud,
    target: &PointCloud,
    init_rotation: [[f64; 3]; 3],
    init_translation: [f64; 3],
    params: &GicpParams,
) -> Result<IcpResult, RegistrationError> {
    if source.is_empty() || target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !(params.epsilon.is_finite() && params.epsilon > 0.0) {
        return Err(RegistrationError::InvalidParameter("epsilon must be > 0"));
    }
    let icp_params = params.icp;
    if !(icp_params.max_correspondence_distance.is_finite()
        && icp_params.max_correspondence_distance > 0.0)
    {
        return Err(RegistrationError::InvalidParameter(
            "max_correspondence_distance must be finite and > 0",
        ));
    }
    if icp_params.max_iterations == 0 {
        return Err(RegistrationError::InvalidParameter(
            "max_iterations must be > 0",
        ));
    }

    let source_cov = covariances(source, params)?;
    let target_cov = covariances(target, params)?;
    let tree = KdTree::new(target.points());
    let max_d2 = icp_params.max_correspondence_distance.powi(2);

    let mut rotation = init_rotation;
    let mut translation = init_translation;
    let mut converged = false;
    let mut iterations = 0u32;

    while iterations < icp_params.max_iterations {
        iterations += 1;
        let (h, g, count) = normal_equations(
            source,
            target,
            &source_cov,
            &target_cov,
            &tree,
            &rotation,
            &translation,
            max_d2,
        );
        if count < icp_params.min_correspondences {
            break;
        }
        let delta = solve6(&h, &g)?;
        let (v, w) = (
            [delta[0], delta[1], delta[2]],
            [delta[3], delta[4], delta[5]],
        );
        let wn = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        let delta_rot = if wn < 1e-12 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            axis_angle(w, wn)?
        };
        let new_rotation = mul3(&delta_rot, &rotation);
        let new_translation = [
            translation[0] + v[0],
            translation[1] + v[1],
            translation[2] + v[2],
        ];
        let step = pose_delta(&rotation, &translation, &new_rotation, &new_translation);
        rotation = new_rotation;
        translation = new_translation;
        if step < icp_params.tolerance {
            converged = true;
            break;
        }
    }

    // Final statistics and information at the resulting pose.
    let (h, _, count) = normal_equations(
        source,
        target,
        &source_cov,
        &target_cov,
        &tree,
        &rotation,
        &translation,
        max_d2,
    );
    let mut sum_d2 = 0.0;
    let mut inliers = 0usize;
    for &s in source.points() {
        let ts = apply(&rotation, &translation, s);
        if let Some((_, d2)) = tree.nearest(&ts) {
            if d2 <= max_d2 {
                inliers += 1;
                sum_d2 += d2;
            }
        }
    }
    let rmse = if inliers == 0 {
        f64::INFINITY
    } else {
        (sum_d2 / inliers as f64).sqrt()
    };
    let information = (count > 0).then_some(h);

    Ok(IcpResult {
        rotation,
        translation,
        fitness: count as f64 / source.len() as f64,
        rmse,
        iterations,
        converged,
        information,
    })
}

/// Per-point covariance `(I − nnᵀ) + ε·nnᵀ` from estimated normals.
fn covariances(
    cloud: &PointCloud,
    params: &GicpParams,
) -> Result<Vec<[[f64; 3]; 3]>, RegistrationError> {
    let normals = estimate_normals(cloud, params.normal_k).ok_or(
        RegistrationError::InvalidParameter("cloud too small for normals"),
    )?;
    let eps = params.epsilon;
    Ok(normals
        .iter()
        .map(|n| {
            let v = n.normal;
            let mut c = [[0.0f64; 3]; 3];
            for r in 0..3 {
                for col in 0..3 {
                    let identity = if r == col { 1.0 } else { 0.0 };
                    let outer = v[r] * v[col];
                    c[r][col] = (identity - outer) + eps * outer;
                }
            }
            c
        })
        .collect())
}

/// Accumulates `H = Σ JᵀWJ` and `g = Σ JᵀW e₀` over inlier correspondences,
/// where `W = (R Cₛ Rᵀ + C_q)⁻¹` and `J = [−I | [x]×]`.
#[allow(clippy::too_many_arguments)]
fn normal_equations(
    source: &PointCloud,
    target: &PointCloud,
    source_cov: &[[[f64; 3]; 3]],
    target_cov: &[[[f64; 3]; 3]],
    tree: &KdTree,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    max_d2: f64,
) -> ([[f64; 6]; 6], [f64; 6], usize) {
    let mut h = [[0.0f64; 6]; 6];
    let mut g = [0.0f64; 6];
    let mut count = 0usize;
    for (i, &s) in source.points().iter().enumerate() {
        let x = apply(rotation, translation, s);
        let Some((idx, d2)) = tree.nearest(&x) else {
            continue;
        };
        if d2 > max_d2 {
            continue;
        }
        count += 1;
        let q = target.points()[idx];
        let e0 = [q[0] - x[0], q[1] - x[1], q[2] - x[2]];

        // Combined covariance R Cₛ Rᵀ + C_q, inverted.
        let rc = mul3(rotation, &source_cov[i]);
        let rt = transpose(rotation);
        let rcr = mul3(&rc, &rt);
        let mut cov = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] = rcr[r][c] + target_cov[idx][r][c];
            }
        }
        let w = invert3(&cov);

        // J = [−I | [x]×]; e(δ) ≈ e₀ − δt + (x × δω).
        let xc = [[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]];
        let mut j = [[0.0f64; 6]; 3];
        for r in 0..3 {
            j[r][r] = -1.0;
            for c in 0..3 {
                j[r][3 + c] = xc[r][c];
            }
        }
        // Wj = W J (3×6), then H += Jᵀ (W J), g += Jᵀ (W e₀).
        let mut wj = [[0.0f64; 6]; 3];
        for r in 0..3 {
            for c in 0..6 {
                for k in 0..3 {
                    wj[r][c] += w[r][k] * j[k][c];
                }
            }
        }
        let mut we = [0.0f64; 3];
        for r in 0..3 {
            for k in 0..3 {
                we[r] += w[r][k] * e0[k];
            }
        }
        for a in 0..6 {
            for b in 0..6 {
                for r in 0..3 {
                    h[a][b] += j[r][a] * wj[r][b];
                }
            }
            for r in 0..3 {
                g[a] += j[r][a] * we[r];
            }
        }
    }
    (h, g, count)
}

fn solve6(h: &[[f64; 6]; 6], g: &[f64; 6]) -> Result<[f64; 6], RegistrationError> {
    // A tiny relative diagonal term keeps weakly constrained directions
    // solvable without changing well-conditioned solutions measurably.
    let mut flat = Vec::with_capacity(36);
    for (i, row) in h.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let extra = if i == j {
                1e-9 * v.abs().max(1e-12)
            } else {
                0.0
            };
            flat.push(v + extra);
        }
    }
    let a = DMatrix::from_row_slice(6, 6, &flat);
    let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
    let x = a
        .solve(&DVector::from_row_slice(&rhs))
        .map_err(|e| RegistrationError::LinearSolveFailed(alloc::format!("{e}")))?;
    Ok([x[0], x[1], x[2], x[3], x[4], x[5]])
}

fn transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Closed-form 3×3 inverse. Falls back to a regularised diagonal when the
/// matrix is singular, which happens for points without a usable normal.
fn invert3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c0 = cross3(&m[1], &m[2]);
    let c1 = cross3(&m[2], &m[0]);
    let c2 = cross3(&m[0], &m[1]);
    let det = m[0][0] * c0[0] + m[0][1] * c0[1] + m[0][2] * c0[2];
    if det.abs() < 1e-12 {
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            out[i][i] = 1.0 / (m[i][i].abs().max(1e-6));
        }
        return out;
    }
    // With rows r₀, r₁, r₂, the cross products cᵢ = r_{i+1} × r_{i+2} satisfy
    // M·[c₀ c₁ c₂]ᵀ = det·I, so the rows of M⁻¹ are cᵢ / det.
    let inv = 1.0 / det;
    [
        [c0[0] * inv, c0[1] * inv, c0[2] * inv],
        [c1[0] * inv, c1[1] * inv, c1[2] * inv],
        [c2[0] * inv, c2[1] * inv, c2[2] * inv],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_surface() -> PointCloud {
        let mut c = PointCloud::new();
        let vals: Vec<f64> = (-20..=20i32).map(|i| i as f64 * 0.05).collect();
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

    #[test]
    fn invert3_is_an_inverse() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let inv = invert3(&m);
        for r in 0..3 {
            for c in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += m[r][k] * inv[k][c];
                }
                let expect = if r == c { 1.0 } else { 0.0 };
                assert!((acc - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn gicp_recovers_small_pose() {
        let target = box_surface();
        let source = target.transformed_rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.01, -0.008, 0.004],
        );
        let params = GicpParams {
            icp: IcpParams {
                max_correspondence_distance: 0.25,
                tolerance: 1e-9,
                ..IcpParams::default()
            },
            ..GicpParams::default()
        };
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let res = gicp(&source, &target, id, [0.0; 3], &params).unwrap();
        assert!(res.converged, "{res:?}");
        assert!(res.rmse < 1e-4, "rmse {}", res.rmse);
        for (k, want) in [-0.01, 0.008, -0.004].iter().enumerate() {
            assert!(
                (res.translation[k] - want).abs() < 2e-3,
                "t = {:?}",
                res.translation
            );
        }
        assert!(res.information.is_some());
    }

    #[test]
    fn gicp_rejects_empty_input() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(
            gicp(
                &PointCloud::new(),
                &box_surface(),
                id,
                [0.0; 3],
                &GicpParams::default()
            ),
            Err(RegistrationError::EmptyCloud)
        );
    }
}
