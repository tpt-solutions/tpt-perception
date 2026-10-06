//! Rigid and similarity alignment from point correspondences.
//!
//! [`kabsch`] solves the *orthogonal Procrustes* problem — the closed-form
//! rigid alignment used inside ICP, loop closure verification and extrinsic
//! calibration. [`umeyama`] extends it with a uniform scale (hand–eye
//! calibration between differently-scaled sensor models).
//!
//! Both take weighted 3-D correspondence sets `(srcᵢ, dstᵢ)` and return the
//! transform minimising `Σ wᵢ ‖dstᵢ − (s·R·srcᵢ + t)‖²`.
#![allow(clippy::needless_range_loop)]

use tpt_math_linalg_fixed::Matrix3;

use crate::error::{CoreError, CoreResult};
use crate::linalg3::{det3, sym_eigen3};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

/// Minimum condition threshold: singular values below this fraction of the
/// largest singular value are treated as zero (rank-deficient alignment).
const RANK_TOL: f64 = 1e-10;

/// A rigid transform `dst ≈ R·src + t` fitted to correspondences.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidAlignment {
    /// Rotation (proper, orthonormal).
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres, applied after rotation).
    pub translation: [f64; 3],
}

/// A similarity transform `dst ≈ s·R·src + t` fitted to correspondences.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityAlignment {
    /// Rotation (proper, orthonormal).
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
    /// Uniform scale (dimensionless).
    pub scale: f64,
}

/// Weighted Kabsch alignment: the rigid transform minimising
/// `Σ wᵢ ‖dstᵢ − (R·srcᵢ + t)‖²`.
///
/// Contract: with ≥ 3 non-collinear correspondences the minimiser is unique
/// and returned exactly (up to rounding). Rank-deficient (collinear or
/// repeated) configurations return [`CoreError::Degenerate`].
pub fn kabsch_weighted(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    weights: &[f64],
) -> CoreResult<RigidAlignment> {
    if weights.len() != src.len() {
        return Err(CoreError::DimensionMismatch {
            what: "src and weights must have equal lengths",
        });
    }
    kabsch_impl(src, dst, &|i: usize| weights[i])
}

/// Unweighted [`kabsch_weighted`].
pub fn kabsch(src: &[[f64; 3]], dst: &[[f64; 3]]) -> CoreResult<RigidAlignment> {
    kabsch_impl(src, dst, &|_: usize| 1.0)
}

/// Weight accessor shared by the weighted/unweighted solvers.
type WeightFn<'a> = &'a dyn Fn(usize) -> f64;

fn kabsch_impl(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    weight: WeightFn<'_>,
) -> CoreResult<RigidAlignment> {
    if src.len() != dst.len() {
        return Err(CoreError::DimensionMismatch {
            what: "src and dst must have equal lengths",
        });
    }
    if src.is_empty() {
        return Err(CoreError::Degenerate {
            what: "no correspondences",
        });
    }
    let n = src.len();

    let mut w_sum = 0.0;
    let mut cs = [0.0; 3];
    let mut cd = [0.0; 3];
    for i in 0..n {
        let w = weight(i);
        if !(w.is_finite() && w >= 0.0) {
            return Err(CoreError::InvalidInput {
                what: "weights must be finite and non-negative",
            });
        }
        w_sum += w;
        for j in 0..3 {
            cs[j] += w * src[i][j];
            cd[j] += w * dst[i][j];
        }
    }
    if w_sum <= 0.0 {
        return Err(CoreError::Degenerate {
            what: "total weight is zero",
        });
    }
    for j in 0..3 {
        cs[j] /= w_sum;
        cd[j] /= w_sum;
    }

    // Cross-covariance H = Σ w (src − cs)(dst − cd)ᵀ.
    let mut h = [[0.0_f64; 3]; 3];
    for i in 0..n {
        let w = weight(i);
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += w * (src[i][r] - cs[r]) * (dst[i][c] - cd[c]);
            }
        }
    }

    let (u, _sigma, v) = svd3_from_eigen(&h)?;
    // D flips the last axis when det(V Uᵀ) < 0 to keep det R = +1.
    let det_v_u_t = {
        let u_t = transpose3(&u);
        det3(&mul3(&v, &u_t))
    };
    let mut d = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    d[2][2] = det_v_u_t.signum();

    // R = V D Uᵀ.
    let u_t = transpose3(&u);
    let rotation = mul3(&mul3(&v, &d), &u_t);
    // Rank-deficient correspondence sets (collinear/repeated points) were
    // rejected by `svd3_from_eigen`; the fitted transform is unique here.

    let translation = [
        cd[0] - dot_row(&rotation, 0, &cs),
        cd[1] - dot_row(&rotation, 1, &cs),
        cd[2] - dot_row(&rotation, 2, &cs),
    ];
    Ok(RigidAlignment {
        rotation,
        translation,
    })
}

/// Unweighted [`umeyama_weighted`].
pub fn umeyama(src: &[[f64; 3]], dst: &[[f64; 3]]) -> CoreResult<SimilarityAlignment> {
    umeyama_impl(src, dst, &|_: usize| 1.0)
}

/// Weighted Umeyama: uniform-scale extension of [`kabsch_impl`].
fn umeyama_impl(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    weight: WeightFn<'_>,
) -> CoreResult<SimilarityAlignment> {
    let rigid = kabsch_impl(src, dst, weight)?;
    // Optimal scale for the fitted rotation:
    //   s = Σ w (R src)·(dst − cd) / Σ w ‖src − cs‖²
    // (numerator equals tr(D Σ) of the Umeyama SVD at the Kabsch optimum).
    let (cs, cd) = centroids(src, dst, weight)?;
    let n = src.len();
    let mut variance = 0.0;
    let mut numerator = 0.0;
    for i in 0..n {
        let w = weight(i);
        let rs = [
            dot_row(&rigid.rotation, 0, &src[i]),
            dot_row(&rigid.rotation, 1, &src[i]),
            dot_row(&rigid.rotation, 2, &src[i]),
        ];
        numerator += w
            * (rs[0] * (dst[i][0] - cd[0])
                + rs[1] * (dst[i][1] - cd[1])
                + rs[2] * (dst[i][2] - cd[2]));
        variance += w
            * ((src[i][0] - cs[0]).powi(2)
                + (src[i][1] - cs[1]).powi(2)
                + (src[i][2] - cs[2]).powi(2));
    }
    if variance <= 1e-300 {
        return Err(CoreError::Degenerate {
            what: "source points coincide; scale is undefined",
        });
    }
    let scale = numerator / variance;
    if scale <= 0.0 {
        return Err(CoreError::Degenerate {
            what: "fitted scale is non-positive",
        });
    }
    let translation = [
        cd[0] - scale * dot_row(&rigid.rotation, 0, &cs),
        cd[1] - scale * dot_row(&rigid.rotation, 1, &cs),
        cd[2] - scale * dot_row(&rigid.rotation, 2, &cs),
    ];
    Ok(SimilarityAlignment {
        rotation: rigid.rotation,
        translation,
        scale,
    })
}

/// Weighted Umeyama alignment: the similarity transform minimising
/// `Σ wᵢ ‖dstᵢ − (s·R·srcᵢ + t)‖²` over uniform `s > 0` and rigid `(R, t)`.
pub fn umeyama_weighted(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    weights: &[f64],
) -> CoreResult<SimilarityAlignment> {
    if weights.len() != src.len() {
        return Err(CoreError::DimensionMismatch {
            what: "src and weights must have equal lengths",
        });
    }
    umeyama_impl(src, dst, &|i: usize| weights[i])
}

fn centroids(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    weight: WeightFn<'_>,
) -> CoreResult<([f64; 3], [f64; 3])> {
    let mut w_sum = 0.0;
    let mut cs = [0.0; 3];
    let mut cd = [0.0; 3];
    for (i, (s, d)) in src.iter().zip(dst).enumerate() {
        let w = weight(i);
        w_sum += w;
        for j in 0..3 {
            cs[j] += w * s[j];
            cd[j] += w * d[j];
        }
    }
    if w_sum <= 0.0 {
        return Err(CoreError::Degenerate {
            what: "total weight is zero",
        });
    }
    for j in 0..3 {
        cs[j] /= w_sum;
        cd[j] /= w_sum;
    }
    Ok((cs, cd))
}

/// Thin SVD `H = U Σ Vᵀ` of a 3×3 matrix.
///
/// Returns `(U, σ descending, V)` — the third element is **V itself** (its
/// columns are the right singular vectors), not `Vᵀ`. Computed via the
/// eigendecomposition of `HᵀH`; a rank-2 `H` gets its `U` completed
/// orthonormally. Errors only for rank ≤ 1.
pub fn svd3(h: &[[f64; 3]; 3]) -> CoreResult<Svd3> {
    svd3_from_eigen(h)
}

/// Thin SVD `H = U Σ Vᵀ` of a 3×3 matrix via the eigendecomposition of
/// `HᵀH` (right singular vectors) with `U = H V Σ⁻¹`. Returns
/// `(U, σ descending, Vᵀ)`; errors when `H` is rank < 2.
/// `(U, σ descending, V)` factors of a 3×3 matrix.
pub type Svd3 = ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]);

fn svd3_from_eigen(h: &[[f64; 3]; 3]) -> CoreResult<Svd3> {
    let ht = transpose3(h);
    let hth = mul3(&ht, h);
    let e = sym_eigen3(&Matrix3::new(hth));

    let sigma = [
        e.values[0].max(0.0).sqrt(),
        e.values[1].max(0.0).sqrt(),
        e.values[2].max(0.0).sqrt(),
    ];
    // Rank ≤ 1 (collinear/repeated correspondences) has no unique rotation.
    // Rank 2 (coplanar — including every minimal 3-point set) is fine: the
    // rotation is unique up to the reflection fix below.
    if sigma[0] <= 0.0 || sigma[1] <= RANK_TOL * sigma[0] {
        return Err(CoreError::Degenerate {
            what: "cross-covariance is rank-deficient; alignment is not unique",
        });
    }
    // V matrix: v[r][col] = component r of eigenvector col (= right singular
    // vector) for σ[col].
    let mut v = [[0.0_f64; 3]; 3];
    for col in 0..3 {
        for r in 0..3 {
            v[r][col] = e.vectors[col].data[r];
        }
    }
    let mut u = [[0.0_f64; 3]; 3];
    for col in 0..3 {
        if sigma[col] > RANK_TOL * sigma[0] {
            for r in 0..3 {
                u[r][col] =
                    (h[r][0] * v[0][col] + h[r][1] * v[1][col] + h[r][2] * v[2][col]) / sigma[col];
            }
        }
    }
    if sigma[2] <= RANK_TOL * sigma[0] {
        // Complete U with its cross product so the third singular direction
        // (unconstrained by the planar data) is still orthonormal.
        let u1 = [u[0][0], u[1][0], u[2][0]];
        let u2 = [u[0][1], u[1][1], u[2][1]];
        let u3 = [
            u1[1] * u2[2] - u1[2] * u2[1],
            u1[2] * u2[0] - u1[0] * u2[2],
            u1[0] * u2[1] - u1[1] * u2[0],
        ];
        let n = (u3[0] * u3[0] + u3[1] * u3[1] + u3[2] * u3[2]).sqrt();
        if n > 1e-12 {
            for r in 0..3 {
                u[r][2] = u3[r] / n;
            }
        }
    }
    Ok((u, sigma, v))
}

fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn dot_row(m: &[[f64; 3]; 3], row: usize, v: &[f64; 3]) -> f64 {
    m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::World;
    use crate::iso::Rotation3;
    use alloc::{vec, vec::Vec};

    fn apply(m: &[[f64; 3]; 3], t: &[f64; 3], p: [f64; 3]) -> [f64; 3] {
        [
            dot_row(m, 0, &p) + t[0],
            dot_row(m, 1, &p) + t[1],
            dot_row(m, 2, &p) + t[2],
        ]
    }

    #[test]
    fn kabsch_recovers_exact_transform() {
        let r = Rotation3::<World, World>::from_axis_angle([0.2, 1.0, -0.4], 0.9)
            .unwrap()
            .matrix();
        let t = [3.0, -7.0, 11.0];
        let src: Vec<[f64; 3]> = (0..25)
            .map(|i| {
                let f = i as f64;
                [f.sin() * 5.0, f * 0.3 - 2.0, (f * 1.7).cos() * 3.0]
            })
            .collect();
        let dst: Vec<[f64; 3]> = src.iter().map(|&p| apply(&r, &t, p)).collect();

        let fit = kabsch(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let got = apply(&fit.rotation, &fit.translation, *s);
            for i in 0..3 {
                assert!((got[i] - d[i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn kabsch_collinear_rejected() {
        let src: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        let dst = src.clone();
        assert!(matches!(
            kabsch(&src, &dst),
            Err(CoreError::Degenerate { .. })
        ));
    }

    #[test]
    fn umeyama_recovers_similarity() {
        let scale = 2.5;
        let r = Rotation3::<World, World>::from_axis_angle([0.0, 0.0, 1.0], 0.5)
            .unwrap()
            .matrix();
        let t = [1.0, 2.0, 3.0];
        let src: Vec<[f64; 3]> = (0..30)
            .map(|i| {
                let f = i as f64;
                [(f * 0.7).cos() * 2.0, f.sin() * 1.5, f * 0.11]
            })
            .collect();
        let dst: Vec<[f64; 3]> = src
            .iter()
            .map(|&p| {
                let q = apply(&r, &[0.0; 3], p);
                [
                    scale * q[0] + t[0],
                    scale * q[1] + t[1],
                    scale * q[2] + t[2],
                ]
            })
            .collect();

        let fit = umeyama(&src, &dst).unwrap();
        assert!((fit.scale - scale).abs() < 1e-9);
        for (s, d) in src.iter().zip(&dst) {
            // Predicted: fit.scale · R · s + fit.translation.
            let rs = apply(&fit.rotation, &[0.0; 3], *s);
            let got = [
                fit.scale * rs[0] + fit.translation[0],
                fit.scale * rs[1] + fit.translation[1],
                fit.scale * rs[2] + fit.translation[2],
            ];
            for i in 0..3 {
                assert!((got[i] - d[i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn kabsch_minimal_three_point_set() {
        // Any 3-point set is coplanar: H is rank 2 and the SVD rank check
        // must not reject it. The fit must still be exact.
        let r = Rotation3::<World, World>::from_axis_angle([0.3, -0.7, 0.2], 1.1)
            .unwrap()
            .matrix();
        let t = [1.0, -2.0, 3.0];
        let src: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.5], [-1.0, 2.0, 0.0], [0.3, -0.6, 1.7]];
        let dst: Vec<[f64; 3]> = src.iter().map(|&p| apply(&r, &t, p)).collect();
        let fit = kabsch(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let got = apply(&fit.rotation, &fit.translation, *s);
            for i in 0..3 {
                assert!((got[i] - d[i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn weighted_kabsch_ignores_outliers() {
        let r = Rotation3::<World, World>::from_axis_angle([1.0, 0.0, 0.0], 0.3)
            .unwrap()
            .matrix();
        let src: Vec<[f64; 3]> = (0..20)
            .map(|i| [i as f64 * 0.5, (i as f64).cos(), i as f64 % 3.0])
            .collect();
        let mut dst: Vec<[f64; 3]> = src.iter().map(|&p| apply(&r, &[1.0; 3], p)).collect();
        dst[0] = [1000.0, -1000.0, 500.0]; // gross outlier
        let mut weights = vec![1.0; 20];
        weights[0] = 0.0; // rejected

        let clean: Vec<[f64; 3]> = src[1..].to_vec();
        let clean_dst: Vec<[f64; 3]> = dst[1..].to_vec();
        let expected = kabsch(&clean, &clean_dst).unwrap();
        let got = kabsch_weighted(&src, &dst, &weights).unwrap();
        for r_ in 0..3 {
            for c in 0..3 {
                assert!((got.rotation[r_][c] - expected.rotation[r_][c]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn length_mismatch_errors() {
        let a = vec![[0.0; 3]; 3];
        let b = vec![[0.0; 3]; 2];
        assert!(matches!(
            kabsch(&a, &b),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }
}
