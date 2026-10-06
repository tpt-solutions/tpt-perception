//! Small dense linear algebra helpers for 3-D geometry.
//!
//! The perception stack needs one operation `tpt-math` does not expose for
//! fixed-size 3×3 matrices: the eigendecomposition of a *symmetric* 3×3
//! matrix (used everywhere for covariance analysis / PCA). This module
//! provides a cyclic Jacobi implementation — unconditionally robust for
//! symmetric input, no allocation, no external solver.
#![allow(clippy::needless_range_loop)]

use tpt_math_linalg_fixed::{Matrix3, Vector3};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

/// An eigenpair decomposition of a symmetric 3×3 matrix.
///
/// Invariant: `A ≈ Σ λᵢ vᵢ vᵢᵀ` and `‖vᵢ‖ = 1` (up to convergence
/// tolerance, `‖A − VΛVᵀ‖∞ ≤ tol · ‖A‖∞`).
#[derive(Clone, Debug, PartialEq)]
pub struct Eigen3 {
    /// Eigenvalues in *descending* order.
    pub values: [f64; 3],
    /// Unit eigenvectors, `vectors[i]` corresponds to `values[i]`.
    pub vectors: [Vector3<f64>; 3],
}

/// Eigendecomposition of a symmetric 3×3 matrix via cyclic Jacobi rotations.
///
/// Symmetry of the input is assumed; the lower triangle is read (the upper
/// part is symmetrised away). Converges to machine precision in a handful of
/// sweeps for well-scaled matrices; capped at 64 sweeps.
pub fn sym_eigen3(a: &Matrix3<f64>) -> Eigen3 {
    // Working copies; `m` is symmetrised from both triangles' average to be
    // forgiving of near-symmetric input.
    let mut m = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = 0.5 * (a.data[i][j] + a.data[j][i]);
        }
    }
    // v starts as identity; columns accumulate the rotations (eigenvectors).
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let scale = {
        let s = m[0][0].abs()
            + m[1][1].abs()
            + m[2][2].abs()
            + m[0][1].abs()
            + m[0][2].abs()
            + m[1][2].abs();
        if s == 0.0 {
            1.0
        } else {
            s
        }
    };

    for _ in 0..64 {
        // Largest off-diagonal magnitude.
        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
        if off <= 1e-15 * scale {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = m[p][q];
            if apq.abs() <= 1e-18 * scale {
                continue;
            }
            // Jacobi rotation angle that zeroes m[p][q].
            let theta = (m[q][q] - m[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            apply_jacobi(&mut m, &mut v, p, q, c, s);
        }
    }

    // Order eigenpairs descending with a 3-element sorting network (no alloc).
    let d = [m[0][0], m[1][1], m[2][2]];
    let mut order = [0usize, 1, 2];
    if d[order[0]] < d[order[1]] {
        order.swap(0, 1);
    }
    if d[order[1]] < d[order[2]] {
        order.swap(1, 2);
    }
    if d[order[0]] < d[order[1]] {
        order.swap(0, 1);
    }

    let values = [d[order[0]], d[order[1]], d[order[2]]];
    let vectors = [
        Vector3::new([v[0][order[0]], v[1][order[0]], v[2][order[0]]]),
        Vector3::new([v[0][order[1]], v[1][order[1]], v[2][order[1]]]),
        Vector3::new([v[0][order[2]], v[1][order[2]], v[2][order[2]]]),
    ];
    Eigen3 { values, vectors }
}

/// Applies one Jacobi rotation `J(p,q,θ)` to `m` from the right and `Jᵀ`
/// from the left, accumulating the rotation into `v`.
fn apply_jacobi(m: &mut [[f64; 3]; 3], v: &mut [[f64; 3]; 3], p: usize, q: usize, c: f64, s: f64) {
    for k in 0..3 {
        let mkp = m[k][p];
        let mkq = m[k][q];
        m[k][p] = c * mkp - s * mkq;
        m[k][q] = s * mkp + c * mkq;
    }
    for k in 0..3 {
        let mpk = m[p][k];
        let mqk = m[q][k];
        m[p][k] = c * mpk - s * mqk;
        m[q][k] = s * mpk + c * mqk;
    }
    for k in 0..3 {
        let vkp = v[k][p];
        let vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
    }
}

/// Builds a `Matrix3` from row-major rows (interop helper).
#[doc(hidden)]
pub fn __matrix3(m: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(m)
}

/// Determinant of a 3×3 matrix (row-major).
pub fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve a 3×3 linear system `A x = b` via Cramer's rule.
///
/// Returns `None` when `A` is (near-)singular relative to its scale.
pub fn solve3(a: &[[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let d = det3(a);
    let scale = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |acc, x| acc.max(x.abs()))
        .max(b.iter().fold(0.0_f64, |acc, x| acc.max(x.abs())));
    if scale == 0.0 || d.abs() <= 1e-12 * scale * scale * scale {
        return None;
    }
    let mut x = [0.0; 3];
    for i in 0..3 {
        let mut ai = *a;
        for r in 0..3 {
            ai[r][i] = b[r];
        }
        x[i] = det3(&ai) / d;
    }
    Some(x)
}

/// The skew-symmetric cross-product matrix `[v]×` with `[v]× w = v × w`.
pub fn skew(v: [f64; 3]) -> Matrix3<f64> {
    Matrix3::new([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
}

/// Outer product `v wᵀ` as a symmetric-contributing 3×3 matrix.
pub fn outer3(v: [f64; 3], w: [f64; 3]) -> Matrix3<f64> {
    Matrix3::new([
        [v[0] * w[0], v[0] * w[1], v[0] * w[2]],
        [v[1] * w[0], v[1] * w[1], v[1] * w[2]],
        [v[2] * w[0], v[2] * w[1], v[2] * w[2]],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(data: [[f64; 3]; 3]) -> Matrix3<f64> {
        Matrix3::new(data)
    }

    fn reconstruct(e: &Eigen3) -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            let lam = e.values[i];
            let v = e.vectors[i];
            for r in 0..3 {
                for c in 0..3 {
                    out[r][c] += lam * v.data[r] * v.data[c];
                }
            }
        }
        out
    }

    #[test]
    fn diagonal_matrix_eigen() {
        let e = sym_eigen3(&mat([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]));
        assert!((e.values[0] - 4.0).abs() < 1e-10);
        assert!((e.values[1] - 1.0).abs() < 1e-10);
        assert!((e.values[2] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn dense_symmetric_reconstruction() {
        let a = mat([[2.0, 1.0, 0.3], [1.0, 3.0, -0.7], [0.3, -0.7, 1.5]]);
        let e = sym_eigen3(&a);
        let rec = reconstruct(&e);
        for r in 0..3 {
            for c in 0..3 {
                assert!((rec[r][c] - a.data[r][c]).abs() < 1e-10);
            }
        }
        // Orthonormal eigenvectors.
        for i in 0..3 {
            let n = e.vectors[i].dot(&e.vectors[i]);
            assert!((n - 1.0).abs() < 1e-9);
            for j in (i + 1)..3 {
                let d = e.vectors[i].dot(&e.vectors[j]);
                assert!(d.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rank_one_matrix() {
        // vvᵀ has eigenvalue ‖v‖² along v, zero elsewhere.
        let v = [1.0, 2.0, 3.0];
        let a = outer3(v, v);
        let e = sym_eigen3(&a);
        assert!((e.values[0] - 14.0).abs() < 1e-9);
        assert!(e.values[1] < 1e-9 && e.values[2] < 1e-9);
    }

    #[test]
    fn solve3_roundtrip() {
        let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let x = [1.0, -2.0, 3.0];
        let b = [
            a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
            a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
            a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
        ];
        let got = solve3(&a, b).unwrap();
        for i in 0..3 {
            assert!((got[i] - x[i]).abs() < 1e-10);
        }
        assert!(solve3(&[[0.0; 3]; 3], [1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn skew_crosses() {
        let v = [1.0, -2.0, 0.5];
        let w = [3.0, 1.0, -4.0];
        let lhs = skew(v) * Vector3::new(w);
        let expect = Vector3::new(v).cross(&Vector3::new(w));
        for i in 0..3 {
            assert!((lhs.data[i] - expect.data[i]).abs() < 1e-12);
        }
    }
}
