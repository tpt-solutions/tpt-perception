//! Degeneracy analysis for registration.
//!
//! The information matrix of a point-to-plane solve tells you how well each
//! of the six pose directions is constrained. Its eigenvectors are the
//! directions and its eigenvalues the constraint strength. A direction whose
//! eigenvalue is tiny relative to the strongest one is *weakly observed* —
//! the classic cases are a long corridor (motion along the corridor is
//! unobservable) and a flat floor (translation in the plane is unobservable
//! and yaw is ill-conditioned).
//!
//! Callers such as an EKF can use the weak directions to down-weight or
//! reject the corresponding components of a registration result instead of
//! trusting them.

#![allow(clippy::needless_range_loop)]

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use alloc::vec::Vec;

/// Eigen-decomposition of an information matrix plus the weak directions.
#[derive(Clone, Debug, PartialEq)]
pub struct Degeneracy {
    /// Eigenvalues in ascending order.
    pub eigenvalues: [f64; 6],
    /// Unit eigenvectors; `eigenvectors[k]` belongs to `eigenvalues[k]`.
    /// Components are ordered `(v_x, v_y, v_z, ω_x, ω_y, ω_z)`.
    pub eigenvectors: [[f64; 6]; 6],
    /// Indices `k` (into the ascending arrays above) whose eigenvalue is
    /// `≤ rel_tol · λ_max`.
    pub weak: Vec<usize>,
}

/// Eigen-decomposes a symmetric information matrix and flags weak
/// directions. Returns `None` when the matrix is all zeros or non-finite.
///
/// `rel_tol` is relative to the largest eigenvalue; values around `1e-3`
/// are a reasonable starting point.
pub fn analyse(information: &[[f64; 6]; 6], rel_tol: f64) -> Option<Degeneracy> {
    if !information.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    let (values, vectors) = jacobi_eigen(*information);
    let max = values[5].max(0.0);
    if max <= 0.0 {
        return None;
    }
    let weak = (0..6)
        .filter(|&k| values[k] <= rel_tol * max)
        .collect::<Vec<_>>();
    Some(Degeneracy {
        eigenvalues: values,
        eigenvectors: vectors,
        weak,
    })
}

/// Cyclic Jacobi eigen-solver for a symmetric 6×6 matrix. Returns the
/// eigenvalues ascending and the matching eigenvectors as columns of the
/// returned matrix, transposed so that `vectors[k]` is the k-th eigenvector.
fn jacobi_eigen(mut a: [[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    const N: usize = 6;
    const MAX_SWEEPS: usize = 64;

    let mut v = [[0.0f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..N {
            for q in (p + 1)..N {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..N {
            for q in (p + 1)..N {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..N {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut order: [usize; N] = [0, 1, 2, 3, 4, 5];
    order.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
    let mut values = [0.0f64; N];
    let mut vectors = [[0.0f64; N]; N];
    for (k, &src) in order.iter().enumerate() {
        values[k] = a[src][src];
        for r in 0..N {
            vectors[k][r] = v[r][src];
        }
    }
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn diagonal_matrix_eigenvalues_are_its_diagonal() {
        let mut m = [[0.0f64; 6]; 6];
        for (i, d) in [5.0, 0.0001, 3.0, 2.0, 4.0, 1.0].iter().enumerate() {
            m[i][i] = *d;
        }
        let d = analyse(&m, 1e-3).unwrap();
        assert_eq!(d.eigenvalues, [0.0001, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(d.weak, vec![0]);
    }

    #[test]
    fn coupled_2x2_block_is_diagonalised() {
        // [[2, 1], [1, 2]] (eigenvalues 1 and 3) coupled into a 6x6 with
        // diagonal axes 2, 5, 6, 10 alongside it.
        let mut m = [[0.0f64; 6]; 6];
        m[0][0] = 2.0;
        m[0][1] = 1.0;
        m[1][0] = 1.0;
        m[1][1] = 2.0;
        m[2][2] = 10.0;
        m[3][3] = 2.0;
        m[4][4] = 5.0;
        m[5][5] = 6.0;
        let d = analyse(&m, 1e-3).unwrap();
        let expected = [1.0, 2.0, 3.0, 5.0, 6.0, 10.0];
        for (got, want) in d.eigenvalues.iter().zip(expected) {
            assert!((got - want).abs() < 1e-9, "{:?}", d.eigenvalues);
        }
        // Eigenvector for λ = 1 is (1, −1)/√2 in the (0, 1) plane.
        let v = d.eigenvectors[0];
        assert!((v[0] + v[1]).abs() < 1e-9);
        assert!((v[0].abs() - 1.0 / 2f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn zero_or_nonfinite_matrix_gives_none() {
        assert!(analyse(&[[0.0; 6]; 6], 1e-3).is_none());
        let mut m = [[0.0f64; 6]; 6];
        m[0][0] = f64::NAN;
        assert!(analyse(&m, 1e-3).is_none());
    }
}
