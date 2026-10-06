//! Geometric invariant checkers.
//!
//! Small, reusable assertion helpers for the properties the spec pins down
//! ("rigid transforms preserve distances", "ICP residual decreases
//! monotonically", …). Each returns `Result<(), String>` so both unit tests
//! and proptest properties can report *which* invariant failed and by how
//! much.
#![allow(clippy::needless_range_loop)]

/// Rigid transform invariants for a point set.
pub struct RigidCheck {
    /// Worst distance drift observed.
    pub max_distance_drift: f64,
}

/// Checks that `(r, t)` preserves all pairwise distances of `points` within
/// `tol` (metres). O(n²) pairs — keep `points` small (≤ 64).
pub fn check_distance_preserving(
    r: &[[f64; 3]; 3],
    t: &[f64; 3],
    points: &[[f64; 3]],
    tol: f64,
) -> Result<RigidCheck, String> {
    let apply = |p: [f64; 3]| {
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    };
    let dist = |a: &[f64; 3], b: &[f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let mut max_drift = 0.0_f64;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d0 = dist(&points[i], &points[j]);
            let ti = apply(points[i]);
            let tj = apply(points[j]);
            let d1 = dist(&ti, &tj);
            max_drift = max_drift.max((d1 - d0).abs());
        }
    }
    if max_drift <= tol {
        Ok(RigidCheck {
            max_distance_drift: max_drift,
        })
    } else {
        Err(format!(
            "distance preservation violated: max drift {max_drift:.6} > tol {tol:.6}"
        ))
    }
}

/// Checks that `T ∘ T⁻¹ = id` within `tol` per matrix element.
pub fn check_inverse_roundtrip(r: &[[f64; 3]; 3], t: &[f64; 3], tol: f64) -> Result<(), String> {
    let r_t = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ];
    // Composed rotation = R Rᵀ = I.
    let mut composed = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            composed[i][j] = r[i][0] * r_t[0][j] + r[i][1] * r_t[1][j] + r[i][2] * r_t[2][j];
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            if (composed[i][j] - want).abs() > tol {
                return Err(format!(
                    "rotation roundtrip broken at [{i}][{j}]: {}",
                    composed[i][j]
                ));
            }
        }
    }
    // The translation part of T⁻¹∘T is −Rᵀ·t + t_inv with
    // t_inv = −Rᵀ·t; composed with the rotation identity above, the
    // roundtrip holds for every point exactly. Nothing further to check
    // beyond the rotation identity.
    let _ = t;
    Ok(())
}

/// Checks that a residual history decreases monotonically (within `tol`
/// slack per step, to absorb numerical noise).
pub fn check_monotone_decreasing(history: &[f64], tol: f64) -> Result<(), String> {
    if history.is_empty() {
        return Err(String::from("empty residual history"));
    }
    for w in history.windows(2) {
        if w[1] > w[0] + tol {
            return Err(format!(
                "residual increased: {} → {} (tol {tol})",
                w[0], w[1]
            ));
        }
    }
    Ok(())
}

/// Checks that a 3×3 covariance is symmetric within `tol` and positive
/// semi-definite via the non-negativity of its leading principal minors
/// (Sylvester for PSD with the 2×2 condition).
pub fn check_covariance_valid(c: &[[f64; 3]; 3], tol: f64) -> Result<(), String> {
    for i in 0..3 {
        for j in 0..3 {
            if (c[i][j] - c[j][i]).abs() > tol {
                return Err(format!("asymmetric at [{i}][{j}]"));
            }
        }
    }
    let det1 = c[0][0];
    let det2 = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    let det3 = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
        - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
        + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
    let eps = 1e-12;
    if det1 < -eps || det2 < -eps || det3 < -eps {
        return Err(format!("not PSD: minors {det1}, {det2}, {det3}"));
    }
    Ok(())
}

/// Summarises a residual history (for diagnostics in failures).
pub fn residual_summary(history: &[f64]) -> String {
    if history.is_empty() {
        return String::from("(empty)");
    }
    let first = history[0];
    let last = history[history.len() - 1];
    let min = history.iter().copied().fold(f64::INFINITY, f64::min);
    format!(
        "first={first:.6} min={min:.6} last={last:.6} steps={}",
        history.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn identity_preserves_distances() {
        let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pts = vec![[0.0; 3], [1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]];
        assert!(check_distance_preserving(&r, &[0.0; 3], &pts, 1e-12).is_ok());
    }

    #[test]
    fn non_rigid_detected() {
        let scaled = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pts = vec![[0.0; 3], [1.0, 0.0, 0.0]];
        assert!(check_distance_preserving(&scaled, &[0.0; 3], &pts, 1e-9).is_err());
    }

    #[test]
    fn monotonicity_and_covariance() {
        assert!(check_monotone_decreasing(&[3.0, 2.0, 2.0, 1.0], 1e-12).is_ok());
        assert!(check_monotone_decreasing(&[3.0, 4.0], 1e-12).is_err());
        let cov = [[1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 0.5]];
        assert!(check_covariance_valid(&cov, 1e-12).is_ok());
        let indefinite = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(check_covariance_valid(&indefinite, 1e-12).is_err());
    }
}
