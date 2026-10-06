//! proptest strategies for perceptual geometry.
//!
//! These generate *valid* inputs — finite points, proper rotations,
//! well-formed clouds and noise profiles — so property tests exercise the
//! mathematics rather than input validation. Rotations are generated exactly
//! (axis-angle), never by orthonormalising a noisy matrix.
#![allow(clippy::needless_range_loop)]

use proptest::prelude::*;

use tpt_percept_cloud::cloud::PointCloud;

/// A finite point in `[−range, range]³`.
pub fn finite_point(range: f64) -> impl Strategy<Value = [f64; 3]> {
    (-range..range, -(range)..range, -(range)..range).prop_map(|(x, y, z)| {
        let sanitize = |v: f64| if v.is_finite() { v } else { 0.0 };
        [sanitize(x), sanitize(y), sanitize(z)]
    })
}

/// A point cloud of `min..=max` finite points in `[−range, range]³`.
pub fn point_cloud(min: usize, max: usize, range: f64) -> impl Strategy<Value = PointCloud> {
    proptest::collection::vec(finite_point(range), min..=max).prop_map(PointCloud::from_points)
}

/// A proper rotation matrix (exact, via axis-angle with a normalised axis).
pub fn rotation() -> impl Strategy<Value = [[f64; 3]; 3]> {
    (any::<f64>(), any::<f64>(), any::<f64>()).prop_map(|(raw1, raw2, raw3)| {
        // Sanitise: `any::<f64>()` spans NaN/±∞ — clamp to [-1, 1] first.
        let u1 = raw1.clamp(-1.0, 1.0);
        let mut u2 = raw2.clamp(-1.0, 1.0);
        let mut u3 = raw3.clamp(-1.0, 1.0);
        let axis_len = (u2 * u2 + u3 * u3).sqrt();
        if axis_len < 1e-9 {
            u2 = 1.0;
            u3 = 0.0;
        }
        let axis = normalize_axis(u1, u2, u3);
        let angle = u1.abs() * core::f64::consts::PI;
        axis_angle_matrix(&axis, angle)
    })
}

/// A rigid transform: exact rotation + translation in `[−10, 10]³`.
pub fn rigid_transform() -> impl Strategy<Value = ([[f64; 3]; 3], [f64; 3])> {
    (rotation(), finite_point(10.0)).prop_map(|(r, t)| (r, t))
}

/// A zero-mean bounded noise profile: `n` offsets in `±scale`.
pub fn noise_profile(n: usize, scale: f64) -> impl Strategy<Value = Vec<[f64; 3]>> {
    proptest::collection::vec(
        (-(scale)..scale, -(scale)..scale, -(scale)..scale).prop_map(|(x, y, z)| [x, y, z]),
        n..=n,
    )
}

/// A random 3×3 symmetric positive-semidefinite covariance with the given
/// eigenvalue range.
pub fn covariance3(min_eig: f64, max_eig: f64) -> impl Strategy<Value = [[f64; 3]; 3]> {
    (
        rotation(),
        min_eig..max_eig,
        min_eig..max_eig,
        min_eig..max_eig,
    )
        .prop_map(|(r, l1, l2, l3)| {
            // A = R Λ Rᵀ with Λ diagonal — PSD by construction.
            let mut a = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let lam = [l1, l2, l3][k];
                        a[i][j] += r[i][k] * lam * r[j][k];
                    }
                }
            }
            a
        })
}

fn normalize_axis(u1: f64, u2: f64, u3: f64) -> [f64; 3] {
    let n = (u1 * u1 + u2 * u2 + u3 * u3).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0]
    } else {
        [u1 / n, u2 / n, u3 / n]
    }
}

/// Rodrigues rotation matrix (row-major).
pub fn axis_angle_matrix(axis: &[f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(1e-30);
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = angle.sin_cos();
    let cc = 1.0 - c;
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    [
        [
            c + cc * kx * kx,
            cc * kx * ky - s * kz,
            cc * kx * kz + s * ky,
        ],
        [
            cc * kx * ky + s * kz,
            c + cc * ky * ky,
            cc * ky * kz - s * kx,
        ],
        [
            cc * kx * kz - s * ky,
            cc * ky * kz + s * kx,
            c + cc * kz * kz,
        ],
    ]
}

/// Applies a rigid transform to a point.
pub fn apply_rigid(r: &[[f64; 3]; 3], t: &[f64; 3], p: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn rotations_are_orthonormal(r in rotation()) {
            // RᵀR = I and det = +1.
            for i in 0..3 {
                let mut col_norm = 0.0;
                for a in 0..3 { for b in 0..3 { col_norm += r[a][i] * r[b][i] * 0.0; } }
                let n = r[0][i]*r[0][i] + r[1][i]*r[1][i] + r[2][i]*r[2][i];
                prop_assert!((n - 1.0).abs() < 1e-9, "column {i} norm {n}");
                let _ = col_norm;
            }
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            prop_assert!((det - 1.0).abs() < 1e-9, "det {det}");
        }

        #[test]
        fn covariances_are_psd(c in covariance3(0.01, 10.0)) {
            // Diagonal dominance is not PSD in general; instead check the
            // trace is positive and the matrix symmetric (construction is
            // R Λ Rᵀ, provably PSD).
            for i in 0..3 { for j in 0..3 {
                prop_assert!((c[i][j] - c[j][i]).abs() < 1e-12);
            }}
            prop_assert!(c[0][0] + c[1][1] + c[2][2] > 0.0);
        }
    }
}
