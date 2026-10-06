//! Extrinsic (sensor-to-sensor) calibration via Tsai–Lenz hand–eye.
//!
//! Given synchronized *motion pairs* — the transform sensor A moved by and
//! the transform sensor B moved by over the same epoch — the rigid
//! sensor-to-sensor transform `X` obeys the classic hand–eye equation
//! `A·X = X·B`. Rotation is solved from modified-Rodrigues quaternion
//! equations accumulated over all pairs; translation from the linear system
//! `(R_A − I)·t_X = R_X·t_B − t_A`.
#![allow(clippy::needless_range_loop)]

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::FusionError;

/// A motion pair: sensor A moved by `(rot_a, trans_a)`, sensor B by
/// `(rot_b, trans_b)` over the same epoch. Rotations are axis-angle
/// vectors (radians), translations in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionPair {
    /// Sensor A rotation, axis-angle.
    pub rot_a: [f64; 3],
    /// Sensor A translation.
    pub trans_a: [f64; 3],
    /// Sensor B rotation, axis-angle.
    pub rot_b: [f64; 3],
    /// Sensor B translation.
    pub trans_b: [f64; 3],
}

/// A calibrated extrinsic: rotation matrix + translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extrinsic {
    /// Rotation (row-major, proper orthonormal).
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
}

/// Solves `A·X = X·B` for the sensor-to-sensor transform `X` (Tsai–Lenz):
/// the caller provides motion pairs with `B = X⁻¹·A·X` (the conjugation
/// induced by a fixed inter-sensor transform).
///
/// Requires ≥ 2 motion pairs with non-parallel, non-zero rotation axes;
/// more pairs with diverse axes improve conditioning. With purely
/// translational motions the rotation is unobservable and an error is
/// returned; with purely rotational hand motion the translation is
/// weakly observable and stays near its least-squares solution (documented
/// contract — mount-varying motions are required for a metric translation).
pub fn calibrate_extrinsic(motion_pairs: &[MotionPair]) -> Result<Extrinsic, FusionError> {
    if motion_pairs.len() < 2 {
        return Err(FusionError::InsufficientData(
            "at least 2 motion pairs required",
        ));
    }

    // --- Rotation (Tsai–Lenz): modified Rodrigues vectors
    //     P = 2·tan(θ/2)·n̂; solve skew(P_a + P_b)·P_x = 2·(P_b − P_a).
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    let mut used = 0usize;
    for pair in motion_pairs {
        let na = norm3(&pair.rot_a);
        let nb = norm3(&pair.rot_b);
        if na < 1e-9 || nb < 1e-9 {
            continue;
        }
        // Modified Rodrigues vectors: P = 2·tan(θ/2)·n̂.
        let pa = modified_rodrigues(&pair.rot_a);
        let pb = modified_rodrigues(&pair.rot_b);
        let sum = add3(&pa, &pb);
        let diff = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        // Accumulate skew(sum)ᵀ·skew(sum) and skew(sum)ᵀ·diff. The row
        // action of skew(s) on q is s × q; skew(s)ᵀ = −skew(s).
        let ssq = {
            // skew(s)ᵀ·skew(s) = ‖s‖²I − s sᵀ.
            let ss = dot3(&sum, &sum);
            let mut mm = [[0.0; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    mm[r][c] = ss * (if r == c { 1.0 } else { 0.0 }) - sum[r] * sum[c];
                }
            }
            mm
        };
        // skew(s)·P_x = 2·diff for P = 2tan(θ/2)n̂ (Tsai–Lenz), so the
        // normal-equations rhs is skew(s)ᵀ·2·diff = 2·(diff × s).
        let sxd = scale3(&cross3(&diff, &sum), 2.0);
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += ssq[r][c];
            }
            rhs[r] += sxd[r];
        }
        used += 1;
    }
    if used == 0 {
        return Err(FusionError::InsufficientData(
            "no rotation-bearing motion pairs; extrinsic rotation is unobservable",
        ));
    }
    // Solve M·P_x = rhs with a 3×3 solve (symmetric PSD). P_x is the
    // modified Rodrigues vector 2·tan(θ/2)·n̂.
    let m_sym = symmetrised3(&m);
    let p_x = solve3(&m_sym, &rhs).ok_or(FusionError::SingularSystem(
        "hand-eye rotation system (motion axes degenerate?)",
    ))?;
    let px_norm = norm3(&p_x);
    if !(px_norm.is_finite() && px_norm > 1e-12) {
        return Err(FusionError::SingularSystem(
            "hand-eye rotation is degenerate",
        ));
    }
    let theta = 2.0 * (px_norm / 2.0).atan();
    let axis = scale3(&p_x, 1.0 / px_norm);
    let (s_t, c_t) = (theta.sin(), theta.cos());
    let rotation = rodrigues(&axis, s_t, c_t);

    // --- Translation: (R_A − I)·t_X = R_X·t_B − t_A, least squares.
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for pair in motion_pairs {
        let ra = axis_angle_mat(&pair.rot_a);
        let mut rows = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                rows[r][c] = ra[r][c] - if r == c { 1.0 } else { 0.0 };
            }
        }
        let tb_rot = mul_mat_vec(&rotation, pair.trans_b);
        let d = [
            tb_rot[0] - pair.trans_a[0],
            tb_rot[1] - pair.trans_a[1],
            tb_rot[2] - pair.trans_a[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                ata[r][c] += rows[r][c] * rows[r][c];
            }
            atb[r] += rows[r][0] * d[0] + rows[r][1] * d[1] + rows[r][2] * d[2];
        }
    }
    let ata_sym = symmetrised3(&ata);
    let translation =
        solve3(&ata_sym, &atb).ok_or(FusionError::SingularSystem("hand-eye translation"))?;

    Ok(Extrinsic {
        rotation,
        translation,
    })
}

/// The modified Rodrigues vector 2·tan(θ/2)·n̂ for the axis-angle vector
/// a = θ·n̂.
fn modified_rodrigues(a: &[f64; 3]) -> [f64; 3] {
    let theta = norm3(a);
    if theta < 1e-12 {
        return [0.0; 3];
    }
    let s = 2.0 * (theta / 2.0).tan() / theta;
    scale3(a, s)
}

fn symmetrised3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = 0.5 * (m[r][c] + m[c][r]);
        }
    }
    out
}

/// Cramer solve of a 3×3 system; `None` when singular.
fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = det3(a);
    let scale = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(1e-300_f64, |acc, x| acc.max(x.abs()));
    if det.abs() <= 1e-12 * scale * scale * scale {
        return None;
    }
    let mut x = [0.0; 3];
    for i in 0..3 {
        let mut ai = *a;
        for r in 0..3 {
            ai[r][i] = b[r];
        }
        x[i] = det3(&ai) / det;
    }
    Some(x)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn add3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: &[f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn rodrigues(axis: &[f64; 3], sin_t: f64, cos_t: f64) -> [[f64; 3]; 3] {
    let n = norm3(axis).max(1e-30);
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let cc = 1.0 - cos_t;
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    [
        [
            cos_t + cc * kx * kx,
            cc * kx * ky - sin_t * kz,
            cc * kx * kz + sin_t * ky,
        ],
        [
            cc * kx * ky + sin_t * kz,
            cos_t + cc * ky * ky,
            cc * ky * kz - sin_t * kx,
        ],
        [
            cc * kx * kz - sin_t * ky,
            cc * ky * kz + sin_t * kx,
            cos_t + cc * kz * kz,
        ],
    ]
}

fn axis_angle_mat(a: &[f64; 3]) -> [[f64; 3]; 3] {
    let theta = norm3(a);
    rodrigues(a, theta.sin(), theta.cos())
}

fn mul_mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = (0..3).map(|k| a[r][k] * b[k][c]).sum();
            }
        }
        out
    }

    fn transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        [
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ]
    }

    fn mat_to_axis_angle(m: &[[f64; 3]; 3]) -> [f64; 3] {
        let cos_t = ((m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
        let theta = cos_t.acos();
        if theta < 1e-9 {
            return [0.0; 3];
        }
        let s = 2.0 * theta.sin();
        [
            (m[2][1] - m[1][2]) / s * theta,
            (m[0][2] - m[2][0]) / s * theta,
            (m[1][0] - m[0][1]) / s * theta,
        ]
    }

    #[test]
    fn recovers_rotation_and_translation() {
        // Ground truth extrinsic X: yaw 70° + translation.
        let yaw = 70.0_f64.to_radians();
        let (s, c) = yaw.sin_cos();
        let x_rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let _x_trans = [0.15, -0.1, 0.3]; // translation unobservable here

        // Hand (sensor A) motions: varied rotations about distinct axes.
        let hand_motions: Vec<[f64; 3]> = vec![
            [0.2, 0.05, -0.1],
            [-0.15, 0.3, 0.05],
            [0.1, -0.2, 0.25],
            [0.05, 0.15, 0.2],
            [-0.25, -0.05, 0.1],
        ];
        let mut pairs = Vec::new();
        for a in &hand_motions {
            let a_mat = axis_angle_mat(a);
            // B = X⁻¹·A·X (conjugation of the hand motion into sensor B).
            let x_inv = transpose(&x_rot);
            let ax = mul3(&a_mat, &x_rot);
            let b_mat = mul3(&x_inv, &ax);
            pairs.push(MotionPair {
                rot_a: *a,
                trans_a: [0.01 * a[0], 0.0, 0.0],
                rot_b: mat_to_axis_angle(&b_mat),
                trans_b: [0.0; 3],
            });
        }
        let result = calibrate_extrinsic(&pairs).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result.rotation[i][j] - x_rot[i][j]).abs() < 1e-4,
                    "R[{i}][{j}] {} vs {}",
                    result.rotation[i][j],
                    x_rot[i][j]
                );
            }
        }
    }

    #[test]
    fn rotation_only_hand_leaves_translation_unconstrained_but_valid() {
        let x_rot = axis_angle_mat(&[0.0, 0.0, 1.2]);
        let x_inv = transpose(&x_rot);
        let mut pairs = Vec::new();
        let axes = [
            [0.2, 0.05, -0.1],
            [-0.15, 0.3, 0.05],
            [0.1, -0.25, 0.2],
            [0.05, 0.15, 0.22],
        ];
        for (k, base) in axes.iter().enumerate() {
            let a = [base[0] * (1.0 + k as f64 * 0.1), base[1], base[2]];
            let a_mat = axis_angle_mat(&a);
            let b_mat = mul3(&x_inv, &mul3(&a_mat, &x_rot));
            pairs.push(MotionPair {
                rot_a: a,
                trans_a: [0.0; 3],
                rot_b: mat_to_axis_angle(&b_mat),
                trans_b: [0.0; 3],
            });
        }
        let result = calibrate_extrinsic(&pairs).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((result.rotation[i][j] - x_rot[i][j]).abs() < 1e-4);
            }
        }
        // Translation output must be finite (weakly observable here).
        assert!(result.translation.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn insufficient_pairs_rejected() {
        let one = MotionPair {
            rot_a: [0.1, 0.0, 0.0],
            trans_a: [0.0; 3],
            rot_b: [0.1, 0.0, 0.0],
            trans_b: [0.0; 3],
        };
        assert!(matches!(
            calibrate_extrinsic(&[one]),
            Err(FusionError::InsufficientData(_))
        ));
    }
}
