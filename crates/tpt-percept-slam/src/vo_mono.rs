//! Monocular visual odometry: essential-matrix estimation from tracked
//! 2-D-2-D correspondences.
//!
//! Pipeline: normalized 8-point algorithm (smallest singular vector of the
//! constraint matrix via inverse iteration) → singular-value projection onto
//! the essential manifold → cheirality-checked decomposition into the four
//! (R, t) candidates → RANSAC over correspondence sets. The motion is
//! recovered up to scale (monocular inevitability); the returned translation
//! is unit-norm.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

use tpt_math_linalg_dense::DMatrix;
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::SlamError;
use tpt_percept_core::align::svd3;
use tpt_percept_core::rng::XorShift64Star;

/// A normalized-image correspondence: `(u₁, v₁)` in frame 1, `(u₂, v₂)` in
/// frame 2 (both divided by the focal length, principal point removed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedCorrespondence {
    /// First view, normalized coordinates.
    pub from: [f64; 2],
    /// Second view, normalized coordinates.
    pub to: [f64; 2],
}

/// Monocular motion parameters.
#[derive(Clone, Copy, Debug)]
pub struct MonoParams {
    /// RANSAC iterations.
    pub ransac_iterations: u32,
    /// Sampson-distance inlier threshold (normalized units).
    pub inlier_threshold: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for MonoParams {
    fn default() -> Self {
        MonoParams {
            ransac_iterations: 1000,
            inlier_threshold: 1e-3,
            seed: 0xA0761D6478BD642F,
        }
    }
}

/// Estimated inter-frame motion (up to scale).
#[derive(Clone, Debug, PartialEq)]
pub struct MonoMotion {
    /// Rotation from view 1 to view 2.
    pub rotation: [[f64; 3]; 3],
    /// Unit-norm translation direction (expressed in view 1).
    pub translation: [f64; 3],
    /// Number of RANSAC inliers supporting the motion.
    pub inliers: usize,
}

/// Estimates the essential-matrix motion from normalized correspondences
/// with RANSAC.
pub fn estimate_motion_mono(
    correspondences: &[NormalizedCorrespondence],
    params: &MonoParams,
) -> Result<MonoMotion, SlamError> {
    if correspondences.len() < 8 {
        return Err(SlamError::InsufficientData(
            "at least 8 correspondences required",
        ));
    }
    if !(params.inlier_threshold.is_finite() && params.inlier_threshold > 0.0) {
        return Err(SlamError::InvalidParameter("inlier_threshold must be > 0"));
    }
    let mut rng = XorShift64Star::new(params.seed);
    let mut best: Option<(MonoMotion, usize)> = None;

    for _ in 0..params.ransac_iterations {
        // 8-point sample (distinct indices).
        let mut idx = [0usize; 8];
        for k in 0..8 {
            loop {
                let j = rng.below(correspondences.len());
                if !idx[..k].contains(&j) {
                    idx[k] = j;
                    break;
                }
            }
        }
        let sample: Vec<NormalizedCorrespondence> =
            idx.iter().map(|&i| correspondences[i]).collect();

        let Ok((r, t)) = essential_motion(&sample) else {
            continue;
        };
        let inliers = correspondences
            .iter()
            .filter(|c| sampson_distance(&r, &t, c) <= params.inlier_threshold)
            .count();
        if best.as_ref().is_none_or(|(_, n)| inliers > *n) {
            best = Some((
                MonoMotion {
                    rotation: r,
                    translation: t,
                    inliers,
                },
                inliers,
            ));
        }
    }

    best.map(|(m, _)| m).ok_or(SlamError::NoSolution)
}

/// Sampson distance of a correspondence under the essential motion
/// (first-order geometric error).
pub fn sampson_distance(
    rotation: &[[f64; 3]; 3],
    t: &[f64; 3],
    c: &NormalizedCorrespondence,
) -> f64 {
    let e = essential_matrix(rotation, t);
    let x1 = [c.from[0], c.from[1], 1.0];
    let x2 = [c.to[0], c.to[1], 1.0];
    let e_x1 = mul_mat_vec(&e, &x1);
    let e_t_x2 = mul_mat_vec(&transpose3(&e), &x2);
    let num = (x2[0] * e_x1[0] + x2[1] * e_x1[1] + x2[2] * e_x1[2]).abs();
    let denom =
        e_x1[0] * e_x1[0] + e_x1[1] * e_x1[1] + e_t_x2[0] * e_t_x2[0] + e_t_x2[1] * e_t_x2[1];
    if denom < 1e-18 {
        return f64::INFINITY;
    }
    num / denom.sqrt()
}

/// The essential matrix `E = [t]× R`.
pub fn essential_matrix(rotation: &[[f64; 3]; 3], t: &[f64; 3]) -> [[f64; 3]; 3] {
    let tx = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
    let mut e = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            e[r][c] =
                tx[r][0] * rotation[0][c] + tx[r][1] * rotation[1][c] + tx[r][2] * rotation[2][c];
        }
    }
    e
}

/// Runs the 8-point algorithm on a minimal (or larger) set and picks the
/// cheirality-consistent decomposition.
pub fn essential_motion(
    correspondences: &[NormalizedCorrespondence],
) -> Result<([[f64; 3]; 3], [f64; 3]), SlamError> {
    let _n = correspondences.len();
    // Constraint rows [x2·x1, x2·y1, x2, y2·x1, y2·y1, y2, x1, y1, 1];
    // accumulate the 9×9 normal matrix AᵀA directly.
    let mut ata = [[0.0f64; 9]; 9];
    for c in correspondences {
        let (x1, y1) = (c.from[0], c.from[1]);
        let (x2, y2) = (c.to[0], c.to[1]);
        let row = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0];
        for i in 0..9 {
            for j in 0..9 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    // Smallest right singular vector of A = smallest eigenvector of AᵀA.
    let flat: Vec<f64> = ata.iter().flat_map(|r| r.iter().copied()).collect();
    let m = DMatrix::from_row_slice(9, 9, &flat);
    let e_vec = smallest_eigenvector(&m)?;

    // Project onto the essential manifold: E = U diag(1, 1, 0) Vᵀ.
    let mut em = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            em[r][c] = e_vec[r * 3 + c];
        }
    }
    let (u, _s, v) =
        svd3(&em).map_err(|e| SlamError::DecompositionFailed(alloc::format!("{e}")))?;
    // Project onto the essential manifold: E = U · diag(1, 1, 0) · Vᵀ
    // (`v` holds V, columns are right singular vectors → v[c][k] is
    // component c of the k-th vector).
    let mut e = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            e[r][c] = u[r][0] * v[c][0] + u[r][1] * v[c][1];
        }
    }
    decompose_essential(&e, correspondences).ok_or(SlamError::NoSolution)
}

/// Smallest eigenvector of a symmetric 9×9 matrix via cyclic Jacobi
/// rotations — shift-free and robust for the rank-deficient normal matrix
/// of a minimal 8-point epipolar system.
fn smallest_eigenvector(m: &DMatrix<f64>) -> Result<Vec<f64>, SlamError> {
    const N: usize = 9;
    if m.nrows() != N || m.ncols() != N {
        return Err(SlamError::DecompositionFailed(
            "expected a 9×9 normal matrix".into(),
        ));
    }
    let mut a = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = m[(i, j)];
        }
    }
    let mut v = [[0.0f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale: f64 = (0..N)
        .flat_map(|i| (0..N).map(move |j| (i, j)))
        .map(|(i, j)| a[i][j].abs())
        .sum::<f64>()
        .max(1e-300);

    for _ in 0..64 {
        let off: f64 = (0..N)
            .flat_map(|i| (i + 1..N).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j].abs())
            .sum::<f64>();
        if off <= 1e-15 * scale {
            break;
        }
        for p in 0..N {
            for q in (p + 1)..N {
                let apq = a[p][q];
                if apq.abs() <= 1e-20 * scale {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // M ← Jᵀ M J with the rotation in the (p, q) plane; V ← V J.
                for k in 0..N {
                    let mkp = a[k][p];
                    let mkq = a[k][q];
                    a[k][p] = c * mkp - s * mkq;
                    a[k][q] = s * mkp + c * mkq;
                }
                for k in 0..N {
                    let mpk = a[p][k];
                    let mqk = a[q][k];
                    a[p][k] = c * mpk - s * mqk;
                    a[q][k] = s * mpk + c * mqk;
                }
                for k in 0..N {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    // Smallest diagonal entry → its column in V.
    let mut best = 0usize;
    for i in 1..N {
        if a[i][i] < a[best][best] {
            best = i;
        }
    }
    let out: Vec<f64> = (0..N).map(|r| v[r][best]).collect();
    let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 1e-12) {
        return Err(SlamError::DecompositionFailed(
            "degenerate eigenvector".into(),
        ));
    }
    Ok(out.iter().map(|x| x / norm).collect())
}

/// The four (R, t) decompositions of E; returns the one whose triangulated
/// points are in front of both cameras (cheirality).
fn decompose_essential(
    e: &[[f64; 3]; 3],
    correspondences: &[NormalizedCorrespondence],
) -> Option<([[f64; 3]; 3], [f64; 3])> {
    let (mut u, _s, mut v) = svd3(e).ok()?;
    // Proper-orientation fix: flip the column paired with σ₃ = 0, which
    // changes det(U)/det(V) while leaving E = UΣVᵀ exact (flipping all
    // columns would negate E and break the factorisation).
    if det3(&u) < 0.0 {
        for r in 0..3 {
            u[r][2] = -u[r][2];
        }
    }
    if det3(&v) < 0.0 {
        for r in 0..3 {
            v[r][2] = -v[r][2];
        }
    }
    let v_t = transpose3(&v);

    // W = [[0,-1,0],[1,0,0],[0,0,1]].
    let w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let u_wt = mul3(&u, &transpose3(&w));
    let uw = mul3(&u, &w);
    let candidates = [
        (mul3(&u_wt, &v_t), col(&u, 2)),
        (mul3(&u_wt, &v_t), scale3(&col(&u, 2), -1.0)),
        (mul3(&uw, &v_t), col(&u, 2)),
        (mul3(&uw, &v_t), scale3(&col(&u, 2), -1.0)),
    ];

    let mut best: Option<([[f64; 3]; 3], [f64; 3], usize)> = None;
    for (r, t) in candidates {
        let mut in_front = 0usize;
        for c in correspondences {
            if let Some((p1, p2)) = triangulate(&r, &t, c.from, c.to) {
                if p1[2] > 0.0 && p2[2] > 0.0 {
                    in_front += 1;
                }
            }
        }
        if best.is_none_or(|(_, _, n)| in_front > n) {
            best = Some((r, t, in_front));
        }
    }
    let (r, t, n) = best?;
    if n * 2 < correspondences.len() {
        return None; // no candidate explains the majority
    }
    Some((r, t))
}

/// Linear (DLT-midpoint) triangulation of a normalized-coordinate
/// correspondence under `(R, t)`; returns the point in both camera frames.
fn triangulate(
    r: &[[f64; 3]; 3],
    t: &[f64; 3],
    x1: [f64; 2],
    x2: [f64; 2],
) -> Option<([f64; 3], [f64; 3])> {
    // Depth triangulation: solve λ₂·x̂₂ = λ₁·R·x̂₁ + t. Crossing both sides
    // with x̂₂ eliminates λ₂: −λ₁·(x̂₂ × R·x̂₁) = x̂₂ × t, giving λ₁ by
    // least squares over the three rows. Cheirality = both depths positive.
    let n1 = norm2(&x1);
    let n2 = norm2(&x2);
    if n1 < 1e-12 || n2 < 1e-12 {
        return None;
    }
    let xh1 = [x1[0] / n1, x1[1] / n1, 1.0];
    let xh2 = [x2[0] / n2, x2[1] / n2, 1.0];
    let rx1 = mul_mat_vec(r, &xh1);
    let n = cross3(&xh2, &rx1);
    let b = cross3(&xh2, t);
    let nn = dot3(&n, &n);
    if nn < 1e-15 {
        return None; // rays (near-)parallel
    }
    // From the cross-product elimination: −λ₁·n = b ⇒ λ₁ = −⟨n, b⟩/⟨n, n⟩.
    let lambda1 = -dot3(&n, &b) / nn;
    if !(lambda1.is_finite() && lambda1 > 1e-9) {
        return None; // behind camera 1
    }
    let p1 = scale3(&xh1, lambda1);
    let p2 = [
        lambda1 * rx1[0] + t[0],
        lambda1 * rx1[1] + t[1],
        lambda1 * rx1[2] + t[2],
    ];
    if p2[2] <= 1e-9 {
        return None; // behind camera 2
    }
    Some((p1, p2))
}

#[cfg(test)]
pub(crate) fn project_for_test(
    p: [f64; 3],
    rotation: [[f64; 3]; 3],
    t: [f64; 3],
) -> Option<[f64; 2]> {
    let xc = mul_mat_vec(&rotation, &p);
    let xc = [xc[0] + t[0], xc[1] + t[1], xc[2] + t[2]];
    if xc[2] <= 1e-6 {
        return None;
    }
    Some([xc[0] / xc[2], xc[1] / xc[2]])
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mul_mat_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
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

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn col(m: &[[f64; 3]; 3], c: usize) -> [f64; 3] {
    [m[0][c], m[1][c], m[2][c]]
}

fn scale3(v: &[f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm2(v: &[f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projects world points into normalized camera coordinates.
    fn project(p_world: [f64; 3], rotation: &[[f64; 3]; 3], t: &[f64; 3]) -> Option<[f64; 2]> {
        let xc = mul_mat_vec(rotation, &p_world);
        let xc = [xc[0] + t[0], xc[1] + t[1], xc[2] + t[2]];
        if xc[2] <= 1e-6 {
            return None; // behind the camera
        }
        Some([xc[0] / xc[2], xc[1] / xc[2]])
    }

    #[test]
    fn recovers_planar_motion_up_to_scale() {
        // Camera moves 1 m along +x with a small yaw; world points in front.
        let yaw = 0.1;
        let (s, c) = yaw.sin_cos();
        // View 2 pose: rotation R, translation t (camera-2 pose in world? we
        // define world = view 1; view 2: x2 = R (x1) + t).
        let r = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let t_true = [0.99, 0.02, -0.05];

        let points: Vec<[f64; 3]> = (0..60)
            .map(|i| {
                let f = i as f64;
                [
                    ((f * 0.7).cos()) * 2.0,
                    (f * 0.31).sin() * 2.0,
                    4.0 + (f * 0.53).abs() % 6.0,
                ]
            })
            .collect();

        let mut corrs = Vec::new();
        for &p in &points {
            if let (Some(a), Some(b)) = (
                project(
                    p,
                    &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    &[0.0; 3],
                ),
                project(p, &r, &t_true),
            ) {
                corrs.push(NormalizedCorrespondence { from: a, to: b });
            }
        }
        assert!(corrs.len() >= 8);

        let motion = estimate_motion_mono(&corrs, &MonoParams::default()).unwrap();
        // Rotation recovered directly.
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (motion.rotation[i][j] - r[i][j]).abs() < 1e-3,
                    "R[{i}][{j}] {} vs {}",
                    motion.rotation[i][j],
                    r[i][j]
                );
            }
        }
        // Translation direction up to scale and sign.
        let mut t_dir = motion.translation;
        let dot = t_dir[0] * t_true[0] + t_dir[1] * t_true[1] + t_dir[2] * t_true[2];
        if dot < 0.0 {
            t_dir = scale3(&t_dir, -1.0);
        }
        let nl = (t_true[0] * t_true[0] + t_true[1] * t_true[1] + t_true[2] * t_true[2]).sqrt();
        let expect = [t_true[0] / nl, t_true[1] / nl, t_true[2] / nl];
        for i in 0..3 {
            assert!((t_dir[i] - expect[i]).abs() < 5e-3, "t[{i}] {}", t_dir[i]);
        }
        assert!(motion.inliers >= corrs.len() * 9 / 10);
    }

    #[test]
    fn debug_decompose_exact_e() {
        let yaw = 0.1_f64;
        let (s, c) = yaw.sin_cos();
        let r = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let t_true = [0.99, 0.02, -0.05];
        let ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let points: Vec<[f64; 3]> = (0..60)
            .map(|i| {
                let f = i as f64;
                [
                    ((f * 0.7).cos()) * 2.0,
                    (f * 0.31).sin() * 2.0,
                    4.0 + (f * 0.53).abs() % 6.0,
                ]
            })
            .collect();
        let mut corrs = Vec::new();
        for &p in &points {
            if let (Some(a), Some(b)) = (
                crate::vo_mono::project_for_test(p, ident, [0.0; 3]),
                crate::vo_mono::project_for_test(p, r, t_true),
            ) {
                corrs.push(NormalizedCorrespondence { from: a, to: b });
            }
        }
        let e = essential_matrix(&r, &t_true);
        let got = decompose_essential(&e, &corrs);
        match got {
            Some((rg, tg)) => {
                // t is unit-norm (monocular scale); compare directions.
                let tl =
                    (t_true[0] * t_true[0] + t_true[1] * t_true[1] + t_true[2] * t_true[2]).sqrt();
                assert!(
                    (rg[0][0] - c).abs() < 1e-6 && (tg[0] - t_true[0] / tl).abs() < 1e-6,
                    "decompose picked wrong candidate: R00={} t0={}",
                    rg[0][0],
                    tg[0]
                );
            }
            None => panic!("decompose_essential returned None for exact E"),
        }
    }

    #[test]
    fn too_few_correspondences_rejected() {
        let cs: Vec<NormalizedCorrespondence> = (0..7)
            .map(|i| NormalizedCorrespondence {
                from: [i as f64 * 0.1, 0.0],
                to: [i as f64 * 0.1, 0.1],
            })
            .collect();
        assert!(matches!(
            estimate_motion_mono(&cs, &MonoParams::default()),
            Err(SlamError::InsufficientData(_))
        ));
    }
}
