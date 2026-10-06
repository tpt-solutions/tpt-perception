//! SHOT — Signature of Histograms of Oriented SHapes (Tombari et al. 2010),
//! here the SHOT-352 arrangement: a local reference frame with 32 spatial
//! bins (4 azimuth × 4 elevation × 2 radius) and 11 normal-similarity bins
//! per sector (32 × 11 = 352 dimensions).
//!
//! The local reference frame (LRF) is built from the distance-weighted
//! scatter matrix of the neighbourhood with the standard sign-disambiguation
//! rule, making the descriptor covariant with rotations.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::normal::estimate_normals;
use crate::FeatureError;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;
use tpt_percept_core::linalg3::outer3;

/// Azimuth bins.
const AZ: usize = 4;
/// Elevation bins.
const EL: usize = 4;
/// Radial bins.
const RAD: usize = 2;
/// Normal-similarity bins per spatial sector.
const HBINS: usize = 11;
/// Descriptor dimensionality: 32 spatial sectors × 11 shape bins.
pub const SHOT_DIM: usize = AZ * EL * RAD * HBINS;

/// Computes a SHOT descriptor for every point over a support radius
/// `radius` (metres).
pub fn compute_shot(cloud: &PointCloud, radius: f64) -> Result<Vec<[f64; SHOT_DIM]>, FeatureError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(FeatureError::InvalidParameter(
            "radius must be finite and > 0",
        ));
    }
    if cloud.len() < 6 {
        return Err(FeatureError::InvalidParameter("cloud too small for SHOT"));
    }
    let k = (cloud.len() - 1).min(64);
    let normals = estimate_normals(cloud, k).ok_or(FeatureError::InvalidParameter(
        "cloud too small for normal estimation",
    ))?;
    let tree = KdTree::new(cloud.points());
    let r2 = radius * radius;

    let mut out = Vec::with_capacity(cloud.len());
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.radius(&p, r2);
        if neighbors.len() < 6 {
            out.push([0.0; SHOT_DIM]);
            continue;
        }
        let frame = local_reference_frame(cloud, &neighbors, p);
        let mut hist = [0.0; SHOT_DIM];
        let mut total = 0.0;
        for &(idx, d2) in &neighbors {
            if idx == i {
                continue;
            }
            let q = cloud.get(idx).expect("in range");
            let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            // Coordinates in the LRF.
            let x = dot3(&frame[0], &d);
            let y = dot3(&frame[1], &d);
            let z = dot3(&frame[2], &d);
            let r = d2.sqrt();
            let az = y.atan2(x); // (-π, π]
            let el = z.atan2((x * x + y * y).sqrt()); // (-π/2, π/2)
            let ia =
                ((az + core::f64::consts::PI) / (2.0 * core::f64::consts::PI) * AZ as f64) as usize;
            let ie =
                ((el + core::f64::consts::FRAC_PI_2) / core::f64::consts::PI * EL as f64) as usize;
            let ir = if r <= radius / 2.0 { 0 } else { 1 };
            let ia = ia.min(AZ - 1);
            let ie = ie.min(EL - 1);

            // Interpolation-free hard binning of the normal-similarity
            // cosine into HBINS bins over [0, π].
            let nn = &normals[idx].normal;
            let cos = (dot3(&normals[i].normal, nn)).clamp(-1.0, 1.0);
            let angle = cos.acos();
            let ih = ((angle / core::f64::consts::PI) * HBINS as f64) as usize;
            let ih = ih.min(HBINS - 1);

            let sector = ((ia * EL) + ie) * RAD + ir;
            hist[sector * HBINS + ih] += 1.0;
            total += 1.0;
        }
        if total > 0.0 {
            for v in &mut hist {
                *v /= total;
            }
        }
        out.push(hist);
    }
    Ok(out)
}

/// The local reference frame: `[x_axis, y_axis, z_axis]` unit vectors.
///
/// Weighted scatter (weight `1/(R − r)`), eigendecomposition, then the
/// standard sign disambiguation `s = Σ sign(axis·d)·(axis·d)²` over the
/// neighbourhood.
fn local_reference_frame(
    cloud: &PointCloud,
    neighbors: &[(usize, f64)],
    center: [f64; 3],
) -> [[f64; 3]; 3] {
    let radius = neighbors
        .iter()
        .map(|&(_, d2)| d2.sqrt())
        .fold(0.0_f64, f64::max);
    let denom = (radius - 0.0).max(1e-9);
    let mut scatter = [[0.0; 3]; 3];
    let mut wsum = 0.0;
    for &(idx, d2) in neighbors {
        let q = cloud.get(idx).expect("in range");
        let d = [q[0] - center[0], q[1] - center[1], q[2] - center[2]];
        let w = (denom - d2.sqrt()) / denom; // 1 at centre → 0 at rim
        let o = outer3(d, d);
        for r in 0..3 {
            for c in 0..3 {
                scatter[r][c] += w * o.data[r][c];
            }
        }
        wsum += w;
    }
    if wsum > 0.0 {
        for v in &mut scatter {
            for x in v {
                *x /= wsum;
            }
        }
    }
    let eig = tpt_percept_core::linalg3::sym_eigen3(&tpt_math_linalg_fixed::Matrix3::new(scatter));
    // Eigenvalues descending; axes = eigenvectors, sign-disambiguated.
    let mut axes = [[0.0; 3]; 3];
    for (a, ev) in axes.iter_mut().zip(eig.vectors.iter()) {
        *a = disambiguate(ev.data, cloud, neighbors, center);
    }
    // Re-orthogonalise y against x to undo numerical drift from the two
    // independent sign choices.
    let x = axes[0];
    let y = {
        let raw = axes[1];
        let d = dot3(&x, &raw);
        let r = [raw[0] - d * x[0], raw[1] - d * x[1], raw[2] - d * x[2]];
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if n < 1e-12 {
            [0.0; 3]
        } else {
            [r[0] / n, r[1] / n, r[2] / n]
        }
    };
    let z = cross3(&x, &y);
    [x, y, z]
}

/// Sign rule: choose the orientation maximising `Σ sign(a·d)(a·d)²`.
fn disambiguate(
    axis: [f64; 3],
    cloud: &PointCloud,
    neighbors: &[(usize, f64)],
    center: [f64; 3],
) -> [f64; 3] {
    let mut s = 0.0;
    for &(idx, _) in neighbors {
        let q = cloud.get(idx).expect("in range");
        let d = [q[0] - center[0], q[1] - center[1], q[2] - center[2]];
        let a = dot3(&axis, &d);
        s += a.signum() * a * a;
    }
    if s < 0.0 {
        [-axis[0], -axis[1], -axis[2]]
    } else {
        axis
    }
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn grid() -> PointCloud {
        let mut c = PointCloud::new();
        for x in -4..=4i32 {
            for y in -4..=4i32 {
                c.push([x as f64 * 0.2, y as f64 * 0.2, 0.0]);
            }
        }
        c
    }

    #[test]
    fn dimensions_and_normalisation() {
        let cloud = grid();
        let shot = compute_shot(&cloud, 0.9).unwrap();
        assert_eq!(shot.len(), cloud.len());
        assert_eq!(SHOT_DIM, 352);
        for desc in &shot {
            let sum: f64 = desc.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9 || sum == 0.0, "sum {sum}");
        }
    }

    #[test]
    fn translation_invariance() {
        let a = grid();
        let b: Vec<[f64; 3]> = a
            .points()
            .iter()
            .map(|&p| [p[0] + 50.0, p[1] - 20.0, p[2] + 3.0])
            .collect();
        let b = PointCloud::from_points(b);
        let fa = compute_shot(&a, 0.9).unwrap();
        let fb = compute_shot(&b, 0.9).unwrap();
        let centre = (4 * 9 + 4) as usize; // (0,0) in the 9×9 grid
        for j in 0..SHOT_DIM {
            assert!(
                (fa[centre][j] - fb[centre][j]).abs() < 1e-12,
                "component {j}"
            );
        }
    }

    #[test]
    fn small_cloud_rejected() {
        let c = PointCloud::from_points(vec![[0.0; 3]; 3]);
        assert!(compute_shot(&c, 1.0).is_err());
        assert!(compute_shot(&grid(), -1.0).is_err());
    }
}
