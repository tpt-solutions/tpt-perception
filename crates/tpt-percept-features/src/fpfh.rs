//! FPFH — Fast Point Feature Histograms (Rusu et al. 2009).
//!
//! For every point, a 33-bin histogram summarises the angular relationships
//! between the point's normal frame and its neighbours' normals (the
//! *simplified PFH*, SPFH). The full FPFH re-weights a point's SPFH with the
//! SPFHs of its neighbours by inverse distance — an O(k) approximation of
//! the O(k²) PFH that preserves most of its descriptive power.
//!
//! Each component is normalised to `[0, 1]`, which makes descriptors
//! comparable across neighbourhood sizes.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::normal::{estimate_normals, Normal};
use crate::FeatureError;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;

/// Number of histogram bins per angular feature.
const BINS: usize = 11;
/// Descriptor dimensionality: 3 angular features × 11 bins.
pub const FPFH_DIM: usize = 3 * BINS;

/// Computes the FPFH descriptor for every point.
///
/// Returns one `FPFH_DIM`-dimensional descriptor per point, each component
/// in `[0, 1]`.
pub fn compute_fpfh(cloud: &PointCloud, k: usize) -> Result<Vec<[f64; FPFH_DIM]>, FeatureError> {
    if k < 5 || k >= cloud.len() {
        return Err(FeatureError::InvalidParameter(
            "k must be >= 5 and < point count",
        ));
    }
    let normals = estimate_normals(cloud, k).ok_or(FeatureError::InvalidParameter(
        "cloud too small for normal estimation",
    ))?;
    let tree = KdTree::new(cloud.points());

    // SPFH for every point.
    let mut spfhs: Vec<[f64; FPFH_DIM]> = Vec::with_capacity(cloud.len());
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.knn(&p, k + 1);
        let mut h = [0.0; FPFH_DIM];
        let mut count = 0.0;
        for &(idx, _) in &neighbors {
            if idx == i {
                continue;
            }
            accumulate_spfh(
                &mut h,
                &p,
                &normals[i],
                cloud.get(idx).expect("in range"),
                &normals[idx],
            );
            count += 1.0;
        }
        if count > 0.0 {
            for v in &mut h {
                *v /= count;
            }
        }
        spfhs.push(h);
    }

    // FPFH = own SPFH + distance-weighted neighbour SPFHs.
    let mut out = Vec::with_capacity(cloud.len());
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.knn(&p, k + 1);
        let mut h = spfhs[i];
        let mut weight_sum = 0.0;
        for &(idx, d2) in &neighbors {
            if idx == i || d2 <= 0.0 {
                continue;
            }
            let w = 1.0 / d2.sqrt();
            for j in 0..FPFH_DIM {
                h[j] += w * spfhs[idx][j];
            }
            weight_sum += w;
        }
        if weight_sum > 0.0 {
            let norm = (1.0 + weight_sum).recip();
            for v in &mut h {
                *v *= norm;
            }
        }
        out.push(h);
    }
    Ok(out)
}

/// Accumulates one (p_i, p_j) angular triple into the histogram.
fn accumulate_spfh(
    h: &mut [f64; FPFH_DIM],
    pi: &[f64; 3],
    ni: &Normal,
    pj: &[f64; 3],
    nj: &Normal,
) {
    let u = ni.normal;
    let d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let v = cross3(&d, &u);
    let vl = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if vl < 1e-12 {
        return;
    }
    let v = [v[0] / vl, v[1] / vl, v[2] / vl];
    let w = cross3(&u, &v);

    let nnj = nj.normal;
    let f1 = v[0] * nnj[0] + v[1] * nnj[1] + v[2] * nnj[2]; // v · n_j ∈ [-1, 1]
    let f2 = w[0] * nnj[0] + w[1] * nnj[1] + w[2] * nnj[2]; // w · n_j ∈ [-1, 1]
    let f3 = (u[0] * nnj[0] + u[1] * nnj[1] + u[2] * nnj[2]).atan2(f1); // atan2(u·n_j, f1) ∈ (-π, π]

    // Map to [0, 1) and bin.
    let b1 = bin01((f1 + 1.0) * 0.5);
    let b2 = bin01((f2 + 1.0) * 0.5);
    let b3 = bin01((f3 + core::f64::consts::PI) / (2.0 * core::f64::consts::PI));
    h[b1] += 1.0;
    h[BINS + b2] += 1.0;
    h[2 * BINS + b3] += 1.0;
}

fn bin01(x: f64) -> usize {
    let clamped = x.clamp(0.0, 0.999_999);
    (clamped * BINS as f64) as usize
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

    #[test]
    fn dimensions_and_range() {
        let mut c = PointCloud::new();
        for x in -3..=3i32 {
            for y in -3..=3i32 {
                for z in -3..=3i32 {
                    c.push([x as f64 * 0.3, y as f64 * 0.3, z as f64 * 0.3]);
                }
            }
        }
        let fpfh = compute_fpfh(&c, 10).unwrap();
        assert_eq!(fpfh.len(), c.len());
        for desc in &fpfh {
            for &v in desc.iter() {
                assert!((0.0..=1.0 + 1e-9).contains(&v), "value {v}");
            }
        }
    }

    #[test]
    fn identical_local_geometry_gives_identical_descriptors() {
        // Two plane patches (same sampling) at different positions: their
        // interior points must share (nearly) the same descriptor.
        let mut a = PointCloud::new();
        let mut b = PointCloud::new();
        for x in -3..=3i32 {
            for y in -3..=3i32 {
                a.push([x as f64 * 0.3, y as f64 * 0.3, 0.0]);
                b.push([x as f64 * 0.3 + 100.0, y as f64 * 0.3, 7.0]);
            }
        }
        let fa = compute_fpfh(&a, 8).unwrap();
        let fb = compute_fpfh(&b, 8).unwrap();
        let centre = (4 * 7 + 3) as usize; // (x=0,y=0) in a 7×7 grid
        for j in 0..FPFH_DIM {
            assert!(
                (fa[centre][j] - fb[centre][j]).abs() < 1e-9,
                "component {j}: {} vs {}",
                fa[centre][j],
                fb[centre][j]
            );
        }
    }

    #[test]
    fn fold_point_differs_from_face_interior() {
        // FPFH encodes normal relationships, so geometry with *differing
        // normals* is required to separate points: a 3-D fold (floor + wall)
        // vs a point in the floor's interior.
        let mut c = PointCloud::new();
        for x in -5..=5i32 {
            for y in -5..=5i32 {
                c.push([x as f64 * 0.3, y as f64 * 0.3, 0.0]); // floor
            }
        }
        for x in 1..=5i32 {
            for y in -5..=5i32 {
                // Wall rising from the floor along z, meeting it at x = 0.
                c.push([0.0, y as f64 * 0.3, x as f64 * 0.3]);
            }
        }
        let fpfh = compute_fpfh(&c, 12).unwrap();
        let fold = c
            .points()
            .iter()
            .position(|p| p[0].abs() < 1e-9 && p[1].abs() < 1e-9 && p[2].abs() < 1e-9)
            .unwrap();
        let flat = c
            .points()
            .iter()
            .position(|p| (p[0] - 0.9).abs() < 1e-9 && p[1].abs() < 1e-9 && p[2].abs() < 1e-9)
            .unwrap();
        let dist: f64 = fpfh[fold]
            .iter()
            .zip(&fpfh[flat])
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            dist > 0.05,
            "fold and interior descriptors too similar: {dist}"
        );
    }

    #[test]
    fn parameter_validation() {
        let c = PointCloud::from_points(vec![[0.0; 3]; 5]);
        assert!(compute_fpfh(&c, 3).is_err());
        assert!(compute_fpfh(&c, 10).is_err());
    }
}
