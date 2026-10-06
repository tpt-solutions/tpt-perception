//! Local surface analysis: PCA, normals and curvature.
//!
//! Everything here reduces to the eigendecomposition of the 3×3 scatter
//! matrix of a neighbourhood (via [`tpt_percept_core::linalg3::sym_eigen3`]):
//!
//! * the eigenvector of the **smallest** eigenvalue is the surface normal;
//! * the **curvature** is `λ_min / (λ₀+λ₁+λ₂)` — 0 for a plane, ~1/3 for an
//!   isotropic blob.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;
use tpt_percept_core::linalg3::{outer3, sym_eigen3, Eigen3};

/// A surface normal with its curvature estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    /// Unit normal (orientation is arbitrary unless oriented elsewhere).
    pub normal: [f64; 3],
    /// Curvature in `[0, 1]`: ratio of the smallest eigenvalue to the
    /// eigenvalue sum (0 = planar, 1/3 = isotropic).
    pub curvature: f64,
}

/// Principal component analysis of a point set: centroid + scatter
/// eigendecomposition (eigenvalues descending).
pub fn pca(points: &[[f64; 3]]) -> Option<([f64; 3], Eigen3)> {
    if points.is_empty() {
        return None;
    }
    let mut c = [0.0; 3];
    for p in points {
        for i in 0..3 {
            c[i] += p[i];
        }
    }
    let n = points.len() as f64;
    c = [c[0] / n, c[1] / n, c[2] / n];

    let mut scatter = [[0.0; 3]; 3];
    for p in points {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let o = outer3(d, d);
        for r in 0..3 {
            for cc in 0..3 {
                scatter[r][cc] += o.data[r][cc];
            }
        }
    }
    for row in &mut scatter {
        for v in row.iter_mut() {
            *v /= n;
        }
    }
    Some((c, sym_eigen3(&tpt_math_linalg_fixed::Matrix3::new(scatter))))
}

/// Estimates a normal (with curvature) for every point from its `k` nearest
/// neighbours.
///
/// Returns `None` if `k < 3` (a normal needs a non-degenerate 3-point
/// neighbourhood) or the cloud is smaller than 4 points. Degenerate
/// neighbourhoods (collinear/repeated points) produce a zero normal and
/// curvature 1/3 rather than an error — callers can filter on
/// `curvature < 1.0/3.0 - eps` to reject them.
pub fn estimate_normals(cloud: &PointCloud, k: usize) -> Option<Vec<Normal>> {
    if k < 3 || cloud.len() < 4 {
        return None;
    }
    let tree = KdTree::new(cloud.points());
    let mut out = Vec::with_capacity(cloud.len());
    for p in cloud.points() {
        let neighbors = tree.knn(p, k + 1); // +1 to drop the point itself
        let patch: Vec<[f64; 3]> = neighbors
            .iter()
            .filter(|&&(idx, _)| cloud.get(idx) != Some(p))
            .map(|&(idx, _)| *cloud.get(idx).expect("knn indices are in range"))
            .collect();
        match pca(&patch) {
            Some((_, eig)) => {
                // Eigenvalues descending; smallest is last.
                let sum = eig.values[0] + eig.values[1] + eig.values[2];
                let v = eig.vectors[2];
                let curvature = if sum > 0.0 {
                    eig.values[2] / sum
                } else {
                    1.0 / 3.0
                };
                let n = v.norm();
                let normal = if n > 1e-12 {
                    [v.data[0] / n, v.data[1] / n, v.data[2] / n]
                } else {
                    [0.0; 3]
                };
                out.push(Normal { normal, curvature });
            }
            None => out.push(Normal {
                normal: [0.0; 3],
                curvature: 1.0 / 3.0,
            }),
        }
    }
    Some(out)
}

/// Re-orients each normal to point towards `viewpoint` (standard LiDAR /
/// camera convention: `n · (viewpoint − p) > 0`).
pub fn orient_normals_towards(normals: &mut [Normal], cloud: &PointCloud, viewpoint: [f64; 3]) {
    for (n, &p) in normals.iter_mut().zip(cloud.points()) {
        let to_vp = [
            viewpoint[0] - p[0],
            viewpoint[1] - p[1],
            viewpoint[2] - p[2],
        ];
        if n.normal[0] * to_vp[0] + n.normal[1] * to_vp[1] + n.normal[2] * to_vp[2] < 0.0 {
            n.normal = [-n.normal[0], -n.normal[1], -n.normal[2]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_cloud() -> PointCloud {
        let mut c = PointCloud::new();
        for x in -4..=4i32 {
            for y in -4..=4i32 {
                c.push([x as f64, y as f64, 2.0]);
            }
        }
        c
    }

    #[test]
    fn plane_normals_are_z() {
        let cloud = plane_cloud();
        let normals = estimate_normals(&cloud, 10).unwrap();
        for n in &normals {
            assert!((n.normal[2].abs() - 1.0).abs() < 1e-9, "normal {n:?}");
            assert!(n.curvature < 1e-9, "curvature {n:?}");
        }
    }

    #[test]
    fn sphere_normals_are_radial() {
        let mut cloud = PointCloud::new();
        // Fibonacci sphere.
        let n = 400;
        let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let a = golden * i as f64;
            cloud.push([r * a.cos(), r * a.sin(), z]);
        }
        let normals = estimate_normals(&cloud, 12).unwrap();
        for (norm, &p) in normals.iter().zip(cloud.points()) {
            let pl = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let radial = [p[0] / pl, p[1] / pl, p[2] / pl];
            let dot = (radial[0] * norm.normal[0]
                + radial[1] * norm.normal[1]
                + radial[2] * norm.normal[2])
                .abs();
            assert!(dot > 0.99, "at {p:?}: dot {dot}");
        }
    }

    #[test]
    fn orientation_towards_viewpoint() {
        let cloud = plane_cloud();
        let mut normals = estimate_normals(&cloud, 10).unwrap();
        orient_normals_towards(&mut normals, &cloud, [0.0, 0.0, 100.0]);
        for (n, &p) in normals.iter().zip(cloud.points()) {
            // All normals must point +z towards the viewpoint above.
            assert!(n.normal[2] > 0.0);
            let _ = p;
        }
    }

    #[test]
    fn too_few_neighbors_rejected() {
        let cloud = plane_cloud();
        assert!(estimate_normals(&cloud, 2).is_none());
        assert!(estimate_normals(&PointCloud::new(), 10).is_none());
    }

    #[test]
    fn pca_major_axis() {
        let pts: Vec<[f64; 3]> = (0..50).map(|i| [i as f64, 0.0, 0.0]).collect();
        let (_, eig) = pca(&pts).unwrap();
        // Line: first eigenvalue dominates, eigenvector ≈ x.
        assert!(eig.vectors[0].data[0].abs() > 0.999);
        assert!(eig.values[1] < 1e-9 && eig.values[2] < 1e-9);
    }
}
