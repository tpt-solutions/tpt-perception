//! Edge / boundary detection in point clouds.
//!
//! Two complementary criteria:
//!
//! * **Boundary estimation** (angle criterion): project the neighbourhood
//!   onto the local tangent plane; if the largest angular gap between
//!   consecutive neighbour directions exceeds a threshold, the point sits on
//!   a boundary — its neighbourhood does not surround it.
//! * **Linear-structure edges** (principal curvature): points whose
//!   neighbourhood scatter is strongly elongated (`λ₁ ≥ ratio · λ₂`, the
//!   two largest eigenvalues) lie on thin/one-dimensional structures —
//!   wires, poles, scan-line remnants. Folds between surfaces are *not*
//!   anisotropic in this sense (their neighbourhood spreads in two
//!   directions); use [`detect_boundaries`] or Harris corners for those.

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::normal::{estimate_normals, pca};
use crate::FeatureError;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;

/// Detects boundary points with the angle criterion.
///
/// A point is a boundary point when the maximum angular gap between
/// consecutive neighbours (sorted by tangent-plane azimuth) exceeds
/// `angle_threshold` radians. Returns the indices of boundary points.
pub fn detect_boundaries(
    cloud: &PointCloud,
    k: usize,
    angle_threshold: f64,
) -> Result<Vec<usize>, FeatureError> {
    if k < 6 || k >= cloud.len() {
        return Err(FeatureError::InvalidParameter(
            "k must be >= 6 and < point count",
        ));
    }
    if !(angle_threshold > 0.0 && angle_threshold < core::f64::consts::PI) {
        return Err(FeatureError::InvalidParameter(
            "angle threshold must be in (0, π)",
        ));
    }
    let normals = estimate_normals(cloud, k).ok_or(FeatureError::InvalidParameter(
        "cloud too small for normal estimation",
    ))?;
    let tree = KdTree::new(cloud.points());

    let mut boundary = Vec::new();
    for (i, &p) in cloud.points().iter().enumerate() {
        let n = &normals[i].normal;
        let neighbors = tree.knn(&p, k + 1);
        // Tangent-plane basis (u, v) ⊥ n.
        let mut helper = [1.0, 0.0, 0.0];
        if n[0].abs() > 0.9 {
            helper = [0.0, 1.0, 0.0];
        }
        let u = cross3(n, &helper);
        let ul = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        if ul < 1e-12 {
            continue;
        }
        let u = [u[0] / ul, u[1] / ul, u[2] / ul];
        let v = cross3(n, &u);

        let mut angles: Vec<f64> = neighbors
            .iter()
            .filter(|&&(idx, _)| idx != i)
            .map(|&(idx, _)| {
                let q = cloud.get(idx).expect("knn indices in range");
                let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                // Skip neighbours essentially on the normal axis.
                let du = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
                let dv = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
                du.atan2(dv)
            })
            .collect();
        if angles.len() < 3 {
            boundary.push(i);
            continue;
        }
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let two_pi = 2.0 * core::f64::consts::PI;
        let mut max_gap = two_pi + angles[0] - angles[angles.len() - 1];
        for w in angles.windows(2) {
            max_gap = max_gap.max(w[1] - w[0]);
        }
        if max_gap > angle_threshold {
            boundary.push(i);
        }
    }
    Ok(boundary)
}

/// Detects linear-structure edges via principal-curvature anisotropy:
/// points whose neighbourhood is strongly elongated (`λ₁ ≥ ratio · λ₂`
/// with `λ₁` the largest eigenvalue) lie on thin, one-dimensional
/// structures.
pub fn detect_crease_edges(
    cloud: &PointCloud,
    k: usize,
    ratio: f64,
) -> Result<Vec<usize>, FeatureError> {
    if k < 6 || k >= cloud.len() {
        return Err(FeatureError::InvalidParameter(
            "k must be >= 6 and < point count",
        ));
    }
    if !(ratio.is_finite() && ratio > 1.0) {
        return Err(FeatureError::InvalidParameter(
            "ratio must be finite and > 1",
        ));
    }
    let tree = KdTree::new(cloud.points());
    let mut edges = Vec::new();
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.knn(&p, k + 1);
        let patch: Vec<[f64; 3]> = neighbors
            .iter()
            .filter(|&&(idx, _)| idx != i)
            .map(|&(idx, _)| *cloud.get(idx).expect("knn indices in range"))
            .collect();
        if let Some((_, eig)) = pca(&patch) {
            let l1 = eig.values[0].max(0.0);
            let l2 = eig.values[1].max(0.0);
            if l2 <= 1e-15 {
                // Perfectly linear neighbourhood — a thin structure edge.
                edges.push(i);
            } else if l1 >= ratio * l2 {
                edges.push(i);
            }
        }
    }
    Ok(edges)
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

    /// A filled square sample in the z = 0 plane.
    fn square_cloud() -> PointCloud {
        let mut c = PointCloud::new();
        for x in -5..=5i32 {
            for y in -5..=5i32 {
                c.push([x as f64 * 0.2, y as f64 * 0.2, 0.0]);
            }
        }
        c
    }

    #[test]
    fn square_boundary_found_on_rim() {
        let cloud = square_cloud();
        let boundary = detect_boundaries(&cloud, 12, 2.4).unwrap();
        assert!(!boundary.is_empty(), "no boundary points found");
        // Every reported boundary point must lie on the rim.
        for &i in &boundary {
            let p = cloud.get(i).unwrap();
            assert!(
                p[0].abs() >= 0.99 || p[1].abs() >= 0.99,
                "interior point {p:?} flagged as boundary"
            );
        }
        // Corners (extremes in both axes) must be among them.
        assert!(boundary.iter().any(|&i| {
            let p = cloud.get(i).unwrap();
            p[0] > 0.9 && p[1] > 0.9
        }));
    }

    #[test]
    fn interior_not_flagged_with_reasonable_threshold() {
        let cloud = square_cloud();
        let boundary = detect_boundaries(&cloud, 12, 2.4).unwrap();
        let interior_centre = cloud
            .points()
            .iter()
            .position(|p| p[0].abs() < 0.1 && p[1].abs() < 0.1)
            .unwrap();
        assert!(!boundary.contains(&interior_centre));
    }

    #[test]
    fn crease_edges_on_thin_rod() {
        // A thin rod along y: every rod point's neighbourhood is a line.
        let mut c = PointCloud::new();
        for y in -10..=10i32 {
            c.push([0.0, y as f64 * 0.1, 0.0]);
        }
        // Plus a flat sheet far away so the cloud has > k points in total.
        for x in -3..=3i32 {
            for z in -3..=3i32 {
                c.push([3.0 + x as f64 * 0.2, -2.0, 3.0 + z as f64 * 0.2]);
            }
        }
        let cloud = c;
        let edges = detect_crease_edges(&cloud, 8, 4.0).unwrap();
        // Rod points (y from -1 to 1) must be detected.
        assert!(
            edges.iter().any(|&i| {
                let p = cloud.get(i).unwrap();
                p[0].abs() < 1e-9 && p[2].abs() < 1e-9 && p[1].abs() < 1.0
            }),
            "rod not detected as edges; got {} edges",
            edges.len()
        );
    }

    #[test]
    fn parameter_validation() {
        let cloud = square_cloud();
        assert!(detect_boundaries(&cloud, 3, 2.0).is_err());
        assert!(detect_boundaries(&cloud, 12, 5.0).is_err());
        assert!(detect_crease_edges(&cloud, 12, 0.5).is_err());
    }
}
