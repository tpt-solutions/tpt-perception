//! Point cloud filters: statistical outlier removal, radius filtering and
//! pass-through cropping.
//!
//! All filters return the *indices of surviving points* so they compose and
//! so per-point auxiliary data can be filtered with the same index list.

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use alloc::vec::Vec;

use crate::cloud::PointCloud;
use crate::kdtree::KdTree;

/// Statistical outlier removal (Rusu & Cousins' classic): computes for every
/// point the mean distance to its `k` nearest neighbours, then rejects
/// points whose mean distance exceeds `global_mean + std_ratio · global_std`.
///
/// Returns indices of the kept points. Requires `k >= 1` and `k < len`
/// (each point excludes itself); otherwise returns an error.
pub fn statistical_outlier_removal(
    cloud: &PointCloud,
    k: usize,
    std_ratio: f64,
) -> Result<Vec<usize>, FilterError> {
    if k == 0 {
        return Err(FilterError::InvalidParameter("k must be >= 1"));
    }
    let n = cloud.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if k >= n {
        return Err(FilterError::InvalidParameter(
            "k must be smaller than the point count",
        ));
    }

    let tree = KdTree::new(cloud.points());
    let mut means = Vec::with_capacity(n);
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.knn(&p, k + 1);
        let sum: f64 = neighbors
            .iter()
            .filter(|&&(idx, _)| idx != i)
            .take(k)
            .map(|&(_, d)| d.sqrt())
            .sum();
        means.push(sum / k as f64);
    }

    let mean = means.iter().sum::<f64>() / n as f64;
    let variance = means.iter().map(|&m| (m - mean) * (m - mean)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    let threshold = mean + std_ratio * std;

    Ok((0..n).filter(|&i| means[i] <= threshold).collect())
}

/// Radius outlier removal: keeps points with at least `min_neighbors`
/// strictly inside `radius` (metres).
///
/// Returns indices of the kept points; errors on `radius <= 0`.
pub fn radius_filter(
    cloud: &PointCloud,
    radius: f64,
    min_neighbors: usize,
) -> Result<Vec<usize>, FilterError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(FilterError::InvalidParameter(
            "radius must be finite and > 0",
        ));
    }
    if cloud.is_empty() {
        return Ok(Vec::new());
    }
    let tree = KdTree::new(cloud.points());
    let mut kept = Vec::new();
    for (i, &p) in cloud.points().iter().enumerate() {
        let count = tree.radius_count_excluding(&p, radius * radius, i);
        if count >= min_neighbors {
            kept.push(i);
        }
    }
    Ok(kept)
}

/// Axis selector for the pass-through filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// X axis.
    X,
    /// Y axis.
    Y,
    /// Z axis.
    Z,
}

impl Axis {
    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// Pass-through filter: keeps points with `min <= p[axis] < max` (metres).
///
/// Returns indices of the kept points; errors when `min >= max`.
pub fn pass_through(
    cloud: &PointCloud,
    axis: Axis,
    min: f64,
    max: f64,
) -> Result<Vec<usize>, FilterError> {
    if !(min.is_finite() && max.is_finite()) || min >= max {
        return Err(FilterError::InvalidParameter(
            "pass-through requires finite min < max",
        ));
    }
    let a = axis.index();
    Ok((0..cloud.len())
        .filter(|&i| {
            let p = cloud.get(i).expect("index in range");
            p[a] >= min && p[a] < max
        })
        .collect())
}

/// Filter parameter errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterError {
    /// A parameter violated its documented constraint.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FilterError::InvalidParameter(what) => write!(f, "invalid filter parameter: {what}"),
        }
    }
}

impl core::error::Error for FilterError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense blob at the origin plus one far-away outlier.
    fn blob_with_outlier() -> PointCloud {
        let mut c = PointCloud::new();
        for x in -5..5 {
            for y in -5..5 {
                c.push([x as f64 * 0.1, y as f64 * 0.1, 0.0]);
            }
        }
        c.push([50.0, 50.0, 50.0]);
        c
    }

    #[test]
    fn statistical_removes_isolated_point() {
        let cloud = blob_with_outlier();
        let kept = statistical_outlier_removal(&cloud, 8, 1.0).unwrap();
        assert_eq!(kept.len(), 100); // outlier dropped, all blob points kept
    }

    #[test]
    fn radius_removes_isolated_point() {
        let cloud = blob_with_outlier();
        let kept = radius_filter(&cloud, 0.5, 3).unwrap();
        assert_eq!(kept.len(), 100);
    }

    #[test]
    fn passthrough_crops_range() {
        let cloud = blob_with_outlier();
        let kept = pass_through(&cloud, Axis::X, -0.15, 0.15).unwrap();
        assert_eq!(kept.len(), 30); // x in {-0.1, 0.0, 0.1} × 10 y rows
    }

    #[test]
    fn parameter_validation() {
        let cloud = blob_with_outlier();
        assert!(radius_filter(&cloud, 0.0, 1).is_err());
        assert!(statistical_outlier_removal(&cloud, 0, 1.0).is_err());
        assert!(statistical_outlier_removal(&cloud, 200, 1.0).is_err());
        assert!(pass_through(&cloud, Axis::Z, 1.0, 1.0).is_err());
    }
}
