//! 3-D Harris corner detection.
//!
//! Follows the Sipiran & Bustos (2011) recipe in spirit: estimate normals,
//! build a second-moment matrix from normal variation over each
//! neighbourhood, and score points with the classical Harris response
//! `det(M) − α · trace(M)²`. Corners — where surface orientation varies in
//! *all* directions — maximise the response; planar interiors minimise it.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::normal::estimate_normals;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;
use tpt_percept_core::linalg3::{det3, outer3};

/// Parameters for [`harris3d`].
#[derive(Clone, Copy, Debug)]
pub struct HarrisParams {
    /// Neighbourhood size (nearest neighbours) for the second-moment matrix.
    pub k: usize,
    /// Harris sensitivity constant (classically 0.04–0.2).
    pub alpha: f64,
    /// Non-max suppression: a corner must beat all neighbours within this
    /// radius (metres).
    pub nms_radius: f64,
    /// Minimum response for a point to be reported as a corner.
    pub min_response: f64,
}

impl Default for HarrisParams {
    fn default() -> Self {
        HarrisParams {
            k: 16,
            alpha: 0.04,
            nms_radius: 0.3,
            min_response: 1e-6,
        }
    }
}

/// A detected corner: its index into the source cloud and Harris response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// Index into the source cloud.
    pub index: usize,
    /// Harris response `det(M) − α·trace(M)²`.
    pub response: f64,
}

/// Detects 3-D Harris corners.
///
/// Requires `params.k < cloud.len()`. Returns corners sorted by descending
/// response after non-max suppression.
pub fn harris3d(cloud: &PointCloud, params: &HarrisParams) -> Result<Vec<Corner>, FeatureError> {
    if params.k < 4 || params.k >= cloud.len() {
        return Err(FeatureError::InvalidParameter(
            "k must be >= 4 and smaller than the point count",
        ));
    }
    let normals = estimate_normals(cloud, params.k).ok_or(FeatureError::InvalidParameter(
        "cloud too small for normal estimation",
    ))?;

    let tree = KdTree::new(cloud.points());
    let mut responses: Vec<f64> = Vec::with_capacity(cloud.len());
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.knn(&p, params.k + 1);
        // Mean normal over the neighbourhood (orientation-agnostic: flip each
        // neighbour normal to agree with the centre's).
        let n0 = &normals[i].normal;
        let mut m = [[0.0; 3]; 3];
        let mut count = 0.0;
        let mut mean = [0.0; 3];
        for &(idx, d2) in neighbors.iter() {
            if idx == i {
                continue;
            }
            let w = 1.0 / (d2.sqrt() + 1e-12);
            let mut g = normals[idx].normal;
            if dot3(&g, n0) < 0.0 {
                g = [-g[0], -g[1], -g[2]];
            }
            for j in 0..3 {
                mean[j] += w * g[j];
            }
            count += w;
        }
        if count == 0.0 {
            responses.push(0.0);
            continue;
        }
        for j in 0..3 {
            mean[j] /= count;
        }
        for &(idx, d2) in neighbors.iter() {
            if idx == i {
                continue;
            }
            let w = 1.0 / (d2.sqrt() + 1e-12);
            let mut g = normals[idx].normal;
            if dot3(&g, n0) < 0.0 {
                g = [-g[0], -g[1], -g[2]];
            }
            let d = [g[0] - mean[0], g[1] - mean[1], g[2] - mean[2]];
            let o = outer3(d, d);
            for r in 0..3 {
                for c in 0..3 {
                    m[r][c] += w * o.data[r][c];
                }
            }
        }
        let det = det3(&m);
        let trace = m[0][0] + m[1][1] + m[2][2];
        responses.push(det - params.alpha * trace * trace);
    }

    // Non-max suppression within `nms_radius`.
    let mut candidates: Vec<Corner> = (0..cloud.len())
        .filter(|&i| responses[i] > params.min_response)
        .map(|i| Corner {
            index: i,
            response: responses[i],
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Corner> = Vec::new();
    for c in candidates {
        let p = cloud.get(c.index).expect("index in range");
        let dominated = kept.iter().any(|k: &Corner| {
            let q = cloud.get(k.index).expect("index in range");
            let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
            d2 <= params.nms_radius * params.nms_radius
        });
        if !dominated {
            kept.push(c);
        }
    }
    Ok(kept)
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Feature extraction errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureError {
    /// A parameter violated its documented constraint.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FeatureError::InvalidParameter(what) => {
                write!(f, "invalid feature parameter: {what}")
            }
        }
    }
}

impl core::error::Error for FeatureError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box surface sample: corners should respond more than face centres.
    fn box_cloud(step: f64, half: f64) -> PointCloud {
        let mut c = PointCloud::new();
        let mut vals = Vec::new();
        let mut v = -half;
        while v <= half {
            vals.push(v);
            v += step;
        }
        // 6 faces.
        for &a in &vals {
            for &b in &vals {
                for (p, sign) in [([a, b, half], 1.0), ([a, b, -half], -1.0)] {
                    let _ = sign;
                    c.push(p);
                    c.push([b, half, a]);
                    c.push([half, a, b]);
                    c.push([b, -half, a]);
                    c.push([-half, a, b]);
                }
            }
        }
        c
    }

    #[test]
    fn box_corners_beat_face_centre() {
        let cloud = box_cloud(0.25, 1.0);
        let params = HarrisParams {
            k: 20,
            nms_radius: 0.2,
            ..Default::default()
        };
        let corners = harris3d(&cloud, &params).unwrap();
        // The extreme corner point (1.0, 1.0, 1.0)-ish must be among the
        // detections.
        assert!(
            corners.iter().any(|c| {
                let p = cloud.get(c.index).unwrap();
                p[0] > 0.9 && p[1] > 0.9 && p[2] > 0.9
            }),
            "corner of the box not detected; got {corners:?}"
        );
    }

    #[test]
    fn parameter_validation() {
        let cloud = box_cloud(0.5, 1.0);
        assert!(harris3d(
            &cloud,
            &HarrisParams {
                k: 3,
                ..Default::default()
            }
        )
        .is_err());
        assert!(harris3d(
            &cloud,
            &HarrisParams {
                k: 10_000,
                ..Default::default()
            }
        )
        .is_err());
    }

    #[test]
    fn deterministic_output() {
        let cloud = box_cloud(0.25, 1.0);
        let params = HarrisParams::default();
        let a = harris3d(&cloud, &params).unwrap();
        let b = harris3d(&cloud, &params).unwrap();
        assert_eq!(a, b);
    }
}
