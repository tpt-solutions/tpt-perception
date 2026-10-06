//! Plane segmentation via RANSAC.
//!
//! Samples minimal 3-point plane hypotheses, scores them by inlier count
//! within a distance threshold, and iteratively extracts dominant planes.
//! The internal RNG is a deterministic xorshift64*, so identical inputs and
//! seeds produce identical segmentations — property tests and CI stay
//! reproducible.

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::FeatureError;
use tpt_percept_cloud::cloud::PointCloud;

/// A fitted plane `n · p + d = 0` with unit normal `n`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit normal.
    pub normal: [f64; 3],
    /// Offset: `n · p + d = 0`.
    pub d: f64,
}

impl Plane {
    /// Signed distance of `p` to the plane (positive on the normal side).
    pub fn signed_distance(&self, p: &[f64; 3]) -> f64 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }
}

/// A segmented plane with its inliers.
#[derive(Clone, Debug)]
pub struct PlaneSegment {
    /// Fitted plane (least-squares-refined over the inliers).
    pub plane: Plane,
    /// Indices of inlier points into the source cloud.
    pub inliers: Vec<usize>,
}

/// Parameters for plane segmentation.
#[derive(Clone, Copy, Debug)]
pub struct RansacParams {
    /// Inlier distance threshold (metres).
    pub distance_threshold: f64,
    /// RANSAC iterations per plane extraction.
    pub max_iterations: u32,
    /// A plane must gather at least this many inliers to be reported.
    pub min_inliers: usize,
    /// RNG seed (deterministic segmentation).
    pub seed: u64,
}

impl Default for RansacParams {
    fn default() -> Self {
        RansacParams {
            distance_threshold: 0.02,
            max_iterations: 1000,
            min_inliers: 10,
            seed: 0x9E3779B97F4A7C15,
        }
    }
}

/// Segments up to `max_planes` dominant planes from the cloud.
///
/// Planes are extracted greedily (largest inlier set first); each extraction
/// removes its inliers from the pool. Returns fewer planes when no further
/// plane gathers `min_inliers` support.
pub fn segment_planes(
    cloud: &PointCloud,
    params: &RansacParams,
    max_planes: usize,
) -> Result<Vec<PlaneSegment>, FeatureError> {
    if !(params.distance_threshold.is_finite() && params.distance_threshold > 0.0) {
        return Err(FeatureError::InvalidParameter(
            "distance threshold must be > 0",
        ));
    }
    if params.max_iterations == 0 {
        return Err(FeatureError::InvalidParameter("max_iterations must be > 0"));
    }
    if cloud.len() < 3 {
        return Ok(Vec::new());
    }

    let mut remaining: Vec<usize> = (0..cloud.len()).collect();
    let mut out = Vec::new();
    for _ in 0..max_planes {
        if remaining.len() < params.min_inliers.max(3) {
            break;
        }
        let Some((plane, mut inliers)) = ransac_once(cloud, &remaining, params) else {
            break;
        };
        if inliers.len() < params.min_inliers {
            break;
        }
        inliers.sort_unstable();
        // Refine with a least-squares fit over all inliers (PCA of the patch).
        let patch: Vec<[f64; 3]> = inliers
            .iter()
            .map(|&i| *cloud.get(i).expect("inlier index in range"))
            .collect();
        if let Some((c, eig)) = crate::normal::pca(&patch) {
            if let Some(n) = normalize3(&eig.vectors[2].data) {
                let d = -(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]);
                // Keep the refined plane's normal on the same side as the
                // hypothesis so distances stay consistent across iterations.
                let refined = Plane { normal: n, d };
                let consistent = inliers.iter().all(|&i| {
                    let p = cloud.get(i).expect("inlier index in range");
                    refined.signed_distance(p).abs() <= params.distance_threshold
                });
                if consistent {
                    out.push(PlaneSegment {
                        plane: refined,
                        inliers,
                    });
                } else {
                    out.push(PlaneSegment { plane, inliers });
                }
            } else {
                out.push(PlaneSegment { plane, inliers });
            }
        } else {
            out.push(PlaneSegment { plane, inliers });
        }

        // Remove the extracted inliers from the pool.
        let inlier_set: Vec<usize> = out.last().expect("just pushed").inliers.clone();
        remaining.retain(|i| !inlier_set.contains(i));
    }
    Ok(out)
}

/// One RANSAC round over the candidate pool; returns the best hypothesis
/// and its inliers.
fn ransac_once(
    cloud: &PointCloud,
    pool: &[usize],
    params: &RansacParams,
) -> Option<(Plane, Vec<usize>)> {
    if pool.len() < 3 {
        return None;
    }
    let mut rng = XorShift64Star::new(params.seed);
    let mut best: Option<(Plane, Vec<usize>)> = None;
    for _ in 0..params.max_iterations {
        let i0 = rng.below(pool.len());
        let i1 = loop {
            let j = rng.below(pool.len());
            if j != i0 {
                break j;
            }
        };
        let i2 = loop {
            let j = rng.below(pool.len());
            if j != i0 && j != i1 {
                break j;
            }
        };
        let p0 = *cloud.get(pool[i0])?;
        let p1 = *cloud.get(pool[i1])?;
        let p2 = *cloud.get(pool[i2])?;

        let n = cross3(
            &[p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
            &[p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
        );
        let Some(n) = normalize3(&n) else { continue }; // collinear sample
        let d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);
        let plane = Plane { normal: n, d };

        let t2 = params.distance_threshold * params.distance_threshold;
        let inliers: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|&i| {
                let p = cloud.get(i).expect("pool indices in range");
                let s = plane.signed_distance(p);
                s * s <= t2
            })
            .collect();
        if best.as_ref().is_none_or(|(_, bi)| inliers.len() > bi.len()) {
            best = Some((plane, inliers));
        }
    }
    best
}

/// Re-export for pipeline ergonomics: deterministic RANSAC sampling.
pub use tpt_percept_core::rng::XorShift64Star;

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: &[f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 || !n.is_finite() {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A z = 1 plane with mild noise plus scattered clutter.
    fn scene_with_plane() -> PointCloud {
        let mut c = PointCloud::new();
        let mut rng = XorShift64Star::new(42);
        for x in -10..=10i32 {
            for y in -10..=10i32 {
                c.push([
                    x as f64 * 0.1,
                    y as f64 * 0.1,
                    1.0 + (rng.next_u64() % 3) as f64 * 0.001,
                ]);
            }
        }
        for i in 0..50 {
            c.push([
                5.0 + i as f64 * 0.01,
                -5.0 + (i % 7) as f64 * 0.3,
                3.0 + i as f64 * 0.02,
            ]);
        }
        c
    }

    #[test]
    fn dominant_plane_found() {
        let cloud = scene_with_plane();
        let params = RansacParams {
            distance_threshold: 0.01,
            min_inliers: 50,
            ..Default::default()
        };
        let planes = segment_planes(&cloud, &params, 2).unwrap();
        assert!(!planes.is_empty());
        let first = &planes[0];
        // z = 1 plane: normal ±z.
        assert!(
            first.plane.normal[2].abs() > 0.999,
            "normal {:?}",
            first.plane.normal
        );
        assert!((first.plane.d.abs() - 1.0) < 0.02, "d = {}", first.plane.d);
        assert!(
            first.inliers.len() >= 400,
            "inliers {}",
            first.inliers.len()
        );
        // The clutter (50 points) must not be inliers of the z=1 plane.
        assert!(first.inliers.len() <= 441);
    }

    #[test]
    fn two_planes_extracted_greedily() {
        let mut c = PointCloud::new();
        for x in -10..=10i32 {
            for y in -10..=10i32 {
                c.push([x as f64 * 0.1, y as f64 * 0.1, 0.0]); // floor
                c.push([x as f64 * 0.1, 0.0, y as f64 * 0.1 + 5.0]); // wall
            }
        }
        let cloud = c;
        let params = RansacParams {
            distance_threshold: 0.01,
            min_inliers: 50,
            ..Default::default()
        };
        let planes = segment_planes(&cloud, &params, 3).unwrap();
        assert!(planes.len() >= 2, "planes {}", planes.len());
        // Both dominant planes recovered with disjoint inlier sets.
        for &i in &planes[0].inliers {
            assert!(
                !planes[1].inliers.contains(&i),
                "inlier sets overlap at {i}"
            );
        }
    }

    #[test]
    fn degenerate_collinear_cloud_yields_no_planes() {
        let c = PointCloud::from_points(vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let params = RansacParams {
            min_inliers: 3,
            ..Default::default()
        };
        // All hypotheses are collinear → no valid plane ever scores; the
        // function must return empty, not panic.
        let planes = segment_planes(&c, &params, 1).unwrap();
        assert!(planes.is_empty());
    }

    #[test]
    fn deterministic_for_same_seed() {
        let cloud = scene_with_plane();
        let params = RansacParams::default();
        let a = segment_planes(&cloud, &params, 1).unwrap();
        let b = segment_planes(&cloud, &params, 1).unwrap();
        assert_eq!(a.len(), b.len());
        for (sa, sb) in a.iter().zip(&b) {
            assert_eq!(sa.inliers, sb.inliers);
        }
    }

    #[test]
    fn invalid_params_rejected() {
        let cloud = scene_with_plane();
        assert!(segment_planes(
            &cloud,
            &RansacParams {
                distance_threshold: 0.0,
                ..Default::default()
            },
            1
        )
        .is_err());
        assert!(segment_planes(
            &cloud,
            &RansacParams {
                max_iterations: 0,
                ..Default::default()
            },
            1
        )
        .is_err());
    }

    #[test]
    fn rng_uniform_enough() {
        let mut rng = XorShift64Star::new(7);
        let mut sum = 0u64;
        let n = 10_000;
        for _ in 0..n {
            sum += rng.below(10) as u64;
        }
        let mean = sum as f64 / n as f64;
        assert!((mean - 4.5).abs() < 0.2, "mean {mean}");
    }
}
