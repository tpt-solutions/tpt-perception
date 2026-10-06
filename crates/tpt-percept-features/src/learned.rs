//! AI-native hooks: learned descriptors and differentiable cost functions.
//!
//! The geometric descriptors in this crate ([`crate::fpfh`], [`crate::shot`])
//! are hand-crafted. Learning-based perception agents (`tpt-eve`,
//! `tpt-anima`) plug in through the traits here:
//!
//! * [`LearnedDescriptor`] — embed a local geometry context into a learned
//!   feature vector (the embedding itself is owned by the agent crate);
//! * [`ScalarCost`] — an objective over a parameter vector, with an optional
//!   analytical gradient hook so optimisers can fall back from finite
//!   differences;
//! * [`DescriptorContext`] — the neighbourhood summary handed to embedded
//!   models, so agents never need raw-cloud access.

use alloc::vec::Vec;

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;

/// Neighbourhood summary for one point, sufficient for learned embeddings.
#[derive(Clone, Debug)]
pub struct DescriptorContext {
    /// Index of the query point in the source cloud.
    pub index: usize,
    /// Query point position (metres).
    pub center: [f64; 3],
    /// Neighbour positions relative to `center`, nearest first.
    pub relative_points: Vec<[f64; 3]>,
    /// Neighbour squared distances, aligned with `relative_points`.
    pub squared_distances: Vec<f64>,
}

/// Collects [`DescriptorContext`]s for every point of a cloud.
pub fn contexts(
    cloud: &PointCloud,
    k: usize,
) -> Result<Vec<DescriptorContext>, crate::FeatureError> {
    if k == 0 || k >= cloud.len() {
        return Err(crate::FeatureError::InvalidParameter(
            "k must be >= 1 and < point count",
        ));
    }
    let tree = KdTree::new(cloud.points());
    let mut out = Vec::with_capacity(cloud.len());
    for (i, &p) in cloud.points().iter().enumerate() {
        let neighbors = tree.knn(&p, k + 1);
        let mut ctx = DescriptorContext {
            index: i,
            center: p,
            relative_points: Vec::with_capacity(k),
            squared_distances: Vec::with_capacity(k),
        };
        for &(idx, d2) in &neighbors {
            if idx == i {
                continue;
            }
            let q = cloud.get(idx).expect("knn indices in range");
            ctx.relative_points
                .push([q[0] - p[0], q[1] - p[1], q[2] - p[2]]);
            ctx.squared_distances.push(d2);
            if ctx.relative_points.len() == k {
                break;
            }
        }
        out.push(ctx);
    }
    Ok(out)
}

/// A learned descriptor: maps local geometry to an embedding owned by the
/// agent crate (weights live behind the implementation).
///
/// Implementations must be deterministic for a fixed context.
pub trait LearnedDescriptor {
    /// Embedding dimensionality (may depend on the trained model).
    fn dim(&self) -> usize;

    /// Embed one context.
    fn embed(&self, ctx: &DescriptorContext) -> Vec<f64>;
}

/// A scalar objective over a parameter vector, with an optional analytic
/// gradient hook (autodiff-ready: an agent crate can return gradients from
/// its reversed-mode tape here).
pub trait ScalarCost {
    /// Evaluate the cost at `params`.
    fn evaluate(&self, params: &[f64]) -> f64;

    /// Analytic gradient, if the implementation provides one. The default
    /// returns `None`, which lets optimisers fall back to finite
    /// differences.
    fn gradient(&self, _params: &[f64]) -> Option<Vec<f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeatureError;
    use alloc::vec;

    #[test]
    fn contexts_collect_neighborhoods() {
        let mut c = PointCloud::new();
        for i in 0..20 {
            c.push([i as f64 * 0.1, 0.0, 0.0]);
        }
        let ctxs = contexts(&c, 4).unwrap();
        assert_eq!(ctxs.len(), 20);
        let mid = &ctxs[10];
        assert_eq!(mid.relative_points.len(), 4);
        assert_eq!(mid.squared_distances.len(), 4);
        // Nearest listed first.
        assert!(mid.squared_distances[0] <= mid.squared_distances[3]);
        // Relative coordinates.
        assert!((mid.relative_points[0][0].abs() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn context_parameter_validation() {
        let c = PointCloud::from_points(vec![[0.0; 3]; 4]);
        assert!(matches!(
            contexts(&c, 0),
            Err(FeatureError::InvalidParameter(_))
        ));
        assert!(matches!(
            contexts(&c, 4),
            Err(FeatureError::InvalidParameter(_))
        ));
    }

    /// Sum-of-squares cost with analytic gradient — the shape agents will
    /// provide.
    struct SqCost;
    impl ScalarCost for SqCost {
        fn evaluate(&self, params: &[f64]) -> f64 {
            params.iter().map(|p| p * p).sum()
        }
        fn gradient(&self, params: &[f64]) -> Option<Vec<f64>> {
            Some(params.iter().map(|p| 2.0 * p).collect())
        }
    }

    #[test]
    fn cost_hook_contract() {
        let cost = SqCost;
        let params = [1.0, -2.0, 3.0];
        assert!((cost.evaluate(&params) - 14.0).abs() < 1e-12);
        let g = cost.gradient(&params).unwrap();
        assert!((g[1] - (-4.0)).abs() < 1e-12);
        // Default implementations report no gradient.
        struct NoGrad;
        impl ScalarCost for NoGrad {
            fn evaluate(&self, _: &[f64]) -> f64 {
                0.0
            }
        }
        assert!(NoGrad.gradient(&params).is_none());
    }
}
