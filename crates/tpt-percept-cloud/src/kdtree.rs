//! k-d tree for exact nearest-neighbour queries.
//!
//! Static (build-once) balanced k-d tree over a point slice. Queries:
//! [`KdTree::nearest`], [`KdTree::knn`] (k smallest by distance) and
//! [`KdTree::radius`] (all points within a ball).
//!
//! Distances are reported and queried as **squared** metres² — callers avoid
//! a square root per candidate.

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

/// Leaf bucket size: leaves hold at most this many point ids.
const LEAF_SIZE: usize = 16;

/// A static balanced k-d tree.
#[derive(Clone, Debug)]
pub struct KdTree {
    points: Vec<[f64; 3]>,
    /// Point indices, arranged so each leaf owns a contiguous range.
    ids: Vec<u32>,
    nodes: Vec<KdNode>,
}

#[derive(Clone, Debug)]
enum KdNode {
    /// Leaf bucket over `ids[start..start+len]`.
    Leaf { start: u32, len: u32 },
    /// Internal split: points with `p[axis] < split` are in `left`.
    Split {
        axis: u8,
        split: f64,
        left: u32,
        right: u32,
    },
}

impl KdTree {
    /// Build a k-d tree over the given points (copies the slice).
    pub fn new(points: &[[f64; 3]]) -> KdTree {
        let points = points.to_vec();
        let mut ids: Vec<u32> = (0..points.len() as u32).collect();
        let mut ordered = Vec::with_capacity(ids.len());
        let mut nodes = Vec::new();
        if !ids.is_empty() {
            build(&points, &mut ids, &mut ordered, &mut nodes);
        }
        KdTree {
            points,
            ids: ordered,
            nodes,
        }
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The single closest point to `q`: `(index, squared_distance)`.
    pub fn nearest(&self, q: &[f64; 3]) -> Option<(usize, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut worst = f64::INFINITY;
        let mut heap: Vec<(usize, f64)> = Vec::new();
        self.knn_node(0, q, 1, &mut heap, &mut worst);
        heap.first().copied()
    }

    /// The `k` closest points to `q`, ascending by squared distance.
    /// Returns fewer than `k` entries when the tree holds fewer points.
    pub fn knn(&self, q: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<(usize, f64)> = Vec::new();
        let mut worst = f64::INFINITY;
        self.knn_node(0, q, k, &mut out, &mut worst);
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        out
    }

    /// All points within squared radius `r2` of `q`, unsorted, as
    /// `(index, squared_distance)` pairs. Errors are impossible; a negative
    /// or non-finite `r2` yields an empty result.
    pub fn radius(&self, q: &[f64; 3], r2: f64) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        if !(r2.is_finite() && r2 >= 0.0) || self.nodes.is_empty() {
            return out;
        }
        self.radius_node(0, q, r2, &mut |entry| out.push(entry));
        out
    }

    /// Count of points within squared radius `r2` of `q`, excluding
    /// `exclude` (used by filters to skip the query point itself).
    pub fn radius_count_excluding(&self, q: &[f64; 3], r2: f64, exclude: usize) -> usize {
        let mut count = 0usize;
        if !(r2.is_finite() && r2 >= 0.0) || self.nodes.is_empty() {
            return 0;
        }
        self.radius_node(0, q, r2, &mut |entry| {
            if entry.0 != exclude {
                count += 1;
            }
        });
        count
    }

    /// k-NN descent. `out` doubles as a max-heap keyed by distance (root =
    /// `out[0]` is the current k-th best while full).
    fn knn_node(
        &self,
        node: u32,
        q: &[f64; 3],
        k: usize,
        out: &mut Vec<(usize, f64)>,
        worst: &mut f64,
    ) {
        match &self.nodes[node as usize] {
            KdNode::Leaf { start, len } => {
                for &id in &self.ids[*start as usize..(*start + *len) as usize] {
                    let p = &self.points[id as usize];
                    let d2 = dist2(p, q);
                    if out.len() < k {
                        out.push((id as usize, d2));
                        if out.len() == k {
                            // Root of the slice-as-heap = current worst.
                            let mut worst_i = 0;
                            for i in 1..out.len() {
                                if out[i].1 > out[worst_i].1 {
                                    worst_i = i;
                                }
                            }
                            out.swap(0, worst_i);
                            *worst = out[0].1;
                        }
                    } else if d2 < *worst {
                        out[0] = (id as usize, d2);
                        let mut worst_i = 0;
                        for i in 1..out.len() {
                            if out[i].1 > out[worst_i].1 {
                                worst_i = i;
                            }
                        }
                        out.swap(0, worst_i);
                        *worst = out[0].1;
                    }
                }
            }
            KdNode::Split {
                axis,
                split,
                left,
                right,
            } => {
                let d = q[*axis as usize] - *split;
                let (near, far) = if d < 0.0 {
                    (*left, *right)
                } else {
                    (*right, *left)
                };
                self.knn_node(near, q, k, out, worst);
                if d * d < *worst || out.len() < k {
                    self.knn_node(far, q, k, out, worst);
                }
            }
        }
    }

    /// Radius descent, invoking `visit` for each point inside the ball.
    fn radius_node(&self, node: u32, q: &[f64; 3], r2: f64, visit: &mut impl FnMut((usize, f64))) {
        match &self.nodes[node as usize] {
            KdNode::Leaf { start, len } => {
                for &id in &self.ids[*start as usize..(*start + *len) as usize] {
                    let p = &self.points[id as usize];
                    let d2 = dist2(p, q);
                    if d2 <= r2 {
                        visit((id as usize, d2));
                    }
                }
            }
            KdNode::Split {
                axis,
                split,
                left,
                right,
            } => {
                let d = q[*axis as usize] - *split;
                // A subtree whose points are all on the far side of the
                // splitting plane is at least |d| away along this axis.
                if d < 0.0 {
                    self.radius_node(*left, q, r2, visit);
                    if d * d <= r2 {
                        self.radius_node(*right, q, r2, visit);
                    }
                } else {
                    self.radius_node(*right, q, r2, visit);
                    if d * d <= r2 {
                        self.radius_node(*left, q, r2, visit);
                    }
                }
            }
        }
    }
}

/// Squared Euclidean distance.
pub(crate) fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Recursively partitions `ids` into the balanced subtree, appending leaf
/// id ranges to `ordered` in leaf order. Returns the node index.
fn build(
    points: &[[f64; 3]],
    ids: &mut [u32],
    ordered: &mut Vec<u32>,
    nodes: &mut Vec<KdNode>,
) -> u32 {
    if ids.len() <= LEAF_SIZE {
        let start = ordered.len() as u32;
        ordered.extend_from_slice(ids);
        nodes.push(KdNode::Leaf {
            start,
            len: ids.len() as u32,
        });
        return (nodes.len() - 1) as u32;
    }

    // Split along the axis with the largest spread, at the median.
    let (min, max) = ids.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(mut min, mut max), &id| {
            let p = &points[id as usize];
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
            (min, max)
        },
    );
    let axis = {
        let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        if extents[0] >= extents[1] && extents[0] >= extents[2] {
            0
        } else if extents[1] >= extents[2] {
            1
        } else {
            2
        }
    };
    ids.sort_by(|&a, &b| {
        points[a as usize][axis]
            .partial_cmp(&points[b as usize][axis])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mid = ids.len() / 2;
    let split = points[ids[mid] as usize][axis];

    // Reserve this node's slot first so the root ends up at index 0.
    nodes.push(KdNode::Leaf { start: 0, len: 0 });
    let my = (nodes.len() - 1) as u32;
    let (lo, hi) = ids.split_at_mut(mid);
    let left = build(points, lo, ordered, nodes);
    let right = build(points, hi, ordered, nodes);
    nodes[my as usize] = KdNode::Split {
        axis: axis as u8,
        split,
        left,
        right,
    };
    my
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Deterministic pseudo-random points in [0, 10)³.
    fn sample_cloud(n: usize) -> Vec<[f64; 3]> {
        let mut state = 0x12345678u64;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (state >> 33) as f64 / (u32::MAX as f64) * 10.0;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = (state >> 33) as f64 / (u32::MAX as f64) * 10.0;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let z = (state >> 33) as f64 / (u32::MAX as f64) * 10.0;
            pts.push([x, y, z]);
        }
        pts
    }

    fn brute_force_knn(points: &[[f64; 3]], q: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dist2(p, q)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }

    #[test]
    fn knn_matches_brute_force() {
        let pts = sample_cloud(500);
        let tree = KdTree::new(&pts);
        for qi in 0..20 {
            let q = pts[qi * 7];
            let got = tree.knn(&q, 9);
            let want = brute_force_knn(&pts, &q, 9);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0, "knn mismatch at q={q:?}");
                assert!((g.1 - w.1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn radius_matches_brute_force() {
        let pts = sample_cloud(400);
        let tree = KdTree::new(&pts);
        let q = [5.0, 5.0, 5.0];
        let got = tree.radius(&q, 4.0);
        let want: Vec<usize> = pts
            .iter()
            .enumerate()
            .filter(|(_, p)| dist2(p, &q) <= 4.0)
            .map(|(i, _)| i)
            .collect();
        let mut got_idx: Vec<usize> = got.iter().map(|e| e.0).collect();
        got_idx.sort_unstable();
        assert_eq!(got_idx, want);
    }

    #[test]
    fn nearest_is_knn_one() {
        let pts = sample_cloud(300);
        let tree = KdTree::new(&pts);
        let q = [3.3, 7.7, 1.1];
        let n = tree.nearest(&q).unwrap();
        let k1 = tree.knn(&q, 1);
        assert_eq!(n.0, k1[0].0);
    }

    #[test]
    fn empty_tree_queries() {
        let tree = KdTree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.nearest(&[0.0; 3]), None);
        assert!(tree.knn(&[0.0; 3], 5).is_empty());
        assert!(tree.radius(&[0.0; 3], 1.0).is_empty());
    }

    #[test]
    fn duplicate_points_all_found() {
        let pts = vec![[1.0; 3]; 40]; // forces multiple leaves
        let tree = KdTree::new(&pts);
        assert_eq!(tree.radius(&[1.0; 3], 0.0).len(), 40);
        let knn = tree.knn(&[1.0; 3], 40);
        assert!(knn.iter().all(|&(_, d)| d == 0.0));
    }

    #[test]
    fn invalid_radius_returns_empty() {
        let pts = sample_cloud(50);
        let tree = KdTree::new(&pts);
        assert!(tree.radius(&[0.0; 3], -1.0).is_empty());
        assert!(tree.radius(&[0.0; 3], f64::NAN).is_empty());
    }
}
