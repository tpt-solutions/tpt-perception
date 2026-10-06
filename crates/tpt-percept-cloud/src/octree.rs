//! Octree: adaptive hierarchical subdivision of a bounded volume.
//!
//! A region octree over an axis-aligned cube. Internal nodes subdivide into
//! the 8 axis-aligned octants; leaves hold up to `capacity` point indices.
//! Useful for coarse range queries and for structures that are queried by
//! volume (frustums, boxes) rather than by exact nearest neighbours.
#![allow(clippy::needless_range_loop)]

use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

/// A region octree built over a point slice.
#[derive(Clone, Debug)]
pub struct Octree {
    points: Vec<[f64; 3]>,
    root: Option<Node>,
    /// Cube centre and half-extent chosen to enclose all points.
    center: [f64; 3],
    half: f64,
}

#[derive(Clone, Debug)]
enum Node {
    /// Leaf bucket of point indices.
    Leaf(Vec<u32>),
    /// Fully subdivided internal node; children ordered by
    /// `(x ? 4 : 0) | (y ? 2 : 0) | (z ? 1 : 0)` relative to the node centre.
    Internal(Box<[Node; 8]>),
}

impl Octree {
    /// Build an octree enclosing all points.
    ///
    /// `capacity` is the maximum number of points per leaf (>= 1);
    /// subdivision stops at `max_depth` even for overloaded leaves (real
    /// clouds with coincident points are bounded this way).
    pub fn new(points: &[[f64; 3]], capacity: usize, max_depth: u8) -> Octree {
        let points = points.to_vec();
        if points.is_empty() {
            return Octree {
                points,
                root: None,
                center: [0.0; 3],
                half: 0.0,
            };
        }
        let capacity = capacity.max(1);
        let (mut lo, mut hi) = (points[0], points[0]);
        for p in &points {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let center = [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ];
        let half = 0.5
            * (hi[0] - lo[0])
                .max(hi[1] - lo[1])
                .max(hi[2] - lo[2])
                .max(f64::EPSILON * 4.0)
            * 1.000_001; // keep boundary points strictly inside

        let ids: Vec<u32> = (0..points.len() as u32).collect();
        let root = build(&points, ids, center, half, capacity, max_depth);
        Octree {
            points,
            root: Some(root),
            center,
            half,
        }
    }

    /// Number of points held.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// All points within squared radius `r2` of `q` as `(index, dist²)`.
    pub fn radius(&self, q: &[f64; 3], r2: f64) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        if let Some(root) = &self.root {
            if r2.is_finite() && r2 >= 0.0 {
                radius_node(root, &self.points, self.center, self.half, q, r2, &mut out);
            }
        }
        out
    }

    /// All points inside the axis-aligned box `[min, max)` (inclusive min,
    /// exclusive max), as indices.
    pub fn box_query(&self, min: [f64; 3], max: [f64; 3]) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(root) = &self.root {
            box_node(
                root,
                &self.points,
                self.center,
                self.half,
                min,
                max,
                &mut out,
            );
        }
        out
    }
}

fn child_center(center: [f64; 3], half: f64, octant: usize) -> ([f64; 3], f64) {
    let h = half * 0.5;
    let sx = if octant & 4 != 0 { h } else { -h };
    let sy = if octant & 2 != 0 { h } else { -h };
    let sz = if octant & 1 != 0 { h } else { -h };
    ([center[0] + sx, center[1] + sy, center[2] + sz], h)
}

fn octant_of(center: &[f64; 3], p: &[f64; 3]) -> usize {
    (if p[0] >= center[0] { 4 } else { 0 })
        | (if p[1] >= center[1] { 2 } else { 0 })
        | (if p[2] >= center[2] { 1 } else { 0 })
}

fn build(
    points: &[[f64; 3]],
    ids: Vec<u32>,
    center: [f64; 3],
    half: f64,
    capacity: usize,
    depth: u8,
) -> Node {
    if ids.len() <= capacity || depth == 0 || half <= f64::EPSILON {
        return Node::Leaf(ids);
    }
    let mut buckets: Vec<Vec<u32>> = (0..8).map(|_| Vec::new()).collect();
    for id in ids {
        let o = octant_of(&center, &points[id as usize]);
        buckets[o].push(id);
    }
    let mut children: Vec<Node> = Vec::with_capacity(8);
    for (o, bucket) in buckets.into_iter().enumerate() {
        let (cc, ch) = child_center(center, half, o);
        if bucket.is_empty() {
            children.push(Node::Leaf(Vec::new()));
        } else {
            children.push(build(points, bucket, cc, ch, capacity, depth - 1));
        }
    }
    let arr: [Node; 8] = children.try_into().expect("exactly 8 children");
    Node::Internal(Box::new(arr))
}

/// Conservative cube-vs-ball overlap test: is any point of the cube within
/// `r2` of `q`? (Clamp q to the cube and measure the distance.)
fn cube_intersects_ball(center: &[f64; 3], half: f64, q: &[f64; 3], r2: f64) -> bool {
    let mut d2 = 0.0;
    for i in 0..3 {
        let lo = center[i] - half;
        let hi = center[i] + half;
        let clipped = q[i].clamp(lo, hi);
        let d = q[i] - clipped;
        d2 += d * d;
    }
    d2 <= r2
}

fn cube_intersects_box(center: &[f64; 3], half: f64, min: &[f64; 3], max: &[f64; 3]) -> bool {
    for i in 0..3 {
        let lo = center[i] - half;
        let hi = center[i] + half;
        if hi <= min[i] || lo >= max[i] {
            return false;
        }
    }
    true
}

fn radius_node(
    node: &Node,
    points: &[[f64; 3]],
    center: [f64; 3],
    half: f64,
    q: &[f64; 3],
    r2: f64,
    out: &mut Vec<(usize, f64)>,
) {
    if !cube_intersects_ball(&center, half, q, r2) {
        return;
    }
    match node {
        Node::Leaf(ids) => {
            for &id in ids {
                let p = &points[id as usize];
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= r2 {
                    out.push((id as usize, d2));
                }
            }
        }
        Node::Internal(children) => {
            for (o, child) in children.iter().enumerate() {
                let (cc, ch) = child_center(center, half, o);
                radius_node(child, points, cc, ch, q, r2, out);
            }
        }
    }
}

fn box_node(
    node: &Node,
    points: &[[f64; 3]],
    center: [f64; 3],
    half: f64,
    min: [f64; 3],
    max: [f64; 3],
    out: &mut Vec<usize>,
) {
    if !cube_intersects_box(&center, half, &min, &max) {
        return;
    }
    match node {
        Node::Leaf(ids) => {
            for &id in ids {
                let p = &points[id as usize];
                if (min[0]..max[0]).contains(&p[0])
                    && (min[1]..max[1]).contains(&p[1])
                    && (min[2]..max[2]).contains(&p[2])
                {
                    out.push(id as usize);
                }
            }
        }
        Node::Internal(children) => {
            for (o, child) in children.iter().enumerate() {
                let (cc, ch) = child_center(center, half, o);
                box_node(child, points, cc, ch, min, max, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample(n: usize) -> Vec<[f64; 3]> {
        let mut state = 0xdeadbeefu64;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = (state >> 33) as f64 / (u32::MAX as f64) * 10.0;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = (state >> 33) as f64 / (u32::MAX as f64) * 10.0;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let z = (state >> 33) as f64 / (u32::MAX as f64) * 10.0;
                [x, y, z]
            })
            .collect()
    }

    #[test]
    fn radius_matches_brute_force() {
        let pts = sample(600);
        let tree = Octree::new(&pts, 8, 12);
        let q = [5.0, 5.0, 5.0];
        let got = tree.radius(&q, 4.0);
        let mut got_idx: Vec<usize> = got.iter().map(|e| e.0).collect();
        got_idx.sort_unstable();
        let want: Vec<usize> = pts
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                let d = [p[0] - 5.0, p[1] - 5.0, p[2] - 5.0];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= 4.0
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(got_idx, want);
    }

    #[test]
    fn box_query_matches_brute_force() {
        let pts = sample(600);
        let tree = Octree::new(&pts, 4, 12);
        let (min, max) = ([2.0, 3.0, 1.0], [4.0, 5.0, 6.0]);
        let mut got = tree.box_query(min, max);
        got.sort_unstable();
        let mut want: Vec<usize> = pts
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p[0] >= min[0]
                    && p[0] < max[0]
                    && p[1] >= min[1]
                    && p[1] < max[1]
                    && p[2] >= min[2]
                    && p[2] < max[2]
            })
            .map(|(i, _)| i)
            .collect();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn coincident_points_bounded_depth() {
        let pts = vec![[1.0; 3]; 1000];
        // Must terminate despite all points mapping to one octant chain.
        let tree = Octree::new(&pts, 4, 16);
        assert_eq!(tree.radius(&[1.0; 3], 0.0).len(), 1000);
    }

    #[test]
    fn empty_octree() {
        let tree = Octree::new(&[], 8, 8);
        assert!(tree.is_empty());
        assert!(tree.radius(&[0.0; 3], 1.0).is_empty());
    }
}
