//! R-tree over axis-aligned bounding boxes (static STR bulk load).
//!
//! An R-tree indexes *boxes* rather than points: object hypotheses, crop
//! regions, map segments. This implementation is static — boxes are packed
//! once with the Sort-Tile-Recursive (STR) heuristic, which gives excellent
//! query behaviour for immutable geometry (the common perception case:
//! build a map, query it many times).

use alloc::vec::Vec;

/// An axis-aligned bounding box `[min, max]` (metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Lower corner (inclusive).
    pub min: [f64; 3],
    /// Upper corner (inclusive).
    pub max: [f64; 3],
}

impl Aabb {
    /// A degenerate box at a single point.
    pub fn point(p: [f64; 3]) -> Aabb {
        Aabb { min: p, max: p }
    }

    /// The union of two boxes.
    pub fn union(a: Aabb, b: Aabb) -> Aabb {
        Aabb {
            min: [
                a.min[0].min(b.min[0]),
                a.min[1].min(b.min[1]),
                a.min[2].min(b.min[2]),
            ],
            max: [
                a.max[0].max(b.max[0]),
                a.max[1].max(b.max[1]),
                a.max[2].max(b.max[2]),
            ],
        }
    }

    /// True if the boxes overlap (closed boxes: touching counts).
    pub fn intersects(&self, other: &Aabb) -> bool {
        (0..3).all(|i| self.min[i] <= other.max[i] && other.min[i] <= self.max[i])
    }

    /// True if the point lies inside (inclusive).
    pub fn contains(&self, p: &[f64; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }

    /// Box centre.
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
}

/// Maximum number of entries per R-tree node.
const NODE_CAPACITY: usize = 8;

/// A static R-tree over `(Aabb, payload)` entries.
#[derive(Clone, Debug)]
pub struct RTree<T: Clone> {
    root: Option<Node<T>>,
}

#[derive(Clone, Debug)]
enum Node<T: Clone> {
    Leaf(Vec<(Aabb, T)>),
    Branch {
        bbox: Aabb,
        children: Vec<(Aabb, Node<T>)>,
    },
}

impl<T: Clone> RTree<T> {
    /// Bulk-load `entries` with STR packing. Empty input yields an empty
    /// tree.
    pub fn new(entries: Vec<(Aabb, T)>) -> RTree<T> {
        if entries.is_empty() {
            return RTree { root: None };
        }
        let root = bulk_build(entries, 0);
        RTree { root: Some(root) }
    }

    /// All payloads whose boxes intersect `query`.
    pub fn query(&self, query: &Aabb) -> Vec<&T> {
        let mut out = Vec::new();
        if let Some(root) = &self.root {
            query_node(root, query, &mut out);
        }
        out
    }

    /// True if nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }
}

/// Bulk load: sort by the axis for this level, split into a balanced fan-out
/// of groups, recurse per group on the next axis. Produces a height-balanced
/// tree with node fan-out bounded by [`NODE_CAPACITY`].
fn bulk_build<T: Clone>(mut entries: Vec<(Aabb, T)>, depth: u8) -> Node<T> {
    if entries.len() <= NODE_CAPACITY {
        return Node::Leaf(entries);
    }
    let axis = (depth % 3) as usize;
    entries.sort_by(|a, b| {
        a.0.center()[axis]
            .partial_cmp(&b.0.center()[axis])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let fan_out = entries
        .len()
        .div_ceil(NODE_CAPACITY)
        .clamp(2, NODE_CAPACITY);
    let group_len = entries.len().div_ceil(fan_out);
    let mut children = Vec::with_capacity(fan_out);
    for group in entries.chunks(group_len) {
        let bbox = bbox_of_group(group);
        let node = bulk_build(group.to_vec(), depth + 1);
        children.push((bbox, node));
    }
    let bbox = children
        .iter()
        .fold(children[0].0, |acc, &(b, _)| Aabb::union(acc, b));
    Node::Branch { bbox, children }
}

fn bbox_of_group<T: Clone>(group: &[(Aabb, T)]) -> Aabb {
    group
        .iter()
        .fold(group[0].0, |acc, &(b, _)| Aabb::union(acc, b))
}

fn query_node<'a, T: Clone>(node: &'a Node<T>, query: &Aabb, out: &mut Vec<&'a T>) {
    match node {
        Node::Leaf(entries) => {
            for (bbox, payload) in entries {
                if bbox.intersects(query) {
                    out.push(payload);
                }
            }
        }
        Node::Branch { bbox, children } => {
            if !bbox.intersects(query) {
                return;
            }
            for (child_bbox, child) in children {
                if child_bbox.intersects(query) {
                    query_node(child, query, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn grid_boxes(n: usize) -> Vec<(Aabb, usize)> {
        (0..n)
            .map(|i| {
                let c = i as f64;
                let b = Aabb {
                    min: [c, c, c],
                    max: [c + 1.0, c + 1.0, c + 1.0],
                };
                (b, i)
            })
            .collect()
    }

    #[test]
    fn query_finds_overlaps() {
        let entries = grid_boxes(200);
        let tree = RTree::new(entries);
        let q = Aabb {
            min: [10.5, 10.5, 10.5],
            max: [12.5, 12.5, 12.5],
        };
        let mut hits: Vec<usize> = tree.query(&q).into_iter().copied().collect();
        hits.sort_unstable();
        // Boxes 10..12 intersect [10.5, 12.5]; box 13 starts at 13 > 12.5.
        assert_eq!(hits, vec![10, 11, 12]);
    }

    #[test]
    fn point_query_finds_single_box() {
        let entries = grid_boxes(100);
        let tree = RTree::new(entries);
        let q = Aabb::point([55.9, 55.9, 55.9]);
        let hits: Vec<usize> = tree.query(&q).into_iter().copied().collect();
        assert_eq!(hits, vec![55]);
    }

    #[test]
    fn empty_tree() {
        let tree: RTree<usize> = RTree::new(Vec::new());
        assert!(tree.is_empty());
        let hits: Vec<&usize> = tree.query(&Aabb {
            min: [0.0; 3],
            max: [1.0; 3],
        });
        assert!(hits.is_empty());
    }

    #[test]
    fn all_boxes_found_by_huge_query() {
        let n = 300;
        let tree = RTree::new(grid_boxes(n));
        let q = Aabb {
            min: [-1.0; 3],
            max: [n as f64 + 1.0; 3],
        };
        assert_eq!(tree.query(&q).len(), n);
    }
}
