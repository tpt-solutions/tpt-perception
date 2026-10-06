//! Branch-and-bound global registration over translation + yaw.
//!
//! Global registration without an initial guess, restricted to rotation
//! about the z-axis — the standard setup for levelled LiDAR (ground robots,
//! survey scanners), where roll/pitch are fixed by gravity alignment and
//! only yaw, x and y are unknown.
//!
//! The search space `(x, y, yaw)` is recursively subdivided into cells; for
//! each cell a *lower bound* on the trimmed mean squared residual is
//! computed from per-point stability margins (`‖p‖·δθ + ‖δt‖`), so subtrees
//! whose best-possible score cannot beat the incumbent are pruned. The
//! bound is valid: any transform inside the cell scores at least the bound,
//! which makes the returned solution ε-optimal in the trimmed-RMSE sense
//! (up to the cell subdivision floor).

use alloc::vec::Vec;
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::RegistrationError;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::kdtree::KdTree;

/// Search-space and pruning configuration.
#[derive(Clone, Copy, Debug)]
pub struct BnbParams {
    /// Half-width of the x/y search window (metres).
    pub translation_range: f64,
    /// Half-width of the yaw search window (radians).
    pub yaw_range: f64,
    /// Stop subdividing cells below this edge size (metres / radians).
    pub resolution: f64,
    /// Trimmed registration: use the best `trim_ratio` fraction of
    /// correspondences (robust to partial overlap); in (0, 1].
    pub trim_ratio: f64,
    /// Maximum expansion steps (safety bound; typical runs need thousands).
    pub max_expansions: u64,
}

impl Default for BnbParams {
    fn default() -> Self {
        BnbParams {
            translation_range: 4.0,
            yaw_range: core::f64::consts::PI,
            resolution: 0.05,
            trim_ratio: 0.8,
            max_expansions: 200_000,
        }
    }
}

/// A globally-registered pose.
#[derive(Clone, Debug, PartialEq)]
pub struct BnbResult {
    /// Rotation part of `target ≈ Rz(yaw) · source + t`.
    pub rotation: [[f64; 3]; 3],
    /// Translation part.
    pub translation: [f64; 3],
    /// Trimmed mean squared residual at the returned pose (metres²).
    pub score: f64,
    /// Number of cells expanded (diagnostic).
    pub expansions: u64,
}

/// Globally registers `source` onto `target` over `(x, y, yaw)` with
/// branch-and-bound.
pub fn register_bnb(
    source: &PointCloud,
    target: &PointCloud,
    params: &BnbParams,
) -> Result<BnbResult, RegistrationError> {
    if source.is_empty() || target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if source.len() > 4000 {
        // Subsample deterministically for tractability; documented behaviour.
        return register_bnb(&subsample(source, 4000), target, params);
    }
    if !(params.translation_range.is_finite() && params.translation_range > 0.0) {
        return Err(RegistrationError::InvalidParameter(
            "translation_range must be > 0",
        ));
    }
    if !(params.yaw_range.is_finite() && params.yaw_range > 0.0) {
        return Err(RegistrationError::InvalidParameter("yaw_range must be > 0"));
    }
    if !(params.trim_ratio > 0.0 && params.trim_ratio <= 1.0) {
        return Err(RegistrationError::InvalidParameter(
            "trim_ratio must be in (0, 1]",
        ));
    }

    let tree = KdTree::new(target.points());
    let src_scale = {
        let (lo, hi) = source.aabb().expect("non-empty");
        let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6)
    };

    let evaluate = |cell: &Cell| -> (f64, f64) {
        // Returns (lower_bound, score_at_centre).
        let (sin, cos) = cell.cyaw.sin_cos();
        let r00 = cos;
        let r01 = -sin;
        let r10 = sin;
        let r11 = cos;
        let t = [cell.cx, cell.cy, 0.0];

        let mut residuals: Vec<f64> = Vec::with_capacity(source.len());
        let mut bound_sq: Vec<f64> = Vec::with_capacity(source.len());
        let margin_rot = src_scale * cell.hyaw; // rotation effect bound
        let margin_t = (cell.hx * cell.hx + cell.hy * cell.hy).sqrt();
        for &s in source.points() {
            let ts = [
                r00 * s[0] + r01 * s[1] + t[0],
                r10 * s[0] + r11 * s[1] + t[1],
                s[2],
            ];
            match tree.nearest(&ts) {
                Some((_, d2)) => {
                    let margin =
                        (s[0] * s[0] + s[1] * s[1]).sqrt() * cell.hyaw + margin_t.max(margin_rot);
                    residuals.push(d2.sqrt());
                    bound_sq.push((d2.sqrt() - margin).max(0.0));
                }
                None => return (f64::INFINITY, f64::INFINITY),
            }
        }
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let keep = ((residuals.len() as f64 * params.trim_ratio).ceil() as usize).max(1);
        let centre_score = residuals[..keep].iter().map(|d| d * d).sum::<f64>() / keep as f64;
        bound_sq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let lower = bound_sq[..keep].iter().map(|d| d * d).sum::<f64>() / keep as f64;
        (lower, centre_score)
    };

    let root = Cell {
        cx: 0.0,
        cy: 0.0,
        cyaw: 0.0,
        hx: params.translation_range,
        hy: params.translation_range,
        hyaw: params.yaw_range,
    };
    let (root_bound, root_score) = evaluate(&root);

    // Incumbent from the root centre.
    let (mut best_score, mut best_cell) = (root_score, root);
    let mut expansions = 0u64;

    // Priority queue ordered by lower bound (manual binary heap over
    // (bound, cell) with a total-order wrapper).
    let mut heap: Heap = Heap::new();
    heap.push(root_bound, root);

    while let Some((bound, cell)) = heap.pop() {
        if bound > best_score || expansions >= params.max_expansions {
            break;
        }
        expansions += 1;
        // Subdivide the largest half-extent.
        let axis = largest_of(cell.hx, cell.hy, cell.hyaw);
        if (match axis {
            0 => cell.hx,
            1 => cell.hy,
            _ => cell.hyaw,
        }) <= params.resolution
        {
            continue; // at floor; the centre was already evaluated
        }
        let halves: [(f64, f64, f64); 2] = match axis {
            0 => [(cell.hx / 2.0, 0.0, 0.0), (-cell.hx / 2.0, 0.0, 0.0)],
            1 => [(0.0, cell.hy / 2.0, 0.0), (0.0, -cell.hy / 2.0, 0.0)],
            _ => [(0.0, 0.0, cell.hyaw / 2.0), (0.0, 0.0, -cell.hyaw / 2.0)],
        };
        for (dx, dy, dyaw) in halves {
            let child = Cell {
                cx: cell.cx + dx,
                cy: cell.cy + dy,
                cyaw: cell.cyaw + dyaw,
                hx: if axis == 0 { cell.hx / 2.0 } else { cell.hx },
                hy: if axis == 1 { cell.hy / 2.0 } else { cell.hy },
                hyaw: if axis == 2 {
                    cell.hyaw / 2.0
                } else {
                    cell.hyaw
                },
            };
            let (lb, cs) = evaluate(&child);
            if cs < best_score {
                best_score = cs;
                best_cell = child;
            }
            if lb < best_score {
                heap.push(lb, child);
            }
        }
    }

    let (sin, cos) = best_cell.cyaw.sin_cos();
    let rotation = [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]];
    Ok(BnbResult {
        rotation,
        translation: [best_cell.cx, best_cell.cy, 0.0],
        score: best_score,
        expansions,
    })
}

/// Deterministic uniform subsample keeping every `len/n`-th point.
fn subsample(cloud: &PointCloud, n: usize) -> PointCloud {
    let len = cloud.len();
    let stride = len.div_ceil(n);
    let mut out = PointCloud::with_capacity(len / stride + 1);
    for i in (0..len).step_by(stride) {
        out.push(*cloud.get(i).expect("index in range"));
    }
    out
}

fn largest_of(a: f64, b: f64, c: f64) -> usize {
    if a >= b && a >= c {
        0
    } else if b >= c {
        1
    } else {
        2
    }
}

/// Minimal binary min-heap over `(f64, Cell)` pairs.
struct Heap {
    items: Vec<(f64, Cell)>,
}

#[derive(Clone, Copy)]
struct Cell {
    cx: f64,
    cy: f64,
    cyaw: f64,
    hx: f64,
    hy: f64,
    hyaw: f64,
}

impl Heap {
    fn new() -> Self {
        Heap { items: Vec::new() }
    }

    fn push(&mut self, key: f64, cell: Cell) {
        self.items.push((key, cell));
        let mut i = self.items.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.items[parent].0 <= self.items[i].0 {
                break;
            }
            self.items.swap(parent, i);
            i = parent;
        }
    }

    fn pop(&mut self) -> Option<(f64, Cell)> {
        if self.items.is_empty() {
            return None;
        }
        let last = self.items.len() - 1;
        self.items.swap(0, last);
        let out = self.items.pop();
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut smallest = i;
            if l < self.items.len() && self.items[l].0 < self.items[smallest].0 {
                smallest = l;
            }
            if r < self.items.len() && self.items[r].0 < self.items[smallest].0 {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            self.items.swap(i, smallest);
            i = smallest;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ground-planar scene: points on two "walls" and a floor pattern.
    fn scene() -> PointCloud {
        let mut c = PointCloud::new();
        for i in -20..=20i32 {
            let f = i as f64 * 0.25;
            c.push([f, 5.0, 0.0]); // far wall
            c.push([f, -5.0, 0.0]); // near wall
            c.push([5.0, f, 0.0]); // right wall
        }
        c
    }

    fn apply(r: &[[f64; 3]; 3], t: &[f64; 3], p: [f64; 3]) -> [f64; 3] {
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    }

    #[test]
    fn finds_yaw_offset_globally() {
        let target = scene();
        let yaw: f64 = 1.1;
        let (s, c) = yaw.sin_cos();
        let source = target.transformed_rigid(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            [0.6, -0.4, 0.0],
        );
        let params = BnbParams {
            resolution: 0.1,
            ..Default::default()
        };
        let res = register_bnb(&source, &target, &params).unwrap();
        assert!(res.score < 1.0, "score {}", res.score);
        // Yaw recovered modulo π-symmetry of this scene: check the rotation
        // maps source onto target (verify with residuals, not raw values).
        let mut total = 0.0;
        let tree = KdTree::new(target.points());
        for &p in source.points() {
            let t = apply(&res.rotation, &res.translation, p);
            let (_, d2) = tree.nearest(&t).unwrap();
            total += d2;
        }
        let rmse = (total / source.len() as f64).sqrt();
        assert!(rmse < 0.2, "rmse {rmse}");
    }

    #[test]
    fn identity_found_when_aligned() {
        let target = scene();
        let params = BnbParams {
            resolution: 0.1,
            ..Default::default()
        };
        let res = register_bnb(&target, &target, &params).unwrap();
        assert!(res.score < 1e-6, "score {}", res.score);
        assert!(res.translation[0].abs() < 1e-3);
        assert!(res.translation[1].abs() < 1e-3);
    }

    #[test]
    fn invalid_params_rejected() {
        let s = scene();
        assert!(register_bnb(
            &s,
            &s,
            &BnbParams {
                trim_ratio: 0.0,
                ..Default::default()
            }
        )
        .is_err());
        assert!(register_bnb(
            &s,
            &s,
            &BnbParams {
                translation_range: -1.0,
                ..Default::default()
            }
        )
        .is_err());
        assert!(matches!(
            register_bnb(&PointCloud::new(), &s, &BnbParams::default()),
            Err(RegistrationError::EmptyCloud)
        ));
    }
}
