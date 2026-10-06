//! Log-odds occupancy grids in 2-D and 3-D.
//!
//! Measurements are inserted along rays from the sensor origin: cells near
//! the hit point get a positive log-odds update, cells traversed along the
//! ray get a negative update, and every cell is clamped to `[lo, hi]` to
//! bound the evidence a single cell can accumulate. The clamp makes the
//! update provably panic-free and bounded (Kani-proved in
//! `tpt-percept-verify`).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

/// Occupancy grid parameters.
#[derive(Clone, Copy, Debug)]
pub struct OccupancyParams {
    /// Voxel edge length (metres).
    pub voxel_size: f64,
    /// Log-odds update on a ray hit.
    pub hit_log_odds: f64,
    /// Log-odds update for traversed (miss) cells.
    pub miss_log_odds: f64,
    /// Clamp bounds on the accumulated log-odds.
    pub clamp_min: f64,
    /// Upper clamp bound on the accumulated log-odds.
    pub clamp_max: f64,
}

impl Default for OccupancyParams {
    fn default() -> Self {
        OccupancyParams {
            voxel_size: 0.1,
            hit_log_odds: 0.85,
            miss_log_odds: -0.4,
            clamp_min: -5.0,
            clamp_max: 5.0,
        }
    }
}

/// Clamps a log-odds value to the parameterised range.
pub fn clamp_log_odds(value: f64, params: &OccupancyParams) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(params.clamp_min, params.clamp_max)
}

/// Log-odds → probability of occupancy.
pub fn log_odds_to_probability(l: f64) -> f64 {
    if !l.is_finite() {
        return 0.5;
    }
    1.0 - 1.0 / (1.0 + l.exp())
}

/// A 2-D log-odds occupancy grid over lattice cells `[i64; 2]`.
#[derive(Clone, Debug, Default)]
pub struct OccupancyGrid2D {
    cells: BTreeMap<[i64; 2], f64>,
    params: OccupancyParams,
    origin: [f64; 2],
}

impl OccupancyGrid2D {
    /// An empty grid anchored at `origin`.
    pub fn new(origin: [f64; 2], params: OccupancyParams) -> Self {
        OccupancyGrid2D {
            cells: BTreeMap::new(),
            params,
            origin,
        }
    }

    /// Grid parameters.
    pub fn params(&self) -> &OccupancyParams {
        &self.params
    }

    /// Number of observed cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// True if no cell was ever observed.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Cell key of a point.
    pub fn cell_of(&self, p: [f64; 2]) -> [i64; 2] {
        [
            ((p[0] - self.origin[0]) / self.params.voxel_size).floor() as i64,
            ((p[1] - self.origin[1]) / self.params.voxel_size).floor() as i64,
        ]
    }

    /// Occupancy probability of a cell (0.5 if never observed).
    pub fn probability(&self, cell: [i64; 2]) -> f64 {
        log_odds_to_probability(self.cells.get(&cell).copied().unwrap_or(0.0))
    }

    /// Occupancy probability at a point.
    pub fn probability_at(&self, p: [f64; 2]) -> f64 {
        self.probability(self.cell_of(p))
    }

    /// Inserts one ray: from `origin` to `hit` (the measured reflection
    /// point). Cells along the ray are marked free, the hit cell occupied.
    pub fn insert_ray(&mut self, origin: [f64; 2], hit: [f64; 2]) {
        for (cell, hit_cell) in self.ray_cells(origin, hit) {
            let update = if hit_cell {
                self.params.hit_log_odds
            } else {
                self.params.miss_log_odds
            };
            let entry = self.cells.entry(cell).or_insert(0.0);
            *entry = clamp_log_odds(*entry + update, &self.params);
        }
    }

    /// Inserts a full scan.
    pub fn insert_scan(&mut self, origin: [f64; 2], hits: &[[f64; 2]]) {
        for &hit in hits {
            self.insert_ray(origin, hit);
        }
    }

    /// Enumerates the cells along the ray (Bresenham in lattice space) as
    /// `(cell, is_hit)` pairs. The final cell has `is_hit = true`.
    pub fn ray_cells(&self, origin: [f64; 2], hit: [f64; 2]) -> Vec<([i64; 2], bool)> {
        let mut out = Vec::new();
        let c0 = self.cell_of(origin);
        let c1 = self.cell_of(hit);
        let (mut x, mut y) = (c0[0], c0[1]);
        let dx = (c1[0] - x).abs();
        let dy = (c1[1] - y).abs();
        let sx = if c1[0] > x { 1 } else { -1 };
        let sy = if c1[1] > y { 1 } else { -1 };
        let mut err = dx - dy;
        // Bounded iteration: the lattice distance is finite by construction;
        // cap at 1e6 cells to guard pathological inputs.
        for _ in 0..1_000_000 {
            let is_hit = x == c1[0] && y == c1[1];
            out.push(([x, y], is_hit));
            if is_hit {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
        out
    }
}

/// A 3-D log-odds occupancy grid over lattice cells `[i64; 3]`.
#[derive(Clone, Debug, Default)]
pub struct OccupancyGrid3D {
    cells: BTreeMap<[i64; 3], f64>,
    params: OccupancyParams,
    origin: [f64; 3],
}

impl OccupancyGrid3D {
    /// An empty grid anchored at `origin`.
    pub fn new(origin: [f64; 3], params: OccupancyParams) -> Self {
        OccupancyGrid3D {
            cells: BTreeMap::new(),
            params,
            origin,
        }
    }

    /// Number of observed cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Cell key of a point.
    pub fn cell_of(&self, p: [f64; 3]) -> [i64; 3] {
        [
            ((p[0] - self.origin[0]) / self.params.voxel_size).floor() as i64,
            ((p[1] - self.origin[1]) / self.params.voxel_size).floor() as i64,
            ((p[2] - self.origin[2]) / self.params.voxel_size).floor() as i64,
        ]
    }

    /// Occupancy probability at a point (0.5 if never observed).
    pub fn probability_at(&self, p: [f64; 3]) -> f64 {
        log_odds_to_probability(self.cells.get(&self.cell_of(p)).copied().unwrap_or(0.0))
    }

    /// Inserts a ray with the Amanatides–Woo voxel traversal.
    pub fn insert_ray(&mut self, origin: [f64; 3], hit: [f64; 3]) {
        for (cell, hit_cell) in self.ray_cells(origin, hit) {
            let update = if hit_cell {
                self.params.hit_log_odds
            } else {
                self.params.miss_log_odds
            };
            let entry = self.cells.entry(cell).or_insert(0.0);
            *entry = clamp_log_odds(*entry + update, &self.params);
        }
    }

    /// Inserts a full scan.
    pub fn insert_scan(&mut self, origin: [f64; 3], hits: &[[f64; 3]]) {
        for &hit in hits {
            self.insert_ray(origin, hit);
        }
    }

    /// Amanatides–Woo voxel traversal: `(cell, is_hit)` along the ray.
    pub fn ray_cells(&self, origin: [f64; 3], hit: [f64; 3]) -> Vec<([i64; 3], bool)> {
        let size = self.params.voxel_size;
        let cell = |p: f64, o: f64| ((p - o) / size).floor() as i64;
        let mut cell_k = [
            cell(origin[0], self.origin[0]),
            cell(origin[1], self.origin[1]),
            cell(origin[2], self.origin[2]),
        ];
        let target = [
            cell(hit[0], self.origin[0]),
            cell(hit[1], self.origin[1]),
            cell(hit[2], self.origin[2]),
        ];

        let dir = [hit[0] - origin[0], hit[1] - origin[1], hit[2] - origin[2]];
        // NB: `f64::signum(0.0)` is 1.0, not 0 — zero-direction axes must
        // get step 0 and t_max = +∞ explicitly.
        let sgn = |x: f64| {
            if x > 0.0 {
                1i64
            } else if x < 0.0 {
                -1i64
            } else {
                0i64
            }
        };
        let step = [sgn(dir[0]), sgn(dir[1]), sgn(dir[2])];
        // t_max: distance to the first lattice boundary per axis; t_delta:
        // distance between boundaries. Axes parallel to the ray stay at ∞.
        let mut t_max = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for i in 0..3 {
            if dir[i].abs() > 1e-12 {
                let boundary = (cell_k[i] as f64 + (if step[i] > 0 { 1.0 } else { 0.0 })) * size
                    + self.origin[i];
                t_max[i] = (boundary - origin[i]) / dir[i];
                t_delta[i] = (size / dir[i].abs()).max(1e-12);
            }
        }

        let mut out = Vec::new();
        for _ in 0..1_000_000 {
            let is_hit = cell_k == target;
            out.push((cell_k, is_hit));
            if is_hit {
                break;
            }
            // Advance along the smallest t_max.
            let axis = if t_max[0] < t_max[1] {
                if t_max[0] < t_max[2] {
                    0
                } else {
                    2
                }
            } else if t_max[1] < t_max[2] {
                1
            } else {
                2
            };
            if !t_max[axis].is_finite() {
                break; // ray parallel to all remaining axes
            }
            cell_k[axis] += step[axis];
            t_max[axis] += t_delta[axis];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_cell_occupied_miss_cells_free() {
        let mut grid = OccupancyGrid2D::new([0.0, 0.0], OccupancyParams::default());
        grid.insert_ray([0.0, 0.0], [0.95, 0.0]);
        // Cell just before the hit must be free; the hit cell occupied.
        let free = grid.probability_at([0.85, 0.05]);
        let occupied = grid.probability_at([0.95, 0.05]);
        assert!(free < 0.5, "free {free}");
        assert!(occupied > 0.5, "occupied {occupied}");
    }

    #[test]
    fn repeated_observations_strengthen_and_clamp() {
        let mut grid = OccupancyGrid2D::new([0.0, 0.0], OccupancyParams::default());
        for _ in 0..100 {
            grid.insert_ray([0.0, 0.0], [0.95, 0.0]);
        }
        // Clamped: probability saturates below 1.
        let p = grid.probability_at([0.95, 0.05]);
        assert!(p > 0.99, "p {p}");
        assert!(p < 1.0);
        assert_eq!(
            grid.cells.values().copied().fold(f64::MIN, f64::max),
            clamp_log_odds(1e9, &OccupancyParams::default())
        );
    }

    #[test]
    fn conflicting_evidence_converges_to_half() {
        let params = OccupancyParams {
            hit_log_odds: 0.4,
            miss_log_odds: -0.4,
            ..Default::default()
        };
        let mut grid = OccupancyGrid2D::new([0.0, 0.0], params);
        for i in 0..10 {
            // Alternating hits/misses on the same cell.
            if i % 2 == 0 {
                grid.insert_ray([0.0, 0.0], [1.0, 0.0]);
            } else {
                grid.insert_ray([0.0, 0.0], [1.5, 0.0]); // hit further away
            }
        }
        let mid = grid.probability_at([1.0, 0.05]);
        assert!((mid - 0.5).abs() < 1e-9, "mid {mid}");
    }

    #[test]
    fn grid3d_ray_marks_expected_cells() {
        let mut grid = OccupancyGrid3D::new([0.0; 3], OccupancyParams::default());
        grid.insert_ray([0.0; 3], [0.95, 0.0, 0.0]);
        assert!(grid.probability_at([0.95, 0.05, 0.05]) > 0.5);
        assert!(grid.probability_at([0.45, 0.05, 0.05]) < 0.5);
        assert_eq!(grid.len(), 10);
    }

    #[test]
    fn grid3d_diagonal_ray() {
        let mut grid = OccupancyGrid3D::new([0.0; 3], OccupancyParams::default());
        grid.insert_ray([0.0; 3], [0.95, 0.95, 0.0]);
        // Hit cell occupied, origin cell free.
        assert!(grid.probability_at([0.95, 0.95, 0.05]) > 0.5);
        assert!(grid.probability_at([0.05, 0.05, 0.05]) < 0.5);
    }

    #[test]
    fn log_odds_roundtrip() {
        for &l in &[-2.0, 0.0, 3.5] {
            let p = log_odds_to_probability(l);
            let back = (p / (1.0 - p)).ln();
            assert!((back - l).abs() < 1e-9);
        }
    }
}
