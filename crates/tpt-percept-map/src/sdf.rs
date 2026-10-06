//! Signed distance fields for collision checking and trajectory
//! optimization.
//!
//! A dense axis-aligned SDF grid over a bounded box: cell values hold the
//! (truncated) distance to the nearest obstacle, built from an observed
//! point cloud by nearest-neighbour search. Queries interpolate trilinearly
//! and expose the distance gradient (numerical, one cell per axis) — the
//! input needed by trajectory optimizers for collision costs.
//!
//! This complements [`crate::tsdf`]: the TSDF is the *fused* streaming
//! representation; this grid is a *derived* query structure.

use alloc::vec;
use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::MapError;
use tpt_percept_cloud::kdtree::KdTree;

/// A dense signed distance field over `[min, max]³`.
#[derive(Clone, Debug)]
pub struct SignedDistanceField {
    /// Grid origin corner.
    pub min: [f64; 3],
    /// Grid extent corner.
    pub max: [f64; 3],
    /// Cell count per axis (≥ 2).
    pub dims: [usize; 3],
    values: Vec<f32>,
}

impl SignedDistanceField {
    /// Builds an SDF over `[min, max]` with `dims` cells per axis from the
    /// obstacle point set: each cell centre stores the (capped) distance to
    /// the nearest obstacle point; cells beyond `max_distance` from every
    /// obstacle store `+max_distance` without a search.
    pub fn from_points(
        obstacles: &[[f64; 3]],
        min: [f64; 3],
        max: [f64; 3],
        dims: [usize; 3],
        max_distance: f32,
    ) -> Result<Self, MapError> {
        if obstacles.is_empty() {
            return Err(MapError::InvalidParameter("obstacle set is empty"));
        }
        if dims.iter().any(|&d| d < 2) {
            return Err(MapError::InvalidParameter("dims must be >= 2 per axis"));
        }
        for i in 0..3 {
            if !(min[i].is_finite() && max[i].is_finite() && max[i] > min[i]) {
                return Err(MapError::InvalidParameter("invalid grid extent"));
            }
        }
        if !(max_distance.is_finite() && max_distance > 0.0) {
            return Err(MapError::InvalidParameter("max_distance must be > 0"));
        }
        let tree = KdTree::new(obstacles);
        let mut values = vec![max_distance; dims[0] * dims[1] * dims[2]];
        let cell = [
            (max[0] - min[0]) / (dims[0] - 1) as f64,
            (max[1] - min[1]) / (dims[1] - 1) as f64,
            (max[2] - min[2]) / (dims[2] - 1) as f64,
        ];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = [
                        min[0] + i as f64 * cell[0],
                        min[1] + j as f64 * cell[1],
                        min[2] + k as f64 * cell[2],
                    ];
                    if let Some((_, d2)) = tree.nearest(&p) {
                        let d = (d2.sqrt() as f32).min(max_distance);
                        values[(k * dims[1] + j) * dims[0] + i] = d;
                    }
                }
            }
        }
        Ok(SignedDistanceField {
            min,
            max,
            dims,
            values,
        })
    }

    /// Grid spacing per axis.
    pub fn cell_size(&self) -> [f64; 3] {
        [
            (self.max[0] - self.min[0]) / (self.dims[0] - 1) as f64,
            (self.max[1] - self.min[1]) / (self.dims[1] - 1) as f64,
            (self.max[2] - self.min[2]) / (self.dims[2] - 1) as f64,
        ]
    }

    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    fn cell_value(&self, i: usize, j: usize, k: usize) -> Option<f64> {
        self.values.get(self.index(i, j, k)).map(|&v| v as f64)
    }

    /// Trilinearly interpolated distance at `p`; `None` outside the grid.
    pub fn sample(&self, p: [f64; 3]) -> Option<f64> {
        let cell = self.cell_size();
        let gx = (p[0] - self.min[0]) / cell[0];
        let gy = (p[1] - self.min[1]) / cell[1];
        let gz = (p[2] - self.min[2]) / cell[2];
        if !(0.0..=(self.dims[0] - 1) as f64).contains(&gx) {
            return None;
        }
        if !(0.0..=(self.dims[1] - 1) as f64).contains(&gy) {
            return None;
        }
        if !(0.0..=(self.dims[2] - 1) as f64).contains(&gz) {
            return None;
        }
        let (i, j, k) = (
            gx.floor() as usize,
            gy.floor() as usize,
            gz.floor() as usize,
        );
        let i1 = (i + 1).min(self.dims[0] - 1);
        let j1 = (j + 1).min(self.dims[1] - 1);
        let k1 = (k + 1).min(self.dims[2] - 1);
        let (fx, fy, fz) = (gx - i as f64, gy - j as f64, gz - k as f64);
        let mut acc = 0.0;
        for (di, wx) in [(0usize, 1.0 - fx), (1usize, fx)] {
            for (dj, wy) in [(0usize, 1.0 - fy), (1usize, fy)] {
                for (dk, wz) in [(0usize, 1.0 - fz), (1usize, fz)] {
                    let (ii, jj, kk) = (
                        if di == 0 { i } else { i1 },
                        if dj == 0 { j } else { j1 },
                        if dk == 0 { k } else { k1 },
                    );
                    acc += wx * wy * wz * self.cell_value(ii, jj, kk)?;
                }
            }
        }
        Some(acc)
    }

    /// Central-difference distance gradient at `p` (unit-consistent; `None`
    /// outside the grid or too close to its border).
    pub fn gradient(&self, p: [f64; 3]) -> Option<[f64; 3]> {
        let cell = self.cell_size();
        let mut g = [0.0; 3];
        for axis in 0..3 {
            let mut plus = p;
            let mut minus = p;
            plus[axis] += cell[axis] * 0.5;
            minus[axis] -= cell[axis] * 0.5;
            let vp = self.sample(plus)?;
            let vm = self.sample(minus)?;
            g[axis] = (vp - vm) / cell[axis];
        }
        Some(g)
    }

    /// Collision check: `true` when every sample of the given points is at
    /// least `margin` from the obstacles (all queries inside the grid).
    pub fn is_free(&self, points: &[[f64; 3]], margin: f64) -> Option<bool> {
        for &p in points {
            let d = self.sample(p)?;
            if d < margin {
                return Some(false);
            }
        }
        Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall_points() -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for x in -40..=40i32 {
            for y in -40..=40i32 {
                pts.push([x as f64 * 0.025, y as f64 * 0.025, 1.0]);
            }
        }
        pts
    }

    #[test]
    fn zero_at_wall_positive_away() {
        let sdf = SignedDistanceField::from_points(
            &wall_points(),
            [-0.5, -0.5, 0.4],
            [0.5, 0.5, 1.6],
            [40, 40, 40],
            0.5,
        )
        .unwrap();
        // On the wall: distance ≈ 0.
        let on = sdf.sample([0.0, 0.0, 1.0]).unwrap();
        assert!(on < 0.03, "on-wall distance {on}");
        // 0.2 m before the wall (z smaller): distance ≈ 0.2.
        let before = sdf.sample([0.0, 0.0, 0.8]).unwrap();
        assert!((before - 0.2).abs() < 0.04, "before {before}");
        // Gradient points TOWARDS the wall (+z): distance decreases as z
        // approaches the wall at z = 1.
        let g = sdf.gradient([0.0, 0.0, 0.8]).unwrap();
        assert!(g[2] < -0.5, "gradient {g:?}");
    }

    #[test]
    fn far_cells_capped() {
        let sdf = SignedDistanceField::from_points(
            &wall_points(),
            [-0.5, -0.5, 0.4],
            [0.5, 0.5, 1.6],
            [20, 20, 20],
            0.3,
        )
        .unwrap();
        let far = sdf.sample([0.0, 0.0, 0.45]).unwrap();
        assert!((far - 0.3).abs() < 1e-6, "far {far}");
    }

    #[test]
    fn collision_check_margins() {
        let sdf = SignedDistanceField::from_points(
            &wall_points(),
            [-0.5, -0.5, 0.4],
            [0.5, 0.5, 1.6],
            [40, 40, 40],
            0.5,
        )
        .unwrap();
        let clear = [[0.0, 0.0, 0.6], [0.1, 0.0, 0.6]];
        let too_close = [[0.0, 0.0, 0.99]];
        assert_eq!(sdf.is_free(&clear, 0.1), Some(true));
        assert_eq!(sdf.is_free(&too_close, 0.1), Some(false));
        // Outside the grid → None.
        assert_eq!(sdf.is_free(&[[9.0; 3]], 0.1), None);
    }

    #[test]
    fn invalid_builds_rejected() {
        assert!(SignedDistanceField::from_points(&[], [0.0; 3], [1.0; 3], [8, 8, 8], 1.0).is_err());
        assert!(
            SignedDistanceField::from_points(&[[0.0; 3]], [0.0; 3], [1.0; 3], [8, 8, 1], 1.0)
                .is_err()
        );
        assert!(
            SignedDistanceField::from_points(&[[0.0; 3]], [1.0; 3], [0.0; 3], [8, 8, 8], 1.0)
                .is_err()
        );
    }
}
