//! Truncated Signed Distance Function (TSDF) voxel maps.
//!
//! Each observed voxel stores a signed distance to the nearest observed
//! surface, truncated to `±truncation`, merged across observations with
//! inverse-sensor-strength weights (`d ← (w₁d₁ + w₂d₂)/(w₁+w₂)`). Surface
//! extraction finds voxels whose neighbourhood contains a sign change and
//! interpolates the zero crossing along the offending axis.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::MapError;

/// TSDF parameters.
#[derive(Clone, Copy, Debug)]
pub struct TsdfParams {
    /// Voxel edge length (metres).
    pub voxel_size: f64,
    /// Truncation distance (metres); measurements further than this from a
    /// voxel do not update it.
    pub truncation: f64,
    /// Maximum weight per voxel (information saturation).
    pub max_weight: f64,
}

impl Default for TsdfParams {
    fn default() -> Self {
        TsdfParams {
            voxel_size: 0.02,
            truncation: 0.06,
            max_weight: 100.0,
        }
    }
}

/// One TSDF voxel: fused signed distance and total weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TsdfVoxel {
    /// Fused signed distance (metres, negative inside the surface).
    pub sdf: f64,
    /// Accumulated weight.
    pub weight: f64,
}

/// A TSDF voxel map (sparse, anchored at the first integration's origin).
#[derive(Clone, Debug, Default)]
pub struct TsdfMap {
    voxels: BTreeMap<[i64; 3], TsdfVoxel>,
    params: TsdfParams,
    origin: [f64; 3],
    anchored: bool,
}

impl TsdfMap {
    /// An empty TSDF with the given parameters.
    pub fn new(params: TsdfParams) -> Self {
        TsdfMap {
            voxels: BTreeMap::new(),
            params,
            origin: [0.0; 3],
            anchored: false,
        }
    }

    /// Grid parameters.
    pub fn params(&self) -> &TsdfParams {
        &self.params
    }

    /// Number of allocated voxels.
    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Integrates one depth-like measurement: a surface point observed from
    /// `origin`. Voxels in front of the surface up to `truncation` get
    /// positive SDF, the surface voxel ~0; voxels behind the surface are
    /// left untouched (unknown).
    pub fn integrate(&mut self, origin: [f64; 3], surface: [f64; 3]) -> Result<(), MapError> {
        for p in [origin, surface] {
            if p.iter().any(|v| !v.is_finite()) {
                return Err(MapError::InvalidParameter("non-finite integration input"));
            }
        }
        if !self.anchored {
            self.origin = surface;
            self.anchored = true;
        }
        let size = self.params.voxel_size;
        let trunc = self.params.truncation;

        let to_point = [
            surface[0] - origin[0],
            surface[1] - origin[1],
            surface[2] - origin[2],
        ];
        let dist =
            (to_point[0] * to_point[0] + to_point[1] * to_point[1] + to_point[2] * to_point[2])
                .sqrt();
        if dist < 1e-9 {
            return Ok(());
        }
        let dir = [to_point[0] / dist, to_point[1] / dist, to_point[2] / dist];

        // Walk from (surface − truncation) to (surface + truncation) along
        // the ray, updating every voxel crossed.
        let start_t = (dist - trunc).max(0.0);
        let end_t = dist + trunc;
        let step = size * 0.5; // sub-voxel sampling along the ray
        let mut t = start_t;
        while t <= end_t {
            let p = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];
            let key = [
                ((p[0] - self.origin[0]) / size).floor() as i64,
                ((p[1] - self.origin[1]) / size).floor() as i64,
                ((p[2] - self.origin[2]) / size).floor() as i64,
            ];
            // Signed distance measured ALONG the ray: the sample at ray
            // parameter `t` lies `dist − t` in front of the surface
            // (positive before it, negative within the band behind it).
            let sdf = (dist - t).clamp(-trunc, trunc);
            // Sensor confidence falls with range.
            let w = (1.0 / dist.max(1e-3)).min(self.params.max_weight);
            let entry = self.voxels.entry(key).or_insert(TsdfVoxel {
                sdf: 0.0,
                weight: 0.0,
            });
            let w_new = (entry.weight + w).min(self.params.max_weight);
            entry.sdf = (entry.sdf * entry.weight + sdf * w) / w_new.max(1e-12);
            entry.weight = w_new;
            t += step;
        }
        Ok(())
    }

    /// Integrates a full scan.
    pub fn integrate_scan(
        &mut self,
        origin: [f64; 3],
        surface_points: &[[f64; 3]],
    ) -> Result<(), MapError> {
        for &p in surface_points {
            self.integrate(origin, p)?;
        }
        Ok(())
    }

    /// Borrows a voxel by key.
    pub fn voxel(&self, key: [i64; 3]) -> Option<&TsdfVoxel> {
        self.voxels.get(&key)
    }

    /// Debug iterator over (key, voxel) pairs.
    #[doc(hidden)]
    pub fn voxels_debug(&self) -> impl Iterator<Item = ([i64; 3], &TsdfVoxel)> {
        self.voxels.iter().map(|(k, v)| (*k, v))
    }

    /// Cell key of a point.
    pub fn key_of(&self, p: [f64; 3]) -> [i64; 3] {
        [
            ((p[0] - self.origin[0]) / self.params.voxel_size).floor() as i64,
            ((p[1] - self.origin[1]) / self.params.voxel_size).floor() as i64,
            ((p[2] - self.origin[2]) / self.params.voxel_size).floor() as i64,
        ]
    }

    /// Trilinearly interpolated SDF at a point; `None` if a touched
    /// neighbour is unobserved.
    pub fn sample(&self, p: [f64; 3]) -> Option<f64> {
        let size = self.params.voxel_size;
        let gx = (p[0] - self.origin[0]) / size - 0.5;
        let gy = (p[1] - self.origin[1]) / size - 0.5;
        let gz = (p[2] - self.origin[2]) / size - 0.5;
        let (i, j, k) = (gx.floor() as i64, gy.floor() as i64, gz.floor() as i64);
        let (fx, fy, fz) = (gx - i as f64, gy - j as f64, gz - k as f64);
        let mut acc = 0.0;
        for (di, wx) in [(0, 1.0 - fx), (1, fx)] {
            for (dj, wy) in [(0, 1.0 - fy), (1, fy)] {
                for (dk, wz) in [(0, 1.0 - fz), (1, fz)] {
                    let voxel = self.voxels.get(&[i + di, j + dj, k + dk])?;
                    acc += wx * wy * wz * voxel.sdf;
                }
            }
        }
        Some(acc)
    }

    /// Extracts the zero-crossing surface as points: for every observed
    /// voxel whose SDF changes sign against any forward neighbour (the 13
    /// neighbours with lexicographically greater keys — axis, face- and
    /// space-diagonals), the zero crossing is interpolated along the line
    /// joining the two voxel centres. Sparse ray-tube observations step
    /// diagonally, so axis-aligned pairs alone would miss crossings.
    pub fn extract_surface(&self) -> Vec<[f64; 3]> {
        let size = self.params.voxel_size;
        let centre = |key: &[i64; 3]| {
            [
                self.origin[0] + (key[0] as f64 + 0.5) * size,
                self.origin[1] + (key[1] as f64 + 0.5) * size,
                self.origin[2] + (key[2] as f64 + 0.5) * size,
            ]
        };
        let mut out = Vec::new();
        for (key, voxel) in &self.voxels {
            for dx in 0..2usize {
                for dy in 0..2usize {
                    for dz in 0..2usize {
                        if dx + dy + dz == 0 {
                            continue;
                        }
                        let nk = [key[0] + dx as i64, key[1] + dy as i64, key[2] + dz as i64];
                        if let Some(other) = self.voxels.get(&nk) {
                            if (voxel.sdf > 0.0) != (other.sdf > 0.0) {
                                let denom = voxel.sdf - other.sdf;
                                let t = if denom.abs() > 1e-12 {
                                    (voxel.sdf / denom).clamp(0.0, 1.0)
                                } else {
                                    0.5
                                };
                                let a = centre(key);
                                let b = centre(&nk);
                                out.push([
                                    a[0] + t * (b[0] - a[0]),
                                    a[1] + t * (b[1] - a[1]),
                                    a[2] + t * (b[2] - a[2]),
                                ]);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A planar surface z = 1 observed from the origin.
    #[test]
    fn plane_zero_crossing_recovers_surface() {
        let params = TsdfParams {
            voxel_size: 0.02,
            truncation: 0.06,
            max_weight: 100.0,
        };
        let mut map = TsdfMap::new(params);
        let mut points = Vec::new();
        for x in -20..=20i32 {
            for y in -20..=20i32 {
                points.push([x as f64 * 0.01, y as f64 * 0.01, 1.0]);
            }
        }
        map.integrate_scan([0.0; 3], &points).unwrap();
        assert!(!map.is_empty());

        let surface = map.extract_surface();
        assert!(!surface.is_empty(), "no surface extracted");
        for p in &surface {
            assert!(
                (p[2] - 1.0).abs() < params.voxel_size,
                "surface point {p:?} off the plane"
            );
        }
    }

    #[test]
    fn sdf_is_positive_before_negative_after() {
        let params = TsdfParams {
            voxel_size: 0.02,
            truncation: 0.06,
            max_weight: 100.0,
        };
        let mut map = TsdfMap::new(params);
        let mut points = Vec::new();
        for x in -30..=30i32 {
            for y in -30..=30i32 {
                points.push([x as f64 * 0.01, y as f64 * 0.01, 1.0]);
            }
        }
        map.integrate_scan([0.0; 3], &points).unwrap();
        // Sample along the z axis through the plane at (0, 0): the sensor
        // side (z < 1) is positive (free space), the far side negative.
        let before = map.sample([0.005, 0.005, 1.0 - 0.03]);
        let after = map.sample([0.005, 0.005, 1.0 + 0.03]);
        assert!(before.unwrap() > 0.0, "before {before:?}");
        assert!(after.unwrap() < 0.0, "after {after:?}");
    }

    #[test]
    fn repeated_integration_converges() {
        let params = TsdfParams {
            voxel_size: 0.02,
            truncation: 0.06,
            max_weight: 100.0,
        };
        let mut map = TsdfMap::new(params);
        let point = [[0.3, 0.0, 1.0]];
        for _ in 0..5 {
            map.integrate_scan([0.0; 3], &point).unwrap();
        }
        let surface = map.extract_surface();
        assert!(!surface.is_empty());
        assert!((surface[0][2] - 1.0).abs() < params.voxel_size);
    }

    #[test]
    fn non_finite_input_rejected() {
        let mut map = TsdfMap::new(TsdfParams::default());
        assert!(map.integrate([0.0; 3], [f64::NAN, 0.0, 1.0]).is_err());
    }

    #[test]
    fn weights_saturate() {
        let params = TsdfParams {
            voxel_size: 0.02,
            truncation: 0.06,
            max_weight: 10.0,
        };
        let mut map = TsdfMap::new(params);
        let point = [[0.3, 0.0, 1.0]];
        for _ in 0..50 {
            map.integrate_scan([0.0; 3], &point).unwrap();
        }
        let max_w = map.voxels.values().map(|v| v.weight).fold(0.0, f64::max);
        assert!(max_w <= 10.0 + 1e-9, "max weight {max_w}");
    }
}
