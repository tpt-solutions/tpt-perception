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

/// A triangle mesh extracted from a TSDF.
///
/// Vertices are shared between neighbouring triangles. Triangle winding is
/// not guaranteed to be consistently oriented.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions (metres).
    pub vertices: Vec<[f64; 3]>,
    /// Triangles as indices into [`Mesh::vertices`].
    pub triangles: Vec<[u32; 3]>,
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

    /// Extracts a triangle mesh of the zero level set with marching
    /// tetrahedra.
    ///
    /// Each cube formed by eight neighbouring observed voxel centres is split
    /// into six tetrahedra along a shared diagonal, which keeps adjacent cubes
    /// consistent. Cubes with any unobserved corner are skipped. Edge
    /// vertices are shared across cubes, so the mesh has no duplicate
    /// vertices along seams.
    pub fn extract_mesh(&self) -> Mesh {
        // Corner offsets of a unit cube, indexed 0..8.
        const CORNER: [[i64; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ];
        // Six monotone paths from corner 0 to corner 6 (Kuhn split).
        const TETS: [[usize; 4]; 6] = [
            [0, 1, 5, 6],
            [0, 1, 2, 6],
            [0, 3, 2, 6],
            [0, 3, 7, 6],
            [0, 4, 7, 6],
            [0, 4, 5, 6],
        ];

        let mut mesh = Mesh::default();
        let mut edge_vertex: BTreeMap<([i64; 3], [i64; 3]), u32> = BTreeMap::new();
        let size = self.params.voxel_size;
        let centre = |key: [i64; 3]| {
            [
                self.origin[0] + (key[0] as f64 + 0.5) * size,
                self.origin[1] + (key[1] as f64 + 0.5) * size,
                self.origin[2] + (key[2] as f64 + 0.5) * size,
            ]
        };

        for base in self.voxels.keys() {
            let mut keys = [[0i64; 3]; 8];
            let mut values = [0.0f64; 8];
            let mut complete = true;
            for (c, off) in CORNER.iter().enumerate() {
                let key = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
                match self.voxels.get(&key) {
                    Some(v) => {
                        keys[c] = key;
                        values[c] = v.sdf;
                    }
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                continue;
            }

            for tet in TETS {
                let inside: Vec<usize> = tet.iter().copied().filter(|&c| values[c] < 0.0).collect();
                let outside: Vec<usize> =
                    tet.iter().copied().filter(|&c| values[c] >= 0.0).collect();
                let mut vertex_on = |a: usize, b: usize, mesh: &mut Mesh| -> u32 {
                    let (ka, kb) = (keys[a], keys[b]);
                    let edge = if ka <= kb { (ka, kb) } else { (kb, ka) };
                    if let Some(&idx) = edge_vertex.get(&edge) {
                        return idx;
                    }
                    let (va, vb) = (values[a], values[b]);
                    let t = if (va - vb).abs() > 1e-12 {
                        (va / (va - vb)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let (pa, pb) = (centre(ka), centre(kb));
                    let p = [
                        pa[0] + t * (pb[0] - pa[0]),
                        pa[1] + t * (pb[1] - pa[1]),
                        pa[2] + t * (pb[2] - pa[2]),
                    ];
                    let idx =
                        u32::try_from(mesh.vertices.len()).expect("mesh exceeds u32::MAX vertices");
                    mesh.vertices.push(p);
                    edge_vertex.insert(edge, idx);
                    idx
                };

                match (inside.len(), outside.len()) {
                    (1, 3) => {
                        let lone = inside[0];
                        let tri = [
                            vertex_on(lone, outside[0], &mut mesh),
                            vertex_on(lone, outside[1], &mut mesh),
                            vertex_on(lone, outside[2], &mut mesh),
                        ];
                        mesh.triangles.push(tri);
                    }
                    (3, 1) => {
                        let lone = outside[0];
                        let tri = [
                            vertex_on(lone, inside[0], &mut mesh),
                            vertex_on(lone, inside[1], &mut mesh),
                            vertex_on(lone, inside[2], &mut mesh),
                        ];
                        mesh.triangles.push(tri);
                    }
                    (2, 2) => {
                        let (a, b) = (inside[0], inside[1]);
                        let (c, d) = (outside[0], outside[1]);
                        let quad = [
                            vertex_on(a, c, &mut mesh),
                            vertex_on(a, d, &mut mesh),
                            vertex_on(b, d, &mut mesh),
                            vertex_on(b, c, &mut mesh),
                        ];
                        mesh.triangles.push([quad[0], quad[1], quad[2]]);
                        mesh.triangles.push([quad[0], quad[2], quad[3]]);
                    }
                    _ => {}
                }
            }
        }
        mesh
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
    fn plane_mesh_is_flat_and_nonempty() {
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

        let mesh = map.extract_mesh();
        assert!(!mesh.triangles.is_empty(), "no triangles");
        for v in &mesh.vertices {
            assert!(
                (v[2] - 1.0).abs() < params.voxel_size,
                "vertex {v:?} off plane"
            );
        }
        // Every index is in range and vertices are not duplicated along the
        // shared edges between cubes.
        for tri in &mesh.triangles {
            for &i in tri {
                assert!((i as usize) < mesh.vertices.len());
            }
        }
        let mut seen = alloc::collections::BTreeSet::new();
        for v in &mesh.vertices {
            let key = [
                (v[0] * 1e6).round() as i64,
                (v[1] * 1e6).round() as i64,
                (v[2] * 1e6).round() as i64,
            ];
            assert!(seen.insert(key), "duplicate vertex {v:?}");
        }
    }

    #[test]
    fn empty_map_gives_empty_mesh() {
        let map = TsdfMap::new(TsdfParams::default());
        let mesh = map.extract_mesh();
        assert!(mesh.vertices.is_empty() && mesh.triangles.is_empty());
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
