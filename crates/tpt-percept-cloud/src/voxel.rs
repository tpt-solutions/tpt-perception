//! Sparse voxel grid hashing.
//!
//! Voxelization maps continuous coordinates onto a uniform 3-D lattice
//! addressed by integer keys `[i64; 3]` — the memory-efficient way to
//! process large-scale clouds (only occupied voxels are stored).
//!
//! Key contract (Kani-proved in `tpt-percept-verify`): [`voxel_index`] never
//! panics and never wraps — non-finite inputs and coordinates beyond the
//! representable lattice return `None` / saturate deterministically.

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::cloud::PointCloud;

/// An occupied voxel: its lattice key and the indices of the points inside.
#[derive(Clone, Debug)]
pub struct Voxel {
    /// Lattice key `[ix, iy, iz]`.
    pub key: [i64; 3],
    /// Indices into the source cloud of the points in this voxel.
    pub members: Vec<usize>,
}

/// Finite-coordinate bound used to saturate lattice keys.
///
/// Coordinates with `|p| > COORD_LIMIT / size` saturate to `±i64::MAX/8`
/// instead of wrapping; the hash map remains a consistent (if meaningless)
/// function of the input for such extreme inputs.
pub const COORD_LIMIT: f64 = 1.0e12;

/// Computes the lattice key of `p` for a grid with edge length `voxel_size`
/// anchored at `origin`.
///
/// Returns `None` when `voxel_size` is not finite/positive or any coordinate
/// is non-finite. Keys for coordinates beyond `±COORD_LIMIT / voxel_size`
/// saturate (documented in [`COORD_LIMIT`]).
pub fn voxel_index(p: [f64; 3], origin: [f64; 3], voxel_size: f64) -> Option<[i64; 3]> {
    if !(voxel_size.is_finite() && voxel_size > 0.0) {
        return None;
    }
    let mut key = [0i64; 3];
    for i in 0..3 {
        if !p[i].is_finite() {
            return None;
        }
        let d = (p[i] - origin[i]) / voxel_size;
        let sat = d.clamp(-COORD_LIMIT, COORD_LIMIT);
        key[i] = sat.floor() as i64;
    }
    Some(key)
}

/// The world-space centre of the voxel with lattice `key`.
pub fn voxel_center(key: [i64; 3], origin: [f64; 3], voxel_size: f64) -> [f64; 3] {
    [
        origin[0] + (key[0] as f64 + 0.5) * voxel_size,
        origin[1] + (key[1] as f64 + 0.5) * voxel_size,
        origin[2] + (key[2] as f64 + 0.5) * voxel_size,
    ]
}

/// Groups points into occupied voxels of edge `voxel_size` anchored at
/// `origin = [0, 0, 0]` (see [`voxelize_at`]).
pub fn voxelize(cloud: &PointCloud, voxel_size: f64) -> Option<alloc::vec::Vec<Voxel>> {
    voxelize_at(cloud, [0.0; 3], voxel_size)
}

/// Groups points into occupied voxels anchored at `origin`.
///
/// The returned voxels are sorted by lattice key, so output order is
/// deterministic for identical input.
pub fn voxelize_at(
    cloud: &PointCloud,
    origin: [f64; 3],
    voxel_size: f64,
) -> Option<alloc::vec::Vec<Voxel>> {
    if cloud.is_empty() {
        return Some(Vec::new());
    }
    let mut map = alloc::collections::BTreeMap::new();
    for (i, &p) in cloud.points().iter().enumerate() {
        let key = voxel_index(p, origin, voxel_size)?;
        map.entry(key).or_insert_with(Vec::new).push(i);
    }
    Some(
        map.into_iter()
            .map(|(key, members)| Voxel { key, members })
            .collect(),
    )
}

/// Voxel-grid downsampling: replaces every occupied voxel by the centroid of
/// its points.
///
/// This is the standard memory-efficient reduction for large LiDAR sweeps —
/// output size is bounded by the number of occupied voxels, not the number
/// of input points.
///
/// Returns `None` for invalid `voxel_size` (non-finite or ≤ 0).
pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f64) -> Option<PointCloud> {
    voxel_downsample_at(cloud, [0.0; 3], voxel_size)
}

/// [`voxel_downsample`] with an explicit grid origin.
pub fn voxel_downsample_at(
    cloud: &PointCloud,
    origin: [f64; 3],
    voxel_size: f64,
) -> Option<PointCloud> {
    let voxels = voxelize_at(cloud, origin, voxel_size)?;
    let mut out = PointCloud::with_capacity(voxels.len());
    for voxel in &voxels {
        let mut c = [0.0; 3];
        for &i in &voxel.members {
            let p = cloud.get(i)?;
            for j in 0..3 {
                c[j] += p[j];
            }
        }
        let n = voxel.members.len() as f64;
        out.push([c[0] / n, c[1] / n, c[2] / n]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn index_basics() {
        assert_eq!(voxel_index([0.1, 0.2, 0.3], [0.0; 3], 1.0), Some([0, 0, 0]));
        assert_eq!(
            voxel_index([1.9, -0.1, 5.0], [0.0; 3], 1.0),
            Some([1, -1, 5])
        );
        assert_eq!(
            voxel_index([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.5),
            Some([-2, -2, -2])
        );
    }

    #[test]
    fn index_rejects_invalid() {
        assert_eq!(voxel_index([0.0; 3], [0.0; 3], 0.0), None);
        assert_eq!(voxel_index([0.0; 3], [0.0; 3], -1.0), None);
        assert_eq!(voxel_index([f64::NAN, 0.0, 0.0], [0.0; 3], 1.0), None);
        assert_eq!(voxel_index([f64::INFINITY, 0.0, 0.0], [0.0; 3], 1.0), None);
    }

    #[test]
    fn extreme_coordinates_saturate_without_panic() {
        let big = 1.0e300;
        let k = voxel_index([big, -big, 0.0], [0.0; 3], 1e-3).unwrap();
        // Saturated rather than wrapped: huge but deterministic.
        assert!(k[0] > 0 && k[1] < 0);
    }

    #[test]
    fn center_roundtrips() {
        let key = [3, -2, 7];
        let c = voxel_center(key, [0.0; 3], 0.5);
        assert_eq!(voxel_index(c, [0.0; 3], 0.5), Some(key));
        assert!(approx(c[0], 1.75) && approx(c[1], -0.75) && approx(c[2], 3.75));
    }

    #[test]
    fn downsample_centroids() {
        let cloud = PointCloud::from_points(vec![
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3], // same voxel as above at size 1
            [5.5, 5.5, 5.5], // own voxel
        ]);
        let ds = voxel_downsample(&cloud, 1.0).unwrap();
        assert_eq!(ds.len(), 2);
        // Deterministic key order: [0,0,0] voxel first.
        assert!(approx(ds.get(0).unwrap()[0], 0.2));
        assert!(approx(ds.get(1).unwrap()[0], 5.5));
    }

    #[test]
    fn voxelize_groups_members() {
        let cloud = PointCloud::from_points(vec![[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]);
        let voxels = voxelize(&cloud, 1.0).unwrap();
        assert_eq!(voxels.len(), 1);
        assert_eq!(voxels[0].members, vec![0, 1]);
    }
}
