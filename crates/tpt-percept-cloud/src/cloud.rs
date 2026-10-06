//! The point cloud container.
//!
//! [`PointCloud`] is the frame-agnostic workhorse: a dense collection of 3-D
//! positions (metres) that every algorithm in `tpt-percept-cloud` operates
//! on. Frame safety is provided at the edges through [`FrameCloud<F>`]: a
//! `PointCloud` tagged with a `Frame` marker type,
//! so a cloud captured in `Lidar<0>` cannot be silently registered against
//! a `World`-frame map without an explicit [`Isometry3`] transform.
//!
//! [`Isometry3`]: tpt_percept_core::iso::Isometry3

use alloc::vec::Vec;

use tpt_percept_core::frame::Frame;
use tpt_percept_core::iso::Isometry3;
use tpt_percept_core::point::Point3D;

/// A dense set of 3-D points (metres).
///
/// Position `i` is `self[i]`; auxiliary per-point data (intensity, ring,
/// timestamp) is kept by the caller indexed in parallel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud {
    points: Vec<[f64; 3]>,
}

impl PointCloud {
    /// An empty cloud.
    pub fn new() -> Self {
        PointCloud { points: Vec::new() }
    }

    /// An empty cloud with pre-allocated capacity for `n` points.
    pub fn with_capacity(n: usize) -> Self {
        PointCloud {
            points: Vec::with_capacity(n),
        }
    }

    /// Build a cloud from raw positions.
    pub fn from_points(points: Vec<[f64; 3]>) -> Self {
        PointCloud { points }
    }

    /// Append a point.
    pub fn push(&mut self, p: [f64; 3]) {
        self.points.push(p);
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if the cloud holds no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Borrow point `i`, or `None` when out of bounds.
    pub fn get(&self, i: usize) -> Option<&[f64; 3]> {
        self.points.get(i)
    }

    /// All positions.
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Mutable access to all positions.
    pub fn points_mut(&mut self) -> &mut [[f64; 3]] {
        &mut self.points
    }

    /// Consume the cloud, returning the raw positions.
    pub fn into_points(self) -> Vec<[f64; 3]> {
        self.points
    }

    /// Keep only the points whose indices are listed (duplicates keep the
    /// first occurrence; out-of-range indices are ignored).
    pub fn select(&self, indices: &[usize]) -> PointCloud {
        let mut out = PointCloud::with_capacity(indices.len());
        for &i in indices {
            if let Some(&p) = self.points.get(i) {
                out.push(p);
            }
        }
        out
    }

    /// Append every point of `other`.
    pub fn extend_from(&mut self, other: &PointCloud) {
        self.points.extend_from_slice(other.points());
    }

    /// Axis-aligned bounding box as `(min, max)`, or `None` if empty.
    pub fn aabb(&self) -> Option<([f64; 3], [f64; 3])> {
        let first = *self.points.first()?;
        let mut min = first;
        let mut max = first;
        for p in &self.points {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }
        Some((min, max))
    }

    /// Centroid of all points, or `None` if empty.
    pub fn centroid(&self) -> Option<[f64; 3]> {
        if self.points.is_empty() {
            return None;
        }
        let mut c = [0.0; 3];
        for p in &self.points {
            for i in 0..3 {
                c[i] += p[i];
            }
        }
        let n = self.points.len() as f64;
        Some([c[0] / n, c[1] / n, c[2] / n])
    }

    /// Apply a raw rigid transform (validated rotation matrix + translation)
    /// to every point in place.
    ///
    /// Frame-typed code should prefer [`FrameCloud::transform_into`].
    pub fn transform_rigid_in_place(&mut self, rotation: [[f64; 3]; 3], translation: [f64; 3]) {
        for p in self.points_mut() {
            *p = [
                rotation[0][0] * p[0]
                    + rotation[0][1] * p[1]
                    + rotation[0][2] * p[2]
                    + translation[0],
                rotation[1][0] * p[0]
                    + rotation[1][1] * p[1]
                    + rotation[1][2] * p[2]
                    + translation[1],
                rotation[2][0] * p[0]
                    + rotation[2][1] * p[1]
                    + rotation[2][2] * p[2]
                    + translation[2],
            ];
        }
    }

    /// Return a rigidly transformed copy.
    pub fn transformed_rigid(&self, rotation: [[f64; 3]; 3], translation: [f64; 3]) -> PointCloud {
        let mut out = self.clone();
        out.transform_rigid_in_place(rotation, translation);
        out
    }
}

/// Applies a frame-typed isometry to raw coordinates (internal helper).
pub(crate) fn apply_iso<F: Frame, T: Frame>(iso: &Isometry3<F, T>, p: [f64; 3]) -> [f64; 3] {
    iso.transform_point(Point3D::from_array(p)).coords()
}

/// A [`PointCloud`] tagged with a coordinate frame.
///
/// All algorithms of this crate run on the underlying `PointCloud`; the
/// wrapper exists to carry the frame through pipeline stages so that
/// downstream consumers cannot mix clouds from different frames.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameCloud<F: Frame> {
    cloud: PointCloud,
    _frame: core::marker::PhantomData<fn() -> F>,
}

impl<F: Frame> FrameCloud<F> {
    /// Tag a raw cloud as living in frame `F`.
    pub fn from_cloud(cloud: PointCloud) -> Self {
        FrameCloud {
            cloud,
            _frame: core::marker::PhantomData,
        }
    }

    /// Build from frame-typed points.
    pub fn from_points(points: impl IntoIterator<Item = Point3D<F>>) -> Self {
        let mut cloud = PointCloud::new();
        for p in points {
            cloud.push(p.coords());
        }
        FrameCloud {
            cloud,
            _frame: core::marker::PhantomData,
        }
    }

    /// Borrow the raw cloud.
    pub fn cloud(&self) -> &PointCloud {
        &self.cloud
    }

    /// Mutable borrow of the raw cloud.
    pub fn cloud_mut(&mut self) -> &mut PointCloud {
        &mut self.cloud
    }

    /// Unwrap into the raw cloud.
    pub fn into_cloud(self) -> PointCloud {
        self.cloud
    }

    /// Iterate frame-typed points.
    pub fn iter_points(&self) -> impl Iterator<Item = Point3D<F>> + '_ {
        self.cloud
            .points()
            .iter()
            .map(|&p| Point3D::<F>::from_array(p))
    }

    /// Transform every point into frame `To`, retagging the cloud.
    pub fn transform_into<To: Frame>(&self, iso: &Isometry3<F, To>) -> FrameCloud<To> {
        let mut out = PointCloud::with_capacity(self.cloud.len());
        for p in self.cloud.points() {
            out.push(apply_iso(iso, *p));
        }
        FrameCloud {
            cloud: out,
            _frame: core::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use tpt_percept_core::frame::{Lidar, World};

    #[test]
    fn basic_container_ops() {
        let mut c = PointCloud::new();
        assert!(c.is_empty());
        c.push([1.0, 2.0, 3.0]);
        c.push([-1.0, 0.5, 2.0]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(1), Some(&[-1.0, 0.5, 2.0]));
        let (min, max) = c.aabb().unwrap();
        assert_eq!(min, [-1.0, 0.5, 2.0]);
        assert_eq!(max, [1.0, 2.0, 3.0]);
        let ctr = c.centroid().unwrap();
        assert_eq!(ctr, [0.0, 1.25, 2.5]);
    }

    #[test]
    fn select_ignores_out_of_range() {
        let c = PointCloud::from_points(vec![[0.0; 3], [1.0; 3], [2.0; 3]]);
        let s = c.select(&[2, 0, 7]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), Some(&[2.0; 3]));
        assert_eq!(s.get(1), Some(&[0.0; 3]));
    }

    #[test]
    fn frame_cloud_transform_retags() {
        let fc = FrameCloud::<Lidar<0>>::from_points(vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ]);
        let iso: Isometry3<Lidar<0>, World> =
            Isometry3::from_translation([10.0, 20.0, 30.0]).unwrap();
        let world = fc.transform_into(&iso);
        assert_eq!(world.cloud().get(0), Some(&[11.0, 20.0, 30.0]));
        // And the original is untouched.
        assert_eq!(fc.cloud().get(0), Some(&[1.0, 0.0, 0.0]));
    }
}
