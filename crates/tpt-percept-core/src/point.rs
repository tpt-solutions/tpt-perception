//! Frame-typed 3D points and vectors.

use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

use tpt_math_linalg_fixed::Vector3;

use crate::error::{CoreError, CoreResult};
use crate::frame::Frame;

/// A 3-D point expressed in frame `F`.
///
/// The frame is a phantom type parameter: points from different frames
/// cannot be mixed, and only an [`crate::iso::Isometry3<F, To>`] can move a
/// point from `F` into `To`.
///
/// Units are metres. Points are affine locations — subtracting two points in
/// the *same* frame yields a displacement [`Vector3D<F>`].
pub struct Point3D<F: Frame> {
    coords: [f64; 3],
    _frame: PhantomData<fn() -> F>,
}

impl<F: Frame> Clone for Point3D<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Frame> Copy for Point3D<F> {}

impl<F: Frame> Point3D<F> {
    /// Create a point from coordinates (metres).
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D {
            coords: [x, y, z],
            _frame: PhantomData,
        }
    }

    /// Create a point from a coordinate array (metres).
    pub const fn from_array(coords: [f64; 3]) -> Self {
        Point3D {
            coords,
            _frame: PhantomData,
        }
    }

    /// The origin of frame `F`.
    pub fn origin() -> Self {
        Point3D::from_array([0.0; 3])
    }

    /// X coordinate (metres).
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Y coordinate (metres).
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Z coordinate (metres).
    pub fn z(&self) -> f64 {
        self.coords[2]
    }

    /// Coordinates as `[x, y, z]` (metres).
    pub fn coords(&self) -> [f64; 3] {
        self.coords
    }

    /// Euclidean distance to another point *in the same frame* (metres).
    pub fn distance_to(&self, other: &Point3D<F>) -> f64 {
        let d = Vector3::new([
            self.coords[0] - other.coords[0],
            self.coords[1] - other.coords[1],
            self.coords[2] - other.coords[2],
        ]);
        d.norm()
    }

    /// Squared Euclidean distance (metres²) — avoids the square root.
    pub fn distance_squared_to(&self, other: &Point3D<F>) -> f64 {
        let dx = self.coords[0] - other.coords[0];
        let dy = self.coords[1] - other.coords[1];
        let dz = self.coords[2] - other.coords[2];
        dx * dx + dy * dy + dz * dz
    }

    /// True if any coordinate is NaN or infinite.
    pub fn is_finite(&self) -> bool {
        self.coords.iter().all(|c| c.is_finite())
    }

    /// Linear interpolation between `self` (at `a`) and `other` (at `b`);
    /// `t ∈ [0, 1]` extraporates.
    pub fn lerp(&self, other: &Point3D<F>, t: f64) -> Point3D<F> {
        Point3D::from_array([
            self.coords[0] * (1.0 - t) + other.coords[0] * t,
            self.coords[1] * (1.0 - t) + other.coords[1] * t,
            self.coords[2] * (1.0 - t) + other.coords[2] * t,
        ])
    }

    /// Reinterpret this point as being expressed in a different frame
    /// *without transforming it*.
    ///
    /// This is the deliberate escape hatch for cases where the frame change
    /// is already known to be a no-op in coordinates (e.g. sensor frames
    /// that share an origin and axes). Misusing it defeats compile-time
    /// frame safety; prefer [`crate::iso::Isometry3::transform_point`].
    pub fn reinterpret_frame<To: Frame>(self) -> Point3D<To> {
        Point3D::from_array(self.coords)
    }
}

impl<F: Frame> PartialEq for Point3D<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl<F: Frame> core::fmt::Debug for Point3D<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Point3D<{}>{{x: {}, y: {}, z: {}}}",
            core::any::type_name::<F>(),
            self.coords[0],
            self.coords[1],
            self.coords[2]
        )
    }
}

impl<F: Frame> Add<Vector3D<F>> for Point3D<F> {
    type Output = Point3D<F>;
    fn add(self, rhs: Vector3D<F>) -> Point3D<F> {
        let v = rhs.coords();
        Point3D::from_array([
            self.coords[0] + v[0],
            self.coords[1] + v[1],
            self.coords[2] + v[2],
        ])
    }
}

impl<F: Frame> Sub for Point3D<F> {
    type Output = Vector3D<F>;
    fn sub(self, rhs: Point3D<F>) -> Vector3D<F> {
        Vector3D::from_array([
            self.coords[0] - rhs.coords[0],
            self.coords[1] - rhs.coords[1],
            self.coords[2] - rhs.coords[2],
        ])
    }
}

/// A 3-D displacement/direction vector expressed in frame `F` (metres).
///
/// Unlike a point, a vector is translation-invariant: it transforms between
/// frames through the rotation part of an isometry only.
pub struct Vector3D<F: Frame> {
    coords: [f64; 3],
    _frame: PhantomData<fn() -> F>,
}

impl<F: Frame> Clone for Vector3D<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Frame> Copy for Vector3D<F> {}

impl<F: Frame> Vector3D<F> {
    /// Create a vector from components (metres).
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3D {
            coords: [x, y, z],
            _frame: PhantomData,
        }
    }

    /// Create a vector from a component array (metres).
    pub const fn from_array(coords: [f64; 3]) -> Self {
        Vector3D {
            coords,
            _frame: PhantomData,
        }
    }

    /// The zero vector.
    pub fn zero() -> Self {
        Vector3D::from_array([0.0; 3])
    }

    /// X component.
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Y component.
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Z component.
    pub fn z(&self) -> f64 {
        self.coords[2]
    }

    /// Components as `[x, y, z]`.
    pub fn coords(&self) -> [f64; 3] {
        self.coords
    }

    /// Euclidean norm (metres).
    pub fn norm(&self) -> f64 {
        Vector3::new(self.coords).norm()
    }

    /// Squared norm (metres²).
    pub fn norm_squared(&self) -> f64 {
        let c = self.coords;
        c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
    }

    /// Dot product with a same-frame vector (metres²).
    pub fn dot(&self, other: &Vector3D<F>) -> f64 {
        Vector3::new(self.coords).dot(&Vector3::new(other.coords))
    }

    /// Cross product with a same-frame vector; the result is expressed in
    /// the same frame and orthogonal to both inputs.
    pub fn cross(&self, other: &Vector3D<F>) -> Vector3D<F> {
        let c = Vector3::new(self.coords).cross(&Vector3::new(other.coords));
        Vector3D::from_array(c.data)
    }

    /// Unit vector in the same direction, or `None` for the zero/near-zero
    /// vector (norm < 1e-12).
    pub fn normalize(&self) -> Option<Vector3D<F>> {
        let n = self.norm();
        if n < 1e-12 {
            return None;
        }
        Some(Vector3D::from_array([
            self.coords[0] / n,
            self.coords[1] / n,
            self.coords[2] / n,
        ]))
    }

    /// True if any component is NaN or infinite.
    pub fn is_finite(&self) -> bool {
        self.coords.iter().all(|c| c.is_finite())
    }

    /// Reinterpret this vector as being expressed in a different frame
    /// *without transforming it*. Escape hatch — see
    /// [`Point3D::reinterpret_frame`] for the caveats.
    pub fn reinterpret_frame<To: Frame>(self) -> Vector3D<To> {
        Vector3D::from_array(self.coords)
    }
}

impl<F: Frame> PartialEq for Vector3D<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl<F: Frame> core::fmt::Debug for Vector3D<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Vector3D<{}>{{x: {}, y: {}, z: {}}}",
            core::any::type_name::<F>(),
            self.coords[0],
            self.coords[1],
            self.coords[2]
        )
    }
}

impl<F: Frame> Add for Vector3D<F> {
    type Output = Vector3D<F>;
    fn add(self, rhs: Vector3D<F>) -> Vector3D<F> {
        let b = rhs.coords();
        let a = self.coords;
        Vector3D::from_array([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
    }
}

impl<F: Frame> Sub for Vector3D<F> {
    type Output = Vector3D<F>;
    fn sub(self, rhs: Vector3D<F>) -> Vector3D<F> {
        let b = rhs.coords();
        let a = self.coords;
        Vector3D::from_array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    }
}

impl<F: Frame> Neg for Vector3D<F> {
    type Output = Vector3D<F>;
    fn neg(self) -> Vector3D<F> {
        let a = self.coords;
        Vector3D::from_array([-a[0], -a[1], -a[2]])
    }
}

impl<F: Frame> Mul<f64> for Vector3D<F> {
    type Output = Vector3D<F>;
    fn mul(self, s: f64) -> Vector3D<F> {
        let a = self.coords;
        Vector3D::from_array([a[0] * s, a[1] * s, a[2] * s])
    }
}

/// Validates that all coordinates are finite.
pub(crate) fn require_finite(what: &'static str, coords: &[f64]) -> CoreResult<()> {
    if coords.iter().all(|c| c.is_finite()) {
        Ok(())
    } else {
        Err(CoreError::InvalidInput { what })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Body, Lidar, World};

    #[test]
    fn same_frame_arithmetic() {
        let a = Point3D::<World>::new(1.0, 2.0, 3.0);
        let b = Point3D::<World>::new(4.0, 6.0, 3.0);
        let d = b - a;
        assert_eq!(d.coords(), [3.0, 4.0, 0.0]);
        assert!((d.norm() - 5.0).abs() < 1e-12);
        let c = a + d;
        assert_eq!(c, b);
    }

    #[test]
    fn cross_and_dot() {
        let x = Vector3D::<Body>::new(1.0, 0.0, 0.0);
        let y = Vector3D::<Body>::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert_eq!(z.coords(), [0.0, 0.0, 1.0]);
        assert!((x.dot(&y)).abs() < 1e-12);
    }

    #[test]
    fn zero_vector_normalize_is_none() {
        let v = Vector3D::<Lidar<0>>::zero();
        assert!(v.normalize().is_none());
        let w = Vector3D::<Lidar<0>>::new(3.0, 4.0, 0.0);
        let n = w.normalize().unwrap();
        assert!((n.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reinterpret_keeps_coordinates() {
        let p = Point3D::<Lidar<2>>::new(1.0, 2.0, 3.0);
        let q: Point3D<Lidar<2>> = p.reinterpret_frame();
        assert_eq!(p, q);
    }
}
