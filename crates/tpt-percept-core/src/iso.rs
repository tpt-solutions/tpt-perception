//! Frame-typed rigid body transformations: rotations and isometries.
//!
//! An [`Isometry3<From, To>`] is a rigid body transform that maps points
//! *expressed in frame `From`* to points *expressed in frame `To`*. The
//! frames are phantom type parameters, so composition only type-checks when
//! the intermediate frames agree:
//!
//! ```
//! use tpt_percept_core::{frame::{Lidar, Body, World}, iso::Isometry3};
//!
//! let lidar_to_body: Isometry3<Lidar<0>, Body> = Isometry3::from_parts(
//!     [0.1, 0.0, 1.7],           // mounted 1.7 m above the base, 10 cm forward
//!     [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], // 90° yaw
//! ).unwrap();
//! let body_to_world: Isometry3<Body, World> = Isometry3::from_parts(
//!     [42.0, 7.0, 0.0],
//!     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
//! ).unwrap();
//!
//! // apply lidar_to_body, then body_to_world:
//! let lidar_to_world: Isometry3<Lidar<0>, World> = body_to_world * lidar_to_body;
//!
//! // Does not compile — `Lidar<0>` points cannot enter `body_to_world`:
//! // let _bad: Isometry3<Lidar<0>, World> = lidar_to_body * body_to_world;
//! ```
//!
//! Conventions follow `tpt-math-geometry`: active rotations, column
//! vectors, right-handed frames, `B * A` means "apply `A`, then `B`".

use core::marker::PhantomData;
use core::ops::Mul;

use tpt_math_geometry as geo;
use tpt_math_linalg_fixed::{Matrix3, Vector3};

use crate::error::{CoreError, CoreResult};
use crate::frame::Frame;
use crate::point::{require_finite, Point3D, Vector3D};

fn to_geo_matrix(m: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(m)
}

fn from_geo_matrix(m: &Matrix3<f64>) -> [[f64; 3]; 3] {
    m.data
}

const ORTHO_TOL: f64 = 1e-9;

/// Validates `RᵀR = I` within [`ORTHO_TOL`] and `det R > 0`.
fn is_proper_rotation(m: &[[f64; 3]; 3]) -> bool {
    // Column norms and pairwise orthogonality (RᵀR diagonal/off-diagonal).
    for col in m.iter() {
        let col_norm_sq = col[0] * col[0] + col[1] * col[1] + col[2] * col[2];
        if (col_norm_sq - 1.0).abs() > ORTHO_TOL {
            return false;
        }
    }
    let d01 = m[0][0] * m[0][1] + m[1][0] * m[1][1] + m[2][0] * m[2][1];
    let d02 = m[0][0] * m[0][2] + m[1][0] * m[1][2] + m[2][0] * m[2][2];
    let d12 = m[0][1] * m[0][2] + m[1][1] * m[1][2] + m[2][1] * m[2][2];
    if d01.abs() > ORTHO_TOL || d02.abs() > ORTHO_TOL || d12.abs() > ORTHO_TOL {
        return false;
    }
    crate::linalg3::det3(m) > 0.0
}

/// A 3-D rotation mapping frame `From` to frame `To` (active convention:
/// `to = R * from` for coordinates expressed in a common basis).
///
/// The underlying 3×3 matrix is orthonormal (`RᵀR = I`, `det R = +1`); this
/// is validated on construction.
pub struct Rotation3<From: Frame, To: Frame> {
    inner: geo::Rotation3<f64>,
    _frames: PhantomData<fn(From) -> To>,
}

impl<From: Frame, To: Frame> Clone for Rotation3<From, To> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<From: Frame, To: Frame> Copy for Rotation3<From, To> {}

impl<F: Frame> Rotation3<F, F> {
    /// The identity rotation (frames unchanged).
    pub fn identity() -> Self {
        Rotation3 {
            inner: geo::Rotation3::identity(),
            _frames: PhantomData,
        }
    }
}

impl<From: Frame, To: Frame> Rotation3<From, To> {
    /// Build a rotation from a right-handed orthonormal rotation matrix
    /// (row-major `[[r00, r01, r02], [r10, …], [r20, …]]`).
    ///
    /// Returns an error if the matrix is not orthonormal within 1e-9 or has
    /// a non-positive determinant (i.e. it is a reflection). The matrix is
    /// *not* orthonormalised — use [`Rotation3::from_matrix_unchecked`] only
    /// with matrices known to be proper rotations.
    pub fn from_matrix(m: [[f64; 3]; 3]) -> CoreResult<Self> {
        for row in &m {
            require_finite("rotation matrix contains non-finite entries", row)?;
        }
        if !is_proper_rotation(&m) {
            return Err(CoreError::InvalidInput {
                what: "rotation matrix is not a proper orthonormal rotation (orthonormality or det > 0 violated)",
            });
        }
        Ok(Rotation3 {
            inner: geo::Rotation3::from_matrix_unchecked(to_geo_matrix(m)),
            _frames: PhantomData,
        })
    }

    /// Build a rotation from a matrix without validating orthonormality.
    ///
    /// Mathematical contract: the caller guarantees `RᵀR = I` and
    /// `det R = +1` within numerical tolerance; violating this is undefined
    /// behaviour *of the math* (never memory-unsafe) and will silently
    /// distort every downstream result.
    pub fn from_matrix_unchecked(m: [[f64; 3]; 3]) -> Self {
        Rotation3 {
            inner: geo::Rotation3::from_matrix_unchecked(to_geo_matrix(m)),
            _frames: PhantomData,
        }
    }

    /// Rotation about `axis` (any non-zero length, normalised internally)
    /// by `angle` radians, right-hand rule.
    pub fn from_axis_angle(axis: [f64; 3], angle_rad: f64) -> CoreResult<Self> {
        require_finite("rotation axis contains non-finite components", &axis)?;
        if !(angle_rad.is_finite()) {
            return Err(CoreError::InvalidInput {
                what: "rotation angle is non-finite",
            });
        }
        let v = Vector3::new(axis);
        let n = v.norm();
        if n < 1e-12 {
            return Err(CoreError::InvalidInput {
                what: "rotation axis is (near-)zero length",
            });
        }
        let uq = geo::UnitQuaternion::from_axis_angle(&v, angle_rad);
        Ok(Rotation3 {
            inner: uq.to_rotation_matrix(),
            _frames: PhantomData,
        })
    }

    /// The underlying rotation matrix (row-major).
    pub fn matrix(&self) -> [[f64; 3]; 3] {
        from_geo_matrix(self.inner.matrix())
    }

    /// Rotate a point (rotation about the frame origin; no translation).
    pub fn transform_point(&self, p: Point3D<From>) -> Point3D<To> {
        let gp = geo::Point3::new(Vector3::new(p.coords()));
        Point3D::from_array(self.inner.transform_point(&gp).coords.data)
    }

    /// Rotate a vector (direction only).
    pub fn transform_vector(&self, v: Vector3D<From>) -> Vector3D<To> {
        let gv = Vector3::new(v.coords());
        Vector3D::from_array(self.inner.transform_vector(&gv).data)
    }

    /// The inverse rotation, mapping `To` back to `From`.
    pub fn inverse(&self) -> Rotation3<To, From> {
        Rotation3 {
            inner: self.inner.inverse(),
            _frames: PhantomData,
        }
    }

    /// Underlying `tpt-math-geometry` rotation (interop escape hatch).
    pub fn into_geo(self) -> geo::Rotation3<f64> {
        self.inner
    }

    /// Wrap a `tpt-math-geometry` rotation (interop escape hatch); the
    /// matrix is assumed to be a proper rotation.
    pub fn from_geo(inner: geo::Rotation3<f64>) -> Self {
        Rotation3 {
            inner,
            _frames: PhantomData,
        }
    }
}

impl<From: Frame> PartialEq for Rotation3<From, From> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.matrix().data == other.inner.matrix().data
    }
}

impl<From: Frame, To: Frame> core::fmt::Debug for Rotation3<From, To> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Rotation3<{}, {}>{:?}",
            core::any::type_name::<From>(),
            core::any::type_name::<To>(),
            self.matrix()
        )
    }
}

impl<A: Frame, B: Frame, C: Frame> Mul<Rotation3<A, B>> for Rotation3<B, C> {
    type Output = Rotation3<A, C>;
    /// `R_bc * R_ab` — apply `R_ab` first, then `R_bc`.
    fn mul(self, rhs: Rotation3<A, B>) -> Rotation3<A, C> {
        Rotation3 {
            inner: self.inner * rhs.inner,
            _frames: PhantomData,
        }
    }
}

/// A rigid body transformation (rotation + translation) mapping frame
/// `From` to frame `To`: `p_to = R * p_from + t`.
///
/// This is the workhorse type of the crate: sensor extrinsics, odometry
/// poses, registration results and map updates are all `Isometry3`s between
/// named frames.
pub struct Isometry3<From: Frame, To: Frame> {
    inner: geo::Isometry3<f64>,
    _frames: PhantomData<fn(From) -> To>,
}

impl<From: Frame, To: Frame> Clone for Isometry3<From, To> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<From: Frame, To: Frame> Copy for Isometry3<From, To> {}

impl<F: Frame> Isometry3<F, F> {
    /// The identity isometry (frame `F` to itself).
    pub fn identity() -> Self {
        Isometry3 {
            inner: geo::Isometry3::identity(),
            _frames: PhantomData,
        }
    }
}

impl<From: Frame, To: Frame> Isometry3<From, To> {
    /// Build from a translation (metres, expressed in frame `From`) and a
    /// rotation matrix.
    pub fn from_parts(translation: [f64; 3], rotation: [[f64; 3]; 3]) -> CoreResult<Self> {
        let r = Rotation3::<From, To>::from_matrix(rotation)?;
        Ok(Isometry3::from_parts_unchecked(translation, r))
    }

    /// Build from a translation and an already-validated rotation.
    pub fn from_parts_unchecked(translation: [f64; 3], rotation: Rotation3<From, To>) -> Self {
        Isometry3 {
            inner: geo::Isometry3::new(
                geo::Translation::new(Vector3::new(translation)),
                rotation.into_geo(),
            ),
            _frames: PhantomData,
        }
    }

    /// Pure translation (identity rotation).
    pub fn from_translation(translation: [f64; 3]) -> CoreResult<Self> {
        require_finite("translation contains non-finite components", &translation)?;
        Ok(Isometry3 {
            inner: geo::Isometry3::from_translation(geo::Translation::new(Vector3::new(
                translation,
            ))),
            _frames: PhantomData,
        })
    }

    /// Pure rotation (zero translation).
    pub fn from_rotation(rotation: Rotation3<From, To>) -> Self {
        Isometry3 {
            inner: geo::Isometry3::from_rotation(rotation.into_geo()),
            _frames: PhantomData,
        }
    }

    /// The rotation part.
    pub fn rotation(&self) -> Rotation3<From, To> {
        Rotation3::from_geo(self.inner.rotation)
    }

    /// The translation part, expressed in frame `From` (metres).
    pub fn translation(&self) -> Vector3D<From> {
        Vector3D::from_array(self.inner.translation.vector.data)
    }

    /// Transform a point from frame `From` to frame `To`.
    pub fn transform_point(&self, p: Point3D<From>) -> Point3D<To> {
        let gp = geo::Point3::new(Vector3::new(p.coords()));
        Point3D::from_array(self.inner.transform_point(&gp).coords.data)
    }

    /// Transform a direction vector (rotation applied, translation ignored).
    pub fn transform_vector(&self, v: Vector3D<From>) -> Vector3D<To> {
        let gv = Vector3::new(v.coords());
        Vector3D::from_array(self.inner.transform_vector(&gv).data)
    }

    /// The inverse transform, mapping `To` back to `From`.
    pub fn inverse(&self) -> Isometry3<To, From> {
        Isometry3 {
            inner: self.inner.inverse(),
            _frames: PhantomData,
        }
    }

    /// `(rotation_matrix, translation)` raw parts.
    pub fn into_raw(self) -> ([[f64; 3]; 3], [f64; 3]) {
        (self.rotation().matrix(), self.translation().coords())
    }

    /// Underlying `tpt-math-geometry` isometry (interop escape hatch).
    pub fn into_geo(self) -> geo::Isometry3<f64> {
        self.inner
    }

    /// Wrap a `tpt-math-geometry` isometry (interop escape hatch); the
    /// rotation part is assumed to be a proper rotation.
    pub fn from_geo(inner: geo::Isometry3<f64>) -> Self {
        Isometry3 {
            inner,
            _frames: PhantomData,
        }
    }
}

impl<From: Frame> PartialEq for Isometry3<From, From> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<From: Frame, To: Frame> core::fmt::Debug for Isometry3<From, To> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Isometry3<{}, {}>{{t: {:?}, R: {:?}}}",
            core::any::type_name::<From>(),
            core::any::type_name::<To>(),
            self.translation().coords(),
            self.rotation().matrix()
        )
    }
}

impl<A: Frame, B: Frame, C: Frame> Mul<Isometry3<A, B>> for Isometry3<B, C> {
    type Output = Isometry3<A, C>;
    /// `T_bc * T_ab` — apply `T_ab` first, then `T_bc`.
    fn mul(self, rhs: Isometry3<A, B>) -> Isometry3<A, C> {
        Isometry3 {
            inner: self.inner * rhs.inner,
            _frames: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Body, Lidar, World};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn identity_is_identity() {
        let p = Point3D::<World>::new(1.0, -2.0, 3.5);
        let t = Isometry3::<World, World>::identity();
        assert_eq!(t.transform_point(p), p);
    }

    #[test]
    fn frame_mismatch_does_not_compile() {
        // The following is the compile-time contract; expressed here as a
        // type-level tautology since negative tests need trybuild.
        fn assert_frames_differ<A: Frame, B: Frame>() {}
        assert_frames_differ::<Lidar<0>, World>();
        assert_frames_differ::<Body, Lidar<0>>();
    }

    #[test]
    fn translation_moves_points() {
        let t = Isometry3::<Lidar<0>, World>::from_parts(
            [10.0, 0.0, 1.5],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .unwrap();
        let p = t.transform_point(Point3D::<Lidar<0>>::new(1.0, 2.0, 0.5));
        assert_eq!(p.coords(), [11.0, 2.0, 2.0]);
    }

    #[test]
    fn rotation_about_z() {
        // 90° about z: x-axis maps to y-axis.
        let r = Rotation3::<Lidar<0>, Lidar<0>>::from_axis_angle(
            [0.0, 0.0, 1.0],
            core::f64::consts::FRAC_PI_2,
        )
        .unwrap();
        let v = r.transform_vector(Vector3D::<Lidar<0>>::new(1.0, 0.0, 0.0));
        assert!(approx(v.x(), 0.0) && approx(v.y(), 1.0) && approx(v.z(), 0.0));
    }

    #[test]
    fn composition_order_and_frames() {
        let t_ab: Isometry3<Lidar<0>, Body> = Isometry3::from_translation([1.0, 0.0, 2.0]).unwrap();
        let t_bc: Isometry3<Body, World> = Isometry3::from_translation([10.0, 20.0, 30.0]).unwrap();
        let t_ac: Isometry3<Lidar<0>, World> = t_bc * t_ab;
        let p = t_ac.transform_point(Point3D::<Lidar<0>>::new(1.0, 1.0, 1.0));
        assert_eq!(p.coords(), [12.0, 21.0, 33.0]);
    }

    #[test]
    fn inverse_roundtrip() {
        let r = Rotation3::<Lidar<0>, World>::from_axis_angle([0.1, 1.0, -0.3], 1.234)
            .unwrap()
            .matrix();
        let t: Isometry3<Lidar<0>, World> = Isometry3::from_parts([3.0, -4.0, 5.0], r).unwrap();
        let p = Point3D::<Lidar<0>>::new(0.3, -1.2, 2.5);
        let round = t.inverse().transform_point(t.transform_point(p));
        assert!(approx(round.x(), p.x()) && approx(round.y(), p.y()) && approx(round.z(), p.z()));
    }

    #[test]
    fn isometry_preserves_distances() {
        // Mathematical contract: rigid transforms preserve distances.
        let r = Rotation3::<Lidar<0>, World>::from_axis_angle([0.1, 1.0, -0.3], 1.234)
            .unwrap()
            .matrix();
        let t: Isometry3<Lidar<0>, World> = Isometry3::from_parts([3.0, -4.0, 5.0], r).unwrap();
        let a = Point3D::<Lidar<0>>::new(0.3, -1.2, 2.5);
        let b = Point3D::<Lidar<0>>::new(-2.0, 0.7, 1.1);
        let d0 = a.distance_to(&b);
        let d1 = t.transform_point(a).distance_to(&t.transform_point(b));
        assert!(approx(d0, d1));
    }

    #[test]
    fn non_orthonormal_matrix_rejected() {
        let bad = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(Rotation3::<World, World>::from_matrix(bad).is_err());
        let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(Rotation3::<World, World>::from_matrix(reflection).is_err());
    }
}
