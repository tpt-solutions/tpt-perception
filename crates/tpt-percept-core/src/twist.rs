//! Spatial twists: combined linear/angular velocity and SE(3) integration.
//!
//! A [`Twist3<F>`] is the instantaneous velocity of a rigid body expressed
//! in frame `F`: a linear part (m/s) and an angular part (rad/s). It
//! integrates to a rigid transform via the SE(3) exponential map and
//! differentiates from one via the SE(3) logarithm — both implemented
//! exactly (with the standard series fallbacks at the singular points).

use tpt_math_linalg_fixed::Vector3;
use tpt_math_units::prelude::Time;
use tpt_math_units::si::time::second;

use crate::error::{CoreError, CoreResult};
use crate::frame::Frame;
use crate::iso::Isometry3;
use crate::point::Vector3D;
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

/// Angular-rate integration threshold: below this rotation angle the
/// exponential map degenerates to a pure translation (error O(θ²) ≈ 1e-16).
const ANGLE_EPS: f64 = 1e-10;

/// Combined linear (m/s) and angular (rad/s) velocity expressed in frame `F`.
///
/// The body moves as `x(t + Δt) = Twist3::exp(Δt) * x(t)` (state-space /
/// right-multiplicative convention: the twist is expressed in the *moving*
/// frame `F`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Twist3<F: Frame> {
    /// Linear velocity (m/s).
    pub linear: Vector3D<F>,
    /// Angular velocity (rad/s), right-hand rule.
    pub angular: Vector3D<F>,
}

impl<F: Frame> Twist3<F> {
    /// Build a twist from linear and angular parts.
    pub fn new(linear: Vector3D<F>, angular: Vector3D<F>) -> Self {
        Twist3 { linear, angular }
    }

    /// The zero twist (rigid body at rest).
    pub fn zero() -> Self {
        Twist3 {
            linear: Vector3D::zero(),
            angular: Vector3D::zero(),
        }
    }

    /// SE(3) exponential map: the rigid transform achieved by following
    /// this twist for `dt` seconds.
    ///
    /// Mathematical contract: `exp(dt).inverse() * exp(dt) == identity` and
    /// `exp` is exact (up to floating-point rounding) for any finite twist;
    /// the `θ → 0` limit uses the series expansion so no division by zero
    /// occurs. Non-finite twists or durations propagate NaN into the result
    /// rather than panicking.
    pub fn exp(&self, dt: f64) -> Isometry3<F, F> {
        let w = self.angular.coords();
        let v = self.linear.coords();
        let finite = v.iter().chain(w.iter()).all(|c| c.is_finite()) && dt.is_finite();
        if !finite {
            debug_assert!(finite, "Twist3::exp called with non-finite input");
            let nan = f64::NAN;
            return Isometry3::from_parts_unchecked(
                [nan; 3],
                crate::iso::Rotation3::<F, F>::from_matrix_unchecked([[nan; 3]; 3]),
            );
        }
        let wdt = [w[0] * dt, w[1] * dt, w[2] * dt];
        let theta = Vector3::new(wdt).norm();

        if theta < ANGLE_EPS {
            // Non-finite components propagate as NaN (documented contract);
            // no panic on invalid input.
            return Isometry3::from_parts_unchecked(
                [v[0] * dt, v[1] * dt, v[2] * dt],
                crate::iso::Rotation3::<F, F>::identity(),
            );
        }

        let rotation = crate::iso::Rotation3::from_axis_angle(w, theta)
            .expect("non-zero angular rate gives a valid axis");
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let a = (1.0 - cos_t) / (theta * theta);
        let b = (theta - sin_t) / (theta * theta * theta);

        // t = Δt·(v + a·(ωΔt × v) + b·(ωΔt × (ωΔt × v)))
        let wv = cross3(&wdt, &v);
        let w2v = cross3(&wdt, &wv);
        let t = [
            dt * (v[0] + a * wv[0] + b * w2v[0]),
            dt * (v[1] + a * wv[1] + b * w2v[1]),
            dt * (v[2] + a * wv[2] + b * w2v[2]),
        ];
        Isometry3::from_parts_unchecked(t, rotation)
    }

    /// Integrate over a unit-checked duration.
    pub fn integrate(&self, dt: Time) -> Isometry3<F, F> {
        self.exp(dt.get::<second>())
    }

    /// SE(3) logarithm: the constant twist that produces `transform` when
    /// integrated over `dt` seconds.
    ///
    /// Inverse of [`Twist3::exp`]: `log(exp(tw, dt), dt)` reconstructs
    /// `tw` up to floating-point rounding. Errors on non-finite transforms.
    pub fn from_transform(transform: &Isometry3<F, F>, dt: f64) -> CoreResult<Self> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(CoreError::InvalidInput {
                what: "integration time must be positive and finite",
            });
        }
        let r = transform.rotation().matrix();
        let t = transform.translation().coords();
        let tr = r[0][0] + r[1][1] + r[2][2];
        let cos_t = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0);
        let theta = cos_t.acos();

        let (axis, angle) = if theta < ANGLE_EPS {
            ([0.0, 0.0, 0.0], 0.0)
        } else if (core::f64::consts::PI - theta).abs() < 1e-6 {
            // θ ≈ π: sin θ ≈ 0, the skew formula is ill-conditioned. R ≈ 2kkᵀ − I,
            // so kkᵀ ≈ (R + I)/2 and the axis is recovered column-wise.
            let mut k = [0.0_f64; 3];
            for (i, ki) in k.iter_mut().enumerate() {
                *ki = ((r[i][i] + 1.0) / 2.0).max(0.0).sqrt();
            }
            // Resolve signs from the largest component for stability.
            let m = if k[0] >= k[1] && k[0] >= k[2] {
                0
            } else if k[1] >= k[2] {
                1
            } else {
                2
            };
            if k[m] < 1e-12 {
                ([0.0, 0.0, 0.0], 0.0)
            } else {
                let km = k[m];
                for j in 0..3 {
                    if j != m {
                        k[j] = r[m][j] / (2.0 * km);
                    }
                }
                (k, theta)
            }
        } else {
            let s = 2.0 * theta.sin();
            let k = [
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s,
            ];
            (k, theta)
        };

        let w = [
            axis[0] * angle / dt,
            axis[1] * angle / dt,
            axis[2] * angle / dt,
        ];

        // v = V⁻¹ t / Δt with, on the unit axis k,
        //   V⁻¹ = I − (θ/2)[k]× + (1 − θ(1+cosθ)/(2 sinθ))[k]×².
        let v = if angle < ANGLE_EPS {
            [t[0] / dt, t[1] / dt, t[2] / dt]
        } else {
            let sin_t = angle.sin();
            let c = 1.0 - angle * (1.0 + angle.cos()) / (2.0 * sin_t);
            let kt = cross3(&axis, &t);
            let k2t = cross3(&axis, &kt);
            [
                (t[0] - 0.5 * angle * kt[0] + c * k2t[0]) / dt,
                (t[1] - 0.5 * angle * kt[1] + c * k2t[1]) / dt,
                (t[2] - 0.5 * angle * kt[2] + c * k2t[2]) / dt,
            ]
        };

        Ok(Twist3 {
            linear: Vector3D::from_array(v),
            angular: Vector3D::from_array(w),
        })
    }
}

/// `a × b` for raw arrays.
pub(crate) fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Body;
    use crate::point::Point3D;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn pure_translation_exp() {
        let tw = Twist3::<Body>::new(Vector3D::new(1.0, -2.0, 0.5), Vector3D::zero());
        let t = tw.exp(2.0);
        assert_eq!(t.translation().coords(), [2.0, -4.0, 1.0]);
    }

    #[test]
    fn rotation_only_exp() {
        // ω = ẑ at π/2 rad/s for 1 s: quarter turn about z, no drift.
        let tw = Twist3::<Body>::new(
            Vector3D::zero(),
            Vector3D::new(0.0, 0.0, core::f64::consts::FRAC_PI_2),
        );
        let t = tw.exp(1.0);
        let p = t.transform_point(Point3D::<Body>::new(2.0, 0.0, 0.0));
        assert!(approx(p.x(), 0.0) && approx(p.y(), 2.0) && approx(p.z(), 0.0));
    }

    #[test]
    fn exp_log_roundtrip() {
        let tw = Twist3::<Body>::new(Vector3D::new(0.3, -1.1, 2.0), Vector3D::new(0.2, 0.5, -0.8));
        let dt = 0.4;
        let t = tw.exp(dt);
        let back = Twist3::from_transform(&t, dt).unwrap();
        assert!(approx(back.linear.x(), tw.linear.x()));
        assert!(approx(back.linear.y(), tw.linear.y()));
        assert!(approx(back.linear.z(), tw.linear.z()));
        assert!(approx(back.angular.x(), tw.angular.x()));
        assert!(approx(back.angular.y(), tw.angular.y()));
        assert!(approx(back.angular.z(), tw.angular.z()));
    }

    #[test]
    fn exp_log_roundtrip_near_pi() {
        // Rotation by π about a tilted axis exercises the θ ≈ π branch.
        let tw = Twist3::<Body>::new(Vector3D::new(1.0, 2.0, 3.0), Vector3D::new(1.0, 0.5, 0.25));
        let angle = tw.angular.norm();
        let dt = core::f64::consts::PI / angle;
        let t = tw.exp(dt);
        let back = Twist3::from_transform(&t, dt).unwrap();
        for i in 0..3 {
            assert!((back.angular.coords()[i] - tw.angular.coords()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rotation_about_origin_preserves_radius() {
        // A body spinning in place (v = 0, ω = ẑ): every point keeps its
        // distance to the rotation axis.
        let tw = Twist3::<Body>::new(Vector3D::zero(), Vector3D::new(0.0, 0.0, 1.0));
        let p = Point3D::<Body>::new(2.0, 0.0, 1.0);
        let moved = tw.exp(0.7).transform_point(p);
        let r0 = (p.x() * p.x() + p.y() * p.y()).sqrt();
        let r1 = (moved.x() * moved.x() + moved.y() * moved.y()).sqrt();
        assert!(approx(r0, r1));
    }

    #[test]
    fn exp_matches_first_order_dynamics() {
        // Contract: for small Δt, exp(Δt)·p ≈ p + Δt·(v + ω × p).
        let tw = Twist3::<Body>::new(Vector3D::new(0.5, -1.0, 2.0), Vector3D::new(0.3, 0.7, -0.2));
        let p = Point3D::<Body>::new(1.0, 2.0, -0.5);
        let dt = 1e-6;
        let moved = tw.exp(dt).transform_point(p);
        let w = tw.angular.coords();
        let vel = [
            tw.linear.x() + (w[1] * p.z() - w[2] * p.y()),
            tw.linear.y() + (w[2] * p.x() - w[0] * p.z()),
            tw.linear.z() + (w[0] * p.y() - w[1] * p.x()),
        ];
        let expect = [
            p.x() + dt * vel[0],
            p.y() + dt * vel[1],
            p.z() + dt * vel[2],
        ];
        assert!((moved.x() - expect[0]).abs() < 1e-9);
        assert!((moved.y() - expect[1]).abs() < 1e-9);
        assert!((moved.z() - expect[2]).abs() < 1e-9);
    }

    #[test]
    fn integrate_uses_seconds() {
        use tpt_math_units::si::time::second;
        let tw = Twist3::<Body>::new(Vector3D::new(3.0, 0.0, 0.0), Vector3D::zero());
        let t = tw.integrate(Time::new::<second>(0.5));
        assert_eq!(t.translation().coords(), [1.5, 0.0, 0.0]);
    }

    #[test]
    fn log_rejects_nonpositive_dt() {
        let t = Isometry3::<Body, Body>::identity();
        assert!(Twist3::<Body>::from_transform(&t, 0.0).is_err());
        assert!(Twist3::<Body>::from_transform(&t, -1.0).is_err());
    }
}
