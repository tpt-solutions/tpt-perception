//! Unit-safe physical quantities via `tpt-math-units` (a `uom` wrap).
//!
//! Algorithm hot paths use bare `f64` in SI units (metres, radians,
//! seconds) — the universal convention for point cloud processing — but API
//! boundaries where accidental unit mixing is a realistic risk (timestamps,
//! durations, voxel edge lengths, sensor rates) accept and return
//! `uom`-typed quantities so that `Time` + `Length` cannot compile.
//!
//! ```
//! use tpt_percept_core::units::{Meters, Seconds, Radians};
//! use tpt_percept_core::units::si::{length::meter, time::second, angle::radian};
//!
//! let range = Meters::new::<meter>(200.0);
//! let dt = Seconds::new::<second>(0.01);
//! // range + dt  → compile error: length and time do not mix.
//! ```

pub use tpt_math_units;

/// Re-export of the `uom` SI system (quantities and unit markers).
pub use tpt_math_units::si;

/// `f64`-backed length quantity.
pub type Meters = tpt_math_units::si::f64::Length;
/// `f64`-backed time duration quantity.
pub type Seconds = tpt_math_units::si::f64::Time;
/// `f64`-backed angle quantity (radians internally).
pub type Radians = tpt_math_units::si::f64::Angle;
/// `f64`-backed linear velocity quantity (m/s).
pub type MetresPerSecond = tpt_math_units::si::f64::Velocity;
/// `f64`-backed angular velocity quantity (rad/s).
pub type RadiansPerSecond = tpt_math_units::si::f64::AngularVelocity;
/// `f64`-backed frequency quantity (Hz).
pub type Hertz = tpt_math_units::si::f64::Frequency;

#[cfg(test)]
mod tests {
    use super::si::{angle::degree, length::kilometer, length::meter, time::hour, time::second};
    use super::{Meters, Radians, Seconds};
    #[test]
    #[allow(clippy::float_cmp)]
    fn unit_conversions() {
        let d = Meters::new::<kilometer>(3.0);
        assert_eq!(d.get::<meter>(), 3000.0);
        let t = Seconds::new::<hour>(1.0);
        assert_eq!(t.get::<second>(), 3600.0);
        let a = Radians::new::<degree>(180.0);
        assert!(
            (a.get::<tpt_math_units::si::angle::radian>() - core::f64::consts::PI).abs() < 1e-12
        );
    }
}
