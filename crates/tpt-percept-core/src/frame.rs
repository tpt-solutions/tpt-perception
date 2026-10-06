//! Compile-time coordinate frames.
//!
//! A [`Frame`] is a phantom marker type naming a reference frame: [`World`],
//! [`Body`], [`Odometry`], or a numbered sensor frame such as
//! [`Lidar<0>`](Lidar) / [`Camera<1>`](Camera). Geometric quantities carry
//! their frame as a type parameter, so mixing frames is a *compile error*:
//!
//! ```
//! use tpt_percept_core::{frame::{Lidar, World}, point::Point3D, iso::Isometry3};
//!
//! let p = Point3D::<Lidar<0>>::new(1.0, 2.0, 0.5);
//! let lidar_to_world = Isometry3::<Lidar<0>, World>::from_parts(
//!     [10.0, 0.0, 1.5],
//!     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
//! ).unwrap();
//! let p_world = lidar_to_world.transform_point(p);
//!
//! // This does NOT compile — frames do not match:
//! // let bad: Point3D<Lidar<0>> = lidar_to_world.transform_point(p_world);
//! ```
//!
//! Custom frames are one line away:
//!
//! ```
//! use tpt_percept_core::frame::Frame;
//!
//! #[derive(Debug)]
//! struct VehicleRearAxle;
//! impl Frame for VehicleRearAxle {}
//! ```

/// A named reference frame for spatial quantities.
///
/// Implement this for your own marker types to define custom frames. All
/// frame-typed quantities in this crate ([`crate::point::Point3D`],
/// [`crate::iso::Isometry3`], …) are generic over `F: Frame`.
pub trait Frame: core::fmt::Debug + 'static {}

/// The fixed global/world reference frame (e.g. a map frame or ENU origin).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct World;
impl Frame for World {}

/// The vehicle/robot body frame (typically centred on the base link).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Body;
impl Frame for Body {}

/// The accumulated odometry frame: drift-afflicted but locally smooth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Odometry;
impl Frame for Odometry {}

macro_rules! numbered_sensor_frame {
    ($(#[$doc:meta])+ $name:ident) => {
        $(#[$doc])+
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name<const ID: u16>;
        impl<const ID: u16> Frame for $name<ID> {}
    };
}

numbered_sensor_frame! {
    /// A LiDAR sensor frame (`ID` distinguishes multiple units). Typically
    /// x-forward, y-left, z-up, origin at the sensor.
    Lidar
}
numbered_sensor_frame! {
    /// A camera sensor frame (`ID` distinguishes multiple units). Typically
    /// x-right, y-down, z-forward (optical convention).
    Camera
}
numbered_sensor_frame! {
    /// An IMU frame (`ID` distinguishes multiple units).
    Imu
}
numbered_sensor_frame! {
    /// A radar sensor frame (`ID` distinguishes multiple units).
    Radar
}
numbered_sensor_frame! {
    /// A generic named sensor frame for anything the concrete sensor frames
    /// above do not cover.
    Sensor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_distinct_types() {
        fn same_frame<A: Frame, B: Frame>() -> bool {
            core::any::TypeId::of::<A>() == core::any::TypeId::of::<B>()
        }
        assert!(same_frame::<World, World>());
        assert!(!same_frame::<World, Body>());
        assert!(!same_frame::<Lidar<0>, Lidar<1>>());
    }
}
