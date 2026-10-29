//! IMU preintegration between two keyframes.
//!
//! Gyroscope and accelerometer samples between two times are folded into a
//! single relative motion `(ΔR, Δv, Δp)` expressed in the body frame at the
//! first time. Biases are constant over the window and are subtracted from
//! every sample. This is the standard first-order scheme used as an odometry
//! prior. It does not propagate uncertainty, so pair it with a covariance
//! model if a filter needs one.

use crate::error::SlamError;
use crate::pose_graph::Pose;

/// Relative motion accumulated over an IMU window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreintegratedDelta {
    /// Body rotation change `ΔR` from the window start.
    pub rotation: [[f64; 3]; 3],
    /// Velocity change `Δv`, expressed in the start body frame (m/s).
    pub velocity: [f64; 3],
    /// Position change `Δp`, expressed in the start body frame (m).
    pub position: [f64; 3],
    /// Total integrated time (s).
    pub duration: f64,
}

/// Accumulates IMU samples into a [`PreintegratedDelta`].
#[derive(Clone, Debug)]
pub struct ImuPreintegrator {
    gyro_bias: [f64; 3],
    accel_bias: [f64; 3],
    delta: PreintegratedDelta,
}

impl ImuPreintegrator {
    /// A fresh window with the given constant biases.
    pub fn new(gyro_bias: [f64; 3], accel_bias: [f64; 3]) -> Self {
        ImuPreintegrator {
            gyro_bias,
            accel_bias,
            delta: PreintegratedDelta {
                rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                velocity: [0.0; 3],
                position: [0.0; 3],
                duration: 0.0,
            },
        }
    }

    /// Integrates one sample: angular rate (rad/s), specific force (m/s²) and
    /// the time since the previous sample (s).
    ///
    /// # Errors
    ///
    /// Returns [`SlamError::InvalidParameter`] for a non-positive or
    /// non-finite `dt`, or for non-finite samples.
    pub fn integrate(&mut self, gyro: [f64; 3], accel: [f64; 3], dt: f64) -> Result<(), SlamError> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(SlamError::InvalidParameter("dt must be finite and > 0"));
        }
        if gyro.iter().chain(accel.iter()).any(|v| !v.is_finite()) {
            return Err(SlamError::InvalidParameter("IMU samples must be finite"));
        }
        let w = [
            (gyro[0] - self.gyro_bias[0]) * dt,
            (gyro[1] - self.gyro_bias[1]) * dt,
            (gyro[2] - self.gyro_bias[2]) * dt,
        ];
        let a = [
            accel[0] - self.accel_bias[0],
            accel[1] - self.accel_bias[1],
            accel[2] - self.accel_bias[2],
        ];
        let d = &mut self.delta;
        // Δp += Δv·dt + ½ ΔR a dt²; Δv += ΔR a dt; then ΔR ← ΔR·Exp(ω dt).
        let ra = mul_vec(&d.rotation, a);
        for ((p, v), a) in d.position.iter_mut().zip(d.velocity.iter_mut()).zip(ra) {
            *p += *v * dt + 0.5 * a * dt * dt;
            *v += a * dt;
        }
        let step = Pose::exp(&[0.0, 0.0, 0.0, w[0], w[1], w[2]]).rotation;
        d.rotation = mul(&d.rotation, &step);
        d.duration += dt;
        Ok(())
    }

    /// The accumulated relative motion so far.
    pub fn delta(&self) -> PreintegratedDelta {
        self.delta
    }
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_yaw_rate_integrates_to_angle() {
        let mut imu = ImuPreintegrator::new([0.0; 3], [0.0; 3]);
        for _ in 0..100 {
            imu.integrate([0.0, 0.0, 1.0], [0.0, 0.0, 9.81], 0.01)
                .unwrap();
        }
        let d = imu.delta();
        // Yaw of 1 rad: R[0][0] = cos 1, R[1][0] = sin 1.
        assert!((d.rotation[0][0] - 1f64.cos()).abs() < 1e-6);
        assert!((d.rotation[1][0] - 1f64.sin()).abs() < 1e-6);
        assert!((d.duration - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_acceleration_gives_kinematics() {
        let mut imu = ImuPreintegrator::new([0.0; 3], [0.0; 3]);
        let n = 200;
        let dt = 0.005;
        for _ in 0..n {
            imu.integrate([0.0; 3], [2.0, 0.0, 0.0], dt).unwrap();
        }
        let t = n as f64 * dt;
        let d = imu.delta();
        assert!((d.velocity[0] - 2.0 * t).abs() < 1e-9);
        assert!((d.position[0] - t * t).abs() < 1e-3, "{:?}", d.position);
    }

    #[test]
    fn biases_are_removed() {
        let mut imu = ImuPreintegrator::new([0.0; 3], [0.1, 0.0, 0.0]);
        imu.integrate([0.0; 3], [0.1, 0.0, 0.0], 0.1).unwrap();
        assert!(imu.delta().velocity[0].abs() < 1e-12);
    }

    #[test]
    fn invalid_samples_are_rejected() {
        let mut imu = ImuPreintegrator::new([0.0; 3], [0.0; 3]);
        assert!(imu.integrate([0.0; 3], [0.0; 3], 0.0).is_err());
        assert!(imu.integrate([f64::NAN, 0.0, 0.0], [0.0; 3], 0.01).is_err());
    }
}
