//! Loosely coupled IMU + GPS fusion with an extended Kalman filter.
//!
//! State `x = [px, py, vx, vy]`. The IMU acceleration is the control input
//! to the prediction, and noisy GPS fixes correct the drift.
//!
//! Run with: `cargo run -p tpt-perception --example ekf_imu_gps_fusion`

use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_perception::fusion::ekf::Ekf;

/// `scale · I` as a dense `n × n` matrix.
fn eye(n: usize, scale: f64) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |i, j| if i == j { scale } else { 0.0 })
}

/// Dense matrix–vector product `m · x`.
fn matvec(m: &DMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    let mut y = DVector::zeros(m.nrows());
    m.mul_vector_into(x, &mut y);
    y
}

/// Deterministic pseudo-noise in [-1, 1] so the output is reproducible.
fn noise(k: u64) -> f64 {
    let x = k
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        >> 11;
    (x as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn main() {
    let dt = 0.1;
    let gps_sigma = 0.5;
    let accel_true = [0.2, 0.0]; // constant acceleration along x (m/s²)

    let mut ekf = Ekf::new(DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0]), eye(4, 1.0));

    // Constant-velocity transition F, and the IMU control matrix B so that
    // x' = F x + B a.
    let f_matrix = DMatrix::from_row_slice(
        4,
        4,
        &[
            1.0, 0.0, dt, 0.0, //
            0.0, 1.0, 0.0, dt, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
    );
    let b = [0.5 * dt * dt, 0.5 * dt * dt, dt, dt];
    let process_noise = eye(4, 0.01);

    // GPS observes position directly.
    let h_matrix = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let gps_noise = eye(2, gps_sigma * gps_sigma);

    // Ground truth for comparison.
    let (mut tx, mut ty, mut tvx, mut tvy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut k = 0u64;
    for step in 1..=100 {
        tx += tvx * dt + 0.5 * accel_true[0] * dt * dt;
        ty += tvy * dt + 0.5 * accel_true[1] * dt * dt;
        tvx += accel_true[0] * dt;
        tvy += accel_true[1] * dt;

        // IMU reading: truth plus noise, used as the control input.
        let imu = [
            accel_true[0] + 0.05 * noise(k),
            accel_true[1] + 0.05 * noise(k + 1),
        ];
        k += 2;
        let control =
            DVector::from_row_slice(&[b[0] * imu[0], b[1] * imu[1], b[2] * imu[0], b[3] * imu[1]]);
        ekf.predict(
            &mut |x: &DVector<f64>| {
                let y = matvec(&f_matrix, x);
                DVector::from_fn(4, |i| y[i] + control[i])
            },
            &f_matrix,
            &process_noise,
        )
        .expect("predict");

        // GPS at 1 Hz.
        if step % 10 == 0 {
            let fix = DVector::from_row_slice(&[
                tx + gps_sigma * noise(k),
                ty + gps_sigma * noise(k + 1),
            ]);
            k += 2;
            let nis = ekf
                .update(
                    &fix,
                    &mut |x: &DVector<f64>| matvec(&h_matrix, x),
                    &h_matrix,
                    &gps_noise,
                )
                .expect("update");
            let s = ekf.state();
            println!(
                "t = {:4.1} s  est [{:+.2}, {:+.2}]  truth [{:+.2}, {:+.2}]  NIS {:.2}",
                step as f64 * dt,
                s[0],
                s[1],
                tx,
                ty,
                nis
            );
        }
    }
}
