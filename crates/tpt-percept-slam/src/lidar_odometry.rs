//! LiDAR odometry: frame-to-frame scan registration with constant-velocity
//! motion prediction.
//!
//! [`LidarOdometry`] keeps the previous scan and the previous inter-frame
//! transform; each new scan is aligned with point-to-plane ICP seeded by the
//! predicted motion. Poses accumulate in the odometry frame.

use alloc::vec::Vec;

use crate::error::SlamError;
use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_features::normal::estimate_normals;
use tpt_percept_register::icp::{icp, IcpParams, IcpVariant};

/// LiDAR odometry configuration.
#[derive(Clone, Copy, Debug)]
pub struct LidarOdometryParams {
    /// ICP configuration for scan matching.
    pub icp: IcpParams,
    /// Seed ICP with the previous inter-frame transform (constant-velocity
    /// prediction) instead of the identity.
    pub predict_motion: bool,
    /// Neighbourhood size for target-normal estimation.
    pub normal_k: usize,
}

impl Default for LidarOdometryParams {
    fn default() -> Self {
        LidarOdometryParams {
            icp: IcpParams {
                max_iterations: 30,
                max_correspondence_distance: 0.5,
                tolerance: 1e-6,
                min_correspondences: 20,
            },
            predict_motion: true,
            normal_k: 10,
        }
    }
}

/// Incremental LiDAR odometry over a scan stream.
pub struct LidarOdometry {
    params: LidarOdometryParams,
    previous: Option<PointCloud>,
    previous_motion: Option<([[f64; 3]; 3], [f64; 3])>,
    pose: ([[f64; 3]; 3], [f64; 3]),
    scans_processed: usize,
}

/// One odometry step's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct OdometryStep {
    /// Current pose in the odometry frame: `point_in_odom ≈ R · scan + t`.
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
    /// Whether the scan-to-scan ICP converged.
    pub converged: bool,
}

impl LidarOdometry {
    /// Creates an odometry pipeline starting at the identity pose.
    pub fn new(params: LidarOdometryParams) -> Self {
        LidarOdometry {
            params,
            previous: None,
            previous_motion: None,
            pose: (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
            ),
            scans_processed: 0,
        }
    }

    /// Scans processed so far.
    pub fn scans_processed(&self) -> usize {
        self.scans_processed
    }

    /// Current odometry pose `(R, t)`.
    pub fn pose(&self) -> ([[f64; 3]; 3], [f64; 3]) {
        self.pose
    }

    /// Registers the next scan and updates the accumulated pose.
    pub fn process(&mut self, scan: &PointCloud) -> Result<OdometryStep, SlamError> {
        match &self.previous {
            None => {
                self.previous = Some(scan.clone());
                self.scans_processed += 1;
                Ok(OdometryStep {
                    rotation: self.pose.0,
                    translation: self.pose.1,
                    converged: true,
                })
            }
            Some(prev) => {
                let normals = estimate_normals(prev, self.params.normal_k).ok_or(
                    SlamError::InsufficientData("previous scan too small for normals"),
                )?;
                let normal_refs: Vec<[f64; 3]> = normals.iter().map(|n| n.normal).collect();

                let (init_r, init_t) = if self.params.predict_motion {
                    self.previous_motion.unwrap_or((
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [0.0; 3],
                    ))
                } else {
                    (
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [0.0; 3],
                    )
                };

                let res = icp(
                    scan,
                    prev,
                    Some(&normal_refs),
                    IcpVariant::PointToPlane,
                    init_r,
                    init_t,
                    &self.params.icp,
                )?;

                // Accumulate: pose ← motion ∘ pose (motion maps new scan
                // into the previous scan's frame = current odometry frame).
                let (pr, pt) = self.pose;
                let motion_r = res.rotation;
                let motion_t = res.translation;
                let new_r = mul3(&motion_r, &pr);
                let new_t = [
                    motion_r[0][0] * pt[0]
                        + motion_r[0][1] * pt[1]
                        + motion_r[0][2] * pt[2]
                        + motion_t[0],
                    motion_r[1][0] * pt[0]
                        + motion_r[1][1] * pt[1]
                        + motion_r[1][2] * pt[2]
                        + motion_t[1],
                    motion_r[2][0] * pt[0]
                        + motion_r[2][1] * pt[1]
                        + motion_r[2][2] * pt[2]
                        + motion_t[2],
                ];
                self.pose = (new_r, new_t);
                self.previous_motion = Some((motion_r, motion_t));
                self.previous = Some(scan.clone());
                self.scans_processed += 1;

                Ok(OdometryStep {
                    rotation: new_r,
                    translation: new_t,
                    converged: res.converged,
                })
            }
        }
    }
}

fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A structured 3-D scene: bent plane + pillars, dense enough for
    /// point-to-plane ICP.
    fn scene(offset: [f64; 3], yaw: f64) -> PointCloud {
        let mut c = PointCloud::new();
        let (s, co) = yaw.sin_cos();
        for x in -20..=20i32 {
            for y in -20..=20i32 {
                let xf = x as f64 * 0.1;
                let yf = y as f64 * 0.1;
                // ground with gentle relief
                let p = [xf, yf, 0.05 * (xf * 1.3).sin() + 0.05 * (yf * 0.9).cos()];
                for h in 0..6i32 {
                    let q = [p[0], p[1], p[2] + h as f64 * 0.25];
                    // rotate by yaw and translate
                    let rq = [co * q[0] - s * q[1], s * q[0] + co * q[1], q[2]];
                    c.push([rq[0] + offset[0], rq[1] + offset[1], rq[2] + offset[2]]);
                }
            }
        }
        c
    }

    #[test]
    fn odometry_tracks_small_motion() {
        let mut odom = LidarOdometry::new(LidarOdometryParams::default());
        let first = scene([0.0; 3], 0.0);
        odom.process(&first).unwrap();

        // Sensor moves +0.05 m in x, yaws 0.01 rad: the new scan expressed
        // in the previous frame is transformed by the inverse motion.
        let (s, c) = (0.01f64).sin_cos();
        let inv_r = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
        let inv_t = [-0.05, 0.0, 0.0];
        let second = first.transformed_rigid(inv_r, inv_t);
        let step = odom.process(&second).unwrap();

        assert!(step.converged);
        // Accumulated pose ≈ the forward motion (+x, small yaw).
        assert!(
            (step.translation[0] - 0.05).abs() < 0.03,
            "t = {:?}",
            step.translation
        );
        let yaw_recovered = step.rotation[1][0].atan2(step.rotation[0][0]);
        assert!((yaw_recovered - 0.01).abs() < 0.02, "yaw {yaw_recovered}");
    }

    #[test]
    fn first_scan_is_identity() {
        let mut odom = LidarOdometry::new(LidarOdometryParams::default());
        let step = odom.process(&scene([0.0; 3], 0.0)).unwrap();
        assert!(step.converged);
        assert_eq!(step.translation, [0.0; 3]);
        assert_eq!(odom.scans_processed(), 1);
    }
}
