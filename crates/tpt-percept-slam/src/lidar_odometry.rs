//! LiDAR odometry: scan-to-map registration with constant-velocity motion
//! prediction.
//!
//! [`LidarOdometry`] keeps a sliding window of recent scans expressed in the
//! odometry frame, together with their surface normals. Each new scan is
//! aligned against that local map with point-to-plane ICP, seeded by the
//! predicted motion. Matching against a window instead of only the previous
//! scan reduces drift, and the normals are estimated once per scan and then
//! rotated into the odometry frame rather than re-estimated on every step.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::error::SlamError;
use crate::pose_graph::Pose;
use tpt_percept_cloud::cloud::{FrameCloud, PointCloud};
use tpt_percept_core::frame::Frame;
use tpt_percept_core::iso::{Isometry3, Rotation3};
use tpt_percept_features::normal::estimate_normals;
use tpt_percept_register::icp::{icp_prepared, IcpParams, IcpVariant, PreparedTarget};

/// LiDAR odometry configuration.
#[derive(Clone, Copy, Debug)]
pub struct LidarOdometryParams {
    /// ICP configuration for scan-to-map matching.
    pub icp: IcpParams,
    /// Seed ICP with the previous inter-frame transform (constant-velocity
    /// prediction) instead of the current pose.
    pub predict_motion: bool,
    /// Neighbourhood size for per-scan normal estimation.
    pub normal_k: usize,
    /// Number of most recent scans kept in the local map. `1` reproduces
    /// scan-to-scan matching.
    pub map_window: usize,
}

impl Default for LidarOdometryParams {
    fn default() -> Self {
        LidarOdometryParams {
            icp: IcpParams {
                max_iterations: 30,
                max_correspondence_distance: 0.5,
                tolerance: 1e-6,
                min_correspondences: 20,
                huber_delta: 0.0,
            },
            predict_motion: true,
            normal_k: 10,
            map_window: 5,
        }
    }
}

/// One scan stored in the local map, already in the odometry frame.
#[derive(Clone, Debug)]
struct MapScan {
    points: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
}

/// Incremental LiDAR odometry over a scan stream.
pub struct LidarOdometry {
    params: LidarOdometryParams,
    /// Sliding window of recent scans, oldest first.
    map: Vec<MapScan>,
    /// Sensor-to-odometry pose.
    pose: Pose,
    /// Last inter-frame motion `ΔP = P_{k−1}⁻¹ · P_k`.
    previous_motion: Option<Pose>,
    scans_processed: usize,
}

/// One odometry step's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct OdometryStep {
    /// Current pose in the odometry frame: `point_in_odom ≈ R · scan + t`.
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
    /// Whether the scan-to-map ICP converged.
    pub converged: bool,
}

impl LidarOdometry {
    /// Creates an odometry pipeline starting at the identity pose.
    pub fn new(params: LidarOdometryParams) -> Self {
        LidarOdometry {
            params,
            map: Vec::new(),
            pose: Pose::identity(),
            previous_motion: None,
            scans_processed: 0,
        }
    }

    /// Scans processed so far.
    pub fn scans_processed(&self) -> usize {
        self.scans_processed
    }

    /// Scans currently held in the local map.
    pub fn map_scans(&self) -> usize {
        self.map.len()
    }

    /// Current odometry pose `(R, t)`.
    pub fn pose(&self) -> ([[f64; 3]; 3], [f64; 3]) {
        (self.pose.rotation, self.pose.translation)
    }

    /// Registers the next scan and updates the accumulated pose.
    pub fn process(&mut self, scan: &PointCloud) -> Result<OdometryStep, SlamError> {
        if scan.is_empty() {
            return Err(SlamError::InsufficientData("empty scan"));
        }
        let scan_normals = estimate_normals(scan, self.params.normal_k)
            .ok_or(SlamError::InsufficientData("scan too small for normals"))?;
        let local_normals: Vec<[f64; 3]> = scan_normals.iter().map(|n| n.normal).collect();

        if self.map.is_empty() {
            self.push_to_map(scan, &local_normals, self.pose);
            self.scans_processed += 1;
            return Ok(OdometryStep {
                rotation: self.pose.rotation,
                translation: self.pose.translation,
                converged: true,
            });
        }

        let predicted = match (self.params.predict_motion, self.previous_motion) {
            (true, Some(motion)) => self.pose.compose(&motion),
            _ => self.pose,
        };

        let (map_points, map_normals) = self.flatten_map();
        let target_cloud = PointCloud::from_points(map_points);
        let target = PreparedTarget::new(target_cloud, Some(map_normals))?;
        let res = icp_prepared(
            scan,
            &target,
            IcpVariant::PointToPlane,
            predicted.rotation,
            predicted.translation,
            &self.params.icp,
        )?;

        // `res` maps the scan (sensor frame) onto the map (odometry frame).
        let new_pose = Pose {
            rotation: res.rotation,
            translation: res.translation,
        };
        let motion = self.pose.inverse().compose(&new_pose);

        self.push_to_map(scan, &local_normals, new_pose);
        self.previous_motion = Some(motion);
        self.pose = new_pose;
        self.scans_processed += 1;

        Ok(OdometryStep {
            rotation: new_pose.rotation,
            translation: new_pose.translation,
            converged: res.converged,
        })
    }

    /// Transforms a scan and its normals into the odometry frame at `pose`,
    /// appends it to the map, and evicts the oldest scan beyond the window.
    fn push_to_map(&mut self, scan: &PointCloud, normals: &[[f64; 3]], pose: Pose) {
        let points = scan
            .points()
            .iter()
            .map(|&p| apply_pose(&pose, p))
            .collect();
        let normals = normals
            .iter()
            .map(|&n| mul_mat_vec(&pose.rotation, n))
            .collect();
        self.map.push(MapScan { points, normals });
        let window = self.params.map_window.max(1);
        while self.map.len() > window {
            self.map.remove(0);
        }
    }

    /// Concatenates the window into one point and normal list.
    fn flatten_map(&self) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let total: usize = self.map.iter().map(|m| m.points.len()).sum();
        let mut points = Vec::with_capacity(total);
        let mut normals = Vec::with_capacity(total);
        for m in &self.map {
            points.extend_from_slice(&m.points);
            normals.extend_from_slice(&m.normals);
        }
        (points, normals)
    }
}

fn apply_pose(pose: &Pose, p: [f64; 3]) -> [f64; 3] {
    let r = mul_mat_vec(&pose.rotation, p);
    [
        r[0] + pose.translation[0],
        r[1] + pose.translation[1],
        r[2] + pose.translation[2],
    ]
}

fn mul_mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Result of one [`FrameLidarOdometry::process`] call.
#[derive(Clone, Debug)]
pub struct FrameOdometryStep<S: Frame, O: Frame> {
    /// Sensor-to-odometry transform for this scan.
    pub pose: Isometry3<S, O>,
    /// Whether the scan-to-map ICP converged.
    pub converged: bool,
}

/// [`LidarOdometry`] with compile-time frames.
///
/// Scans arrive as `FrameCloud<S>` (sensor frame `S`) and the pose is an
/// `Isometry3<S, O>` into the odometry frame `O`. Passing a cloud from a
/// different frame, or using the pose in the wrong direction, is a compile
/// error rather than a silent bug.
pub struct FrameLidarOdometry<S: Frame, O: Frame> {
    inner: LidarOdometry,
    _frames: PhantomData<(S, O)>,
}

impl<S: Frame, O: Frame> FrameLidarOdometry<S, O> {
    /// Creates an odometry pipeline starting at the identity pose.
    pub fn new(params: LidarOdometryParams) -> Self {
        FrameLidarOdometry {
            inner: LidarOdometry::new(params),
            _frames: PhantomData,
        }
    }

    /// Current sensor-to-odometry transform.
    pub fn pose(&self) -> Isometry3<S, O> {
        let (r, t) = self.inner.pose();
        Isometry3::from_parts_unchecked(t, Rotation3::from_matrix_unchecked(r))
    }

    /// Registers the next scan and returns the updated transform.
    pub fn process(&mut self, scan: &FrameCloud<S>) -> Result<FrameOdometryStep<S, O>, SlamError> {
        let step = self.inner.process(scan.cloud())?;
        Ok(FrameOdometryStep {
            pose: self.pose(),
            converged: step.converged,
        })
    }

    /// The untyped odometry this wraps.
    pub fn inner(&self) -> &LidarOdometry {
        &self.inner
    }
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
        assert_eq!(odom.map_scans(), 1);
    }

    #[test]
    fn map_window_is_bounded() {
        let mut odom = LidarOdometry::new(LidarOdometryParams {
            map_window: 3,
            ..LidarOdometryParams::default()
        });
        let first = scene([0.0; 3], 0.0);
        odom.process(&first).unwrap();
        let (s, c) = (0.01f64).sin_cos();
        let inv_r = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut scan = first.clone();
        for _ in 0..5 {
            scan = scan.transformed_rigid(inv_r, [-0.02, 0.0, 0.0]);
            odom.process(&scan).unwrap();
        }
        assert_eq!(odom.map_scans(), 3);
        assert_eq!(odom.scans_processed(), 6);
    }

    #[test]
    fn scan_to_map_survives_a_noisy_window() {
        // Constant-velocity drive along x. Each scan is shifted by exactly
        // 0.05 m per step; the odometry should track the cumulative motion.
        let mut odom = LidarOdometry::new(LidarOdometryParams::default());
        let base = scene([0.0; 3], 0.0);
        let steps = 4;
        let mut last = None;
        for k in 0..=steps {
            let scan = base.transformed_rigid(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [-0.05 * k as f64, 0.0, 0.0],
            );
            last = Some(odom.process(&scan).unwrap());
        }
        let step = last.unwrap();
        let expected = 0.05 * steps as f64;
        assert!(
            (step.translation[0] - expected).abs() < 0.05,
            "t = {:?}, expected x ≈ {expected}",
            step.translation
        );
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use alloc::vec::Vec;
    use tpt_percept_core::frame::{Lidar, Odometry};

    #[test]
    fn typed_odometry_matches_untyped() {
        let mut points = Vec::new();
        for x in -15..=15i32 {
            for y in -15..=15i32 {
                let (xf, yf) = (x as f64 * 0.1, y as f64 * 0.1);
                points.push([xf, yf, 0.05 * (xf * 1.3).sin()]);
                points.push([xf, yf, 0.6]);
            }
        }
        let base = PointCloud::from_points(points);
        let second = base.transformed_rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-0.05, 0.0, 0.0],
        );

        let mut plain = LidarOdometry::new(LidarOdometryParams::default());
        let mut typed =
            FrameLidarOdometry::<Lidar<0>, Odometry>::new(LidarOdometryParams::default());
        plain.process(&base).unwrap();
        typed.process(&FrameCloud::from_cloud(base)).unwrap();
        let step = typed
            .process(&FrameCloud::from_cloud(second.clone()))
            .unwrap();
        let untyped = plain.process(&second).unwrap();

        let (r, t) = step.pose.into_raw();
        assert_eq!(r, untyped.rotation);
        assert_eq!(t, untyped.translation);
        assert!(step.converged);
    }
}
