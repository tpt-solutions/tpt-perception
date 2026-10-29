//! Motion deskew for spinning LiDAR.
//!
//! A spinning sensor takes a full revolution to capture one scan, so the
//! sensor moves during the capture. Points taken early and late in the scan
//! are therefore expressed in slightly different sensor poses. Deskew
//! re-expresses every point in the pose the sensor had at the end of the scan,
//! assuming constant velocity across the revolution.

use crate::error::SlamError;
use crate::pose_graph::Pose;
use tpt_percept_cloud::attributes::AttributedCloud;
use tpt_percept_cloud::cloud::PointCloud;

/// Re-expresses each point of `scan` in the sensor frame at the end of the
/// scan.
///
/// `motion` is the sensor's pose change over the full scan, from its start
/// pose to its end pose. The scan's timestamp column gives each point's
/// fraction of the revolution in `[0, 1]`; values outside that range are
/// clamped.
///
/// # Errors
///
/// Returns [`SlamError::InsufficientData`] when the scan has no timestamp
/// column, and [`SlamError::InvalidParameter`] when a timestamp is not finite.
pub fn deskew(scan: &AttributedCloud, motion: &Pose) -> Result<PointCloud, SlamError> {
    let times = scan
        .timestamp()
        .ok_or(SlamError::InsufficientData("scan has no timestamp column"))?;
    if times.iter().any(|t| !t.is_finite()) {
        return Err(SlamError::InvalidParameter("timestamps must be finite"));
    }
    // ξ with exp(ξ) = motion. The pose at fraction t is exp(t·ξ), so the
    // correction from time t to the end pose is exp((t − 1)·ξ).
    let xi = motion.log();
    let points = scan
        .cloud()
        .points()
        .iter()
        .zip(times)
        .map(|(&p, &t)| {
            let s = t.clamp(0.0, 1.0) - 1.0;
            let pose = Pose::exp(&xi.map(|v| v * s));
            apply(&pose, p)
        })
        .collect();
    Ok(PointCloud::from_points(points))
}

fn apply(pose: &Pose, p: [f64; 3]) -> [f64; 3] {
    let r = pose.rotation;
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + pose.translation[0],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + pose.translation[1],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + pose.translation[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Builds a scan of static structure `world` taken while the sensor moves
    /// along `motion`, with point `i` captured at fraction `times[i]`.
    fn captured(world: &[[f64; 3]], motion: &Pose, times: &[f64]) -> AttributedCloud {
        let xi = motion.log();
        let pts: Vec<[f64; 3]> = world
            .iter()
            .zip(times)
            .map(|(&p, &t)| {
                // The sensor pose at fraction t is exp(t·ξ); a static point is
                // seen at exp(t·ξ)⁻¹ · p in the sensor frame.
                apply(&Pose::exp(&xi.map(|v| v * t)).inverse(), p)
            })
            .collect();
        AttributedCloud::new(PointCloud::from_points(pts))
            .with_timestamp(times.to_vec())
            .unwrap()
    }

    #[test]
    fn deskew_recovers_end_of_scan_frame() {
        // A 0.3 rad yaw plus travel over the revolution.
        let motion = Pose {
            rotation: [
                [0.955336, -0.295520, 0.0],
                [0.295520, 0.955336, 0.0],
                [0.0, 0.0, 1.0],
            ],
            translation: [1.0, 0.1, 0.0],
        };
        let world: Vec<[f64; 3]> = (0..40)
            .map(|i| {
                [
                    3.0 + 0.1 * i as f64,
                    (i % 7) as f64 - 3.0,
                    0.5 * (i % 3) as f64,
                ]
            })
            .collect();
        let times: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
        let scan = captured(&world, &motion, &times);

        let corrected = deskew(&scan, &motion).unwrap();
        let end = motion.inverse();
        for (i, &p) in world.iter().enumerate() {
            let want = apply(&end, p);
            let got = corrected.points()[i];
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-3,
                    "point {i}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn deskew_needs_timestamps() {
        let scan = AttributedCloud::new(PointCloud::from_points(vec![[1.0, 0.0, 0.0]]));
        assert!(matches!(
            deskew(&scan, &Pose::identity()),
            Err(SlamError::InsufficientData(_))
        ));
    }

    #[test]
    fn exp_and_log_round_trip() {
        let xi = [0.2, -0.1, 0.4, 0.3, -0.2, 0.25];
        let back = Pose::exp(&xi).log();
        for k in 0..6 {
            assert!((back[k] - xi[k]).abs() < 1e-9, "{back:?}");
        }
    }
}
