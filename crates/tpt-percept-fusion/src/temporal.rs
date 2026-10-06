//! Temporal alignment: timestamp interpolation and latency compensation.
//!
//! Sensors report with different latencies and rates; fusing them requires
//! mapping every measurement to a common timeline. [`PoseTimeline`] stores
//! time-stamped poses and interpolates; [`LatencyEstimator`] tracks the
//! rolling latency between a sensor's own clock and arrival time.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::FusionError;
use tpt_math_units::prelude::Time;
use tpt_math_units::si::time::second;

/// A time-stamped pose (seconds, monotonic clock; `R, t` with
/// `p_world = R·p_body + t`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedPose {
    /// Timestamp (seconds).
    pub time: f64,
    /// Rotation (row-major, proper).
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
}

/// A time-ordered pose buffer with interpolation.
#[derive(Clone, Debug, Default)]
pub struct PoseTimeline {
    samples: Vec<TimedPose>,
    max_samples: usize,
}

impl PoseTimeline {
    /// A timeline keeping the most recent `max_samples` poses.
    pub fn new(max_samples: usize) -> Self {
        PoseTimeline {
            samples: Vec::new(),
            max_samples: max_samples.max(2),
        }
    }

    /// Inserts a pose; samples must arrive with non-decreasing timestamps
    /// (violations return an error).
    pub fn push(&mut self, pose: TimedPose) -> Result<(), FusionError> {
        if let Some(last) = self.samples.last() {
            if pose.time < last.time {
                return Err(FusionError::InvalidParameter(
                    "timestamps must be non-decreasing",
                ));
            }
        }
        self.samples.push(pose);
        while self.samples.len() > self.max_samples {
            self.samples.remove(0);
        }
        Ok(())
    }

    /// Number of buffered poses.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Time of the oldest buffered pose.
    pub fn oldest_time(&self) -> Option<f64> {
        self.samples.first().map(|p| p.time)
    }

    /// Time of the newest buffered pose.
    pub fn newest_time(&self) -> Option<f64> {
        self.samples.last().map(|p| p.time)
    }

    /// Interpolates (or extrapolates) the pose at `time`.
    ///
    /// Linear translation interpolation and rotation-component
    /// interpolation followed by orthonormalisation — accurate for the
    /// small inter-sample rotations typical of 10-100 Hz sensors (documented
    /// contract; for large inter-sample rotations use slerp, not provided
    /// here).
    pub fn interpolate(&self, time: f64) -> Result<TimedPose, FusionError> {
        let n = self.samples.len();
        if n == 0 {
            return Err(FusionError::InsufficientData("empty timeline"));
        }
        if n == 1 {
            let p = self.samples[0];
            return Ok(TimedPose { time, ..p });
        }
        // Bracketing samples.
        let mut hi = 0usize;
        while hi < n && self.samples[hi].time < time {
            hi += 1;
        }
        let (a, b, alpha) = match hi {
            0 => {
                let (a, b) = (self.samples[0], self.samples[1]);
                let span = b.time - a.time;
                (a, b, (time - a.time) / span.max(1e-12))
            }
            k if k >= n => {
                let (a, b) = (self.samples[n - 2], self.samples[n - 1]);
                let span = b.time - a.time;
                (a, b, 1.0 + (time - b.time) / span.max(1e-12))
            }
            k => {
                let (a, b) = (self.samples[k - 1], self.samples[k]);
                let span = b.time - a.time;
                (a, b, (time - a.time) / span.max(1e-12))
            }
        };
        let al = alpha.clamp(-4.0, 5.0); // bounded extrapolation
        let mut rotation = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                rotation[r][c] = a.rotation[r][c] * (1.0 - al) + b.rotation[r][c] * al;
            }
        }
        // Gram-Schmidt orthonormalisation (columns), keep determinant +1.
        let mut cols = [[0.0; 3]; 3];
        for (j, col) in cols.iter_mut().enumerate() {
            *col = [rotation[0][j], rotation[1][j], rotation[2][j]];
        }
        for j in 0..3 {
            for i in 0..j {
                let d = dot3(&cols[i], &cols[j]);
                for r in 0..3 {
                    cols[j][r] -= cols[i][r] * d;
                }
            }
            let nrm = (cols[j][0] * cols[j][0] + cols[j][1] * cols[j][1] + cols[j][2] * cols[j][2])
                .sqrt();
            if nrm > 1e-12 {
                for r in 0..3 {
                    cols[j][r] /= nrm;
                }
            }
        }
        let det = {
            let m = &cols;
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        if det < 0.0 {
            for r in 0..3 {
                cols[2][r] = -cols[2][r];
            }
        }
        for j in 0..3 {
            for r in 0..3 {
                rotation[r][j] = cols[j][r];
            }
        }
        let translation = [
            a.translation[0] * (1.0 - al) + b.translation[0] * al,
            a.translation[1] * (1.0 - al) + b.translation[1] * al,
            a.translation[2] * (1.0 - al) + b.translation[2] * al,
        ];
        Ok(TimedPose {
            time,
            rotation,
            translation,
        })
    }
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rolling latency estimate between a sensor's own timestamp and the
/// processing-arrival time.
#[derive(Clone, Copy, Debug, Default)]
pub struct LatencyEstimator {
    mean: f64,
    m2: f64,
    count: u64,
}

impl LatencyEstimator {
    /// Records an observation: the sensor's timestamp and the arrival time
    /// on the processing clock.
    pub fn observe(&mut self, sensor_time: Time, arrival_time: Time) {
        let latency = arrival_time.get::<second>() - sensor_time.get::<second>();
        self.count += 1;
        let delta = latency - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = latency - self.mean;
        self.m2 += delta * delta2;
    }

    /// Estimated mean latency (seconds), if any observations exist.
    pub fn mean(&self) -> Option<Time> {
        if self.count == 0 {
            None
        } else {
            Some(Time::new::<second>(self.mean))
        }
    }

    /// Latency standard deviation (seconds).
    pub fn std_dev(&self) -> Option<Time> {
        if self.count < 2 {
            None
        } else {
            Some(Time::new::<second>(
                (self.m2 / (self.count - 1) as f64).sqrt(),
            ))
        }
    }

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(t: f64, x: f64, yaw: f64) -> TimedPose {
        let (s, c) = yaw.sin_cos();
        TimedPose {
            time: t,
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [x, 0.0, 0.0],
        }
    }

    #[test]
    fn interpolates_midpoints() {
        let mut tl = PoseTimeline::new(10);
        tl.push(pose(0.0, 0.0, 0.0)).unwrap();
        tl.push(pose(1.0, 2.0, 0.2)).unwrap();
        let mid = tl.interpolate(0.5).unwrap();
        assert!((mid.translation[0] - 1.0).abs() < 1e-12);
        let s = (0.1_f64).sin();
        let c = (0.1_f64).cos();
        assert!((mid.rotation[0][0] - c).abs() < 1e-9);
        assert!((mid.rotation[1][0] - s).abs() < 1e-9);
    }

    #[test]
    fn extrapolates_bounded() {
        let mut tl = PoseTimeline::new(10);
        tl.push(pose(0.0, 0.0, 0.0)).unwrap();
        tl.push(pose(1.0, 2.0, 0.0)).unwrap();
        let ex = tl.interpolate(1.5).unwrap();
        assert!((ex.translation[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn interpolation_is_orthonormal() {
        let mut tl = PoseTimeline::new(10);
        tl.push(pose(0.0, 0.0, 0.8)).unwrap();
        tl.push(pose(1.0, 0.5, 1.2)).unwrap();
        let mid = tl.interpolate(0.5).unwrap();
        let r = &mid.rotation;
        for i in 0..3 {
            let n = r[0][i] * r[0][i] + r[1][i] * r[1][i] + r[2][i] * r[2][i];
            assert!((n - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn decreasing_timestamps_rejected() {
        let mut tl = PoseTimeline::new(10);
        tl.push(pose(1.0, 0.0, 0.0)).unwrap();
        assert!(tl.push(pose(0.5, 0.0, 0.0)).is_err());
    }

    #[test]
    fn latency_statistics() {
        use tpt_math_units::si::time::second;
        let mut est = LatencyEstimator::default();
        for i in 0..100u64 {
            est.observe(
                Time::new::<second>(i as f64),
                Time::new::<second>(i as f64 + 0.03 + 0.005 * (i as f64 * 0.7).sin()),
            );
        }
        let mean = est.mean().unwrap().get::<second>();
        assert!((mean - 0.03).abs() < 0.003, "mean {mean}");
        let std = est.std_dev().unwrap().get::<second>();
        assert!(std > 0.001 && std < 0.01, "std {std}");
        assert_eq!(est.count(), 100);
    }
}
