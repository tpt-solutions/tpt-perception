//! Streaming point cloud processing for real-time LiDAR data.
//!
//! LiDAR sensors deliver sweeps at 5–30 Hz; perception pipelines accumulate
//! them into windows (for denser feature extraction), motion-compensate them,
//! and push them through chains of filters. This module provides the
//! plumbing:
//!
//! * [`WindowAggregator`] — a sliding time/count window over incoming scans
//!   emitting merged clouds;
//! * [`StreamStage`] — a composable per-scan transformation, with
//!   implementations for voxel downsampling and pass-through cropping;
//! * [`Pipeline`] — chains stages so `scan → aggregate → filter → out` reads
//!   top-to-bottom.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::cloud::PointCloud;
use crate::filter::{pass_through, Axis};
use crate::voxel::voxel_downsample;

/// A per-scan transformation in a streaming pipeline.
pub trait StreamStage {
    /// Transform one incoming scan.
    fn process(&mut self, scan: &PointCloud) -> PointCloud;
}

/// Voxel-downsampling stage (see [`voxel_downsample`]).
#[derive(Clone, Copy, Debug)]
pub struct VoxelStage {
    /// Voxel edge length (metres).
    pub voxel_size: f64,
}

impl StreamStage for VoxelStage {
    fn process(&mut self, scan: &PointCloud) -> PointCloud {
        voxel_downsample(scan, self.voxel_size).unwrap_or_default()
    }
}

/// Pass-through cropping stage (see [`pass_through`]).
#[derive(Clone, Copy, Debug)]
pub struct CropStage {
    /// Axis to crop along.
    pub axis: Axis,
    /// Inclusive lower bound (metres).
    pub min: f64,
    /// Exclusive upper bound (metres).
    pub max: f64,
}

impl StreamStage for CropStage {
    fn process(&mut self, scan: &PointCloud) -> PointCloud {
        let idx = pass_through(scan, self.axis, self.min, self.max).unwrap_or_default();
        scan.select(&idx)
    }
}

/// A sliding window over the most recent scans.
///
/// Scans enter via [`WindowAggregator::push`]; at any time
/// [`WindowAggregator::snapshot`] returns the merged cloud of everything
/// currently inside the window (by count or by wall time, whichever bound
/// is hit first). This is the standard "accumulate a few sweeps before
/// feature extraction" pattern.
#[derive(Clone, Debug)]
pub struct WindowAggregator {
    scans: VecDeque<PointCloud>,
    times: VecDeque<f64>,
    max_scans: usize,
    max_age: Option<f64>,
}

impl WindowAggregator {
    /// A window holding at most `max_scans` scans.
    ///
    /// Pass `max_age: Some(seconds)` to additionally evict scans older than
    /// `now - max_age` (scan timestamps in seconds, monotonically
    /// increasing).
    pub fn new(max_scans: usize, max_age: Option<f64>) -> Self {
        WindowAggregator {
            scans: VecDeque::new(),
            times: VecDeque::new(),
            max_scans: max_scans.max(1),
            max_age,
        }
    }

    /// Push a scan observed at time `timestamp` (seconds).
    pub fn push(&mut self, scan: PointCloud, timestamp: f64) {
        self.scans.push_back(scan);
        self.times.push_back(timestamp);
        while self.scans.len() > self.max_scans {
            self.scans.pop_front();
            self.times.pop_front();
        }
        if let Some(age) = self.max_age {
            while let Some(&t) = self.times.front() {
                if timestamp - t <= age {
                    break;
                }
                self.scans.pop_front();
                self.times.pop_front();
            }
        }
    }

    /// Number of scans currently buffered.
    pub fn scan_count(&self) -> usize {
        self.scans.len()
    }

    /// Merge the buffered scans into one cloud (allocated fresh each call).
    pub fn snapshot(&self) -> PointCloud {
        let total: usize = self.scans.iter().map(|s| s.len()).sum();
        let mut out = PointCloud::with_capacity(total);
        for scan in &self.scans {
            out.extend_from(scan);
        }
        out
    }

    /// Drop all buffered scans (e.g. after a tracking loss).
    pub fn clear(&mut self) {
        self.scans.clear();
        self.times.clear();
    }
}

/// A chain of stages applied to every scan.
#[derive(Default)]
pub struct Pipeline {
    stages: Vec<Box<dyn StreamStage>>,
}

impl Pipeline {
    /// An empty pipeline (identity).
    pub fn new() -> Self {
        Pipeline { stages: Vec::new() }
    }

    /// Append a stage.
    pub fn push_stage(&mut self, stage: impl StreamStage + 'static) -> &mut Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Run a scan through every stage in order.
    pub fn run(&mut self, scan: &PointCloud) -> PointCloud {
        let mut current = scan.clone();
        for stage in &mut self.stages {
            current = stage.process(&current);
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(offset: f64, n: usize) -> PointCloud {
        PointCloud::from_points(
            (0..n)
                .map(|i| [offset + i as f64 * 0.01, 0.0, 0.0])
                .collect(),
        )
    }

    #[test]
    fn window_evicts_by_count() {
        let mut w = WindowAggregator::new(3, None);
        for i in 0..5 {
            w.push(scan(i as f64, 10), i as f64);
        }
        assert_eq!(w.scan_count(), 3);
        let snap = w.snapshot();
        assert_eq!(snap.len(), 30);
        // Oldest scans evicted: first x should be ~2.0.
        let min_x = snap
            .points()
            .iter()
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min);
        assert!((min_x - 2.0).abs() < 1e-9);
    }

    #[test]
    fn window_evicts_by_age() {
        let mut w = WindowAggregator::new(10, Some(0.5));
        w.push(scan(0.0, 10), 0.0);
        w.push(scan(1.0, 10), 1.0);
        assert_eq!(w.scan_count(), 1); // t=0 evicted at now=1.0
        assert_eq!(w.snapshot().len(), 10);
    }

    #[test]
    fn pipeline_downsamples_then_crops() {
        let mut c = PointCloud::new();
        for x in 0..100 {
            c.push([x as f64 * 0.1, 0.0, 0.0]);
        }
        let mut p = Pipeline::new();
        p.push_stage(VoxelStage { voxel_size: 1.0 });
        p.push_stage(CropStage {
            axis: Axis::X,
            min: 2.0,
            max: 5.0,
        });
        let out = p.run(&c);
        // Voxel centroids at 0.45, 1.45, ... 9.45; crop keeps 2.45, 3.45, 4.45.
        assert_eq!(out.len(), 3);
        assert!((out.get(0).unwrap()[0] - 2.45).abs() < 1e-9);
    }
}
