//! Python bindings for `tpt-perception`.
//!
//! This crate is the only part of the project that uses a foreign-function
//! interface (PyO3). The algorithm crates stay FFI-free. Build a wheel with
//! `maturin build --release` from this directory, or install into the active
//! environment with `maturin develop --release`.
//!
//! Points cross the boundary as lists of `[x, y, z]` triples. Results come
//! back as dictionaries or tuples, so no NumPy dependency is needed.

#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::voxel::voxel_downsample;
use tpt_percept_features::normal::estimate_normals;
use tpt_percept_features::planes::{segment_planes, RansacParams};
use tpt_percept_io::{read_kitti_bin, read_pcd, read_ply, IoError};
use tpt_percept_register::icp::{icp, IcpParams, IcpVariant};
use tpt_percept_slam::lidar_odometry::{LidarOdometry, LidarOdometryParams};

type Points = Vec<[f64; 3]>;

fn to_value_error<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn io_err(e: IoError) -> PyErr {
    match e {
        IoError::Io(err) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Aligns `source` onto `target` with ICP.
///
/// Returns a dict with `rotation` (3×3 list), `translation`, `fitness`,
/// `rmse`, `iterations` and `converged`. When `point_to_plane` is true,
/// target normals are estimated from `target` first.
#[pyfunction]
#[pyo3(signature = (source, target, max_correspondence_distance=None, point_to_plane=false, max_iterations=50))]
fn align_icp<'py>(
    py: Python<'py>,
    source: Points,
    target: Points,
    max_correspondence_distance: Option<f64>,
    point_to_plane: bool,
    max_iterations: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let source = PointCloud::from_points(source);
    let target = PointCloud::from_points(target);
    let mut params = IcpParams::auto(&source, &target)
        .ok_or_else(|| PyValueError::new_err("source and target must be non-empty"))?;
    if let Some(gate) = max_correspondence_distance {
        params.max_correspondence_distance = gate;
    }
    params.max_iterations = max_iterations;

    let (variant, normals) = if point_to_plane {
        let normals = estimate_normals(&target, 10)
            .ok_or_else(|| PyValueError::new_err("target too small for normals"))?
            .iter()
            .map(|n| n.normal)
            .collect::<Vec<_>>();
        (IcpVariant::PointToPlane, Some(normals))
    } else {
        (IcpVariant::PointToPoint, None)
    };
    let result = icp(
        &source,
        &target,
        normals.as_deref(),
        variant,
        identity(),
        [0.0; 3],
        &params,
    )
    .map_err(to_value_error)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "rotation",
        result
            .rotation
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("translation", result.translation.to_vec())?;
    dict.set_item("fitness", result.fitness)?;
    dict.set_item("rmse", result.rmse)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict)
}

/// Scan-to-map LiDAR odometry as a stateful object.
#[pyclass(name = "LidarOdometry")]
struct LidarOdometryPy {
    inner: LidarOdometry,
}

#[pymethods]
impl LidarOdometryPy {
    /// `map_window` sets how many recent scans form the local map.
    #[new]
    #[pyo3(signature = (map_window=5))]
    fn new(map_window: usize) -> Self {
        LidarOdometryPy {
            inner: LidarOdometry::new(LidarOdometryParams {
                map_window,
                ..LidarOdometryParams::default()
            }),
        }
    }

    /// Registers the next scan. Returns `(rotation, translation, converged)`.
    fn process(&mut self, points: Points) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, bool)> {
        let scan = PointCloud::from_points(points);
        let step = self.inner.process(&scan).map_err(to_value_error)?;
        Ok((
            step.rotation.iter().map(|r| r.to_vec()).collect(),
            step.translation.to_vec(),
            step.converged,
        ))
    }

    /// Scans processed so far.
    fn scans_processed(&self) -> usize {
        self.inner.scans_processed()
    }
}

/// Segments up to `max_planes` planes with seeded RANSAC.
///
/// Returns a list of dicts with `normal`, `d` and `inliers` (indices into
/// `points`).
#[pyfunction(name = "segment_planes")]
#[pyo3(signature = (points, distance_threshold=0.02, max_planes=3, seed=None))]
fn segment_planes_py<'py>(
    py: Python<'py>,
    points: Points,
    distance_threshold: f64,
    max_planes: usize,
    seed: Option<u64>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let cloud = PointCloud::from_points(points);
    let mut params = RansacParams {
        distance_threshold,
        ..RansacParams::default()
    };
    if let Some(seed) = seed {
        params.seed = seed;
    }
    let segments = segment_planes(&cloud, &params, max_planes).map_err(to_value_error)?;
    segments
        .into_iter()
        .map(|seg| {
            let dict = PyDict::new(py);
            dict.set_item("normal", seg.plane.normal.to_vec())?;
            dict.set_item("d", seg.plane.d)?;
            dict.set_item("inliers", seg.inliers)?;
            Ok(dict)
        })
        .collect()
}

/// Voxel-grid downsampling (centroid per voxel).
#[pyfunction]
fn downsample(points: Points, voxel_size: f64) -> PyResult<Points> {
    let cloud = PointCloud::from_points(points);
    let reduced = voxel_downsample(&cloud, voxel_size)
        .ok_or_else(|| PyValueError::new_err("voxel_size must be finite and > 0"))?;
    Ok(reduced.into_points())
}

/// Reads a PCD file and returns its points.
#[pyfunction]
fn read_pcd_points(path: &str) -> PyResult<Points> {
    Ok(read_pcd(path).map_err(io_err)?.cloud.into_points())
}

/// Reads a PLY file and returns its points.
#[pyfunction]
fn read_ply_points(path: &str) -> PyResult<Points> {
    Ok(read_ply(path).map_err(io_err)?.cloud.into_points())
}

/// Reads a KITTI `.bin` scan and returns its points.
#[pyfunction]
fn read_kitti_points(path: &str) -> PyResult<Points> {
    Ok(read_kitti_bin(path).map_err(io_err)?.cloud.into_points())
}

#[pymodule]
fn tpt_perception(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(align_icp, m)?)?;
    m.add_function(wrap_pyfunction!(segment_planes_py, m)?)?;
    m.add_function(wrap_pyfunction!(downsample, m)?)?;
    m.add_function(wrap_pyfunction!(read_pcd_points, m)?)?;
    m.add_function(wrap_pyfunction!(read_ply_points, m)?)?;
    m.add_function(wrap_pyfunction!(read_kitti_points, m)?)?;
    m.add_class::<LidarOdometryPy>()?;
    Ok(())
}
