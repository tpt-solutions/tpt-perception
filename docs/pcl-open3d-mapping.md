# Mapping from PCL and Open3D

A reference for users moving from PCL (C++) or Open3D (Python) to
`tpt-perception`. Names are the closest equivalent, not a drop-in API: the
main differences are noted in the right-hand column.

## Input and output

| PCL | Open3D | tpt-perception | Notes |
|-----|--------|----------------|-------|
| `pcl::io::loadPCDFile` | `o3d.io.read_point_cloud` (`.pcd`) | `tpt_percept_io::read_pcd` | Intensity returned in `LoadedCloud::intensity` |
| `pcl::io::loadPLYFile` | `o3d.io.read_point_cloud` (`.ply`) | `tpt_percept_io::read_ply` | ascii and binary little-endian |
| (none; KITTI tools) | (none) | `tpt_percept_io::read_kitti_bin` | |
| `pcl::io::savePCDFile` | `o3d.io.write_point_cloud` | `tpt_percept_io::write_pcd_to` | Writes to any `Write` |
| (none) | (none) | `tpt_percept_io::read_tum` / `write_tum_to` | Trajectories |
| (none) | (none) | `tpt-percept convert` | CLI |

## Filtering and downsampling

| PCL | Open3D | tpt-perception |
|-----|--------|----------------|
| `pcl::VoxelGrid` | `voxel_down_sample` | `tpt_percept_cloud::voxel::voxel_downsample` (centroid per voxel) |
| `pcl::StatisticalOutlierRemoval` | `remove_statistical_outlier` | `cloud::filter::statistical_outlier_removal` |
| `pcl::RadiusOutlierRemoval` | `remove_radius_outlier` | `cloud::filter::radius_filter` |
| `pcl::PassThrough` | `crop` | `cloud::filter::pass_through` |

## Search

| PCL | Open3D | tpt-perception |
|-----|--------|----------------|
| `pcl::KdTreeFLANN` | `KDTreeFlann` | `cloud::kdtree::KdTree` (`nearest`, `knn`, `radius`) |
| `pcl::octree::OctreePointCloudSearch` | `o3d.geometry.KDTreeSearchParamHybrid` (no octree) | `cloud::octree::Octree` (ball and box queries) |

## Features

| PCL | Open3D | tpt-perception |
|-----|--------|----------------|
| `pcl::NormalEstimation` | `estimate_normals` | `features::normal::estimate_normals` |
| `pcl::FPFHEstimation` | `compute_fpfh_feature` | `features::fpfh::compute_fpfh` |
| `pcl::SHOTEstimation` | (none) | `features::shot::compute_shot` |
| `pcl::HarrisKeypoint3D` | (none) | `features::harris3d::harris3d` |
| `pcl::SACSegmentation` (plane) | `segment_plane` | `features::planes::segment_planes` |

## Registration

| PCL | Open3D | tpt-perception | Notes |
|-----|--------|----------------|-------|
| `pcl::IterativeClosestPoint` | `registration_icp` with `TransformationEstimationPointToPoint` | `register::icp::icp` with `IcpVariant::PointToPoint` | Returns `IcpResult` with fitness, rmse and covariance |
| `pcl::IterativeClosestPointWithNormals` | `TransformationEstimationPointToPlane` | `register::icp::icp` with `IcpVariant::PointToPlane` | Needs target normals passed explicitly |
| (none) | (none) | `register::icp::icp_prepared` / `PreparedTarget` | Reuse a target across many calls |
| (none) | (none) | `register::icp::icp_frames` | Frame-typed: `Isometry3<From, To>` |
| `pcl::NormalDistributionsTransform` | (none) | `register::ndt::ndt_register` | |
| `pcl::SampleConsensusInitialAlignment` | `registration_ransac_based_on_feature_matching` | `register::feature_match::register_features` | |
| (none) | `registration_fast_based_on_feature_matching` | `register::bnb` / `register::hough` | Global search |

Key difference: ICP here takes the initial pose explicitly as
`(rotation, translation)` or, through `icp_frames`, as a typed isometry. PCL
and Open3D take a 4×4 matrix.

## Mapping and SLAM

| PCL / Open3D | tpt-perception | Notes |
|--------------|----------------|-------|
| (none; use `pcl::octree` or a voxel grid) | `map::occupancy::OccupancyGrid2D` / `3D` | Log-odds, with ray traversal |
| (none; Kinect Fusion in PCL is separate) | `map::tsdf::TsdfMap` + `extract_mesh()` | Marching tetrahedra |
| (none; g2o / GTSAM external) | `slam::pose_graph::PoseGraph` | Dense; see `docs/which-algorithm.md` |
| (none; LOAM / FAST-LIO external) | `slam::lidar_odometry::LidarOdometry` | Scan-to-map, no IMU yet |

## Things PCL and Open3D have that tpt-perception does not yet

- Point attributes (colour, normals stored in the cloud): normals are passed
  alongside clouds rather than stored in them.
- Single precision storage: all points are `f64`.
- Visualisation: use `rerun` or export to PLY/PCD (planned, Phase 5.4).
- Generalised ICP and IMU preintegration (planned, Phase 5.3).
