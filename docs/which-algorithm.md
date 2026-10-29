# Which algorithm should I use?

Pick by what you know before running anything: how well the two clouds
overlap, how good your initial pose is, and whether you have normals.

## Registration

| Situation | Use | Why |
|-----------|-----|-----|
| Clouds already roughly aligned (a few degrees, a few tens of cm), dense overlap | `register::icp::icp` with `PointToPlane` | Fastest convergence on smooth surfaces; needs target normals |
| Same, but no normals and you do not want to estimate them | `PointToPoint` | Closed-form Kabsch each step; slower to converge |
| Aligned but noisy, with some outliers that pass the gate | `PointToPlane` plus `IcpParams::huber_delta` | Robust kernel bounds the influence of outliers |
| Repeated localisation against one fixed map | `PreparedTarget` + `icp_prepared` | Reuses the k-d tree and normals across calls |
| Large or unknown initial error, no reliable guess | `register::bnb` (branch-and-bound over x, y, yaw) or `register::hough` | Global search; ICP alone will converge to a local minimum |
| No initial guess, but distinctive local features | `register::feature_match` (FPFH or SHOT + RANSAC) | Produces an initial pose for ICP to refine |
| Dense smooth surfaces, want a probabilistic fit | `register::ndt` | Voxel-Gaussian map; good with sparse or noisy samples |

Check the result, not only the return value:

- `IcpResult::converged` and `fitness` (fraction of source points inside the gate). Low fitness usually means the gate is too small or the initial pose is wrong.
- `IcpResult::degeneracy(1e-3)`. Weak directions mean the geometry does not constrain the pose, for example a corridor or a flat floor.

## Odometry and SLAM

| Situation | Use |
|-----------|-----|
| LiDAR only, moderate speed | `slam::lidar_odometry::LidarOdometry` (scan-to-map over a window) |
| Monocular camera, scale unknown | `slam::vo_mono` (up to scale) |
| Stereo camera | `slam::vo_stereo` |
| Accumulated drift, loop revisited | `slam::loop_closure` (Scan Context) to detect, then `slam::pose_graph` to correct |

The pose graph is dense. It suits graphs of a few hundred nodes. For larger
graphs, wait for the sparse solver (see `todo.md`, Phase 5.1).

## Mapping

| Goal | Use |
|------|-----|
| 2-D navigation map from a planar lidar | `map::occupancy::OccupancyGrid2D` |
| 3-D occupancy for planning | `map::occupancy::OccupancyGrid3D` |
| Surface reconstruction from many views | `map::tsdf::TsdfMap`, then `extract_mesh()` |
| Collision checking and trajectory costs | `map::sdf::SignedDistanceField` |
| Labelled objects or classes | `map::semantic::SemanticCloud` |

## Features and segmentation

| Goal | Use |
|------|-----|
| Surface normals | `features::normal::estimate_normals` |
| Walls, floors, tabletops | `features::planes::segment_planes` (RANSAC) |
| Keypoints for matching | `features::harris3d::harris3d` |
| Descriptors for matching | `features::fpfh::compute_fpfh` (fast) or `features::shot::compute_shot` (more discriminative) |
| Outliers | `cloud::filter::statistical_outlier_removal`, `radius_filter` |
| Region of interest | `cloud::filter::pass_through` |
