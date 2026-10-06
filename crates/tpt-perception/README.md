# tpt-perception

Umbrella crate: feature-gated re-exports of the full perception stack.

| Feature | Sub-crate | Contents |
|---------|-----------|----------|
| `core` | tpt-percept-core | frame-typed geometry, rigid transforms, twists, Kabsch/Umeyama |
| `cloud` | tpt-percept-cloud | point clouds, voxel grids, filters, k-d/octree/R-tree, streaming |
| `features` | tpt-percept-features | normals, Harris 3D, edges, RANSAC planes, FPFH, SHOT, learned hooks |
| `register` | tpt-percept-register | ICP, NDT, feature matching, branch-and-bound, Hough loop closure |
| `slam` | tpt-percept-slam | visual/LiDAR odometry, Scan Context loop closure, pose graph |
| `fusion` | tpt-percept-fusion | temporal alignment, EKF/UKF, hand–eye calibration, covariance intersection |
| `map` | tpt-percept-map | occupancy grids, TSDF, SDF, semantic clouds |
| `verify` | tpt-percept-verify | proptest strategies, invariant checkers, Kani harnesses |

`default` = everything except `verify`; `full` = everything. Each feature
can be toggled independently for minimal builds.

```toml
[dependencies]
tpt-perception = { version = "0.1", default-features = false, features = ["register", "slam"] }
```

Part of [tpt-perception](../../). Dual-licensed MIT OR Apache-2.0.
